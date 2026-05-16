import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.helpers import named_apply
from functools import partial
from timm.models.layers import trunc_normal_tf_
import math
"""
各种尺寸的都适用，不管是base还是large
## ConvNeXtV2
base是	[128, 256, 512, 1024]
large是 [192, 384, 768, 1536]
"""
def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = self.in_planes // ratio
        if self.out_planes == None:
            self.out_planes = in_planes
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.activation = act_layer(activation, inplace=True)

        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)

        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        max_pool_out = self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x0 = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x) * x0

#MKDC 多核深度卷积
class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size, stride, kernel_size // 2,
                          groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        # Apply the convolution layers in a loop
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if self.dw_parallel == False:
                x = x + dw_out
        # You can return outputs based on what you intend to do with them
        # For example, you could concatenate or add them; here, we just return the list
        return outputs
# MKIR
class MultiKernelInvertedResidualBlock(nn.Module):
    """
    inverted residual block used in MobileNetV2
    """

    def __init__(self, in_c, out_c, stride, expansion_factor=2, dw_parallel=True, add=True, kernel_sizes=[1, 3, 5],
                 activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        # check stride value
        assert stride in [1, 2]
        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        # Skip connection if stride is 1
        self.use_skip_connection = True if self.stride == 1 else False

        # expansion factor or t as mentioned in the paper
        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )
        self.multi_scale_dwconv = MultiKernelDepthwiseConv(self.ex_c, self.kernel_sizes, self.stride, activation,
                                                           dw_parallel=dw_parallel)

        if self.add == True:
            self.combined_channels = self.ex_c * 1
        else:
            self.combined_channels = self.ex_c * self.n_scales
        self.pconv2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),  #
            nn.BatchNorm2d(self.out_c),
        )
        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)
        if self.add == True:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)
        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out
class ConvBlockResPath(nn.Module):
    def __init__(self, num_filters, kernel_size, padding="same", act=True):
        super(ConvBlockResPath, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = 3
        self.padding = padding
        self.act = act
        self.depthwise_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=self.kernel_size, #改过
            stride=1,
            padding=1, #改过
            groups=num_filters,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.pointwise_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

        if act:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        if self.act:
            x = self.activation(x)

        x = self.pointwise_conv(x)
        x = self.bn2(x)
        if self.act:
            x = self.activation(x)
        return x


class ResPath(nn.Module):
    def __init__(self, num_filters, length):
        super(ResPath, self).__init__()
        self.num_filters = num_filters
        self.length = length

        self.res_layers = nn.ModuleList()
        for i in range(length):
            self.res_layers.append(nn.ModuleList([
                ConvBlockResPath(num_filters, 3, padding="same", act=False),
                ConvBlockResPath(num_filters, 1, padding="same", act=False),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(num_filters)
            ]))

    def forward(self, x):
        for i in range(self.length):
            x0 = x
            conv_3x3, conv_1x1, leaky_relu, bn = self.res_layers[i]

            x1 = conv_3x3(x0)
            sc = conv_1x1(x0)

            x = x1 + sc
            x = leaky_relu(x)
            x = bn(x)

        return x
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1):
        super(ConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate,
                                   dilation=rate, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
class LinearAttention(nn.Module):
    def __init__(self, in_channels):
        super(LinearAttention, self).__init__()
        self.keys = nn.Linear(in_channels, in_channels)
        self.queries = nn.Linear(in_channels, in_channels)
        self.values = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        # 使用线性注意力近似，避免计算 (H*W, H*W) 的大矩阵
        # 标准注意力：softmax(QK^T)V -> 复杂度 O((HW)^2)
        # 线性注意力：Q(K^TV) -> 复杂度 O(HW*C^2)，其中 C<<HW

        # 对 queries 和 keys 应用核函数映射（这里用 ReLU 作为核函数）
        keys = F.relu(keys)
        queries = F.relu(queries)

        # 归一化
        keys_sum = keys.sum(dim=1, keepdim=True) + 1e-6
        queries_sum = queries.sum(dim=-1, keepdim=True) + 1e-6

        # 计算 K^T V (C, C) - 这是一个很小的矩阵
        kv = torch.bmm(keys.transpose(-2, -1), values)

        # 计算 Q (K^T V)
        out = torch.bmm(queries, kv)

        # 归一化因子
        normalizer = torch.bmm(queries, keys_sum.transpose(-2, -1))
        out = out / (normalizer + 1e-6)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class ConvNeXtV2(nn.Module):
    def __init__(self, backbone, num_classes=21):
        super(ConvNeXtV2, self).__init__()

        # 加载预训练的模型
        self.backbone = backbone

        # 获取特征提取层
        self.stem = backbone.stem
        self.stages = backbone.stages

        # 通过实际前向传播确定各层的输出通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)

            # 获取stem输出
            stem_out = self.stem(dummy_input)

            # 获取各stage输出
            stage1_out = self.stages[0](stem_out)
            stage2_out = self.stages[1](stage1_out)
            stage3_out = self.stages[2](stage2_out)
            stage4_out = self.stages[3](stage3_out)

            # 获取各层的通道数,以base为例
            self.feature_channels = [
                int(stage1_out.size(1)/4), # 32
                int(stage1_out.size(1)/2), # 64
                stage1_out.size(1),  # stage1 输出通道数 128
                stage2_out.size(1),  # stage2 输出通道数 256
                stage3_out.size(1),  # stage3 输出通道数 512
                stage4_out.size(1)  # stage4 输出通道数 1024
            ]

        # 上采样模块 以base为例，upsample_blocks是[1024->512, 512->256, 256->128,128->64,64->32]一共5个
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1, 0, -1): # 倒序
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.feature_channels[i], self.feature_channels[i - 1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(self.feature_channels[i - 1]),
                    nn.ReLU(inplace=True)
                )
            )
        # RLAB
        self.RLABS = nn.ModuleList()
        for i in range(len(self.feature_channels) - 2, 0, -1):
            self.RLABS.append(
                nn.Sequential(
                    ResPath(self.feature_channels[i],i),
                    ConvBlock(self.feature_channels[i]*2, self.feature_channels[i]),
                    LinearAttention(self.feature_channels[i])
                )
            )
        #MCSAB
        self.CA5 = ChannelAttention(self.feature_channels[5], ratio=16)
        self.CA4 = ChannelAttention(self.feature_channels[4], ratio=16)
        self.CA3 = ChannelAttention(self.feature_channels[3], ratio=16)
        self.CA2 = ChannelAttention(self.feature_channels[2], ratio=16)
        self.CA1 = ChannelAttention(self.feature_channels[1], ratio=8)
        self.CA0 = ChannelAttention(self.feature_channels[0], ratio=4)
        self.SA = SpatialAttention()
        self.MCSAB5 = MultiKernelInvertedResidualBlock(self.feature_channels[5], self.feature_channels[5],stride=1)
        self.MCSAB4 = MultiKernelInvertedResidualBlock(self.feature_channels[4], self.feature_channels[4],stride=1)
        self.MCSAB3 = MultiKernelInvertedResidualBlock(self.feature_channels[3], self.feature_channels[3], stride=1)
        self.MCSAB2 = MultiKernelInvertedResidualBlock(self.feature_channels[2], self.feature_channels[2], stride=1)
        self.MCSAB1 = MultiKernelInvertedResidualBlock(self.feature_channels[1], self.feature_channels[1], stride=1)
        self.MCSAB0 = MultiKernelInvertedResidualBlock(self.feature_channels[0], self.feature_channels[0], stride=1)
        # 特征融合卷积层 以base为例，fusion_convs[1024->512, 512->256, 256->128,128->64,64->32]一共5个
        self.fusion_convs = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1, 0, -1):
            self.fusion_convs.append(
                nn.Conv2d(self.feature_channels[i], self.feature_channels[i-1], kernel_size=1)
            )

        self.final = nn.Conv2d(self.feature_channels[0], num_classes, kernel_size=1)
    def forward(self, x):
        # 编码器部分 - 提取多尺度特征
        features = []

        # stem 层
        x = self.stem(x)
        features.append(x)

        # stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)

        # 解码器部分 - 级联上采样和注意力融合
        decoder_feature = features[-1]  # 使用最深层特征开始解码

        # Level 5: 从最深层次(1024通道)上采样到512通道
        MCSAB5_feature = self.MCSAB5(self.SA(self.CA5(decoder_feature)))
        LUB5_feature = self.upsample_blocks[0](MCSAB5_feature)
        skip_feat_adjusted = features[3]  # stage3输出
        ResPath_feature = self.RLABS[0][0](skip_feat_adjusted)
        fused_feature = torch.cat([LUB5_feature, ResPath_feature], dim=1)
        convblock_feature = self.RLABS[0][1](fused_feature)
        linear_feature = self.RLABS[0][2](convblock_feature)
        RLAB_feature = linear_feature + LUB5_feature
        decoder_feature = RLAB_feature

        # Level 4: 从512通道上采样到256通道
        MCSAB4_feature = self.MCSAB4(self.SA(self.CA4(decoder_feature)))
        LUB4_feature = self.upsample_blocks[1](MCSAB4_feature)
        skip_feat_adjusted = features[2]  # stage2输出
        ResPath_feature = self.RLABS[1][0](skip_feat_adjusted)
        fused_feature = torch.cat([LUB4_feature, ResPath_feature], dim=1)
        convblock_feature = self.RLABS[1][1](fused_feature)
        linear_feature = self.RLABS[1][2](convblock_feature)
        RLAB_feature = linear_feature + LUB4_feature
        decoder_feature = RLAB_feature

        # Level 3: 从256通道上采样到128通道
        MCSAB3_feature = self.MCSAB3(self.SA(self.CA3(decoder_feature)))
        LUB3_feature = self.upsample_blocks[2](MCSAB3_feature)
        skip_feat_adjusted = features[1]  # stage1输出
        ResPath_feature = self.RLABS[2][0](skip_feat_adjusted)
        fused_feature = torch.cat([LUB3_feature, ResPath_feature], dim=1)
        convblock_feature = self.RLABS[2][1](fused_feature)
        linear_feature = self.RLABS[2][2](convblock_feature)
        RLAB_feature = linear_feature + LUB3_feature
        decoder_feature = RLAB_feature

        # Level 2: 从128通道上采样到64通道 (stem层需要特殊处理)
        MCSAB2_feature = self.MCSAB2(self.SA(self.CA2(decoder_feature)))
        LUB2_feature = self.upsample_blocks[3](MCSAB2_feature)
        skip_feat_adjusted = features[0]  # stem输出
        skip_feat_adjusted = self.upsample_blocks[3](skip_feat_adjusted)  # stem层尺度不一致，需要额外上采样
        ResPath_feature = self.RLABS[3][0](skip_feat_adjusted)
        fused_feature = torch.cat([LUB2_feature, ResPath_feature], dim=1)
        convblock_feature = self.RLABS[3][1](fused_feature)
        linear_feature = self.RLABS[3][2](convblock_feature)
        RLAB_feature = linear_feature + LUB2_feature
        decoder_feature = RLAB_feature

        # Level 1: 从64通道上采样到32通道 (stem层需要特殊处理)
        MCSAB1_feature = self.MCSAB1(self.SA(self.CA1(decoder_feature)))
        LUB1_feature = self.upsample_blocks[-1](MCSAB1_feature)
        # 最终预测,还差两次反卷积，从6*128*128*128->6*64*256*256->6*3*512*512
        MCSAB0_feature = self.MCSAB0(self.SA(self.CA0(LUB1_feature)))
        output = self.final(MCSAB0_feature)

        return output

        return output

# if __name__ == '__main__':
#     model = timm.create_model('convnextv2_base', pretrained=False)
#     model = ConvNeXtV2(model)
#     print(model)