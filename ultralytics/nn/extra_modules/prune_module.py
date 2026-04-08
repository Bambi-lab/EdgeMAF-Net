import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..modules.conv import Conv, DWConv, RepConv, GhostConv, autopad
from ..modules.block import *
from ..modules.block import Bottleneck_Rep1
from ultralytics.nn.backbone.SwinTransformer import SwinTransformerBlock, Mlp, window_partition

class CSP_MFB_infer(nn.Module):
    # CSP Bottleneck with 2 convolutions For Infer
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c1, self.c2 = 0, 0
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c1, self.c2), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class CSP_MFB_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C2_Rep1_v2(nn.Module):
    # CSP Bottleneck with 2 convolutions
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck_Rep1(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        return self.cv2(torch.cat((self.m(y[0]), y[1]), 1))
    

########################################################################

class AdaptiveNoiseSuppression(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 小的卷积核（3x3）捕捉局部信息
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # 非线性激活函数（ReLU）
        self.relu = nn.ReLU()
        # 全连接层输出权重向量
        self.fc = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        # 卷积操作
        local_features = self.conv(x)
        # 激活函数
        local_features = self.relu(local_features)
        # 全局平均池化
        pooled_features = nn.functional.adaptive_avg_pool2d(local_features, (1, 1)).squeeze(-1).squeeze(-1)
        # 全连接层得到权重向量
        weights = self.fc(pooled_features)
        # 调整每个通道的滤波强度
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        output = x * weights
        return output

class PartiallyTransformerBlock(nn.Module):
    def __init__(self, c, tcr, num_heads=4, window_size=7, mlp_ratio=4., drop=0., attn_drop=0., drop_path=0., shortcut=True):
        super().__init__()
        self.t_ch = int(c * tcr)  # Transformer 分支通道数
        self.c_ch = c - self.t_ch  # CNN 分支通道数
        
        self.noise_suppression = nn.ModuleList([
            AdaptiveNoiseSuppression(self.c_ch),
            AdaptiveNoiseSuppression(self.c_ch)
        ])

        self.c_b = nn.ModuleList([
            Bottleneck(self.c_ch, self.c_ch, shortcut=shortcut),
            Bottleneck(self.c_ch, self.c_ch, shortcut=shortcut)
        ])

        self.t_b = nn.ModuleList([
            SwinTransformerBlock(
                dim=self.t_ch,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm
            ),
            SwinTransformerBlock(
                dim=self.t_ch,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                norm_layer=nn.LayerNorm
            )
        ])
        
        self.fuse = MultiScaleGatedAttn(dims=[self.c_ch, self.t_ch], out_dim=c)
        self.window_size = window_size
    
    def forward(self, x):
        B, C, H, W = x.shape
        cnn_branch, transformer_branch = x.split((self.c_ch, self.t_ch), 1)
        
        for i in range(len(self.c_b)):

            cnn_branch = self.noise_suppression[i](cnn_branch)
            cnn_branch = self.c_b[i](cnn_branch)

        transformer_branch = transformer_branch.flatten(2).transpose(1, 2)  # (B, H*W, t_ch)

        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size // 2),
                    slice(-self.window_size // 2, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.window_size // 2),
                    slice(-self.window_size // 2, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        self.t_b[0].H, self.t_b[0].W = H, W
        transformer_branch = self.t_b[0](transformer_branch, None)
        self.t_b[1].H, self.t_b[1].W = H, W
        transformer_branch = self.t_b[1](transformer_branch, attn_mask)

        transformer_branch = transformer_branch.view(B, H, W, self.t_ch).permute(0, 3, 1, 2)

        return self.fuse([cnn_branch, transformer_branch])

class CSP_PTB(nn.Module):
    """CSP-PTB(Partially Transformer Block)."""

    def __init__(self, c1, c2, n=1, tcr=0.25, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(PartiallyTransformerBlock(self.c, tcr, shortcut=shortcut) for _ in range(n))

    def forward(self, x):
        """Forward pass through CSP_MFB layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

def num_trainable_params(model):
    nums = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    return nums

class GlobalExtraction(nn.Module):

  def __init__(self, dim):  # 添加dim参数
    super().__init__()
    self.avgpool = self.globalavgchannelpool
    self.maxpool = self.globalmaxchannelpool
    self.proj = nn.Sequential(
        nn.Conv2d(2, dim, 1, 1),  # 输出通道调整为dim
        nn.BatchNorm2d(dim)
    )
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def globalavgchannelpool(self, x):
    x = x.mean(1, keepdim = True)
    return x

  def globalmaxchannelpool(self, x):
    x = x.max(dim = 1, keepdim=True)[0]
    return x

  def forward(self, x):
    x_ = x.clone()
    x = self.avgpool(x)
    x2 = self.maxpool(x_)

    cat = torch.cat((x,x2), dim = 1)

    proj = self.proj(cat)
    return proj
  
class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride= 1, padding = 1)
    
    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge

class MutilScaleEdgeInformationSelect(nn.Module):
    def __init__(self, inc):
        super().__init__()
        
        # 固定 bins 为 [3, 6, 9, 12]
        bins = [3, 6, 9, 12]
        
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),                    # 自适应平均池化到指定尺寸
                Conv(inc, inc // len(bins), 1),              # 1x1卷积，通道缩减
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))  # 深度卷积
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))  # 为每个尺度创建边缘增强器
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)                  # 局部卷积
        self.dsm = DualDomainSelectionMechanism(inc * 2)     # 双域选择机制
        self.final_conv = Conv(inc * 2, inc)                # 最终卷积，通道数恢复
    
    def forward(self, x):
        x_size = x.size()                                    # 保存输入尺寸
        out = [self.local_conv(x)]                           # 局部特征
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(self.dsm(torch.cat(out, 1)))  # 拼接、选择、输出
    
class DualDomainSelectionMechanism(nn.Module):

    def __init__(self, channel) -> None:
        super().__init__()
        pyramid = 1
        self.spatial_gate = DSM_SpatialGate(channel)
        layers = [DSM_LocalAttention(channel, p=i) for i in range(pyramid-1,-1,-1)]
        self.local_attention = nn.Sequential(*layers)
        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))
        
    def forward(self, x):
        out = self.spatial_gate(x)
        out = self.local_attention(out)
        return self.a*out + self.b*x

class DSM_SpatialGate(nn.Module):
    def __init__(self, channel):
        super(DSM_SpatialGate, self).__init__()
        kernel_size = 3
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size, act=False)
        self.dw1 = nn.Sequential(
            Conv(channel, channel, 5, s=1, d=2, g=channel, act=nn.GELU()),
            Conv(channel, channel, 7, s=1, d=3, g=channel, act=nn.GELU())
        )
        self.dw2 = Conv(channel, channel, kernel_size, g=channel, act=nn.GELU())

    def forward(self, x):
        out = self.compress(x)
        out = self.spatial(out)
        out = self.dw1(x) * out + self.dw2(x)
        return out
    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class DSM_LocalAttention(nn.Module):
    def __init__(self, channel, p) -> None:
        super().__init__()
        self.channel = channel

        self.num_patch = 2 ** p
        self.sig = nn.Sigmoid()

        self.a = nn.Parameter(torch.zeros(channel,1,1))
        self.b = nn.Parameter(torch.ones(channel,1,1))

    def forward(self, x):
        out = x - torch.mean(x, dim=(2,3), keepdim=True)
        return self.a*out*x + self.b*x

class MultiscaleFusion(nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.local= MutilScaleEdgeInformationSelect(dim)
    self.global_ = GlobalExtraction(dim)
    self.bn = nn.BatchNorm2d(dim)
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x, g,):
    x = self.local(x)
    g = self.global_(g)

    fuse = self.bn(x + g)
    return fuse

class AdjustedCSMHSA(nn.Module):

    def __init__(self, dim, heads=8):
        super(AdjustedCSMHSA, self).__init__()
        self.dim = dim

        self.interaction_conv = nn.Sequential(

            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),

            nn.Conv2d(dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, 1),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, interaction, x_):
        interaction_feat = self.interaction_conv(interaction)
        fused = self.fusion(torch.cat([interaction_feat, x_], dim=1))
        attn = self.channel_attn(fused)
        out = fused * attn + x_      
        return out

class MultiScaleGatedAttn(nn.Module):
    def __init__(self, dims, out_dim):  # dims 是输入通道数列表，out_dim 是目标输出通道数
        super().__init__()
        dim = min(dims)  # 内部统一通道数
        self.conv1 = Conv(dims[0], dim) if dims[0] != dim else nn.Identity()
        self.conv2 = Conv(dims[1], dim) if dims[1] != dim else nn.Identity()
        self.multi = MultiscaleFusion(dim)
        self.selection = nn.Conv2d(dim, 2, 1)
        self.bn_2 = nn.BatchNorm2d(dim)
        self.conv_block = nn.Sequential(nn.Conv2d(dim, dim, 1))
        self.adjusted_csmhsa = AdjustedCSMHSA(dim=dim, heads=8)
        self.proj = nn.Conv2d(dim, out_dim, 1)  # 添加投影层，输出通道数为 out_dim

    def forward(self, inputs):
        x, g = inputs
        if x.size(1) != g.size(1):
            x = self.conv1(x)
            g = self.conv2(g)
        x_ = x.clone()
        g_ = g.clone()
        multi = self.multi(x, g)
        multi = self.selection(multi)
        attention_weights = F.softmax(multi, dim=1)
        A, B = attention_weights.split(1, dim=1)
        x_att = A.expand_as(x_) * x_ + x_
        g_att = B.expand_as(g_) * g_ + g_
        x_sig = torch.sigmoid(x_att)
        g_att_2 = x_sig * g_att
        g_sig = torch.sigmoid(g_att)
        x_att_2 = g_sig * x_att
        interaction = x_att_2 * g_att_2
        y = self.adjusted_csmhsa(interaction, x_)
        y = self.conv_block(y)
        y = self.bn_2(y)
        return self.proj(y)  # 通过投影层输出
