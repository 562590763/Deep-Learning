import torch
import numpy as np
from torch import nn
from timm.models.layers import trunc_normal_
from utils.register import register, register_model
from model.FTN.networks.spt import SpatialTransformer
from model.FTN.networks.transformer_block import Block


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1, 'input_size': (3, 512, 512), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'FTN_4': _cfg(),
    'FTN_8': _cfg(),
    'FTN_12': _cfg(),
}


__all__ = ['FTN_4', 'FTN_8', 'FTN_12']

sigmoid = nn.Sigmoid()


class FTN_encoder(nn.Module):
    """
    FTN encoding module
    """

    def __init__(self, img_size=512, in_chans=3, embed_dim=512, token_dim=64):
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim
        self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
        self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.swt_0 = nn.Sequential(nn.Conv2d(in_chans, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)))
        self.swt_1 = nn.Sequential(nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))
        self.swt_2 = nn.Sequential(nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

        self.attention1 = SpatialTransformer(dim=token_dim, in_dim=token_dim, num_heads=1, mlp_ratio=1.0, attn_drop=0.0,
                                             drop_path=0, drop=0.)
        self.attention2 = SpatialTransformer(dim=token_dim, in_dim=token_dim, num_heads=1, mlp_ratio=1.0, attn_drop=0.0,
                                             drop_path=0, drop=0.)

        self.num_patches = (img_size // (4 * 2 * 2)) * (
                img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        encoder_list = []
        B = x.shape[0]
        # step0 encoder 0: apply sliding window tokenization for spatial dimension reduction and local information
        # then utilize Spatial Transformer to aggregate global information
        x = self.swt_0(x).view(B, self.token_dim, -1).transpose(1, 2)
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        # Step1 encoder 1: soft split
        encoder_list.append(x)
        x = self.swt_1(x).view(B, self.token_dim, -1).transpose(1, 2)
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        encoder_list.append(x)

        # Step2: apply sliding window tokenization to
        # simultaneously reduce the spatial dimension and increase channels number
        x = self.swt_2(x).view(B, self.embed_dim, -1).transpose(1, 2)

        return x, encoder_list


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class FTN_decoder(nn.Module):
    """
    FTN decoding module
    """

    def __init__(self, img_size=512, embed_dim=512, token_dim=64):
        super().__init__()
        self.token_dim = token_dim
        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.token_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)))

        self.attention1 = SpatialTransformer(dim=token_dim // 2, in_dim=token_dim, num_heads=1, mlp_ratio=1.0,
                                             attn_drop=0.0, drop_path=0, drop=0.)
        self.attention2 = SpatialTransformer(dim=token_dim // 2, in_dim=token_dim, num_heads=1, mlp_ratio=1.0,
                                             attn_drop=0.0, drop_path=0, drop=0.)

        self.conv1 = DoubleConv(2 * token_dim, token_dim)
        self.conv2 = DoubleConv(2 * token_dim, token_dim)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.num_patches = (img_size // (4 * 2 * 2)) * (
                img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x, encoder_list):
        # step0 decoder 0
        x = self.proj(x)
        B, C, new_HW, new_HW = x.shape

        x = self.conv1(torch.cat([encoder_list[-1], self.upsample(x)], 1)).view(B, self.token_dim, -1).transpose(1, 2)
        x = self.attention1(x)

        # step1 decoder 1
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        x = self.conv2(torch.cat([encoder_list[-2], self.upsample(x)], 1)).view(B, self.token_dim, -1).transpose(1, 2)
        x = self.attention2(x)

        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


@register_model
class FTN(nn.Module):
    def __init__(self, img_size=448, in_chans=3, num_classes=9, embed_dim=512, depth=12,
                 num_heads=12, mlp_ratio=2., qkv_bias=False, qk_scale=256 ** -0.5, drop_rate=0, attn_drop_rate=0,
                 drop_path_rate=0, norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.token_dim = token_dim
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = FTN_encoder(
            img_size=img_size, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, in_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.conv_list = nn.ModuleList([
            nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            for _ in range(depth)])

        self.decoder = FTN_decoder(
            img_size=img_size, embed_dim=embed_dim, token_dim=token_dim)

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=4)
        self.pre_head_norm = nn.BatchNorm2d(token_dim)
        self.head = SegmentationHead(in_channels=self.token_dim,
                                     out_channels=self.num_classes, upsampling=4)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # Dim_reduce
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = SegmentationHead(in_channels=self.token_dim,
                                     out_channels=self.num_classes, upsampling=4) \
            if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x, encoder_lsit = self.tokens_to_token(x)
        for blk, conv in zip(self.blocks, self.conv_list):
            B, new_HW, C = x.shape
            x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
            x = conv(x) + x
            x = x.view(B, C, -1).transpose(1, 2)
            x = blk(x)

        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        return x, encoder_lsit

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, encoder_lsit = self.forward_features(x)
        x = self.decoder(x, encoder_lsit)
        x = self.pre_head_norm(x)
        out = self.head(x)
        return out


@register
def FTN_4(args):  # adopt performer for tokens to token
    model = FTN(img_size=args.img_size, embed_dim=256, depth=4, num_heads=2, mlp_ratio=2., token_dim=64,
                drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path)
    model.default_cfg = default_cfgs['FTN_4']
    return model


@register
def FTN_8(args):  # adopt performer for tokens to token
    model = FTN(img_size=args.img_size, embed_dim=384, depth=8, num_heads=3, mlp_ratio=2., token_dim=64,
                drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path)
    model.default_cfg = default_cfgs['FTN_8']
    return model


@register
def FTN_12(args):  # adopt performer for tokens to token
    model = FTN(img_size=args.img_size, embed_dim=512, depth=12, num_heads=4, mlp_ratio=2., token_dim=64,
                drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate, drop_path_rate=args.drop_path)
    model.default_cfg = default_cfgs['FTN_12']
    return model
