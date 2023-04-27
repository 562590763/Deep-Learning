import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.register import register, register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = ['cntunet_tiny', 'cntunet_small', 'cntunet_medium', 'cntunet_large']


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Mlp2(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, (1, 1))
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, (1, 1))
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=(sr_ratio, sr_ratio), stride=(sr_ratio, sr_ratio))
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BottleneckBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp2(in_features=dim, hidden_features=mlp_hidden_dim,
                        act_layer=act_layer, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):

        B = x.shape[0]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B L C -> B C H W
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, -1, H * W).permute(0, 2, 1).contiguous()  # B C H W -> B L C

        return x


class PrePatchEmbed(nn.Module):
    """
    Image to Visual Word Embedding
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, norm_layer=GroupNorm):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.conv = nn.Conv2d(in_chans, in_chans, kernel_size=(3, 3), padding=1, stride=(1, 1))
        self.norm = norm_layer(in_chans)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.unfold(x)  # B, C*4*4, N
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size)  # B*N, C, p, p
        x = self.conv(x)
        x = x.transpose(1, 2).reshape(B, C, int(math.sqrt(self.num_patches)), -1,
                                      *self.patch_size).transpose(-3, -2).reshape(B, C, H, W)

        return self.norm(x)


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=64):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PatchExpand(nn.Module):
    def __init__(self, img_size, embed_dim=512, dim_reduction=2, scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.scale = scale
        self.expand = nn.Linear(embed_dim, scale ** 2 // dim_reduction * embed_dim, bias=False)
        self.norm = norm_layer(embed_dim // dim_reduction)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = to_2tuple(self.img_size)
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.scale, p2=self.scale, c=C // (self.scale ** 2))
        x = x.view(B, -1, C // (self.scale ** 2))
        x = self.norm(x)

        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


@register_model
class CntUNet(nn.Module):
    def __init__(self, img_size=224, patch_size=4, pool_size=3,
                 in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 depths=[3, 4, 6, 3], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 act_layer=nn.GELU, norm_layer1=nn.LayerNorm, norm_layer2=GroupNorm,
                 sr_ratios=[8, 4, 2, 1], num_stages=4, use_layer_scale=True,
                 layer_scale_init_value=1e-5, ape=False, task="classification"):
        super().__init__()

        self.ape = ape
        self.task = task
        self.img_size = img_size
        self.patches_resolution = img_size / patch_size
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.num_features = embed_dims[-1]
        self.token_dim = embed_dims[0]

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build encoder and bottleneck layers
        self.pre_patch_embed = PrePatchEmbed(img_size=img_size, patch_size=patch_size,
                                             in_chans=in_chans, norm_layer=norm_layer2)
        cur = 0
        for i in range(self.num_stages):
            # split image into non-overlapping patches
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            # absolute position embedding
            num_patches = patch_embed.num_patches
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            if i == self.num_stages - 1:
                block = nn.ModuleList([BasicBlock(
                    dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                    act_layer=act_layer, norm_layer=norm_layer1, sr_ratio=sr_ratios[i])
                    for j in range(depths[i])])
            else:
                block = nn.ModuleList([BottleneckBlock(
                    dim=embed_dims[i], pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                    drop=drop_rate, drop_path=dpr[cur + j], act_layer=act_layer, norm_layer=norm_layer2,
                    use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"encoder_block{i + 1}", block)

        # build decoder layers
        for i in range(self.num_stages - 1, -1, -1):
            index = self.num_stages - i
            cur -= depths[i]
            patch_expand = PatchExpand(img_size=img_size // (2 ** (i + 2)), embed_dim=embed_dims[i],
                                       dim_reduction=1 if i == 0 else 2,
                                       scale=patch_size if i == 0 else 2, norm_layer=norm_layer1)
            concat_linear = nn.Linear(2 * embed_dims[i], embed_dims[i]) if index > 1 else nn.Identity()

            if index == 1:
                block = nn.Identity()
            else:
                block = nn.ModuleList([BasicBlock(
                    dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                    act_layer=act_layer, norm_layer=norm_layer1, sr_ratio=sr_ratios[i])
                    for j in range(depths[i])])

            setattr(self, f"patch_expand{index}", patch_expand)
            setattr(self, f"concat_linear{index}", concat_linear)
            setattr(self, f"decoder_block{index}", block)

        self.norm = norm_layer1(self.num_features)
        self.pre_head_norm = norm_layer1(self.num_features)

        # classification head
        self.classification_head = nn.Linear(self.num_features, self.num_classes) \
            if self.num_classes > 0 else nn.Identity()

        # segmentation head
        self.segmentation_head = SegmentationHead(in_channels=self.token_dim, out_channels=self.num_classes,
                                                  upsampling=1) if self.num_classes > 0 else nn.Identity()

        # init weights
        for i in range(num_stages):
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def get_classifier(self):
        return {'classification': self.classification_head, 'segmentation': self.segmentation_head}

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classification_head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.segmentation_head = SegmentationHead(in_channels=self.token_dim, out_channels=num_classes,
                                                  upsampling=1) if num_classes > 0 else nn.Identity()

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == patch_embed.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    # Encoder and Bottleneck
    def forward_features(self, x):
        B = x.shape[0]

        x_downsample = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"encoder_block{i + 1}")

            if i == 0:
                x = x + self.pre_patch_embed(x)

            x, (H, W) = patch_embed(x)  # B C H W -> B L C
            if self.ape:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
                x = pos_drop(x + pos_embed)
            x_downsample.append(x)

            for blk in block:
                x = blk(x, H, W)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B L C -> B C H W

        x = self.norm(x)  # B L C

        return x, x_downsample

    # Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):

        for i in range(self.num_stages):
            index = self.num_stages - i + 1
            H, W = to_2tuple(self.img_size // (2 ** index))
            patch_expand = getattr(self, f"patch_expand{i + 1}")
            concat_linear = getattr(self, f"concat_linear{i + 1}")
            block = getattr(self, f"decoder_block{i + 1}")

            if i > 0:
                for blk in block:
                    x = blk(x, H, W)
                x = torch.cat([x, x_downsample[self.num_stages - i - 1]], -1)
                x = concat_linear(x)

            x = patch_expand(x)

        B, L, C = x.shape
        H, W = to_2tuple(self.img_size)
        assert L == H * W, "input features has wrong size"
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # B L C -> B C H W

        return x

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, x_downsample = self.forward_features(x)  # B L C
        if self.task == 'segmentation':
            # otuput features of four stages for dense prediction
            x = self.forward_up_features(x, x_downsample)  # B C H W
            x = self.segmentation_head(x)
            return x
        else:
            x = self.pre_head_norm(x)
            cls_out = self.classification_head(x.mean(-2).squeeze(1))
            # for image classification
            return cls_out


def _conv_filter(state_dict, patch_size=4):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@register
def cntunet_tiny(args):
    # 参数:32.5M FLOPs:4.3G
    depths = [2, 2, 2, 8]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [8, 8, 4, 4]
    num_heads = [1, 2, 4, 8]
    sr_ratios = [8, 4, 2, 1]
    model = CntUNet(img_size=args.img_size, patch_size=args.patch_size, pool_size=args.pool_size,
                    in_chans=3, num_classes=args.num_classes, embed_dims=embed_dims,
                    depths=depths, num_heads=num_heads, mlp_ratios=mlp_ratios,
                    qkv_bias=False, qk_scale=None, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
                    drop_path_rate=args.drop_path, act_layer=nn.GELU, norm_layer1=nn.LayerNorm, norm_layer2=GroupNorm,
                    sr_ratios=sr_ratios, num_stages=args.stages, use_layer_scale=args.use_layer_scale,
                    layer_scale_init_value=args.layer_scale_init_value, task=args.task)
    return model


@register
def cntunet_small(args):
    # 参数:40.5M FLOPs:6.2G
    depths = [2, 4, 6, 8]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [8, 8, 4, 4]
    num_heads = [1, 2, 4, 8]
    sr_ratios = [8, 4, 2, 1]
    model = CntUNet(img_size=args.img_size, patch_size=args.patch_size, pool_size=args.pool_size,
                    in_chans=3, num_classes=args.num_classes, embed_dims=embed_dims,
                    depths=depths, num_heads=num_heads, mlp_ratios=mlp_ratios,
                    qkv_bias=False, qk_scale=None, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
                    drop_path_rate=args.drop_path, act_layer=nn.GELU, norm_layer1=nn.LayerNorm, norm_layer2=GroupNorm,
                    sr_ratios=sr_ratios, num_stages=args.stages, use_layer_scale=args.use_layer_scale,
                    layer_scale_init_value=args.layer_scale_init_value, task=args.task)
    return model


@register
def cntunet_medium(args):
    # 参数:57.6M FLOPs:8.2G
    depths = [3, 5, 8, 12]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [8, 8, 4, 4]
    num_heads = [1, 2, 4, 8]
    sr_ratios = [8, 4, 2, 1]
    model = CntUNet(img_size=args.img_size, patch_size=args.patch_size, pool_size=args.pool_size,
                    in_chans=3, num_classes=args.num_classes, embed_dims=embed_dims,
                    depths=depths, num_heads=num_heads, mlp_ratios=mlp_ratios,
                    qkv_bias=False, qk_scale=None, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
                    drop_path_rate=args.drop_path, act_layer=nn.GELU, norm_layer1=nn.LayerNorm, norm_layer2=GroupNorm,
                    sr_ratios=sr_ratios, num_stages=args.stages, use_layer_scale=args.use_layer_scale,
                    layer_scale_init_value=args.layer_scale_init_value, task=args.task)
    return model


@register
def cntunet_large(args):
    # 参数:42.1M FLOPs:16.4G
    depths = [3, 3, 6]
    embed_dims = [128, 256, 512]
    mlp_ratios = [8, 8, 4]
    num_heads = [2, 4, 8]
    sr_ratios = [8, 4, 2]
    model = CntUNet(img_size=args.img_size, patch_size=args.patch_size, pool_size=args.pool_size,
                    in_chans=3, num_classes=args.num_classes, embed_dims=embed_dims,
                    depths=depths, num_heads=num_heads, mlp_ratios=mlp_ratios,
                    qkv_bias=False, qk_scale=None, drop_rate=args.drop_rate, attn_drop_rate=args.attn_drop_rate,
                    drop_path_rate=args.drop_path, act_layer=nn.GELU, norm_layer1=nn.LayerNorm, norm_layer2=GroupNorm,
                    sr_ratios=sr_ratios, num_stages=3, use_layer_scale=args.use_layer_scale,
                    layer_scale_init_value=args.layer_scale_init_value, task=args.task)
    return model
