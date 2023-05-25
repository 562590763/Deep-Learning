import copy
import torch
import torch.nn as nn
from utils.activation import Swish
from model.PoolUNet.networks.resnet import ResNetV2
from model.PoolUNet.networks.dcnn import ConvOffset2D
from utils.register import register, register_model
from model.PoolUNet.networks.token_mixer import Shift, Identical
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

__all__ = ['poolunet_base', 'poolunet_large']


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolUNet
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of Channel MLP with FFN.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop_rate=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, (1, 1))
        self.act = act_layer()
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, (1, 1))
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        # x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = x.permute(0, 3, 1, 2)

        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, in_channels=3, embed_dim=768, patch_size=16, stride=16, padding=0, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, img_size, in_channels=3, patch_size=16, hidden_size=768, depths=None,
                 hybrid=False, drop_rate=0., dpr=[], width_factor=1, act_layer=nn.ReLU):
        super(EncoderBlock, self).__init__()
        self.img_size = img_size
        self.hybrid = hybrid
        if self.hybrid:  # ResNet V2
            self.hybrid_model = ResNetV2(in_channels, block_units=depths, width_factor=width_factor,
                                         dpr=dpr, act_layer=act_layer)
            patch_size = patch_size // 16
            in_channels = self.hybrid_model.width * 16
        self.patch_embedding = PatchEmbed(in_channels=in_channels, embed_dim=hidden_size,
                                          patch_size=patch_size, stride=patch_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embedding(x)
        x = self.dropout(x)

        return x, features


class BottleneckBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4., drop_rate=0., drop_path=0., act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        # self.token_mixer = Shift()
        # self.token_mixer = Identical()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_rate=drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
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

        return x


class DeformableBottleneckModule(nn.Module):
    def __init__(self, cr_ratio, skip_channels, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        embed_dim = skip_channels // cr_ratio
        self.reduction = nn.Conv2d(skip_channels, embed_dim, kernel_size=(1, 1))
        self.dcnn = ConvOffset2D(embed_dim)
        self.restoration = nn.Conv2d(embed_dim, skip_channels, kernel_size=(1, 1))
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(skip_channels)
        self.act = act_layer()

    def forward(self, x):
        deformable_x = self.act(self.norm1(self.reduction(x)))
        deformable_x = self.act(self.norm1(self.dcnn(deformable_x)))
        return self.act(self.norm2(self.restoration(deformable_x)) + x)


class Conv2dAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=(stride, stride), padding=padding)
        act = act_layer()
        bn = norm_layer(out_channels)
        super(Conv2dAct, self).__init__(conv, bn, act)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, scale_factor=2,
                 deformable=False, cr_ratio=4, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.deformable = deformable
        if deformable:
            self.dbm = DeformableBottleneckModule(cr_ratio, skip_channels, act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = Conv2dAct(in_channels + skip_channels, out_channels, kernel_size=3,
                               padding=1, act_layer=act_layer, norm_layer=norm_layer)
        self.conv2 = Conv2dAct(out_channels, out_channels, kernel_size=3,
                               padding=1, act_layer=act_layer, norm_layer=norm_layer)
        self.up = nn.UpsamplingBilinear2d(scale_factor=scale_factor)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if self.deformable:
                skip = self.dbm(skip)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


@register_model
class PoolUNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, patch_size=16, pool_size=3,
                 depths=(3, 4, 9, 4), hidden_size=768, mlp_ratios=4., n_skip=3, hybrid=False,
                 drop_rate=0.1, drop_path_rate=0., skip_channels=[512, 256, 64, 16],
                 deformable=False, cr_ratio=4, decoder_channels=[256, 128, 64, 16],
                 width_factor=1, use_layer_scale=True, norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, layer_scale_init_value=1e-5, task="classification"):
        super(PoolUNet, self).__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.token_dim = decoder_channels[-1]
        self.task = task

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build encoder layers
        self.encoder = EncoderBlock(img_size=img_size, in_channels=in_chans, patch_size=patch_size,
                                    hidden_size=hidden_size, depths=depths[:-1], hybrid=hybrid, drop_rate=drop_rate,
                                    dpr=dpr[:-depths[-1]], width_factor=width_factor, act_layer=act_layer)

        # build bottleneck layers
        self.layer = nn.ModuleList()
        for i in range(depths[-1]):
            layer = BottleneckBlock(dim=hidden_size, pool_size=pool_size, mlp_ratio=mlp_ratios, drop_rate=drop_rate,
                                    drop_path=dpr[i-depths[-1]], norm_layer=norm_layer, act_layer=act_layer,
                                    use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value)
            self.layer.append(copy.deepcopy(layer))
        self.bottleneck_norm = norm_layer(hidden_size)

        # build skip connection
        self.n_skip = n_skip
        self.skip_channels = skip_channels
        for i in range(4 - n_skip):  # re-select the skip channels according to n_skip
            self.skip_channels[3 - i] = 0
        if not hybrid:
            self.skip_channels = [0, 0, 0, 0]

        # build decoder layers
        head_channels = 512
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        self.conv_more = Conv2dAct(hidden_size, head_channels, kernel_size=3,
                                   padding=1, act_layer=act_layer, norm_layer=norm_layer)
        self.decoder = nn.ModuleList([DecoderBlock(in_channels[i], out_channels[i], self.skip_channels[i],
                                                   scale_factor=patch_size // 8 if i == 0 else 2,
                                                   deformable=deformable, cr_ratio=cr_ratio,
                                                   act_layer=act_layer, norm_layer=norm_layer)
                                      for i in range(len(in_channels))])

        # classification head
        self.classification_head = nn.Linear(hidden_size, num_classes) if num_classes > 0 else nn.Identity()

        # segmentation head
        self.segmentation_head = SegmentationHead(in_channels=self.token_dim, out_channels=num_classes,
                                                  upsampling=1) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return {'classification': self.classification_head, 'segmentation': self.segmentation_head}

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.classification_head = nn.Linear(self.hidden_size, num_classes) if num_classes > 0 else nn.Identity()
        self.segmentation_head = SegmentationHead(in_channels=self.token_dim, out_channels=num_classes,
                                                  upsampling=1) if num_classes > 0 else nn.Identity()

    # Encoder and Bottleneck
    def forward_features(self, x):
        x, features = self.encoder(x)
        for bottleneck_block in self.layer:
            x = bottleneck_block(x)
        x = self.bottleneck_norm(x)  # (B, C, H, W)
        return x, features

    # Decoder and Skip Connection
    def forward_up_features(self, x, features=None):
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.decoder):
            if features is not None:
                skip = features[i] if (i < self.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.forward_features(x)
        if self.task == 'segmentation':
            # otuput features of four stages for dense prediction
            x = self.forward_up_features(x, features)
            logits = self.segmentation_head(x)
            return logits
        else:
            # for image classification
            cls_out = self.classification_head(x.mean(-2).squeeze(1))
            return cls_out


@register
def poolunet_base(args):
    depths = [2, 4, 6, 8]
    decoder_channels = [256, 128, 64, 16]
    skip_channels = [512, 256, 64, 16]
    hybrid = True
    deformable = False
    cr_ratio = 4
    act_layer = nn.ReLU
    model = PoolUNet(img_size=args.img_size, in_chans=3, num_classes=args.num_classes, patch_size=args.patch_size,
                     pool_size=args.pool_size, depths=depths, hidden_size=768, mlp_ratios=4., n_skip=args.n_skip,
                     drop_rate=args.drop_rate, drop_path_rate=args.drop_path, skip_channels=skip_channels,
                     deformable=deformable, cr_ratio=cr_ratio, decoder_channels=decoder_channels, width_factor=1,
                     use_layer_scale=args.use_layer_scale, act_layer=act_layer, norm_layer=nn.BatchNorm2d,
                     layer_scale_init_value=args.layer_scale_init_value, task=args.task, hybrid=hybrid)

    return model


@register
def poolunet_large(args):
    depths = [3, 4, 9, 16]
    decoder_channels = [256, 128, 64, 16]
    skip_channels = [512, 256, 64, 16]
    hybrid = True
    deformable = False
    cr_ratio = 4
    act_layer = nn.ReLU
    model = PoolUNet(img_size=args.img_size, in_chans=3, num_classes=args.num_classes, patch_size=args.patch_size,
                     pool_size=args.pool_size, depths=depths, hidden_size=1024, mlp_ratios=4., n_skip=args.n_skip,
                     drop_rate=args.drop_rate, drop_path_rate=args.drop_path, skip_channels=skip_channels,
                     deformable=deformable, cr_ratio=cr_ratio, decoder_channels=decoder_channels, width_factor=1,
                     use_layer_scale=args.use_layer_scale, act_layer=act_layer, norm_layer=nn.BatchNorm2d,
                     layer_scale_init_value=args.layer_scale_init_value, task=args.task, hybrid=hybrid)

    return model
