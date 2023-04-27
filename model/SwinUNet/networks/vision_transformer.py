import copy
import torch
import torch.nn as nn
from model.SwinUNet.networks.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys


class SwinUnet(nn.Module):
    # 参数:41.3M FLOPs:8.7G
    def __init__(self, args):
        super(SwinUnet, self).__init__()
        self.args = args
        self.swin_unet = SwinTransformerSys(img_size=args.img_size,
                                            patch_size=args.patch_size,
                                            in_chans=3,
                                            num_classes=args.num_classes,
                                            embed_dim=args.embed_dim,
                                            depths=[2, 2, 6, 2],
                                            num_heads=[3, 6, 12, 24],
                                            window_size=args.window_size,
                                            mlp_ratio=args.mlp_ratio,
                                            qkv_bias=True,
                                            qk_scale=None,
                                            drop_rate=args.drop_rate,
                                            drop_path_rate=args.drop_path,
                                            ape=False,
                                            patch_norm=True,
                                            use_checkpoint=False)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.swin_unet(x)
        return logits

    def load_from(self, path_state_dict):
        if path_state_dict is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(path_state_dict, map_location=device)
            if "model" not in pretrained_dict:
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
