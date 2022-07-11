import math
import random
import torch
import torch.nn as nn

from cvlep.CLIPT5.clip import load

from PIL import Image

from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers import PretrainedConfig
from timm.models.vision_transformer import resize_pos_embed
# from transformers import ViTModel, CLIPVisionModel
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


import torchvision
from torchvision.transforms import functional as F


class MinMaxResize(object):
    def __init__(self, min_size, max_size):
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target=None):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is None:
            return image
        target = target.resize(image.size)
        return image, target


class PadToSquare(object):
    def __call__(self, image, target=None):
        imsize = image.size
        max_w = max_h = max(imsize[0], imsize[1])
        h_padding = (max_w - imsize[0]) / 2
        v_padding = (max_h - imsize[1]) / 2
        l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
        t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
        r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
        b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5

        padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

        return torchvision.transforms.functional.pad(image, padding)


def _transform(n_px):
    return Compose([
        # PadToSquare(),
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        # MinMaxResize(*n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def get_vis_encoder(backbone='openai/clip-vit-base-patch32', **kwargs):
    # if 'openai/clip' in backbone:
    #     if 'vit' in backbone:
    #         encoder = CLIPViTEncoder.from_pretrained(
    #             backbone,
    #             )

    #         if kwargs is not None:
    #             for k, v in kwargs.items():
    #                 setattr(encoder.config, k, v)
    #     else:
    if backbone.lower().endswith('RN50'.lower()):
        backbone = 'RN50'
    elif backbone.lower().endswith('RN101'.lower()):
        backbone = 'RN101'
    elif backbone.lower().endswith('RN50x4'.lower()):
        backbone = 'RN50x4'
    encoder = CLIPResNetEncoder(backbone, **kwargs)

    # elif 'google/vit' in backbone:
    #     encoder = ViTVisualEncoder.from_pretrained(
    #         backbone,
    #         add_pooling_layer=False
    #     )

    return encoder


class CLIPResNetEncoder(nn.Module):
    def __init__(self, backbone='RN50x4', image_size=224, adapter_type=None, reduction_factor=1, use_bn=True):

        super().__init__()

        self.model, transform = load(backbone, device='cpu', jit=False, adapter_type=adapter_type, reduction_factor=reduction_factor, use_bn=use_bn)
        del self.model.transformer

        # if backbone == 'RN50x4':
        #     image_size = 288

        self.config = PretrainedConfig(
            image_size=image_size,
            patch_size=32,
            hidden_size=self.model.visual.attnpool.positional_embedding.shape[-1]
        )

        num_patches = (int(image_size / 32)) ** 2 #600 * 1000 // 32 // 32
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.model.visual.attnpool.positional_embedding.shape[-1]),)
        pos_embed.weight = resize_pos_embed(self.model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed)
        self.model.visual.attnpool.positional_embedding = pos_embed

        # self.reset_image_size(self.config.image_size, self.config.patch_size)
        # self.model.visual.attnpool.positional_embedding = nn.Parameter(self.model.visual.attnpool.positional_embedding.view(1, -1, self.config.hidden_size))

    def reset_image_size(self, image_size=224, patch_size=32):
        # num_patches = 196 #600 * 1000 // 32 // 32
        # assert num_patches == (image_size // patch_size) ** 2
        old_image_size = self.config.image_size
        new_image_size = image_size

        old_grid = old_image_size // self.config.patch_size
        new_grid = new_image_size // patch_size

        previous_position_embedding_weight = self.model.visual.attnpool.positional_embedding.data[0, 1:]
        previous_position_embedding_weight = previous_position_embedding_weight.transpose(1, 0)
        previous_position_embedding_weight = previous_position_embedding_weight.view(1, self.config.hidden_size, old_grid, old_grid)

        new_position_embedding_weight = torch.nn.functional.interpolate(
            previous_position_embedding_weight,
            size=(new_grid, new_grid),
            mode='bicubic',
            align_corners=False
        )

        new_position_embedding_weight = new_position_embedding_weight.view(
            self.config.hidden_size, new_grid**2).transpose(1, 0)

        new_position_embedding = nn.Parameter(torch.zeros(1 + new_grid**2, self.config.hidden_size))
        # [CLS]
        new_position_embedding.data[0] = self.model.visual.attnpool.positional_embedding.data[0, 0]
        new_position_embedding.data[1:] = new_position_embedding_weight

        self.model.visual.attnpool.positional_embedding = nn.Parameter(new_position_embedding.unsqueeze(0))

        self.config.image_size = new_image_size
        self.config.patch_size = patch_size

    def forward(self, image):
        x, attnpool = self.model.encode_image(image)

        B, C, H, W = x.size()
        # (B, C, H, W) => (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = x.contiguous().view(B, H*W, C)

        x = x # + self.model.visual.attnpool.positional_embedding.view(1, 1+H*W, C)[:, 1:]

        return x, attnpool.unsqueeze(1)


if __name__ == "__main__":
    # model = get_vis_encoder(backbone="RN50")

    # import torch
    
    # x = torch.randn(4, 3, 224, 224)
    # feat, attn = model(x)

    # print(feat.shape)
    # print(attn.shape)

    clip_model = get_vis_encoder("RN50x4", adapter_type="front/middle/back-bottleneck", reduction_factor=26)

    import torch
    
    x = torch.randn(4, 3, 224, 224)
    clip_model(x)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(count_parameters(clip_model)/1000000)

        # model = self.model
        # x = model.visual.conv1(image.half())  # shape = [*, width, grid, grid]
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # x = torch.cat([model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + model.visual.positional_embedding.to(x.dtype)[:x.shape[1], :]
        # x = model.visual.ln_pre(x)

        # x = x.permute(1, 0, 2)  # NLD -> LND

        # for layer_idx, layer in enumerate(model.visual.transformer.resblocks):
        #     x = layer(x)  

        # x = x.permute(1, 0, 2)
        # tmp_fc = x[0, 0, :]
        # tmp_att = x[0, 1:, :].reshape( 14, 14, 768 )
