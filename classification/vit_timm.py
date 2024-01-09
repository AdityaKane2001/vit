import torch

from timm import create_model
from timm.models.registry import register_model
import patch_timm




@register_model
def tome_kvmerge_vit_small_patch16_224_augreg_in21k_ft_in1k(pretrained=True, **kwargs):
    model = create_model(
        "kvmerge_vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained
    )

    patch_timm.apply_patch(model)

    model.r = 16

    return model

