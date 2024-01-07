import torch
from torchsummary import summary

import torch
from fvcore.nn import FlopCountAnalysis
from natten.flops import add_natten_handle

from vit import vit_small_patch16_224
from kvpool_vit import kvpool_vit_small_patch16_224

# from timm.models import create_model



def get_gflops(model, img_size=224, disable_warnings=False, device='cpu'):
    flop_ctr = FlopCountAnalysis(model, torch.randn(1, 3, img_size, img_size).to(device))
    flop_ctr = add_natten_handle(flop_ctr)
    if disable_warnings:
        flop_ctr = flop_ctr.unsupported_ops_warnings(False)
    return flop_ctr.total() / 1e9


def get_mparams(model, **kwargs):
    return sum([m.numel() for m in model.parameters() if m.requires_grad]) / 1e6


model_clss = [
    # create_model(
    #     "vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=False
    # ),
    vit_small_patch16_224(),
    kvpool_vit_small_patch16_224()
    
    # attnprune_vit_small_patch16_224_augreg_in21k,
    # vit_small_patch16_224_augreg_in21k_ft_in1k,
    # tome_vit_small_patch16_224_augreg_in21k_f1_in1k,
    # prunemap_vit_small_patch16_224_augreg_in21k,
    # attnmap_merge_tail_vit_small_patch16_224_augreg_in21k_f1_in1k
    # wintome_nat_s_tiny,
    # dinat_s_tiny,
    # nat_isotropic_small, 
    # dinat_isotropic_small, 
    # vitrpb_small,
    # wintome_nat_s_small,
    # wintome_nat_s_base,
    # wintome_nat_s_large,
    # wintome_nat_s_large_21k,
    # wintome_nat_s_large_384,
    # wintome_dinat_s_tiny,
    # wintome_dinat_s_small,
    # wintome_dinat_s_base,
    # wintome_dinat_s_large,
    # wintome_dinat_s_large_21k,
    # wintome_dinat_s_large_384,
    # dinat_s_large_384
]


for model in model_clss:
    # print(model_cls.__name__)
    # model = model_cls()
    
    # print(model)
    # print(model(torch.rand(2, 3, 224, 224)).shape)
    model.to("cuda:2")
    # print(model)

    # summary(model, input_data=(3, 224, 224), device="cuda:0")
    print(f"flops: ", get_gflops(model, device="cuda:2"))
    print(f"params: ", get_mparams(model, device="cuda:2"))
    print("=" * 80)
    # break

# model = dinat_s_tiny()
# # print(model(torch.rand(2, 3, 224, 224)).shape)
# summary(model, input_data=(3, 224, 224), device="cuda:0")

# print(f"NAT flops: ", get_gflops(model, device="cuda:0"))
# print(f"NAT params: ", get_mparams(model, device="cuda:0"))
