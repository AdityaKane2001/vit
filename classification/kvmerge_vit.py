import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List, Literal


from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, \
    IMAGENET_INCEPTION_STD, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, \
    get_act_layer, get_norm_layer, LayerType
from timm.models import generate_default_cfgs, register_model, register_model_deprecations, \
    named_apply, checkpoint_seq, adapt_input_conv, build_model_with_cfg

# from timm.layers import use_fused_attn

from boilerplate import checkpoint_filter_fn, init_weights, _init_weights, \
    load_pretrained, no_weight_decay, group_matcher, set_grad_checkpointing, \
    get_classifier, _intermediate_layers, get_intermediate_layers, \
    reset_classifier, default_cfgs

from merge import unequal_bipartite_soft_matching, merge_wavg,\
    bipartite_soft_matching, multihead_bipartite_soft_matching

__all__ = ["default_cfgs"]


class KVMergeAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False # use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def merge(self, tokens, size=None):
        # tokens: [B, H, N, C]
        # metric = torch.mean(tokens, dim=1)
        m, u = bipartite_soft_matching(tokens, r=self.kvmerge_info["r"], 
                                               class_token=self.kvmerge_info["has_cls"], 
                                               distill_token=self.kvmerge_info["has_distill"])
        tokens = m(tokens)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        k = self.merge(k)
        v = self.merge(v)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # if self.kvmerge_info["k_size"] is not None:
        #     attn += self.kvmerge_info["k_size"].log()[:, :, None, :, 0]
            
        # if self.kvmerge_info["v_size"] is not None:
        #     attn += self.kvmerge_info["v_size"].log()
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class KVMergeBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = KVMergeAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class KVMergeVisionTransformer(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]
    
    init_weights = init_weights
    _init_weights = _init_weights
    load_pretrained = load_pretrained
    no_weight_decay = no_weight_decay
    group_matcher = group_matcher
    set_grad_checkpointing = set_grad_checkpointing
    get_classifier = get_classifier
    _intermediate_layers = _intermediate_layers
    get_intermediate_layers = get_intermediate_layers
    reset_classifier = reset_classifier

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = KVMergeBlock,
            mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)
        
        
        # KVMerge artifacts
        self.kvmerge_info = dict()
        # self.kvmerge_info["v_size"] = None
        # self.kvmerge_info["k_size"] = None
        self.kvmerge_info["r"] = 98
        self.kvmerge_info["has_cls"] = self.has_class_token
        self.kvmerge_info["has_distill"] = False
        
        for child in self.modules():
            if isinstance(child, KVMergeAttention):
                child.kvmerge_info = self.kvmerge_info
            


    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.kvmerge_info["v_size"] = None
        self.kvmerge_info["k_size"] = None
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs): # -> KVMergeVisionTransformer:
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = True
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    # print(kwargs)
    
    return build_model_with_cfg(
        KVMergeVisionTransformer,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


@register_model
def kvmerge_vit_tiny_patch16_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_tiny_patch16_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Tiny (Vit-Ti/16) @ 384x384.
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
    model = _create_vision_transformer('vit_tiny_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_small_patch32_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Small (ViT-S/32)
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_small_patch32_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Small (ViT-S/32) at 384x384.
    """
    model_args = dict(patch_size=32, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_small_patch16_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_small_patch16_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_small_patch8_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Small (ViT-S/8)
    """
    model_args = dict(patch_size=8, embed_dim=384, depth=12, num_heads=6)
    model = _create_vision_transformer('vit_small_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch32_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch32_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base model (ViT-B/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch8_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base (ViT-B/8) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch8_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch32_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929). No pretrained weights.
    """
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch32_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/32) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=32, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch32_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch16_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch16_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch16_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch14_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/14)
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16)
    model = _create_vision_transformer('vit_large_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch14_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16)
    model = _create_vision_transformer('vit_huge_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_giant_patch14_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_args = dict(patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16)
    model = _create_vision_transformer('vit_giant_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_gigantic_patch14_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Gigantic (big-G) model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    """
    model_args = dict(patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16)
    model = _create_vision_transformer(
        'vit_gigantic_patch14_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_224_miil(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
    model = _create_vision_transformer(
        'vit_base_patch16_224_miil', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_medium_patch16_gap_240(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 240x240
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_medium_patch16_gap_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 256x256
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_medium_patch16_gap_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Medium (ViT-M/16) w/o class token, w/ avg-pool @ 384x384
    """
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=False,
        global_pool='avg', qkv_bias=False, init_values=1e-6, fc_norm=False)
    model = _create_vision_transformer(
        'vit_medium_patch16_gap_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_gap_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base (ViT-B/16) w/o class token, w/ avg-pool @ 224x224
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_base_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch14_gap_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/14) w/ no class token, avg pool
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_huge_patch14_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch16_gap_448(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/16) w/ no class token, avg pool @ 448x448
    """
    model_args = dict(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16, class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_huge_patch16_gap_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_giant_patch16_gap_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Giant (little-gg) model (ViT-g/16) w/ no class token, avg pool
    """
    model_args = dict(
        patch_size=16, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11,
        class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer(
        'vit_giant_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch32_clip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/32 CLIP image tower @ 224x224
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch32_clip_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/32 CLIP image tower @ 256x256
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch32_clip_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/32 CLIP image tower @ 384x384
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch32_clip_448(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/32 CLIP image tower @ 448x448
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch32_clip_448', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_clip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/16 CLIP image tower
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_clip_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/16 CLIP image tower @ 384x384
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch16_clip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch14_clip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/14) CLIP image tower
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_large_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch14_clip_336(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/14) CLIP image tower @ 336x336
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_large_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch14_clip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/14) CLIP image tower.
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch14_clip_336(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 336x336
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch14_clip_378(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 378x378
    """
    model_args = dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_378', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_giant_patch14_clip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Giant (little-g) model (ViT-g/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    Pretrained weights from CLIP image tower.
    """
    model_args = dict(
        patch_size=14, embed_dim=1408, mlp_ratio=48/11, depth=40, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_giant_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_gigantic_patch14_clip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-bigG model (ViT-G/14) from `Scaling Vision Transformers` - https://arxiv.org/abs/2106.04560
    Pretrained weights from CLIP image tower.
    """
    model_args = dict(
        patch_size=14, embed_dim=1664, mlp_ratio=64/13, depth=48, num_heads=16, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_gigantic_patch14_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch32_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/32 CLIP image tower @ 224x224
    """
    model_args = dict(
        patch_size=32, embed_dim=768, depth=12, num_heads=12, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_base_patch32_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/16 CLIP image tower w/ QuickGELU act
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_base_patch16_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch14_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/14) CLIP image tower w/ QuickGELU act
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_large_patch14_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch14_clip_quickgelu_336(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Large model (ViT-L/14) CLIP image tower @ 336x336 w/ QuickGELU act
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_large_patch14_clip_quickgelu_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch14_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/14) CLIP image tower w/ QuickGELU act.
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_huge_patch14_clip_quickgelu_378(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Huge model (ViT-H/14) CLIP image tower @ 378x378 w/ QuickGELU act
    """
    model_args = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_huge_patch14_clip_quickgelu_378', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# Experimental models below

@register_model
def kvmerge_vit_base_patch32_plus_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base (ViT-B/32+)
    """
    model_args = dict(patch_size=32, embed_dim=896, depth=12, num_heads=14, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_base_patch32_plus_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_plus_240(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base (ViT-B/16+)
    """
    model_args = dict(patch_size=16, embed_dim=896, depth=12, num_heads=14, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_base_patch16_plus_240', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# @register_model
# def kvmerge_vit_base_patch16_rpn_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
#     """ ViT-Base (ViT-B/16) w/ residual post-norm
#     """
#     model_args = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, init_values=1e-5,
#         class_token=False, block_fn=ResPostBlock, global_pool='avg')
#     model = _create_vision_transformer(
#         'vit_base_patch16_rpn_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


@register_model
def kvmerge_vit_small_patch16_36x1_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-Base w/ LayerScale + 36 x 1 (36 block serial) config. Experimental, may remove.
    Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
    Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=36, num_heads=6, init_values=1e-5)
    model = _create_vision_transformer(
        'vit_small_patch16_36x1_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# @register_model
# def kvmerge_vit_small_patch16_18x2_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
#     """ ViT-Small w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
#     Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
#     Paper focuses on 24x2 + 48x1 for 'Small' width but those are extremely slow.
#     """
#     model_args = dict(
#         patch_size=16, embed_dim=384, depth=18, num_heads=6, init_values=1e-5, block_fn=ParallelThingsBlock)
#     model = _create_vision_transformer(
#         'vit_small_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def kvmerge_vit_base_patch16_18x2_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
#     """ ViT-Base w/ LayerScale + 18 x 2 (36 block parallel) config. Experimental, may remove.
#     Based on `Three things everyone should know about Vision Transformers` - https://arxiv.org/abs/2203.09795
#     """
#     model_args = dict(
#         patch_size=16, embed_dim=768, depth=18, num_heads=12, init_values=1e-5, block_fn=ParallelThingsBlock)
#     model = _create_vision_transformer(
#         'vit_base_patch16_18x2_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


@register_model
def eva_large_patch14_196(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ EVA-large model https://arxiv.org/abs/2211.07636 /via MAE MIM pretrain"""
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer(
        'eva_large_patch14_196', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def eva_large_patch14_336(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ EVA-large model https://arxiv.org/abs/2211.07636 via MAE MIM pretrain"""
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, global_pool='avg')
    model = _create_vision_transformer('eva_large_patch14_336', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def flexivit_small(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ FlexiViT-Small
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, no_embed_class=True)
    model = _create_vision_transformer('flexivit_small', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def flexivit_base(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ FlexiViT-Base
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, no_embed_class=True)
    model = _create_vision_transformer('flexivit_base', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def flexivit_large(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ FlexiViT-Large
    """
    model_args = dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16, no_embed_class=True)
    model = _create_vision_transformer('flexivit_large', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# @register_model
# def kvmerge_vit_base_patch16_xp_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
#     """ ViT-Large model (ViT-L/14) w/ parallel blocks and qk norm enabled.
#     """
#     model_args = dict(
#         patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, no_embed_class=True,
#         norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
#     )
#     model = _create_vision_transformer(
#         'vit_base_patch16_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def kvmerge_vit_large_patch14_xp_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
#     """ ViT-Large model (ViT-L/14) w/ parallel blocks and qk norm enabled.
#     """
#     model_args = dict(
#         patch_size=14, embed_dim=1024, depth=24, num_heads=16, pre_norm=True, no_embed_class=True,
#         norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
#     )
#     model = _create_vision_transformer(
#         'vit_large_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


# @register_model
# def kvmerge_vit_huge_patch14_xp_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
#     """ ViT-Huge model (ViT-H/14) w/ parallel blocks and qk norm enabled.
#     """
#     model_args = dict(
#         patch_size=14, embed_dim=1280, depth=32, num_heads=16, pre_norm=True, no_embed_class=True,
#         norm_layer=RmsNorm, block_fn=ParallelScalingBlock, qkv_bias=False, qk_norm=True,
#     )
#     model = _create_vision_transformer(
#         'vit_huge_patch14_xp_224', pretrained=pretrained, **dict(model_args, **kwargs))
#     return model


@register_model
def kvmerge_vit_small_patch14_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-S/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_small_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch14_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_base_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch14_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-L/14 for DINOv2
    """
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5, img_size=518)
    model = _create_vision_transformer(
        'vit_large_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_giant_patch14_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-G/14 for DINOv2
    """
    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192
    model_args = dict(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5,
        mlp_ratio=2.66667 * 2, mlp_layer=SwiGLUPacked, img_size=518, act_layer=nn.SiLU
    )
    model = _create_vision_transformer(
        'vit_giant_patch14_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_small_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-S/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=384, depth=12, num_heads=6, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_small_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-B/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=768, depth=12, num_heads=12, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_base_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-L/14 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=14, embed_dim=1024, depth=24, num_heads=16, init_values=1e-5,
        reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_large_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_giant_patch14_reg4_dinov2(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    """ ViT-G/14 for DINOv2
    """
    # The hidden_features of SwiGLU is calculated by:
    # hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
    # When embed_dim=1536, hidden_features=4096
    # With SwiGLUPacked, we need to set hidden_features = 2 * 4096 = 8192
    model_args = dict(
        patch_size=14, embed_dim=1536, depth=40, num_heads=24, init_values=1e-5, mlp_ratio=2.66667 * 2,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU, reg_tokens=4, no_embed_class=True,
    )
    model = _create_vision_transformer(
        'vit_giant_patch14_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_siglip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_siglip_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_siglip_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_siglip_512(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_base_patch16_siglip_512', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch16_siglip_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_large_patch16_siglip_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_large_patch16_siglip_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_large_patch16_siglip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_so400m_patch14_siglip_224(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch14_siglip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_so400m_patch14_siglip_384(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=14, embed_dim=1152, depth=27, num_heads=16, mlp_ratio=3.7362, class_token=False, global_pool='map',
    )
    model = _create_vision_transformer(
        'vit_so400m_patch14_siglip_384', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_medium_patch16_reg4_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8, class_token=True,
        no_embed_class=True, reg_tokens=4,
    )
    model = _create_vision_transformer(
        'vit_medium_patch16_reg4_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_medium_patch16_reg4_gap_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=512, depth=12, num_heads=8,
        class_token=False, no_embed_class=True, reg_tokens=4, global_pool='avg',
    )
    model = _create_vision_transformer(
        'vit_medium_patch16_reg4_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def kvmerge_vit_base_patch16_reg8_gap_256(pretrained: bool = False, **kwargs) -> KVMergeVisionTransformer:
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, class_token=False,
        no_embed_class=True, global_pool='avg', reg_tokens=8,
    )
    model = _create_vision_transformer(
        'vit_base_patch16_reg8_gap_256', pretrained=pretrained, **dict(model_args, **kwargs))
    return model