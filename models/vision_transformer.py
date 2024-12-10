import collections.abc
from functools import partial
from itertools import repeat
from typing import Callable, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def _init_img_size(self, img_size: Union[int, Tuple[int, int]]):
        assert self.patch_size
        img_size = to_2tuple(img_size)
        grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]})."
            
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        bias=True,
        drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        proj_drop: float = 0.,
        attn_drop: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        pos_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[Union[str, Callable, Type[torch.nn.Module]]] = partial(nn.LayerNorm, eps=1e-6),
        act_layer: Optional[Union[str, Callable, Type[torch.nn.Module]]] = nn.GELU,
        block_fn: Type[nn.Module] = Block,
        mlp_layer: Type[nn.Module] = Mlp,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Number of classes for classification head.
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            proj_drop_rate: Projection dropout rate.
            attn_drop_rate: Attention dropout rate.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
            mlp_layer: MLP layer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = 1

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        embed_len = num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * 0.02)
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier Head
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        pos_embed = self.pos_embed
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1)] + [x], dim=1)
        x = x + pos_embed
        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.head_drop(x)
        return self.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
