#!/usr/bin/env python3
import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat

class DWConv2d(nn.Module):

    def __init__(self, dim=64, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        # x = x.permute(0, 3, 1, 2) #(b c h w)
        x = self.conv(x) #(b c h w)
        # x = x.permute(0, 2, 3, 1) #(b h w c)
        # x = x.permute(0, 3, 1, 2)#(b c h w)
        return x

class ResidualAttention(nn.Module):

    def __init__(
            self,
            dim=256,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.5,
            proj_drop=0.5,
            norm_layer=nn.LayerNorm,
            layer_scale=1e-5,
            act_layer=nn.GELU,
            mlp_ratio=4.,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim))
        self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim))
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=attn_drop)
        self.drop_path = DropPath(attn_drop)
        self.DWConv2d = DWConv2d()
        
    def forward(self, x, kv):
        xb, xn, xh, xw = x.shape
        x = self.norm1(x)
        kv = self.norm1(kv)
        kv_local = self.DWConv2d(kv)

        x = x.view(xb, xn, -1).transpose(1, 2)
        kv = kv.view(xb, xn, -1).transpose(1, 2)
        kv_local = kv_local.view(xb, xn, -1).transpose(1, 2)
        
        x_finall = x
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, 1, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        kv_local = self.kv(kv_local).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        
        q = q.squeeze(0)
        k, v = kv.unbind(0)
        k_local, v_local = kv_local.unbind(0)
        q, k_local, k = self.q_norm(q), self.k_norm(k_local), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
            x_local = F.scaled_dot_product_attention(
             q, k_local, v_local,
                dropout_p=self.attn_drop.p,
            )
            x = x + x_local
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + self.drop_path(self.gamma_1 * x)
        x = x.transpose(1, 2).reshape(-1, 256, xh, xw)
        x = self.norm2(x)
        x = x.view(xb, xn, -1).transpose(1, 2)
        x = x_finall + self.drop_path(self.gamma_2 * self.mlp(x))
        x = x.transpose(1, 2).reshape(-1, 256, xh, xw)
        return x
