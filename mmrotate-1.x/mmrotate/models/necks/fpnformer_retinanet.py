# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor

from mmrotate.registry import MODELS
from mmdet.utils import ConfigType, MultiConfig, OptConfigType

import logging
from typing import Optional
import torch
from .window_att import Swin
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from .mamba_vision import crossAttention
from .posembedding import PositionEmbeddingSine

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class Fpndecoder(nn.Module):
    def __init__(self, query_pos, k_pos, v_pos):
        super(Fpndecoder, self).__init__()
        self.dropout = nn.Dropout(0.0)
        self.activation = _get_activation_fn("relu")
        # self.MambaVisionMixer = MambaVisionMixer()
        self.crossMambaAttention = crossAttention()
        self.query = query_pos
        self.key = k_pos
        self.value = v_pos
        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 256)

    def forward(self, q, k, v):
        b, n, h, w = q.shape
        kb, kn, kh, kw = k.shape
        vb, vn, vh, vw = v.shape
        pH = h // 2
        pW = h // 2

        #加入位置编码
        query_pos = self.query.transpose(1, 2).reshape(-1, 256, h, w)
        k_pos = self.key.transpose(1, 2).reshape(-1, 256, kh, kw)
        v_pos = self.value.transpose(1, 2).reshape(-1, 256, vh, vw)

        q = q + query_pos
        x = q
        k = k + k_pos
        v = v + v_pos

        q = q.reshape(b, n, -1).transpose(1, 2)
        k = k.reshape(b, n, -1).transpose(1, 2)
        v = v.reshape(b, n, -1).transpose(1, 2)

        q = self.linear1(q).transpose(1, 2).reshape(-1, 64, h, w)
        k = self.linear1(k).transpose(1, 2).reshape(-1, 64, kh, kw)
        v = self.linear1(v).transpose(1, 2).reshape(-1, 64, vh, vw)

        q_k = F.interpolate(q, size=[kh, kw], mode="bilinear", align_corners=False)
        k = k + self.dropout(self.activation(q_k))

        q_v = F.interpolate(q, size=[vh, vw], mode="bilinear", align_corners=False)
        v = v + self.dropout(self.activation(q_v))

        q = F.interpolate(q, size=[pH, pW], mode="bilinear", align_corners=False)
        k_q = self.dropout(self.activation(F.interpolate(k, size=[pH,pW], mode="bilinear", align_corners=False)))
        v_q = self.dropout(self.activation(F.interpolate(v, size=[pH,pW], mode="bilinear", align_corners=False)))

        # q =  self.MambaVisionMixer(q)
        # k_q = self.MambaVisionMixer(k_q)
        q = self.crossMambaAttention(q, k_q)

        #通道拆分
        # q = self.MambaVisionMixer(q)
        # v_q = self.MambaVisionMixer(v_q)
        q = self.crossMambaAttention(q, v_q)
        q = q.reshape(b, 64, -1).transpose(1, 2)
        q = self.linear2(q).transpose(1, 2).reshape(-1, 256, pH, pW)
        q = F.interpolate(q, size=[h,w], mode="bilinear", align_corners=False)
        q = x + self.dropout(self.activation(q))
        k = k.reshape(b, 64, -1).transpose(1, 2)
        v = v.reshape(b, 64, -1).transpose(1, 2)
        k = self.linear2(k).transpose(1, 2).reshape(-1, 256, kh, kw)
        v = self.linear2(v).transpose(1, 2).reshape(-1, 256, vh, vw)
        return q , k , v

c3 = torch.rand(1, 256, 128, 128)
c4 = torch.rand(1, 256, 64, 64)
c5 = torch.rand(1, 256, 32, 32)
pos = PositionEmbeddingSine()
pos_c3 = pos(c3).flatten(2).transpose(1, 2).to('cuda')
pos_c4 = pos(c4).flatten(2).transpose(1, 2).to('cuda')
pos_c5 = pos(c5).flatten(2).transpose(1, 2).to('cuda')








@MODELS.register_module()
class FPNdecoderformer_swin_double(BaseModule):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        num_outs: int,
        start_level: int = 0,
        end_level: int = -1,
        add_extra_convs: Union[bool, str] = False,
        relu_before_extra_convs: bool = False,
        no_norm_on_lateral: bool = False,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = None,
        upsample_cfg: ConfigType = dict(mode='nearest'),
        init_cfg: MultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

        # self.decoder_c5_c4 = Fpndecoder(input_resolution=to_2tuple(8),query_size=8, key_size=8, pretrain_size=8)
        # self.decoder_c5_c3 = Fpndecoder(input_resolution=to_2tuple(8),query_size=8, key_size=8, pretrain_size=8)
        # self.decoder_c4_c5 = Fpndecoder(input_resolution=to_2tuple(16),query_size=16, key_size=16, pretrain_size=16)
        # self.decoder_c3_c5 = Fpndecoder(input_resolution=to_2tuple(32),query_size=32, key_size=32, pretrain_size=32)
        # self.decoder_c5_c4_2 = Fpndecoder(input_resolution=to_2tuple(8),query_size=8, key_size=8, pretrain_size=8)
        # self.decoder_c5_c3_2 = Fpndecoder(input_resolution=to_2tuple(8),query_size=8, key_size=8, pretrain_size=8)
        # self.decoder_c4_c5_2 = Fpndecoder(input_resolution=to_2tuple(16),query_size=16, key_size=16, pretrain_size=16)
        # self.decoder_c3_c5_2 = Fpndecoder(input_resolution=to_2tuple(32),query_size=32, key_size=32, pretrain_size=32)
        # self.decoder_c5_c3 = Fpndecoder(input_resolution=to_2tuple(32), query_size=32, key_size=32,pretrain_size=32)
        # self.decoder_c5_c4 = Fpndecoder(input_resolution=to_2tuple(32), query_size=32, key_size=32,pretrain_size=32)
        # self.decoder_c3_c4_c5 = Fpndecoder(pos_c5, pos_c4, pos_c3, input_resolution=to_2tuple(32), query_size=32, key_size=32, pretrain_size=32)
        self.decoder_c3_c4_c5 = Fpndecoder(pos_c5, pos_c4, pos_c3)
        
    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)  # 比如3，索引0:C3, 1:C4, 2:C5
        highest_idx = used_backbone_levels - 1  # 2，即C5索引

        for i in range(used_backbone_levels - 1):  # 遍历C3和C4索引0和1
            if 'scale_factor' in self.upsample_cfg:
                upsampled = F.interpolate(laterals[highest_idx], **self.upsample_cfg)
            else:
                target_shape = laterals[i].shape[2:]  # 目标层的空间尺寸（C3或C4）
                upsampled = F.interpolate(laterals[highest_idx], size=target_shape, **self.upsample_cfg)

            laterals[i] = laterals[i] + upsampled  # 融合到C3或C4

        # build top-down path
        # used_backbone_levels = len(laterals)
        # for i in range(used_backbone_levels - 1, 0, -1):
        #     # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
        #     #  it cannot co-exist with `size` in `F.interpolate`.
        #     if 'scale_factor' in self.upsample_cfg:
        #         # fix runtime error of "+=" inplace operation in PyTorch 1.10
        #         laterals[i - 1] = laterals[i - 1] + F.interpolate(
        #             laterals[i], **self.upsample_cfg)
        #
        #     else:
        #         prev_shape = laterals[i - 1].shape[2:]
        #         laterals[i - 1] = laterals[i - 1] + F.interpolate(
        #             laterals[i], size=prev_shape, **self.upsample_cfg)
        #     print(f"After fusion, laterals[{i - 1}] shape: {laterals[i - 1].shape}")

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        outs[2], outs[1], outs[0] = self.decoder_c3_c4_c5(outs[2], outs[1], outs[0])
        # outs[2],outs[1] = self.decoder_c5_c4(outs[2], outs[1])
        # outs[2],outs[0] = self.decoder_c5_c3(outs[2], outs[0])
        # outs[2], outs[1] = self.decoder_c5_c4(outs[2], outs[1])
        # outs[2], outs[0] = self.decoder_c5_c3(outs[2], outs[0])
        # outs[0] = self.decoder_c3_c5(outs[0], outs[2])
        # outs[1] = self.decoder_c4_c5(outs[1], outs[2])
        # outs[2] = self.decoder_c5_c4(outs[2], outs[1])
        # outs[2] = self.decoder_c5_c3(outs[2], outs[0])
        # outs[0] = self.decoder_c3_c5_2(outs[0], outs[2])
        # outs[1] = self.decoder_c4_c5_2(outs[1], outs[2])
        # outs[2] = self.decoder_c5_c4_2(outs[2], outs[1])
        # outs[2] = self.decoder_c5_c3_2(outs[2], outs[0])
        outs[0], outs[1], outs[2] = outs[0].contiguous(), outs[1].contiguous(), outs[2].contiguous()
        
        return tuple(outs)
