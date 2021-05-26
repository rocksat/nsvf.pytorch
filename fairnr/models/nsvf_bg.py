# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
logger = logging.getLogger(__name__)

import cv2, math, time, copy, json
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.utils import item, with_torch_seed
from fairnr.data.geometry import compute_normal_map, fill_in
from fairnr.models.nsvf import NSVFModel, base_architecture, nerf_style_architecture
from fairnr.models.fairnr_model import get_encoder, get_field, get_reader, get_renderer

@register_model('nsvf_bg')
class NSVFBGModel(NSVFModel):
    # add a background field sub-module neural network

    def __init__(self, args, setups):
        super().__init__(args, setups)

        args_copy = copy.deepcopy(args)
        if getattr(args, "bg_field_args", None) is not None:
            args_copy.__dict__.update(json.loads(args.bg_field_args))
        else:
            args_copy.inputs_to_density = "pos:10"
            args_copy.inputs_to_texture = "feat:0:256, ray:4:3:b"
            args_copy.feature_layers = 0 # can define the number of layers
            args_copy.texture_layers = 2
            args_copy.has_density_predictor = True
        self.bg_field  = get_field("radiance_field")(args_copy) # generate the new neural network field
        

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--near', type=float, help='near distance of the volume')
        parser.add_argument('--far',  type=float, help='far distance of the volume')
        parser.add_argument('--nerf-steps', type=int, help='additional nerf steps')
        parser.add_argument('--bg-field-args', type=str, default=None, help='override args for bg field')

    
@register_model_architecture("nsvf_bg", "nsvf_bg")
def base_bg_architecture(args):
    # field
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:6:64")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    args.feature_embed_dim = getattr(args, "feature_embed_dim", 256)
    args.density_embed_dim = getattr(args, "density_embed_dim", 128)
    args.texture_embed_dim = getattr(args, "texture_embed_dim", 256)

    # API Update: fix the number of layers
    args.feature_layers = getattr(args, "feature_layers", 1)
    args.texture_layers = getattr(args, "texture_layers", 3)
    
    base_architecture(args)






