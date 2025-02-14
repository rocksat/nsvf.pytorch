# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
  
logger = logging.getLogger(__name__)

import copy

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairnr.models.nsvf import NSVFImageModel, nsvf_image_architecture
from fairnr.modules.field import RaidanceField


@register_model('nsvf_image_bg')
class NSVFImageBGModel(NSVFImageModel):
    # add a background field to sub-module neural network

    def __init__(self, args, setups):
        super().__init__(args, setups)
        self.enable_bg_field = getattr(self.args, 'enable_bg_field', False)
        if self.enable_bg_field:
            # construct bg_field
            bg_args = copy.deepcopy(args)
            for bg_arg in vars(bg_args):
                if bg_arg.startswith('bg'):
                    arg = bg_arg.split('bg_')[-1]
                    if hasattr(bg_args, arg):
                        setattr(bg_args, arg, getattr(bg_args, bg_arg))
            self.bg_field = RaidanceField(bg_args)
        else:
            self.bg_field = None


@register_model_architecture('nsvf_image_bg', 'nsvf_image_bg')
def nsvf_image_bg_architecture(args):
    # parameter need to be changed
    args.enable_bg_field = getattr(args, "enable_bg_field", True)

    # background field
    args.num_videos = sum([1 for _ in open(args.object_id_path)])
    args.bg_inputs_to_density = getattr(args, "bg_inputs_to_density", "pos:10, emb:0:{}".format(args.num_videos))
    args.bg_inputs_to_texture = getattr(args, "bg_inputs_to_texture", "feat:0:256, ray:4:3:b")
    args.bg_feature_layers = getattr(args, "bg_feature_layers", 0)
    args.bg_texture_layers = getattr(args, "bg_texture_layers", 2)
    args.has_density_predictor = getattr(args, "has_density_predictor", True)
    nsvf_image_architecture(args)
