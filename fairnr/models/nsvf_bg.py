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
from fairnr.models.nsvf import NSVFModel, base_architecture
from fairnr.models.fairnr_model import get_field


@register_model('nsvf_bg')
class NSVFBGModel(NSVFModel):
    # add a background field sub-module neural network

    def __init__(self, args, setups):
        super().__init__(args, setups)
        self.enable_bg_field = getattr(self.args, 'enable_bg_field', False)
        if self.enable_bg_field:
            # construct bg_field
            bg_args = copy.deepcopy(args)
            for bg_arg in vars(bg_args):
                if bg_arg.startswith('bg'):
                    arg = bg_arg.split('bg_')[-1]
                    assert hasattr(bg_args, arg)
                    setattr(bg_args, arg, getattr(bg_args, bg_arg))
            self.bg_field = get_field("radiance_field")(bg_args)  # generate the new neural network field

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--nerf-steps', type=int, help='additional nerf steps')


@register_model_architecture("nsvf_bg", "nsvf_bg")
def nerf_bg_architecture(args):
    # parameter need to be changed
    args.enable_bg_field = getattr(args, "enable_bg_field", True)

    # background field
    args.bg_inputs_to_density = getattr(args, "bg_inputs_to_density", "pos:10")
    args.bg_inputs_to_texture = getattr(args, "bg_inputs_to_texture", "feat:0:256, ray:4:3:b")
    args.bg_feature_layers = getattr(args, "bg_feature_layers", 0)
    args.bg_texture_layers = getattr(args, "bg_texture_layers", 2)
    args.has_density_predictor = getattr(args, "has_density_predictor", True)
    base_architecture(args)
