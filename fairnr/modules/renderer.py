# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict

import os
import imageio
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from fairnr.modules.module_utils import FCLayer
from fairnr.data.geometry import ray
from fairnr.data.shape_dataset import ShapeViewDataset
from fairnr.data.data_utils import load_matrix, load_rgb, parse_views
from fairnr.modules.encoder import LocalImageSparseVoxelEncoder, MultiSparseVoxelEncoder

MAX_DEPTH = 10000.0
RENDERER_REGISTRY = {}

def register_renderer(name):
    def register_renderer_cls(cls):
        if name in RENDERER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        RENDERER_REGISTRY[name] = cls
        return cls
    return register_renderer_cls


def get_renderer(name):
    if name not in RENDERER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return RENDERER_REGISTRY[name]


@register_renderer('abstract_renderer')
class Renderer(nn.Module):
    """
    Abstract class for ray marching
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        pass


@register_renderer('volume_rendering')
class VolumeRenderer(Renderer):

    def __init__(self, args):
        super().__init__(args)
        self.chunk_size = 1024 * getattr(args, "chunk_size", 64)
        self.valid_chunk_size = 1024 * getattr(args, "valid_chunk_size", self.chunk_size // 1024)
        self.discrete_reg = getattr(args, "discrete_regularization", False)
        self.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0.0)
        self.trace_normal = getattr(args, "trace_normal", False)
        self.backbone = 'resnet34'
        self.device = 'cpu' if self.args.cpu else 'cuda'

    @staticmethod
    def add_args(parser):
        # ray-marching parameters
        parser.add_argument('--discrete-regularization', action='store_true',
                            help='if set, a zero mean unit variance gaussian will be added to encougrage discreteness')

        # additional arguments
        parser.add_argument('--chunk-size', type=int, metavar='D',
                            help='set chunks to go through the network (~K forward passes). trade time for memory. ')
        parser.add_argument('--valid-chunk-size', type=int, metavar='D',
                            help='chunk size used when no training. In default the same as chunk-size.')
        parser.add_argument('--raymarching-tolerance', type=float, default=0)

        parser.add_argument('--trace-normal', action='store_true')

    # load pre-trained model
    def load_dataset(self, data_path):
        train_view = parse_views(self.args.train_views)
        train_dataset = ShapeViewDataset(paths=data_path, views=train_view, num_view=self.args.view_per_batch,
                                         resolution=self.args.view_resolution, preload=False)
        N, C, H, W = len(train_dataset.views), 3, train_dataset.resolution[0], train_dataset.resolution[1]
        colors = torch.zeros((N, C, H, W), dtype=torch.float)
        extrinsics = torch.zeros((N, 4, 4), dtype=torch.float)

        for iview, (fn_rgb, fn_ext) in enumerate(zip(train_dataset.data[0]['rgb'], train_dataset.data[0]['ext'])):
            colors[iview, :, :, :] = torch.from_numpy(
                load_rgb(fn_rgb, resolution=(H, W), with_alpha=False)[0][:C, :, :])
            extrinsics[iview, :, :] = torch.from_numpy(load_matrix(fn_ext)).reshape(4, 4)
        intrinsics = load_matrix(train_dataset.data[0]['ixt'])

        # resize intrinsics
        img = imageio.imread(train_dataset.data[0]['rgb'][0])[:, :, :3]
        ori_H, ori_W, _ = img.shape
        resized_intrinsics = np.identity(4)
        resized_intrinsics[0, :] = intrinsics[0, :] * (W / ori_W)
        resized_intrinsics[1, :] = intrinsics[1, :] * (H / ori_H)

        return colors, extrinsics, torch.from_numpy(resized_intrinsics)

    @staticmethod
    def extract_resnet34_features(colors):
        resnet34 = torchvision.models.resnet.resnet34(pretrained=True)
        resnet34.fc = nn.Sequential()
        resnet34.avgpool = nn.Sequential()

        x = resnet34.conv1(colors)
        x = resnet34.bn1(x)
        x = resnet34.relu(x)
        latents = [x]

        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = torch.nn.functional.interpolate(
                latents[i],
                latent_sz,
                mode="bilinear",  # self.upsample_interp
                align_corners=False
            )
        features = torch.cat(latents, dim=1)
        return features

    @staticmethod
    def extract_vgg16_features(colors, num_layers=3):
        vgg16 = torchvision.models.vgg.vgg16(pretrained=True)
        features_HxW = vgg16.features[:num_layers](colors)  # [N, 64, H, W]
        upsampler = torch.nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        features = upsampler(features_HxW)
        return features

    def extract_image_features(self, data_path):
        save_path = os.path.join(data_path, 'feature', '{}.pt'.format(self.backbone))
        if not os.path.exists(save_path):
            colors, extrinsics, intrinsics = self.load_dataset(data_path)
            if self.backbone == 'resnet34':
                features = self.extract_resnet34_features(colors)
            elif self.backbone == 'vgg16':
                features = self.extract_vgg16_features(colors)
            else:
                raise ValueError('unknown network backbone type')
            # save to local
            features_dict = {
                'features': features.to(self.device),
                'extrinsics': extrinsics.to(self.device),
                'intrinsics': intrinsics.to(self.device)
            }
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            torch.save(features_dict, save_path)
        else:
            # load precompute
            features_dict = torch.load(save_path, map_location=self.device)
        return features_dict


    def forward_once(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
        early_stop=None, output_types=['sigma', 'texture']
        ):
        """
        chunks: set > 1 if out-of-memory. it can save some memory by time.
        """
        sampled_depth = samples['sampled_point_depth']
        sampled_idx = samples['sampled_point_voxel_idx'].long()

        # only compute when the ray hits
        sample_mask = sampled_idx.ne(-1)
        if early_stop is not None:
            sample_mask = sample_mask & (~early_stop.unsqueeze(-1))
        if sample_mask.sum() == 0:  # miss everything skip
            return None, 0

        sampled_xyz = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), sampled_depth.unsqueeze(2))
        sampled_dir = ray_dir.unsqueeze(1).expand(*sampled_depth.size(), ray_dir.size()[-1])
        samples['sampled_point_xyz'] = sampled_xyz
        samples['sampled_point_ray_direction'] = sampled_dir

        # apply mask    
        samples = {name: s[sample_mask] for name, s in samples.items()}

        # get encoder features as inputs
        field_inputs = input_fn(samples, encoder_states)

        # forward implicit fields
        field_outputs = field_fn(field_inputs, outputs=output_types)
        outputs = {'sample_mask': sample_mask}

        def masked_scatter(mask, x):
            B, K = mask.size()
            if x.dim() == 1:
                return x.new_zeros(B, K).masked_scatter(mask, x)
            return x.new_zeros(B, K, x.size(-1)).masked_scatter(
                mask.unsqueeze(-1).expand(B, K, x.size(-1)), x)

        # post processing
        if 'sigma' in field_outputs:
            sigma, sampled_dists= field_outputs['sigma'], field_inputs['dists']
            noise = 0 if not self.discrete_reg and (not self.training) else torch.zeros_like(sigma).normal_()
            free_energy = torch.relu(noise + sigma) * sampled_dists
            free_energy = free_energy * 7.0  # ? [debug] 
            # (optional) free_energy = (F.elu(sigma - 3, alpha=1) + 1) * dists
            # (optional) free_energy = torch.abs(sigma) * sampled_dists  ## ??
            outputs['free_energy'] = masked_scatter(sample_mask, free_energy)
        if 'sdf' in field_outputs:
            outputs['sdf'] = masked_scatter(sample_mask, field_outputs['sdf'])
        if 'texture' in field_outputs:
            outputs['texture'] = masked_scatter(sample_mask, field_outputs['texture'])
        if 'normal' in field_outputs:
            outputs['normal'] = masked_scatter(sample_mask, field_outputs['normal'])
        if 'feat_n2' in field_outputs:
            outputs['feat_n2'] = masked_scatter(sample_mask, field_outputs['feat_n2'])
        return outputs, sample_mask.sum()

    def forward_chunk(
        self, input_fn, bg_field_fn, field_fn, ray_start, ray_dir, BG_DEPTH, samples, encoder_states,
        gt_depths=None, output_types=['sigma', 'texture'], global_weights=None,
        ):
        if self.trace_normal:
            output_types += ['normal']

        sampled_depth = samples['sampled_point_depth']
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        original_depth = samples.get('original_point_depth', None)
        data_path = os.path.dirname(input_fn.args.initial_boundingbox)
        image_features = self.extract_image_features(data_path)

        tolerance = self.raymarching_tolerance
        chunk_size = self.chunk_size if self.training else self.valid_chunk_size
        early_stop = None
        if tolerance > 0:
            tolerance = -math.log(tolerance)

        hits = sampled_idx.ne(-1).long()
        outputs = defaultdict(lambda: [])
        size_so_far, start_step = 0, 0
        accumulated_free_energy = 0
        accumulated_evaluations = 0
        for i in range(hits.size(1) + 1):
            if ((i == hits.size(1)) or (size_so_far + hits[:, i].sum() > chunk_size)) and (i > start_step):
                _outputs, _evals = self.forward_once(
                        input_fn, field_fn,
                        ray_start, ray_dir,
                        {name: s[:, start_step: i]
                            for name, s in samples.items()},
                        encoder_states=image_features if isinstance(
                            input_fn, (LocalImageSparseVoxelEncoder, MultiSparseVoxelEncoder)) else encoder_states,
                        early_stop=early_stop,
                        output_types=output_types)
                if _outputs is not None:
                    accumulated_evaluations += _evals

                    if 'free_energy' in _outputs:
                        accumulated_free_energy += _outputs['free_energy'].sum(1)
                        if tolerance > 0:
                            early_stop = accumulated_free_energy > tolerance
                            hits[early_stop] *= 0

                    for key in _outputs:
                        outputs[key] += [_outputs[key]]
                else:
                    for key in outputs:
                        outputs[key] += [outputs[key][-1].new_zeros(
                            outputs[key][-1].size(0),
                            sampled_depth[:, start_step: i].size(1),
                            *outputs[key][-1].size()[2:]
                        )]
                start_step, size_so_far = i, 0

            if (i < hits.size(1)):
                size_so_far += hits[:, i].sum()

        outputs = {key: torch.cat(outputs[key], 1) for key in outputs}
        results = {}

        if 'free_energy' in outputs:
            free_energy = outputs['free_energy']
            shifted_free_energy = torch.cat([free_energy.new_zeros(sampled_depth.size(0), 1), free_energy[:, :-1]], dim=-1)  # shift one step
            a = 1 - torch.exp(-free_energy.float())                             # probability of it is not empty here
            b = torch.exp(-torch.cumsum(shifted_free_energy.float(), dim=-1))   # probability of everything is empty up to now
            probs = (a * b).type_as(free_energy)                                # probability of the ray hits something here
            results['transparency'] = b
        else:
            probs = outputs['sample_mask'].type_as(sampled_depth) / sampled_depth.size(-1)  # assuming a uniform distribution

        if global_weights is not None:
            probs = probs * global_weights

        depth = (sampled_depth * probs).sum(-1)
        missed = 1 - probs.sum(-1)

        results.update({
            'probs': probs, 'depths': depth,
            'max_depths': sampled_depth.masked_fill(hits.eq(0), -1).max(1).values,
            'min_depths': sampled_depth.min(1).values,
            'missed': missed, 'ae': accumulated_evaluations
        })
        if original_depth is not None:
            results['z'] = (original_depth * probs).sum(-1)

        if 'texture' in outputs:
            results['colors'] = (outputs['texture'] * probs.unsqueeze(-1)).sum(-2)

        if 'normal' in outputs:
            results['normal'] = (outputs['normal'] * probs.unsqueeze(-1)).sum(-2)
            if not self.trace_normal:
                results['eikonal-term'] = (outputs['normal'].norm(p=2, dim=-1) - 1) ** 2
            else:
                results['eikonal-term'] = torch.log((outputs['normal'] ** 2).sum(-1) + 1e-6)
            results['eikonal-term'] = results['eikonal-term'][sampled_idx.ne(-1)]

        if 'feat_n2' in outputs:
            results['feat_n2'] = (outputs['feat_n2'] * probs).sum(-1)
            results['regz-term'] = outputs['feat_n2'][sampled_idx.ne(-1)]

        # === Add background color with object color ====
        if bg_field_fn is not None:
            assert 'transparency' in results
            bg_color_scale = results['transparency'][:, [-1]]

            # background color estimate
            bg_fiend_inputs = {'pos': ray_start, 'ray': ray_dir}
            bg_field_outputs = bg_field_fn(bg_fiend_inputs, outputs=['texture'])

            # add colors background term
            results['colors'] += bg_color_scale * bg_field_outputs['texture']

            # add depth background term
            results['depths'] += results['transparency'][:, -1] * field_fn.bg_color.depth

        return results

    def forward(self, input_fn, bg_field_fn, field_fn, ray_start, ray_dir, BG_DEPTH, samples, *args, **kwargs):
        chunk_size = self.chunk_size if self.training else self.valid_chunk_size
        if ray_start.size(0) <= chunk_size:
            results = self.forward_chunk(input_fn, bg_field_fn, field_fn, ray_start, ray_dir, BG_DEPTH, samples, *args, **kwargs)
        else:
            # the number of rays is larger than maximum forward passes. pre-chuncking..
            results = [
                self.forward_chunk(input_fn, bg_field_fn, field_fn,
                    ray_start[i: i+chunk_size], ray_dir[i: i+chunk_size], BG_DEPTH,
                    {name: s[i: i+chunk_size] for name, s in samples.items()}, *args, **kwargs)
                for i in range(0, ray_start.size(0), chunk_size)
            ]
            results = {name: torch.cat([r[name] for r in results], 0)
                        if results[0][name].dim() > 0 else sum([r[name] for r in results])
                    for name in results[0]}

        if getattr(input_fn, "track_max_probs", False) and (not self.training):
            input_fn.track_voxel_probs(samples['sampled_point_voxel_idx'].long(), results['probs'])
        return results


@register_renderer('surface_volume_rendering')
class SurfaceVolumeRenderer(VolumeRenderer):

    def forward_chunk(
        self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
        gt_depths=None, output_types=['sigma', 'texture'], global_weights=None,
        ):
        results = super().forward_chunk(
            input_fn, field_fn, ray_start, ray_dir, samples, encoder_states,
            output_types=['sigma', 'normal'])

        # render at the "intersection"
        n_probs = results['probs'].clamp(min=1e-6).masked_fill(samples['sampled_point_voxel_idx'].eq(-1), 0)
        n_depth = (samples['sampled_point_depth'] * n_probs).sum(-1, keepdim=True) / n_probs.sum(-1, keepdim=True).clamp(min=1e-6)
        n_bound = samples['sampled_point_depth'] + samples['sampled_point_distance'] / 2
        n_vidxs = ((n_depth - n_bound) >= 0).sum(-1, keepdim=True)
        n_vidxs = samples['sampled_point_voxel_idx'].gather(1, n_vidxs)

        new_samples = {
            'sampled_point_depth': n_depth,
            'sampled_point_distance': torch.ones_like(n_depth) * 1e-3,  # dummy distance. not useful.
            'sampled_point_voxel_idx': n_vidxs,
        }
        new_results, _ = self.forward_once(input_fn, field_fn, ray_start, ray_dir, new_samples, encoder_states)
        results['colors'] = new_results['texture'].squeeze(1) * (1 - results['missed'][:, None])
        results['normal'] = new_results['normal'].squeeze(1)
        results['eikonal-term'] = torch.cat([results['eikonal-term'], (results['normal'].norm(p=2, dim=-1) - 1) ** 2], 0)
        return results