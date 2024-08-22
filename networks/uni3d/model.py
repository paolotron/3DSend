import torch
import timm
import numpy as np
from torch import nn
# from openpatch.networks.uni3d import losses
from networks.uni3d.point_encoder import PointcloudEncoder
import os.path as osp

class DotConfig:
    """
    Access to dictionary through dot notation - more troubles than benefits
    """

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k):
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v



# code source: https://github.com/baaivision/Uni3D

class Uni3D(nn.Module):
    def __init__(self, point_encoder):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.point_encoder = point_encoder

    # def encode_pc(self, pc):
    #     xyz = pc[:,:,:3].contiguous()
    #     color = pc[:,:,3:].contiguous()
    #     pc_feat = self.point_encoder(xyz, color)
    #     return pc_feat

    # def forward(self, pc, text, image):
    #     text_embed_all = text
    #     image_embed = image
    #     pc_embed = self.encode_pc(pc)
    #     return {'text_embed': text_embed_all,
    #             'pc_embed': pc_embed,
    #             'image_embed': image_embed,
    #             'logit_scale': self.logit_scale.exp()}

    def forward(self, pc, return_penultimate=True):
        xyz = pc[:, :, :3].contiguous()
        if pc.size(-1) == 6:
            color = pc[:, :, 3:].contiguous()
        else:
            color = torch.zeros(pc.size()).to(pc)

        pc_feat = self.point_encoder(xyz, color)

        if return_penultimate:
            return pc_feat
        else:
            raise NotImplementedError


def get_filter_loss(args):
    return losses.Uni3d_Text_Image_Loss()


def get_metric_names(model):
    return ['loss', 'uni3d_loss', 'pc_image_acc', 'pc_text_acc']


def create_uni3d(args):
    # create transformer blocks for point cloud via timm
    point_transformer = timm.create_model(
        args.pc_model, checkpoint_path=args.pretrained_pc, drop_path_rate=args.drop_path_rate
        )

    # create whole point cloud encoder
    point_encoder = PointcloudEncoder(point_transformer, args)

    # uni3d model
    model = Uni3D(point_encoder=point_encoder,)
    return model


def get_uni3d_model(model_name, model_zoo_path):
    args = {
        "pretrained_pc": '',
        "clip-model": 'RN50',
        "drop_path_rate": 0.0,
        'num_group': 512,
        'group_size': 64,
        'pc_encoder_dim': 512,
        'embed_dim': 1024,
        'patch_dropout': 0.
    }

    if model_name in ['uni3d-g', 'uni3d-g-no-lvis']:
        args['pc_model'] = "eva_giant_patch14_560"
        args['pc_feat_dim'] = 1408
    elif model_name in ['uni3d-l', 'uni3d-l-no-lvis']:
        args['pc_model'] = "eva02_large_patch14_448"
        args['pc_feat_dim'] = 1024
    elif model_name in ['uni3d-b', 'uni3d-b-no-lvis']:
        args['pc_model'] = "eva02_base_patch14_448"
        args['pc_feat_dim'] = 768
    elif model_name in ['uni3d-s', 'uni3d-s-no-lvis']:
        args['pc_model'] = "eva02_small_patch14_224"
        args['pc_feat_dim'] = 384
    elif model_name in ['uni3d-ti', 'uni3d-ti-no-lvis']:
        args['pc_model'] = "eva02_tiny_patch14_224"
        args['pc_feat_dim'] = 192
    else:
        raise ValueError(f"Unknown Uni3D model: {model_name}")
    
    ckpt_path = osp.join(model_zoo_path, model_name, "model.pt")
    assert osp.exists(ckpt_path), f"Cannot find checkpoint path at: {ckpt_path}"

    args = DotConfig(args)

    # build model
    model_weights = torch.load(ckpt_path, map_location='cpu')['module']
    model = create_uni3d(args)

    # load pretrained weights
    print("Load model weights: ", model.load_state_dict(model_weights))
    print(model)
    return model

