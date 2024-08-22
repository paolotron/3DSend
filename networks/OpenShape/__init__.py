import torch

from .models import make
import MinkowskiEngine as ME
from huggingface_hub import hf_hub_download
from collections import OrderedDict
import open_clip
import re
from omegaconf import OmegaConf
import torch.nn as nn
from .models.ppat import Projected, PointPatchTransformer


def load_config(yaml_file):
    yaml_confs = OmegaConf.load(yaml_file)
    return yaml_confs


def module(state_dict: dict, name):
    """ this function is used to change some keys in the serialized checkpoint dictionary"""
    return {'.'.join(k.split('.')[1:]): v for k, v in state_dict.items() if k.startswith(name + '.')}

import clip
@torch.no_grad()
def extract_text_feat(texts, clip_model, ):
    text_tokens = clip.tokenize(texts)
    return clip_model.encode_text(text_tokens)


@torch.no_grad()
def extract_image_feat(images, clip_model, clip_preprocess):
    image_tensors = [clip_preprocess(image) for image in images]
    image_tensors = torch.stack(image_tensors, dim=0).float().cuda()
    image_features = clip_model.encode_image(image_tensors)
    image_features = image_features.reshape((-1, image_features.shape[-1]))
    return image_features


def G14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 512 * 3, 256, 384, 0.2, 64, 6),
        nn.Linear(512, 1280)
    )
    print("OpenShape G14 load weight from ckpt: ", model.load_state_dict(module(s['state_dict'], 'module')))
    return model


def L14(s):
    model = Projected(
        PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6),
        nn.Linear(512, 768)
    )
    print("OpenShape L14 load weight from ckpt: ", model.load_state_dict(module(s, 'pc_encoder')))
    return model


def B32(s):
    model = PointPatchTransformer(512, 12, 8, 1024, 128, 64, 0.4, 256, 6)
    print("OpenShape B32 load weight from ckpt: ", model.load_state_dict(module(s, 'pc_encoder')))
    return Projected(model, None)


model_list = {
    "OpenShape/openshape-pointbert-vitb32-rgb": B32,
    # tested it for semantic novelty detection (synth to real) and works very good
    "OpenShape/openshape-pointbert-vitl14-rgb": L14,
    "OpenShape/openshape-pointbert-vitg14-rgb": G14,
}


def load_bert(model_name='OpenShape/pointbert-vitg14-rgb'):
    from argparse import Namespace
    config = {'name': 'PointBERT', 'scaling': 4, 'use_dense': True, 'in_channel': 6, 'out_channel': 1280}
    config = Namespace(model=Namespace(**config))
    model = make(config).cuda()
    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k, v in checkpoint['state_dict'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
    print(model.load_state_dict(model_dict))
    print(model)
    return model


def load_model(model_name="OpenShape/openshape-spconv-all"):
    config = load_config("./networks/OpenShape/train.yaml")
    # model = make(config).cuda()
    checkpoint = torch.load(hf_hub_download(repo_id=model_name, filename="model.pt"))
    model = model_list[model_name](checkpoint)
    print(model)

    # if config.model.name.startswith('Mink'):
    #     model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)  # minkowski only
    # else:
    #     model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    return model
