import torch
import numpy as np
import tqdm
from typing import Callable

def detach(x):
    return x.detach().cpu().numpy()

@torch.no_grad()
def extract_features(extractor, loader, device, extractor_type='pointnet'):
    detach = lambda x: x[1].transpose(2, 1).cpu().numpy(),
    feature_arr, label_arr = [], []
    for sample, label in tqdm.tqdm(loader, 'extracting features'):
        features = extractor(sample.float().transpose(2, 1).to(device).contiguous())
        features = next(iter(features.values()))
        feature_arr.append(detach(features))
        label_arr.append(label)
    return np.concatenate(feature_arr), np.concatenate(label_arr)  # N x P x C, N


def minkowski_collate(list_data, batch_coordinates=True, swap_axis=True):
    import MinkowskiEngine as ME
    xyz_list, feature_list, label_list = [], [], []
    for xyz, label in list_data:
        xyz = xyz.T
        if swap_axis:
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        else:
            pass
            # xyz[:, [0, 1]] = xyz[:, [1, 0]]
        feats = np.concatenate([xyz, np.ones_like(xyz) * 0], axis=1)
        xyz_list.append(torch.tensor(xyz))
        label_list.append(torch.tensor(label))
        feature_list.append(torch.tensor(feats))
    if batch_coordinates:
        xyz = ME.utils.batched_coordinates(xyz_list, dtype=torch.float32)
        feats = torch.cat(feature_list)
        labels = torch.tensor(label_list)
    else:
        xyz = torch.tensor(np.array(xyz_list))
        feats = torch.tensor(np.array(feature_list))
        labels = torch.tensor(np.array(label_list))

    return xyz, feats, labels


@torch.no_grad()
def extract_features_epn(extractor, loader, device):
    global ix
    global time
    global splits
    ix = 0
    feature_arr, label_arr = [], []
    for xyz, label in tqdm.tqdm(loader, 'extracting features'):
        ix += 1
        features = extractor(xyz.transpose(2, 1).float().to(device))
        features = next(iter(features.values()))
        features = features
        features = features.feats.permute(0, 1, 3, 2)
        features, _ = torch.max(features, dim=2)
        features = features.transpose(2, 1)
        feature_arr.append(features.cpu().numpy())
        label_arr.append(label)
        
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels


@torch.no_grad()
def extract_features_sparse(extractor, loader, device, stride=3, kernel=3):
    import MinkowskiEngine as ME
    loader.collate_fn = lambda x: minkowski_collate(x, batch_coordinates=True)
    feature_arr, label_arr = [], []
    aggregator = ME.MinkowskiAvgPooling(kernel, stride=stride, dimension=3).to(device)
    for xyz, feats, label in tqdm.tqdm(loader, 'extracting features'):
        features = extractor(xyz.float().to(device), feats.float().to(device), device=device, quantization_size=0.02)
        features = next(iter(features.values()))
        features = aggregator(features)
        feature_arr.append([f.cpu().numpy() for f in features.decomposed_features])
        label_arr.append(label)
    feats = [item for sublist in feature_arr for item in sublist]
    labels = np.concatenate(label_arr)
    return feats, labels


@torch.no_grad()
def extract_features_bert(extractor, loader, device, swap_axis=False):
    loader.collate_fn = lambda x: minkowski_collate(x, batch_coordinates=False, swap_axis=swap_axis)
    feature_arr, label_arr = [], []
    for xyz, feats, label in tqdm.tqdm(loader, 'extracting features'):
        features = extractor(xyz.float().to(device), feats.float().to(device))
        features = next(iter(features.values()))
        if len(features.shape) == 2:
            features = features[:, None, :]
        feature_arr.append(features.cpu().numpy())
        label_arr.append(label)
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels


@torch.no_grad()
def extract_features_uni3d(extractor, loader, device):
    feature_arr, label_arr = [], []
    for xyz, label in tqdm.tqdm(loader, 'extracting features'):
        features = extractor(xyz.permute(0, 2, 1).float().to(device))
        features = next(iter(features.values()))
        if len(features.shape) == 2:
            features = features[:, None, :]
        feature_arr.append(detach(features))
        label_arr.append(label)
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels

def get_extractor_loop(args) -> Callable:
    backbone_name = args.backbone
    sparse = backbone_name in ['OpenShape_spconv', 'OpenShape_ShapeNet'] or backbone_name in 'minkowski'
    sparse_args = {'kernel': args.kernel, 'stride': args.stride} if sparse else {}
    swap_args = {}
    
    if backbone_name.startswith('OpenShape_Bert'):
        extractor_type = 'bert'
        if backbone_name == 'OpenShape_Bert':
            swap_args['swap_axis'] = True
    elif backbone_name.startswith('OpenShape'):
        extractor_type = 'sparse'
        sparse = True
    elif backbone_name.startswith('spconv'):
        extractor_type = 'spconv'
    elif backbone_name.startswith('uni3d'):
        extractor_type = 'uni3d'
    
    ex_feat = {'normal': extract_features,
                'bert': extract_features_bert,
                'sparse': extract_features_sparse,
                'spconv': extract_features_epn,
                'uni3d': extract_features_uni3d}[extractor_type]
    
    def loop(*args, **kwargs):
        return ex_feat(*args, **kwargs, **sparse_args, **swap_args)
    
    return loop, sparse