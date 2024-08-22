import torch
import sys

sys.path.append('./')

import data.ood_metrics as ood_metrics
from data.scanobject import SONN_ix_dict
from data import get_ood_dataset, minkowski_collate
from data.ood_datasets import set_train_loader_3d
from data.ood_datasets import set_test_loader_3d
import networks as backbones
import argparse
import tqdm
import numpy as np

import open_clip
import clip

from data.sncore_splits import ALL



def get_backbone(name: str):
    backbone = backbones.load(name)
    return backbone


def detach(x):
    return x.detach().cpu().numpy()


@torch.no_grad()
def extract_features_openshape(extractor, loader, device):
    global ix
    global time
    loader.collate_fn = lambda x: minkowski_collate(x, batch_coordinates=True, swap_axis=False)
    feature_arr, label_arr = [], []
    # for xyz, feats, label in tqdm.tqdm(loader, 'extracting features'):
    #     features = extractor(xyz.float().to(device), feats.float().to(device))
    for xyz, feats, label in tqdm.tqdm(loader, 'extracting features'):
        features = extractor(xyz.float().to(device), feats.float().to(device), device=device, quantization_size=0.02)
        feature_arr.append(detach(features))
        label_arr.append(label)
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels


@torch.no_grad()
def extract_features_uni3d(extractor, loader, device):
    global ix
    global time
    feature_arr, label_arr = [], []
    for xyz, label in tqdm.tqdm(loader, 'extracting features'):
        features = extractor(xyz.permute(0, 2, 1).float().to(device))
        feature_arr.append(detach(features))
        label_arr.append(label)
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels


@torch.no_grad()
def extract_features_bert(extractor, loader, device):
    loader.collate_fn = lambda x: minkowski_collate(x, batch_coordinates=False, swap_axis=True)
    feature_arr, label_arr = [], []
    for xyz, feats, label in tqdm.tqdm(loader, 'extracting features'):
        features = extractor(xyz.float().to(device), feats.float().to(device))
        feature_arr.append(features.cpu().numpy())
        label_arr.append(label)
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--in_dataset', type=str)
    parser.add_argument('--backbone', type=str, default='OpenShape_spconv')
    parser.add_argument('-bs', '--batch_size', default=16, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--sample_rate', default=1, type=float)
    parser.add_argument('--num_points', default=1024, type=int)
    parser.add_argument('--clip_model', default='EVA02-E-14-plus', type=str)
    parser.add_argument('--clip_pretrain', default='laion2b_s9b_b144k')
    args = parser.parse_args()
    sparse = args.backbone.startswith('OpenShape')
    train_loader, test_in_dataloader, test_out_dataloader = get_ood_dataset(data_path=args.data_path,
                                                                            in_dataset=args.in_dataset,
                                                                            batch_size=args.batch_size,
                                                                            num_points=args.num_points,
                                                                            num_workers=args.num_workers,
                                                                            sparse=sparse
                                                                            )
    device = 'cuda'

    if args.in_dataset.startswith('Real2Real'):
        names = train_loader.dataset.dataset.label_names
    else:
        in_classes = train_loader.dataset.class_choice
        names = list(in_classes.keys())
    from networks.OpenShape import extract_text_feat
    temperature = [1]
    print(names)
    text_encoder = open_clip.create_model(args.clip_model, pretrained=args.clip_pretrain)
    backbone = get_backbone(args.backbone)
    backbone = backbone.cuda()
    backbone.eval()
    # train_feats, train_labels = extract_features(backbone, train_loader, device=device)
    if args.backbone.startswith('uni3d'):
        extract_features = extract_features_uni3d
    elif args.backbone == 'OpenShape_SPConv':
        extract_features = extract_features_openshape
    elif args.backbone == 'OpenShape_Bert':
        extract_features = extract_features_bert
    #  extract_features = extract_features_bert
    test_in_feats, _ = extract_features(backbone, test_in_dataloader, device=device)
    test_out_feats, _ = extract_features(backbone, test_out_dataloader, device=device)
    text_prototypes = extract_text_feat(texts=names, clip_model=text_encoder).numpy()

    test_in_feats = test_in_feats / np.linalg.norm(test_in_feats, axis=-1, keepdims=True)
    test_out_feats = test_out_feats / np.linalg.norm(test_out_feats, axis=-1, keepdims=True)
    text_prototypes = text_prototypes / np.linalg.norm(text_prototypes, axis=-1, keepdims=True)
    from scipy.special import softmax
    feats = np.concatenate([test_in_feats, test_out_feats])
    for t in temperature:
        in_probs = softmax(feats @ text_prototypes.T / t, axis=-1)
        ood_score = np.max(in_probs, axis=1)
        labels = np.concatenate([np.ones_like(test_in_feats[:, 0]), np.zeros_like(test_out_feats[:, 0])])
        metrics = ood_metrics.calc_metrics(ood_score, labels)
        print(f'Temperature: {t} metrics: {metrics}')


if __name__ == '__main__':
    main()
