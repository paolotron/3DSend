import argparse
import sys



sys.path.append('./')

import pandas as pd
import torch
import numpy as np
import EVM
import networks
from tqdm import tqdm
from data.ood_datasets import set_train_loader_3d, set_test_loader_3d
import data.ood_metrics
from networks.extractor_loops import get_extractor_loop


@torch.no_grad()
def extract_features_uni3d(model, loader):
    output_num = 1024
    device = 'cuda'
    ds_size = len(loader.dataset)
    batch_size = loader.batch_size

    feats_np = -np.ones(dtype=np.float32, shape=(ds_size, output_num))
    gts_np = -np.ones(dtype=int, shape=(ds_size,))

    for batch_idx, (images, target) in enumerate(tqdm(loader)):
        images = images.to(device).float().transpose(2, 1)
        this_batch_size = len(images)
        feats = model(images)
        pos_start = batch_idx * batch_size
        pos_end = batch_idx * batch_size + this_batch_size
        feats_np[pos_start:pos_end] = feats.cpu().numpy()
        gts_np[pos_start:pos_end] = target.numpy()

    feats_list, gt_list = feats_np, gts_np

    return feats_list, gt_list


def minkowski_collate(list_data, batch_coordinates=False):
    import MinkowskiEngine as ME
    xyz_list, feature_list, label_list = [], [], []
    for xyz, label in list_data:
        xyz = xyz.T
        xyz[:, [1, 2]] = xyz[:, [2, 1]]
        feats = np.concatenate([xyz, np.ones_like(xyz) * 0.4], axis=1)
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

    return (xyz, feats), labels


def extract_features_sparse(extractor, loader, device, stride=3, kernel=3):
    import MinkowskiEngine as ME
    loader.collate_fn = lambda x: minkowski_collate(x, batch_coordinates=True)
    feature_arr, label_arr = [], []
    for (xyz, feats), label in tqdm(loader, 'extracting features'):
        features = extractor(xyz.float().to(device), feats.float().to(device), device=device, quantization_size=0.02)
        feature_arr.append(features.cpu().numpy())
        label_arr.append(label)
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels


def extract_features_bert(extractor, loader, device):
    loader.collate_fn = lambda x: minkowski_collate(x, batch_coordinates=False)
    feature_arr, label_arr = [], []
    for (xyz, feats), label in loader:
        features = extractor(xyz.float().to(device), feats.float().to(device))
        feature_arr.append(features.cpu().numpy())
        label_arr.append(label)
    feats = np.concatenate(feature_arr)
    labels = np.concatenate(label_arr)
    return feats, labels




@torch.no_grad()
def EVM_evaluator(train_loader, test_loader, device, model, model_zoo_path):
    # this evaluator defines an Extreme Value Classifier and use it to estimate the normality of a given test sample
    # ref paper: http://doi.org/10.1109/TPAMI.2017.2707495
    
    print("Running EVM evaluator")

    import scipy
    import scipy.spatial
    
    if model.startswith('OpenShape_Bert'):
        m_type = 'bert'
    elif model.startswith('OpenShape'):
        m_type = 'spconv'
    elif model.startswith('uni3d'):
        m_type = 'uni3d'
    
    model = networks.load(model, model_zoo_path).cuda()

    model = model.eval()
    run_model = {
        'bert': extract_features_bert,
        'spconv': extract_features_sparse,
        'uni3d': extract_features_uni3d,
    }[m_type]
    
    # first we extract features for both support and test data
    train_feats, train_lbls = run_model(model, loader=train_loader, device=device)
    test_feats_in, test_lbls_in = run_model(model, loader=test_loader[0], device=device)
    test_feats_out, test_lbls_out = run_model(model, loader=test_loader[1],  device=device)
    
    del model

    # we need to divide train sample by class
    known_labels = np.unique(train_lbls)
    known_labels.sort()
    train_classes = [train_feats[train_lbls ==
                                 lbl] for lbl in known_labels]
    # create and train the classifier
    mevm = EVM.MultipleEVM(tailsize=len(train_feats), distance_function=scipy.spatial.distance.euclidean)
    # knn = scipy.spatial.KDTree(train_feats)
    mevm.train(train_classes, parallel=8)

    test_feats = np.concatenate([test_feats_in, test_feats_out])
    # estimate probabilities for test data
    pred_prob, indices = mevm.max_probabilities(test_feats, parallel=8)
    # pred_distances, _ = knn.query(test_feats)
    # pred_distances = -pred_distances
    pred_prob = np.array(pred_prob)
    # cs_preds = np.stack(indices)[:, 0]
    # cs_preds[pred_prob == 0] = len(known_labels)

    # known labels have 1 for known samples and 0 for unknown ones
    ood_labels = np.concatenate([np.ones(test_lbls_in.shape[0]), np.zeros(test_lbls_out.shape[0])])

    # closed set accuracy
    # known_mask = ood_labels == 1
    # cs_acc = closed_set_accuracy(cs_preds[known_mask], test_lbls[known_mask])

    # print(f"Num known: {ood_labels.sum()}. Num unknown: {len(test_lbls) - ood_labels.sum()}.")

    metrics = networks.ood_metrics.calc_metrics(
        predictions=pred_prob, labels=ood_labels
    )

    return metrics




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--in_dataset', type=str)
    parser.add_argument('--backbone', type=str, default='OpenShape_Bert')
    parser.add_argument('-bs', '--batch_size', default=8, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_points', default=2048, type=int)
    parser.add_argument('--model_zoo_path', type=str)

    args = parser.parse_args()
    
    data_path = args.data_path
    in_dataset = args.in_dataset
    num_workers = args.num_workers
    seed_dataset = args.seed
    padding = 0
    sparse = False
    numpoints = args.num_points
    batch_size = args.batch_size
    model = args.backbone

    train_loader = set_train_loader_3d(data_root=data_path,
                                        in_dataset=in_dataset,              
                                        num_workers=num_workers,
                                        batch_size=batch_size, num_points=numpoints, split_classes=False,
                                        sparse=sparse, padding=padding,
                                        seed_dataset=args.seed)
    test_loader = set_test_loader_3d(data_root=data_path,
                                        in_dataset=in_dataset,
                                        num_workers=num_workers,
                                        batch_size=batch_size, num_points=numpoints,
                                        sparse=sparse, padding=padding,)
    results = EVM_evaluator(train_loader=train_loader, test_loader=test_loader, device='cuda', model=model, model_zoo_path=args.model_zoo_path)
    print(results)