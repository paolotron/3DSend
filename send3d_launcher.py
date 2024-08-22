import argparse
import sys

import numpy as np
import pandas as pd
import tqdm
import torch

import data.ood_metrics as ood_metrics
from send_3d_modules.common import FaissNN
from data.ood_datasets import set_train_loader_3d, set_test_loader_3d
import networks
import os.path as osp
import send_3d_modules.network_extractor as network_extractor
import send_3d_modules.patch_scoring as p
from send_3d_modules.sampler import ApproximateGreedyCoresetSampler
from networks.extractor_loops import get_extractor_loop
import time
import time

from send_3d_modules.utils import set_random_seed


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='path to dataset folder')
    parser.add_argument('--in_dataset', type=str, help='dataset split to be considered in distribution')
    parser.add_argument('--num_points', default=1024, type=int, help='number of points for the sampled pointclouds')
    
    parser.add_argument('--backbone', type=str, help='name of the feature extractor')
    parser.add_argument('--layer', type=str, help='name of extraction layer')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='path to checkpoint')
    
    parser.add_argument('-bs', '--batch_size', default=16, type=int, help='batch size for feature extraction loop, affects only speed as there is no training')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int, help='number of dataloader workers')
    
    parser.add_argument('--sample_rate', default=1, type=float, help='memory bank coreset subsampling, 1 means no sampling')
    parser.add_argument('--normalize', action='store_true', default=False, help='false for l2 distance true for cosine similarity')
    
    parser.add_argument('--device', default='cuda')
    
    parser.add_argument('--model_zoo_path', default='./checkpoints', help='uni3d models want their checkpoint already downloaded in a specific folder')
    
    # Aggregator Only for Conv Like Feature extractors
    parser.add_argument('--stride', default=3, type=int, help='stride of mean pooling for conv-like extractors')
    parser.add_argument('--kernel', default=3, type=int, help='kernel of mean pooling for conv-like extractors')
    
    return parser.parse_args()


def get_ood_dataset(data_path: str, in_dataset: str, batch_size=16, num_points=1024,
                    num_workers=8, k_shot=-1, seed_dataset=3, sparse=False):
    train_loaders = set_train_loader_3d(data_root=data_path, in_dataset=in_dataset, num_workers=num_workers,
                                        batch_size=batch_size, num_points=num_points, split_classes=False,
                                        padding=0,
                                        k=k_shot, seed_dataset=seed_dataset, sparse=sparse)
    test_in_dataloader, test_out_dataloader = set_test_loader_3d(data_root=data_path, in_dataset=in_dataset,
                                                                 num_workers=num_workers,
                                                                 batch_size=batch_size, num_points=num_points,
                                                                 padding=0, sparse=sparse)

    return train_loaders, test_in_dataloader, test_out_dataloader


def sample_memory_bank(memory_bank, labels, percentage: float, device):

    sampler = ApproximateGreedyCoresetSampler(percentage, device=device)
    unique_labels = np.unique(labels)
    sampled_bank_list = []
    for l in tqdm.tqdm(unique_labels):
        sampled_bank = sampler.run(memory_bank[labels == l])
        sampled_bank_list.append((sampled_bank, np.ones(sampled_bank.shape[0]) * l))
    sampled_memory_bank, sampled_labels = zip(*sampled_bank_list)
    return np.concatenate(sampled_memory_bank), np.concatenate(sampled_labels)

def merge_memory_bank(memory_bank, labels):
    unique_labels = np.unique(labels)
    sampled_bank_list = []
    for l in tqdm.tqdm(unique_labels):
        sampled_bank = np.mean(memory_bank[labels == l], axis=0, keepdims=True)
        sampled_bank_list.append((sampled_bank, l))
    sampled_memory_bank, sampled_labels = zip(*sampled_bank_list)
    return np.concatenate(sampled_memory_bank), np.array(sampled_labels)

def score_samples(mem_bank, bank_labels, test_features, normalize=False):
    M, C = mem_bank.shape
    print(f'MemBank: {M}x{C}')
    N, P = len(test_features), [t.shape[0] for t in test_features]
    if normalize:
        # L2 distance + Normalization is equal to cosine distance with our specified metrics
        mem_bank /= np.linalg.norm(mem_bank, axis=-1, keepdims=True)
        test_features = [t / np.linalg.norm(t, axis=-1, keepdims=True) for t in test_features]

    test_features = np.ascontiguousarray(np.concatenate(test_features))
    
    nn = FaissNN(on_gpu=True, metric='l2')
    nn.fit(np.ascontiguousarray(mem_bank, dtype=np.float32))

    print('Searching NN')
    start = time.time()
    test_dist, ix = nn.run(1, test_features)
    duration = time.time() - start
    print(f'Finished in {duration}')


    test_dist, ix = test_dist[:, 0], ix[:, 0]
    test_label = bank_labels[ix]
    args = {'threshold': -np.inf}

    scorers = {'Entropy': p.EntropyScorer(**args), 'WEntropy': p.WeightedEntropyScorer(**args)}
    total_scores = {'Entropy': [], 'WEntropy': []}

    curr_dim = 0
    
    # Weird wat but handles also non-uniform number of patches 
    for i, patch_dim in tqdm.tqdm(enumerate(P), 'scoring'):
        patch_scores = test_dist[curr_dim:curr_dim + patch_dim]
        patch_labels = test_label[curr_dim:curr_dim + patch_dim]
        anomaly_scores = {k: v.score(patch_scores, np.zeros_like(patch_scores), patch_labels)
                          for k, v in scorers.items()}
        for k in anomaly_scores:
            ak = anomaly_scores[k][0] if type(anomaly_scores[k]) == tuple else anomaly_scores[k]
            total_scores[k].append(ak)
        curr_dim += patch_dim
    total_scores = {k: -np.array(v) for k, v in total_scores.items()}
    return total_scores


def main(args, seed_dataset=3):

    device = args.device
    set_random_seed(args.seed, deterministic=True)
    
    # THREE DATASETS SupportSet / TestInDistribution / TestOutOfDistribution
    train_loader, test_in_dataloader, test_out_dataloader = get_ood_dataset(data_path=args.data_path,
                                                                            in_dataset=args.in_dataset,
                                                                            batch_size=args.batch_size,
                                                                            num_points=args.num_points,
                                                                            num_workers=args.num_workers,
                                                                            seed_dataset=seed_dataset,
                                                                            )

    backbone = args.backbone
    model = networks.get_backbone(backbone, model_zoo_path=args.model_zoo_path)
    # If model has no default checkpoints load it here
    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt)
    
    # Put forward hook at desired extraction layer and get the correct output loop 
    layer = [args.layer]
    network = network_extractor.NetworkFeatureAggregator(model, layer, device=device).to(device).eval()
    extractor_loop, sparse = get_extractor_loop(args)
    
    # Generate Support features
    support_features, support_labels = extractor_loop(extractor=network,
                                                loader=train_loader,
                                                device=device)
    if sparse:
        # Edge case for non uniform number of patches (SPCONV)
        N = len(support_features)
        P = [i.shape[0] for i in support_features]
        support_features = np.concatenate(support_features)
        support_labels = np.concatenate([np.ones(p) * l for p, l in zip(P, support_labels)])
    else:
        N, P, C = support_features.shape
        support_features = support_features.reshape(-1, support_features.shape[-1])
        support_labels = support_labels.repeat(P)

    # Generate test features
    test_in_features, _ = extractor_loop(extractor=network, loader=test_in_dataloader, device=device)
    test_out_features, _ = extractor_loop(extractor=network, loader=test_out_dataloader, device=device)

    # Coreset Subsampling if specified
    if args.sample_rate < 1.0:
        support_features, support_labels = sample_memory_bank(memory_bank=support_features,
                                                            labels=support_labels,
                                                            percentage=args.sample_rate, device=device)
    
    if sparse:
        test_features = [*test_in_features, *test_out_features]
    else:
        test_features = np.concatenate([test_in_features, test_out_features])
        test_features = [test_features[i] for i in range(test_features.shape[0])]

    ood_labels = np.concatenate([np.ones(len(test_in_features)), np.zeros(len(test_out_features))])
    normalize = args.normalize
    # Score Samples with Entropy and Weighted Entropy
    scores = score_samples(mem_bank=support_features, bank_labels=support_labels, test_features=test_features,
    
    normalize=normalize)
    results = []
    names = []
    # Calculate Metrics 
    for name, s in scores.items():
        metrics = ood_metrics.calc_metrics(np.asarray(s), ood_labels)
        results.append({**metrics})
        names.append(name)
    return pd.DataFrame(index=names, data=results)


if __name__ == '__main__':
    args = parser()
    print(main(args))
