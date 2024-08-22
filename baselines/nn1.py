
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('./')

from data import ood_metrics
from data.ood_metrics import calc_metrics
import networks
from tqdm import tqdm
from data.ood_datasets import set_train_loader_3d, set_test_loader_3d


from networks.extractor_loops import get_extractor_loop, minkowski_collate
from send3d_launcher import get_ood_dataset
from send_3d_modules import network_extractor
from send_3d_modules.common import FaissNN
from send_3d_modules.patch_scoring import MeanScorer

def get_layer(backbone: str):
    if backbone.startswith('epn'):
        layer = 'backbone.4'
        dist = 'eucli'
    elif backbone.startswith('OpenShape'):
        layer = 'proj'
        dist = 'cos'
    elif backbone.startswith('uni3d'):
        layer = 'point_encoder'
        dist = 'cos'
    return layer, dist
        
def score_samples(mem_bank, bank_labels, test_features, normalize=False):
    
    M, C = mem_bank.shape
    print(f'MemBank: {M}x{C}')
    P, N ,C = test_features.shape
    if normalize:
        mem_bank /= np.linalg.norm(mem_bank, axis=-1, keepdims=True)
        test_features = [t / np.linalg.norm(t, axis=-1, keepdims=True) for t in test_features]

    test_features = np.ascontiguousarray(np.concatenate(test_features))
    
    nn = FaissNN(on_gpu=True, metric='l2')
    nn.fit(np.ascontiguousarray(mem_bank, dtype=np.float32))
    test_dist, ix = nn.run(1, test_features)
    test_dist, ix = test_dist[:, 0], ix[:, 0]
    return (-test_dist).tolist()

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
    parser.add_argument('--checkpoint_path', default=None, type=str)
    args = parser.parse_args()
    args.kernel = 1
    args.stride = 1
    
    sparse = args.backbone.startswith('OpenShape')
    train_loader, test_in_dataloader, test_out_dataloader = get_ood_dataset(data_path=args.data_path,
                                                                        in_dataset=args.in_dataset,
                                                                        batch_size=args.batch_size,
                                                                        num_points=args.num_points,
                                                                        num_workers=args.num_workers,
                                                                        sparse=sparse
                                                                        )
    
    model = networks.get_backbone(args.backbone, model_zoo_path=args.model_zoo_path)
    device = 'cuda'
    
    if args.checkpoint_path is not None:
        ckpt = torch.load(args.checkpoint_path)
        model.load_state_dict(ckpt)
    
    layer, dist = get_layer(args.backbone)
    network = network_extractor.NetworkFeatureAggregator(model, [layer], device=device).to(device).eval()
    extractor_loop, sparse = get_extractor_loop(args)
    support_features, support_labels = extractor_loop(extractor=network,
                                                loader=train_loader,
                                                device=device)
    N, _, C = support_features.shape
    support_features = support_features.reshape(N, C)
    test_in_features, _ = extractor_loop(extractor=network, loader=test_in_dataloader, device=device)
    test_out_features, _ = extractor_loop(extractor=network, loader=test_out_dataloader, device=device)
    test_features = np.concatenate([test_in_features, test_out_features])
    ood_labels = np.concatenate([np.ones(len(test_in_features)), np.zeros(len(test_out_features))])
    normalize = dist == 'cos'
    scores = score_samples(mem_bank=support_features, bank_labels=support_labels, test_features=test_features, normalize=normalize)
    results = []
    names = []

    metrics = ood_metrics.calc_metrics(np.asarray(scores), ood_labels)
    print(metrics)