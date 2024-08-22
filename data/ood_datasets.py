import math

try:
    import MinkowskiEngine as ME
except ImportError:
    pass

import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split
from data import ShapeNetCore4k, ModelNet40_OOD, ScanObject
from data.data_utils import AugmRotate


def sparse_collate_fn(list_data):
    coordinates_batch, features_batch, labels_batch = ME.utils.sparse_collate(
        [d["coordinates"] for d in list_data],
        [d["features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    return {
        "coordinates": coordinates_batch,
        "features": features_batch,
    }, labels_batch


TRI_DATA = {'ShapeNet_SN1', 'ShapeNet_SN2', 'ShapeNet_SN3', 'Syn2Real_SR1', 'Syn2Real_SR2',
            'Real2Real_RR1', 'Real2Real_RR2', 'Real2Real_RR3'}


class PCPadTransform:
    def __init__(self, padding_dim=6):
        self.padding_dim = padding_dim

    def __call__(self, points, *args, **kwargs):
        n, _ = points.shape
        points = points.T  # [bs,n,3] => [bs,3,n]
        padding = np.ones((self.padding_dim, n)) * 0.5
        points = np.concatenate([points, padding], axis=0)
        return points.T


class PointTranspose:
    def __call__(self, points, *args, **kwargs):
        return points.T


def set_dataset_transforms(dataset, transformer):
    if isinstance(dataset, torch.utils.data.Subset):
        dataset.dataset.transforms = transformer
    else:
        dataset.transforms = transformer
    return dataset


def get_k_dataset(dataset, k, seed_dataset=3):
    rn = np.random.default_rng(seed=seed_dataset)
    if k >= 1:
        k = int(k)
        k_indices = rn.choice(len(dataset), k, replace=len(dataset) < k)
    else:
        tot = math.ceil(k * len(dataset))
        k_indices = rn.choice(len(dataset), tot, replace=len(dataset) < tot)

    return torch.utils.data.Subset(dataset, indices=k_indices)


def set_train_loader_3d(data_root,
                        in_dataset,
                        batch_size,
                        num_points,
                        num_workers,
                        split_classes=False,
                        sparse=False,
                        padding=3,
                        augment_rot=None,
                        k=-1,
                        seed_dataset=3):
    drop_last, shuffle = False, False
    transforms = [PCPadTransform(padding_dim=padding), PointTranspose()]
    if augment_rot is not None:
        transforms.insert(0, AugmRotate(augment_rot))

    transforms = Compose(transforms)
    data_args = {
        'data_root': data_root,
        'num_points': num_points,
        'transforms': transforms
    }

    if in_dataset.startswith('ShapeNet'):
        class_choice = in_dataset.split('_')[1]
        train_data = ShapeNetCore4k(
            **data_args,
            class_choice=class_choice,
            split='train',
            apply_fix_cellphone=True,
        )

        targets = train_data.targets()

    elif in_dataset.startswith('Syn2Real_SR'):  # 3DOS syn2real
        # sets SR1, SR2
        class_choice = in_dataset.split('_')[1]
        assert class_choice in ['SR1', 'SR2']
        train_data = ModelNet40_OOD(  # sampling performed as dataugm
            train=True,
            class_choice=class_choice,
            **data_args
        )

        targets = train_data.labels

    elif in_dataset.startswith("Real2Real_RR"):
        # sets RR1, RR2, RR3
        class_choice = in_dataset.split('_')[1]
        assert class_choice in ['RR1', 'RR2', 'RR3']
        sonn_args = {
            'sonn_split': 'main_split',
            'h5_file': "objectdataset.h5",
            **data_args
        }

        source_splits = {
            "RR1": "SR23",
            "RR2": "SR13",
            "RR3": "SR12",
        }

        whole_train_data = ScanObject(split='train', class_choice=source_splits[class_choice], **sonn_args)

        # split whole train into train and val (deterministic)
        total_len = len(whole_train_data)
        num_val = int(total_len * 10 / 100)
        train_idx, val_idx = train_test_split(np.arange(total_len), test_size=num_val, shuffle=True, random_state=42)
        train_data = torch.utils.data.Subset(whole_train_data, train_idx)

        targets = np.array(whole_train_data.labels)[train_idx]
    else:
        raise NotImplementedError

    if k > 0:
        if k > 1:
            k = int(k)
        idx = np.load('./ksplits.npz')[f'{in_dataset}_k{k}_seed{seed_dataset}']
        train_data = torch.utils.data.Subset(train_data, idx)
        targets = targets[idx]

    # we create a dataset for each class
    if split_classes:
        classes = set(targets)
        indices = [
            list(map(lambda i: i[0],
                     filter(lambda i: i[1] == cl,
                            enumerate(targets))))
            for cl in classes
        ]

        datasets = [torch.utils.data.Subset(train_data, sample_ids) for sample_ids in indices]

        train_loader = [torch.utils.data.DataLoader(  # CIUFF CIUFF
            d,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=sparse_collate_fn if sparse else None
        ) for d in datasets]

    else:
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            drop_last=drop_last,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=sparse_collate_fn if sparse else None
        )

    # for seed_dataset in [3, 7, 13, 31, 23, 71, 1027, 100, 9, 11]:
    #     for k in [5, 10, 20, 50, 0.1, 0.05, 0.01, 0.02]:
    #         datasets_splid = [get_k_dataset(t, k=k, seed_dataset=seed_dataset) for t in datasets]
    #         ixes = np.concatenate([np.asarray(i.dataset.indices)[i.indices] for i in datasets_splid])
    #         if type(k) is int:
    #             assert np.all(np.unique(np.asarray([train_data[i][1] for i in ixes]), return_counts=True)[1] == k)
    #         np.save(f'./k_split/{in_dataset}_k{k}_seed{seed_dataset}', ixes)
    # exit(0)

    return train_loader


def set_test_loader_3d(data_root, in_dataset, batch_size, num_points, num_workers, sparse=False,
                       padding=3):
    """
    Returns all dataloaders used for evaluation
    Return values:
        test_loader: test ID data loader - no augm, no shuffle, no drop_last
        tar1_loader: test OOD 1 data loader - no augm, shuffle, no drop_last
        tar2_loader: test OOD 2 data loader - no augm, shuffle, no drop_last

    """

    drop_last = False
    transforms = [PCPadTransform(padding_dim=padding), PointTranspose()]

    transforms = Compose(transforms)

    base_data_params = {
        'data_root': data_root,
        'num_points': num_points,
        'transforms': transforms,
        'apply_fix_cellphone': True,
    }

    if in_dataset.startswith('ShapeNet'):
        source = in_dataset.split('_')[1]
        t_1, t_2 = {'SN1', 'SN2', 'SN3'} - {source}
        # In-Distribution test data
        src_data = ShapeNetCore4k(**base_data_params, split='test', class_choice=source)
        # Out-Of-Distribution test data
        tar1_data = ShapeNetCore4k(**base_data_params, split='test', class_choice=t_1)
        tar2_data = ShapeNetCore4k(**base_data_params, split='test', class_choice=t_2)
        out_dataset = torch.utils.data.ConcatDataset([tar1_data, tar2_data])

    elif in_dataset.startswith('Syn2Real_SR'):  # 3DOS syn2real
        # sets SR1, SR2
        class_choice = in_dataset.split('_')[1]
        assert class_choice in [''
                                'SR1', 'SR2']
        source_splits = {'SR1': 'sonn_2_mdSet1',
                         'SR2': 'sonn_2_mdSet2'}
        target_splits = {'SR1': ['sonn_2_mdSet2', 'sonn_ood_common'],
                         'SR2': ['sonn_2_mdSet1', 'sonn_ood_common']}

        sonn_args = {
            'data_root': data_root,
            'sonn_split': 'main_split',
            'h5_file': "objectdataset.h5",
            'split': 'all',  # we use both training (unused) and test samples during evaluation
            'num_points': num_points,  # default: use all 2048 sonn points to avoid sampling randomicity
            'transforms': transforms,  # no augmentation applied at inference time
        }

        src_data = ScanObject(class_choice=source_splits[class_choice], **sonn_args)
        target_sets = [ScanObject(class_choice=split, **sonn_args) for split in target_splits[class_choice]]
        out_dataset = torch.utils.data.ConcatDataset(target_sets)
    elif in_dataset.startswith("Real2Real_RR"):
        # sets RR1, RR2, RR3
        class_choice = in_dataset.split('_')[1]
        assert class_choice in ['RR1', 'RR2', 'RR3']
        sonn_args = {
            'data_root': data_root,
            'sonn_split': 'main_split',
            'h5_file': "objectdataset.h5",
            'num_points': num_points,  # default: use all 2048 sonn points to avoid sampling randomly
            'transforms': transforms,
        }
        source_splits = {
            "RR1": "SR23",
            "RR2": "SR13",
            "RR3": "SR12"}
        target_splits = {
            "RR1": "sonn_2_mdSet1",
            "RR2": "sonn_2_mdSet2",
            "RR3": "sonn_ood_common"}

        src_data = ScanObject(split="test", class_choice=source_splits[class_choice], **sonn_args)
        out_dataset = ScanObject(split="all", class_choice=target_splits[class_choice], **sonn_args)

    else:
        raise NotImplementedError

    # loaders
    src_loader = torch.utils.data.DataLoader(
        src_data, batch_size=batch_size, drop_last=drop_last, num_workers=num_workers,
        collate_fn=sparse_collate_fn if sparse else None
    )
    out_loader = torch.utils.data.DataLoader(
        out_dataset, batch_size=batch_size, drop_last=drop_last, num_workers=num_workers,
        collate_fn=sparse_collate_fn if sparse else None
    )

    return src_loader, out_loader


if __name__ == '__main__':
    pass
    # x = set_train_loader_2d(root='/home/prabino/data/MCM', in_dataset='ImageNet10', split_classes=True)
    # x1 = set_ood_loader_2d(out_dataset='places365', preprocess=transforms.ToTensor())
    # print(next(iter(x1)))
