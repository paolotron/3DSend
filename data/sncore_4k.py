import os.path as osp
import json
import sys

import h5py
import torch
from torch.utils.data import Dataset
from data.data_utils import *
from data.sncore_splits import *

class ShapeNetCore4k(Dataset):
    def __init__(self,
                 data_root=None,
                 split="train",
                 class_choice=None,
                 num_points=4096,
                 transforms=None,
                 apply_fix_cellphone=True):

        self.whoami = "ShapeNetCore4k"
        self.points = None  # all pointclouds from split
        self.synset_ids = None  # for each shape its synset id
        self.model_ids = None  # for each shape its model id
        assert split.lower() in ['train', 'test', 'val']
        self.split = split
        self.pc_dim = 4096
        self.data_dir = osp.join(data_root, "sncore_fps_4096")
        assert osp.exists(self.data_dir), f"{self.whoami} - {self.data_dir} does not exist"
        self.class_choice = list(eval(class_choice).keys())
        self.class_split = class_choice
        self.num_points = num_points
        self.transforms = transforms

        # load data split
        self.load_split()
        assert self.points is not None  # silent pycharm warnings
        assert self.synset_ids is not None
        assert self.model_ids is not None

        # sub-select pointclouds with synset choice
        if self.class_choice:
            # a list of synset Ids is expected for category selection
            assert isinstance(self.class_choice, list), \
                f"{self.whoami} {self.split} - class_choice should be a list of synset ids"
            chosen_idxs = [index for index, s_id in enumerate(self.synset_ids) if s_id in self.class_choice]
            assert len(chosen_idxs) > 0, f"ShapeNetCore4k {self.split} - No samples for class choice"
            self.synset_ids = [self.synset_ids[i] for i in chosen_idxs]
            self.model_ids = [self.model_ids[i] for i in chosen_idxs]
            self.points = self.points[chosen_idxs]

        if apply_fix_cellphone:
            # merge "cellphone" with "telephone"
            cellphone_sid = "02992529"
            telephone_sid = "04401088"
            cell_idxs = [index for index, s_id in enumerate(self.synset_ids) if s_id == cellphone_sid]
            if len(cell_idxs):
                print(f"{self.whoami} {self.split} - merging cellphone with telephone")
            for j in cell_idxs:
                # substitute synset_id of cellphones with telephone one
                self.synset_ids[j] = telephone_sid

        unique_ids = list(set(self.synset_ids))
        unique_ids.sort()
        self.num_classes = len(unique_ids)
        self.id_2_label = dict(zip(unique_ids, list(range(self.num_classes))))
        self.labels = np.asarray([self.id_2_label[s_id] for s_id in self.synset_ids])


        print(f"{self.whoami} {self.split} - points: {self.points.shape}, synset_ids: {len(self.synset_ids)}")
        print(f"{self.whoami} {self.split} - id_2_label: {self.id_2_label}")
        print(f"{self.whoami} {self.split} - sampled points: {self.num_points}")
        print(f"{self.whoami} {self.split} - transforms: {self.transforms}")

    def load_split(self):
        s_ids_fn = osp.join(self.data_dir, f"sncore_{self.split}_{self.pc_dim}_sids.json")
        assert osp.exists(s_ids_fn), f"synset ids file {s_ids_fn} does not exist"
        with open(s_ids_fn, 'r') as f:
            self.synset_ids = json.load(f)

        m_ids_fn = osp.join(self.data_dir, f"sncore_{self.split}_{self.pc_dim}_mids.json")
        assert osp.exists(m_ids_fn), "model ids file does not exist"
        with open(m_ids_fn, 'r') as f:
            self.model_ids = json.load(f)

        points_fn = osp.join(self.data_dir, f"sncore_{self.split}_{self.pc_dim}_points.h5")
        assert osp.exists(points_fn), "points file does not exist"
        f = h5py.File(points_fn, 'r')

        self.points = f['data'][:].astype('float32')
        f.close()

    def __getitem__(self, item):
        orig_item = item
        point_set = self.points[item]
        lbl = self.labels[orig_item]
        synset_id = self.synset_ids[item]
        model_id = self.model_ids[item]

        # sampling
        point_set = random_sample(point_set, num_points=self.num_points)

        # unit cube normalization
        point_set = pc_normalize(point_set)
        # data augm
        if self.transforms:
            point_set = self.transforms(point_set)

        # if item == 0 and self.class_split == 'SN1' and self.split == 'test':
        #     np.save(f'{len(self.transforms.transforms) - 2}pc', point_set[:3].T)

        return point_set, lbl  # , model_id, synset_id

    def targets(self):
        return self.labels

    def __len__(self):
        return len(self.points)
