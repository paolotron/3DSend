import os
import tqdm
from data.data_utils import *
import h5py


modelnet40_label_dict = {
    'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6,
    'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13,
    'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19,
    'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25,
    'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32,
    'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}

modelnet10_label_dict = {
    'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6,
    'sofa': 7, 'table': 8, 'toilet': 9}


############################################
# Closed Set for Modelnet to SONN experiments

SR1 = {
    "chair": 0,
    "bookshelf": 1,
    "door": 2,
    "sink": 3,
    "sofa": 4
}

SR2 = {
    "bed": 0,
    "toilet": 1,
    "desk": 2,
    "monitor": 3,
    "table": 2
}


# these are always OOD samples in cross-domain experiments!
modelnet_set3 = {
    'bathtub': 404,  # 1,  # simil sink???
    'bottle': 404,  # 5,
    'bowl': 404,  # 6,
    'cup': 404,  # 10,
    'curtain': 404,  # 11,
    'plant': 404,  # 26,  # simil bin???
    'flower_pot': 404,  # 15,  # simil bin???
    'vase': 404,  # 37,  # simil bin???
    'guitar': 404,  # 17,
    'keyboard': 404,  # 18,
    'lamp': 404,  # 19,
    'laptop': 404,  # 20,
    'night_stand': 404,  # 23,  # simil table - hard out-of-distrib.?
    'person': 404,  # 24,
    'piano': 404,  # 25,  # simil table - hard out-of-distrib.?
    'radio': 404,  # 27,
    'stairs': 404,  # 31,
    'tent': 404,  # 34,
    'tv_stand': 404,  # 36,  # simil table - hard out-of-distrib.?
}


################################################

class ModelNet40_OOD(data.Dataset):
    """
    ModelNet40 normal resampled. 10k sampled points for each shape
    Not using LMDB cache!
    """

    def __init__(self, num_points, data_root=None,
                 transforms=None, train=True,
                 class_choice="SR1"):
        super().__init__()
        self.whoami = "ModelNet40_OOD"
        self.split = "train" if train else "test"
        self.num_points = min(int(1e4), num_points)
        self.transforms = transforms
        assert isinstance(class_choice, str) and class_choice.startswith('SR'), \
            f"{self.whoami} - class_choice must be SRX name"
        self.class_choice = eval(class_choice)
        assert isinstance(self.class_choice, dict)
        self.num_classes = len(set(self.class_choice.values()))
        # reading data
        self.data_dir = os.path.join(data_root, "modelnet40_normal_resampled")
        if not osp.exists(self.data_dir):
            raise FileNotFoundError(f"{self.whoami} - {self.data_dir} does not exist")
        # cache
        cache_dir = osp.join(self.data_dir, "ood_sets_cache")  # directory containing cache files
        cache_fn = osp.join(cache_dir, f'{class_choice}_{self.split}.h5')  # path to cache file
        if os.path.exists(cache_fn):
            # read from cache file
            print(f"{self.whoami} - Reading data from h5py file: {cache_fn}")
            f = h5py.File(cache_fn, 'r')
            self.datas = np.asarray(f['data'][:])
            self.labels = np.asarray(f['label'][:], dtype=np.int64)
            f.close()
        else:
            # reading from txt files and building cache for next training/evaluation
            split_file = os.path.join(self.data_dir, f"modelnet40_{self.split}.txt")

            # all paths
            shape_ids = [
                line.rstrip()
                for line in open(
                    os.path.join(self.data_dir, split_file)
                )
            ]

            # all names
            shape_names = ["_".join(x.split("_")[0:-1]) for x in shape_ids]

            # class choice
            chosen_idxs = [index for index, name in enumerate(shape_names) if name in self.class_choice.keys()]
            self.shape_ids = [shape_ids[_] for _ in chosen_idxs]
            self.shape_names = [shape_names[_] for _ in chosen_idxs]
            del shape_ids, shape_names

            # read chosen data samples from disk
            self.datapath = [
                (
                    self.shape_names[i],
                    os.path.join(self.data_dir, self.shape_names[i], self.shape_ids[i])
                    + ".txt",
                )
                for i in range(len(self.shape_ids))
            ]
            self.datas = []
            self.labels = []
            for i in tqdm.trange(len(self.datapath), desc=f"{self.whoami} loading data from txts", dynamic_ncols=True):
                fn = self.datapath[i]
                point_set = np.loadtxt(fn[1], delimiter=",").astype(np.float32)
                point_set = point_set[:, 0:3]  # remove normals
                category_name = self.shape_names[i]  # 'airplane'
                cls = self.class_choice[category_name]
                self.datas.append(point_set)  # [1, 10000, 3]
                self.labels.append(cls)

            self.datas = np.stack(self.datas, axis=0)  # [num_samples, 10000, 3]
            self.labels = np.asarray(self.labels, dtype=np.int64)  # [num_samples, ]

            # make cache
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)
            print(f"Saving h5py datataset to: {cache_fn}")

            with h5py.File(cache_fn, "w") as f:
                f.create_dataset(name='data', data=self.datas, dtype=np.float32, chunks=True)
                f.create_dataset(name='label', data=self.labels, dtype=np.int64, chunks=True)

            print(f"{self.whoami} - Cache built for split: {self.split}, set: {self.class_choice} - "
                  f"datas: {self.datas.shape} labels: {self.labels.shape} ")
        print(f"{self.whoami} - "
              f"split: {self.split}, "
              f"categories: {self.class_choice}")

    def __getitem__(self, item):
        rotation = 0
        orig_item = item
        point_set = self.datas[item]
        lbl = self.labels[orig_item]

        # sampling
        point_set = random_sample(point_set, num_points=self.num_points)
        # unit cube normalization
        point_set = pc_normalize(point_set)

        # data augm
        if self.transforms:
            point_set = self.transforms(point_set)

        return point_set, lbl

    def __len__(self):
        return len(self.labels)
