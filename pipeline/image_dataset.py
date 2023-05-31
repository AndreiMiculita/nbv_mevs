import os
from sys import exit

import lmdb
import matplotlib.pyplot as plt
import msgpack_numpy
import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data.dataset import Dataset

# Base dir is one level up
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


class ModelNet10ImageDataset(Dataset):
    def __init__(self, transforms=None, train=True):
        super().__init__()

        self.transforms = transforms

        self._cache = os.path.join(DATA_DIR, "image-dataset_cache")

        if not os.path.exists(self._cache):
            self.folder = "image-dataset"
            self.data_dir = os.path.join(DATA_DIR, self.folder)

            if not os.path.exists(self.data_dir):
                print("Need data dir: ", self.data_dir)
                exit(1)

            self.train = train

            self.catfile = os.path.join(self.data_dir, "modelnet10_shape_names.txt")
            self.cat = [line.rstrip() for line in open(self.catfile)]
            self.classes = dict(zip(self.cat, range(len(self.cat))))

            os.makedirs(self._cache)

            print("Converted to LMDB for faster dataloading while training")
            for split in ["train", "test"]:
                if split == "train":
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet10_train.txt")
                        )
                    ]
                else:
                    shape_ids = [
                        line.rstrip()
                        for line in open(
                            os.path.join(self.data_dir, "modelnet10_test.txt")
                        )
                    ]

                print("Shape IDs:", shape_ids)

                shape_names = ["_".join(x.split("_")[0:-7]) for x in shape_ids]
                print("Shape names:", shape_names)

                shape_indexes = ["_".join(x.split("_")[-7:-6]) for x in shape_ids]

                # Convert to int
                shape_indexes = [int(x) for x in shape_indexes]

                print(set(shape_indexes))

                # exit()

                # list of (shape_name, shape_txt_file_path) tuple
                self.datapath = [
                    (
                        shape_names[i],
                        os.path.join(self.data_dir, shape_names[i], shape_ids[i])
                        + ".png",
                    )
                    for i in range(len(shape_ids))
                ]

                with lmdb.open(
                        os.path.join(self._cache, split), map_size=1 << 36
                ) as lmdb_env, lmdb_env.begin(write=True) as txn:
                    print(len(self.datapath))

                    idx = 0
                    pbar = tqdm.trange(len(self.datapath))
                    for i in pbar:
                        fn = self.datapath[i]

                        print(fn[1])

                        image = Image.open(fn[1])
                        image = np.array(image)

                        cls = self.classes[self.datapath[i][0]]
                        cls = int(cls)

                        txn.put(
                            str(idx).encode(),
                            msgpack_numpy.packb(
                                dict(image=image, lbl=cls), use_bin_type=True
                            ),
                        )
                        idx += 1

        self._lmdb_file = os.path.join(self._cache, "train" if train else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = None

    def __getitem__(self, idx):
        if self._lmdb_env is None:
            self._lmdb_env = lmdb.open(
                self._lmdb_file, map_size=1 << 36, readonly=True, lock=False
            )

        with self._lmdb_env.begin(buffers=True) as txn:
            ele = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)

        image = ele["image"]

        if self.transforms is not None:
            image = self.transforms(image)

        return image, ele["lbl"]

    def __len__(self):
        return self._len


if __name__ == "__main__":
    from torchvision import transforms

    transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224))
        ]
    )
    dset = ModelNet10ImageDataset(transforms=transforms, train=True)
    # Show some images on a grid
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, 10):
        img, lbl = dset[i]
        ax = fig.add_subplot(3, 3, i)
        ax.imshow(img)
        ax.set_title(lbl)
    dloader = torch.utils.data.DataLoader(dset, batch_size=32, shuffle=True)
