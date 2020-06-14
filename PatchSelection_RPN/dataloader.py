import torch
from torch.utils.data import Dataset
import pandas as pd
from os import listdir
from numpy import clip
from skimage import io
from skimage.color import rgb2gray
from skimage.transform import resize

class DataProcessor(Dataset):
    def __init__(self, imgs_dir=None, csv_path=None, transformations=None, resize_img=False):
        self.imgs_dir = imgs_dir
        self.csv_path = csv_path
        self.transformations = transformations
        self.resize = resize_img
        self.imgs_ids = [file for file in listdir(imgs_dir)]

    @classmethod
    def preprocess(cls, img, new_size=False, normalize=False, img_transforms=None):
        """
        w, h = img.shape
        """
        if new_size:
            img = resize(img, (new_size, new_size))

        if normalize:
            if img.max() > 1:
                img = (img - img.min()) / (img.max() - img.min())
            img = (img - img.mean()) / img.std()
            img = clip(img, -1.0, 1.0)
            img = (img + 1.0) / 2.0

        if img_transforms:
            img = img_transforms(img)
        return img

    def __getitem__(self, i):
        img_idx = self.imgs_ids[i]
        img_file = self.imgs_dir + img_idx
        coord_file = pd.read_csv(self.csv_path)
        img = rgb2gray(io.imread(img_file).astype('float32'))
        img = self.preprocess(img, self.resize, normalize=True,
                              img_transforms=self.transformations)

        # Set up annotations
        file_name = img_file.split("\\")[-1]
        name_filter = coord_file["image_names"] == file_name
        filtered = coord_file[name_filter]
        tensor_coords = []
        for index, row in filtered.iterrows():
            tensor_coords.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
        boxes = torch.as_tensor(tensor_coords, dtype=torch.float32)
        num_objs = boxes.size(0)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        img_id = torch.tensor([i])  # tensor([1])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Set up annotations
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = area
        my_annotation["iscrowd"] = iscrowd
        return img, my_annotation

    def __len__(self):
        return len(self.imgs_ids)