import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import matplotlib.pyplot as plt
import json
import skimage.io as io
from pycocotools.coco import COCO

class CocoPairsDataset(data.Dataset):
    def __init__(self, coco, pairs, transform=None):
        super().__init__()
        self.coco = coco
        self.pairs = pairs 
        self.size = len(self.pairs)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raw = self.pairs[index]
        extract_img = lambda i : io.imread(self.coco.loadImgs(raw[f"img_{i}"]["id"])[0]["coco_url"])
        if self.transform is not None:
            imgs = [self.transform(extract_img(i)) for i in range(1, 3)]
        else : 
            imgs = [extract_img(i) for i in range(1, 3)]
        return imgs + [raw["super_class"]]


class CocoDataset(data.Dataset):
    def __init__(self, coco, image_ids, transform=None):
        self.coco = coco
        self.image_ids = image_ids
        self.transform = transform
        self.size = len(self.image_ids)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data = self.image_ids[index]
        
        img = io.imread(self.coco.loadImgs(data["id"])[0]["coco_url"])
        super_class = data["meta_class"]

        if img.shape[0] == 1:
            img = torch.Tensor(img).repeat(3, 1, 1)

        if self.transform is not None:
            img = self.transform(img)

        return img, super_class
