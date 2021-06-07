import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
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

def coco_pairs_dataset(anns, pairs_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]) 

    return CocoPairsDataset(
        coco=anns if isinstance(anns, COCO) else COCO(anns),
        pairs=json.load(open(pairs_file, "r")),
        transform=transform
    )

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
        id, super_class = data["id"], data["meta_class"]
        img = io.imread(self.coco.loadImgs(id)[0]["coco_url"])
        if self.transform is not None:
            img = self.transform(img)
        return img, super_class


def coco_dataset(anns, ids_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]) 

    return CocoDataset(
        coco=anns if isinstance(anns, COCO) else COCO(anns),
        image_ids=json.load(open(ids_file, "r")),
        transform=transform
    )