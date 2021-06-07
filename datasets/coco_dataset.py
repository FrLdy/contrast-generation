import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import json
import skimage.io as io
from pycocotools.coco import COCO

class CocoPairsDataset(data.Dataset):
    def __init__(self, coco, paires, transform=None):
        super().__init__()
        self.coco = coco
        self.paires = paires 
        self.size = len(self.paires)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raw = self.paires[index]
        extract_img = lambda i : io.imread(self.coco.loadImgs(raw[f"img_{i}"]["id"])[0]["coco_url"])
        if self.transform is not None:
            imgs = [self.transform(extract_img(i)) for i in range(1, 3)]
        else : 
            imgs = [extract_img(i) for i in range(1, 3)]
        return imgs + [raw["super_class"]]

def coco_pairs_dataset(anns_file, paires_file):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]) 

    return CocoPairsDataset(
        coco=COCO(anns_file),
        paires=json.load(open(paires_file, "r")),
        transform=transform
    )
