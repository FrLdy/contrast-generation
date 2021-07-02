from numpy.lib.histograms import _ravel_and_check_weights
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import json
import skimage.io as io
from pycocotools.coco import COCO
import utils.functionnal as f

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
        return *imgs, raw["super_class"]


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
        
        img = np.array(io.imread(self.coco.loadImgs(data["id"])[0]["coco_url"]))
        super_class = data["meta_class"]

        if img.shape[0] == 1:
            img = torch.Tensor(img).repeat(3, 1, 1)

        if self.transform is not None:
            img = self.transform(img)

        return img, super_class

class CocoDatasets():
    coco = None
    def __init__(self, data_dir="/Volumes/F_LEDOYEN/ms_coco/annotations") -> None:
        self.data_dir_pattern = data_dir+"/{}.json"
        self.data_files = {
            "coco_anns_all" : "instances_all2017",
            
            "all" : {
                "singles" : "imgs",
                "pairs" : "pairs"
            },
            
            "sport" : {
                "singles" : "imgs_sport",
                "pairs" : "pairs_sport"
            }  
        }
        self.data_files = f.map_nested_dicts(self.data_files, lambda k, v : self.data_dir_pattern.format(v))

        self.pairs_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Lambda(lambda x : x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.singles_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Lambda(lambda x : x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


        if CocoDatasets.coco is None :
            CocoDatasets.coco = COCO(self.data_files["coco_anns_all"]) 
        self.coco = CocoDatasets.coco 

    def pairs(self, slice):
        return CocoPairsDataset(
            coco=self.coco,
            pairs=json.load(open(self.data_files["sport"]["pairs"], "r"))[slice],
            transform=self.pairs_transform
        )

    def singles(self, slice=slice(None, None)):
        return CocoDataset(
            coco=self.coco,
            image_ids=json.load(open(self.data_files["sport"]["singles"], "r"))[slice],
            transform=self.pairs_transform
        )