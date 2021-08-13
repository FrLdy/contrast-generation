import os
from numpy.lib.histograms import _ravel_and_check_weights
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import json
import skimage.io as io
from pycocotools.coco import COCO
import contrast_generation.utils.functionnal as f

class CocoDataset(data.Dataset):

    def __init__(self, samples, transform=None, coco=None, data_dir=None):
        super().__init__()
        self.coco = coco
        self.data_dir = os.path.abspath(data_dir)
        self.samples = samples
        self.size = len(samples)
        self.transform = transform
        if coco is None and data_dir is None:
            raise ValueError("coco OR data_dir must be set")

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        sample = self.samples[index]

        ids = self._ids(sample)
        urls = self._urls(sample)
        paths = self._local_paths(sample)
        
        imgs = []
        for id, url, path in zip(ids, urls, paths):
            if path is not None:
                img = self._img_by_local_path(path)
            elif url is not None:
                img = self._img_by_url(url)
            elif id:
                img = self._img_by_coco(path)
            imgs.append(img)

        label = self._label(sample)

        return *imgs, label

    def _ids(self, sample):
        raise NotImplementedError

    def _urls(self, sample):
        raise NotImplementedError

    def _label(self, sample):
        raise NotImplementedError
    
    def _local_paths(self, sample):
        raise NotImplementedError

    def _img_by_coco(self, id):
        path = self.coco.loadImgs(id)[0]["coco_url"]
        return self._get_np_img(path)

    def _img_by_url(self, url):
        return self._get_np_img(url)

    def _img_by_local_path(self, path):
        return self._get_np_img(path)

    def _get_np_img(self, path):
        img = io.imread(path)
        if self.transform is not None: 
            img = self.transform(img)

        if img.shape[0] == 1:
            img = torch.Tensor(img).repeat(3, 1, 1)

        return img 

    def _local_path(self, id):
        return os.path.join(self.data_dir, f'{id}.png')
    


class CocoPairsDataset(CocoDataset):
    def __init__(self, samples, transform=None, coco=None, data_dir=None):
        super().__init__(samples, transform, coco, data_dir)

    def _label(self, sample):
        return sample["super_class"]
        
    def _ids(self, sample):
        return self._values(sample, "id") 

    def _urls(self, sample):
        return self._values(sample, "url")

    def _local_paths(self, sample):
        paths = []
        for id in self._ids(sample):
            path = self._local_path(id)
            if not os.path.exists(path):
                path = None
            paths.append(path)

        return paths

    def _values(self, sample, key):
        res = []
        for i in range(1, 3):
            img_dict  = sample[f"img_{i}"]
            if key in img_dict:
                res.append(img_dict[key])
            else:
                res.append(None)
        
        return res 


class CocoSinglesDataset(CocoDataset):
    def __init__(self, samples, transform=None, coco=None, data_dir=None):
        super().__init__(samples, transform, coco, data_dir)

    def _label(self, sample):
        return [sample["meta_class"]]
        
    def _ids(self, sample):
        return [sample["id"]]

    def _urls(self, sample):
        return [sample["url"]]

    def _local_paths(self, sample):
        path = self._local_path(sample["id"])
        if not os.path.exists(path):
            path = None
        return [path]
     

class CocoDatasets():

    def __init__(self, imgs_dir, anns_dir="/Volumes/F_LEDOYEN/ms_coco/annotations", coco=False) -> None:
        self.anns_dir_pattern = anns_dir+"/{}.json"
        self.anns_files = {
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
        self.anns_files = f.map_nested_dicts(self.anns_files, lambda k, v : self.anns_dir_pattern.format(v))

        self.pairs_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((128, 128)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.singles_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.imgs_dir = imgs_dir
        self.coco = COCO(self.anns_files["coco_anns_all"]) if coco else None

    def pairs(self, type="sport", slice=slice(None, None)):
        return CocoPairsDataset(
            coco=self.coco,
            samples=json.load(open(self.anns_files[type]["pairs"], "r"))[slice],
            transform=self.pairs_transform,
            data_dir=self.imgs_dir
        )

    def singles(self, type="sport", slice=slice(None, None)):
        return CocoSinglesDataset(
            coco=self.coco,
            samples=json.load(open(self.anns_files[type]["singles"], "r"))[slice],
            transform=self.pairs_transform,
            data_dir=self.imgs_dir
        )