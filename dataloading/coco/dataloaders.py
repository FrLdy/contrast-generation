import torch
import torch.utils.data as torch_data
from torch.utils.data import dataloader

class CocoDataLoaders():
    def __init__(self, coco_datasets, num_workers, type="sport") -> None:
        self.coco_datasets = coco_datasets
        self.num_workers = num_workers
        self.type = type

    def pairs(self, prop_train=0, prop_val=0, prop_test=0, batch_size=32, shuffle=True, slice=slice(None, None)):
        return self.build_dataloaders("pairs", prop_train, prop_val, prop_test, batch_size, shuffle, slice)

    def singles(self, prop_train=0, prop_val=0, prop_test=0, batch_size=32, shuffle=True, slice=slice(None, None)):
        return self.build_dataloaders("singles", prop_train, prop_val, prop_test, batch_size, shuffle)

    def build_dataloaders(self, type, prop_train, prop_val, prop_test, batch_size, shuffle, slice):
        if type == "pairs" :
            dataset = self.coco_datasets.pairs(slice=slice, type=self.type)
        elif type == "singles" :
            dataset = self.coco_datasets.singles(slice=slice, type=self.type)
        
        return {
            k:torch_data.DataLoader(
                v, 
                batch_size=batch_size, 
                shuffle=(shuffle and k != "test"), 
                num_workers=self.num_workers) 
            for k,v in self.split_dataset(dataset, prop_train, prop_val, prop_test).items()
        }

    @staticmethod    
    def split_dataset(dataset, prop_train=0, prop_val=0, prop_test=0):
        dataset_size = len(dataset)
        parts, lengths = [], []
        for n,p in zip(["train", "val", "test"], [prop_train, prop_val, prop_test]):
            if p > 0:
                parts.append(n)
                lengths.append(dataset_size*p) 


        return dict(zip(
            parts, 
            torch_data.random_split(
            dataset, 
            lengths=lengths,
            generator=torch.Generator().manual_seed(42)
        )))
