import torch
import torch.utils.data as torch_data

class CocoDataLoaders():
    def __init__(self, coco_datasets, num_workers) -> None:
        self.coco_datasets = coco_datasets
        self.num_workers = num_workers

    def pairs(self, prop_train, batch_size, shuffle, slice=slice(None, None)):
        return {k:torch_data.DataLoader(v, batch_size=batch_size, shuffle=(shuffle and k != "test"), num_workers=self.num_workers) 
                for k,v in CocoDataLoaders.split_dataset(self.coco_datasets.pairs(slice=slice), prop_train).items()
        }

    def singles(self, prop_train, batch_size, shuffle, slice=slice(None, None)):
        return {
            k:torch_data.DataLoader(v, batch_size=batch_size, shuffle=(shuffle and k != "test"), num_workers=self.num_workers) 
            for k,v in self.split_dataset(self.coco_datasets.singles(slice=slice), prop_train).items()
        }

    @staticmethod    
    def split_dataset(dataset, prop_train):
        dataset_size = len(dataset)
        lengths = [round(dataset_size*prop) for prop in [prop_train, 1-prop_train]]

        return dict(zip(
            ["train", "test"], 
            torch_data.random_split(
            dataset, 
            lengths=lengths,
            generator=torch.Generator().manual_seed(42)
        )))
