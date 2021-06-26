import torch
import torch.utils.data as torch_data 


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

def coco_pairs_dataloaders(dataset, prop_train, batch_size, shuffle, num_workers):
    return {k:torch_data.DataLoader(v, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) 
            for k,v in split_dataset(dataset, prop_train).items()
    }

def coco_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch_data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) 



