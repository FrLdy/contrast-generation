import json
from datasets.coco_dataset import CocoDataset, CocoPairsDataset
from pycocotools.coco import COCO
import torch
import torch.utils.data as torch_data

def coco_pairs_dataset(anns, pairs_file, transform):
    return CocoPairsDataset(
        coco=anns if isinstance(anns, COCO) else COCO(anns),
        pairs=json.load(open(pairs_file, "r")),
        transform=transform
    )

def coco_dataset(anns, ids_file, transform):
    return CocoDataset(
        coco=anns if isinstance(anns, COCO) else COCO(anns),
        image_ids=json.load(open(ids_file, "r")),
        transform=transform
    )

def split_dataset(dataset, prop_train, prop_test):
    dataset_size = len(dataset)
    lengths = [round(dataset_size*prop) for prop in [prop_train, prop_test]]

    return torch_data.random_split(
        dataset, 
        lengths=lengths,
        generator=torch.Generator().manual_seed(42)
    )

def compute_mean_std(dataloader):
    mean = 0
    std = 0
    nb_samples = 0
    for batch in dataloader:
        batch_size = batch.size(0)
        data = batch.view(batch_size, batch.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_size

    return mean / nb_samples, std / nb_samples