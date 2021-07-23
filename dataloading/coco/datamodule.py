import pytorch_lightning as pl

class Pairs(pl.LightningDataModule):
    def __init__(self, train_transforms, val_transforms, test_transforms, dims):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
    
    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return super().train_dataloader()