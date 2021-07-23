# import some common libraries
import torch
import torch.nn as nn


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torch.nn.functional import mse_loss


class ContentContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = self.__build_cfg() 
        self.predictor = DefaultPredictor(self.cfg)
        self.classes = self.predictor.metadata.thing_classes
        self.nb_classes = len(self.classes)

    def forward(self, x1, x2, gen):
        x_union = self.__build_pred_vector(x1) * self.__build_pred_vector(x2)
        gen = self.__build_pred_vector(gen)
        
        return mse_loss(x_union, gen)
        
    
    def __build_pred_vector(self, x):
        labels = torch.zeros(self.nb_classes)
        preds = self.predictor(x).pred_classes.unique()
        for p in preds:
            labels[p] = 1
        
        return labels

    def __build_cfg(self):
        cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        
        return cfg
        

class TestContentContrastLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1, x2, gen):
        return 0