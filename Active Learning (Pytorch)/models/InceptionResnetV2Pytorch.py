import torch
import torch.nn as nn
from torchvision import models
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class InceptionResnetV2Pytorch(nn.Module):
    def __init__(self, fine_tune=False, num_classes=2):
        super(InceptionResnetV2Pytorch, self).__init__()
        self.fine_tune = fine_tune
        self.num_classes = num_classes

    def get_model(self):
        model_ft = models.inception_v3(pretrained=True)
        set_parameter_requires_grad(model=model_ft, feature_extracting=False)
        self.num_features = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(self.num_features, self.num_classes)
        self.num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(self.num_features, self.num_classes)
        self.pretrained = model_ft
        return model_ft
