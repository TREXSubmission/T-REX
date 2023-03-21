from robustness import datasets, defaults, model_utils, train
from robustness.tools import helpers
from torch import nn
import torch
from torch import Tensor
from typing import Optional, Any, Callable, List, Type, Union

class RobustResNet(nn.Module):
    def __init__(self,num_classes=1000, resume_path=None,
                    final_activation: Union[nn.Sigmoid(),nn.Softmax()] = nn.Sigmoid()):
            super().__init__()
            self.model =  robust_resnet50_builder(num_classes, resume_path=resume_path)
            self.final_activation = final_activation
            self.last_conv_layer = None

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        x = self.final_activation(x)
        return x

    def add_last_conv_layer(self):
        self.last_conv_layer = self.model.layer4[2].conv3


def robust_resnet50_builder(num_classes: int = 1000, resume_path=None):
    model, _ = model_utils.make_and_restore_model(
        arch='resnet50',
        dataset=datasets.ImageNet(''),
        resume_path=resume_path
    )

    # getting rid of adversarial wrapper
    model = model.model

    # replace last classification layer
    with torch.no_grad():
        model.fc = nn.Linear(
            in_features=model.fc.in_features,
            out_features=num_classes,
        )

    return model