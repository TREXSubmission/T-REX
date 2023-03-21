import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from pytorch_grad_cam import GradCAM,ScoreCAM,LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget,ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.metrics.cam_mult_image import CamMultImageConfidenceChange
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform
from PIL import Image
import torch
from src.utils.get_config import get_config,get_config_original
from src.utils.get_model import get_model
import torch
import argparse
from src.modules.classification import ClassificationModule
import random
import tqdm
parser = argparse.ArgumentParser()

parser.add_argument("--config-file", help="Config File")
parser.add_argument("--model", help="Config File")

config,dataset = get_config_original(parser.parse_args())
torch.manual_seed(config.seed)
torch.use_deterministic_algorithms(True)
model_name = get_model(parser.parse_args(),config.num_classes,config.last_layer,final_activation=config.final_activation)
model = ClassificationModule(config, model_name)
ckpt = torch.load('/home/User/Downloads/Sigmoid/logs/deit/2023-01-10-17-01-50/last.ckpt', map_location='cpu')
state_dict = ckpt['state_dict']
print(model.load_state_dict(state_dict, strict=True))
model.eval()

average_drop_wrong_prediction = {
    'scores':[]
}
data = set()
len_d = len(dataset)
for sidx in tqdm.tqdm(range(len_d)):
    iters=0
    idx=sidx
    input_tensor = dataset[idx]['image'].cpu()[None]
    iters+=1
    res = model(input_tensor)[0].detach().numpy().argmax()
    average_drop_wrong_prediction['scores'].append(model(input_tensor)[0].detach().numpy().tolist())

    targets = [ClassifierOutputTarget(res)]

    target_layers = [model.model.model[0].blocks[-1].norm1]
    img = (input_tensor[0]).numpy().swapaxes(0,1).swapaxes(1,2)[:,:,::-1]
    with ScoreCAM(model=model, target_layers=target_layers,reshape_transform=vit_reshape_transform) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
        cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
    cam = np.uint8(255*grayscale_cams[0, :])
    cam = cv2.merge([cam, cam, cam])
    images = np.hstack((np.uint8(255*img), cam , cam_image))
    # Now lets see how to evaluate this explanation:
    from pytorch_grad_cam.metrics.road import *
    for x in range(10,61,10):
        cam_metric = ROADMostRelevantFirst(percentile=100-x)
        scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
        score = scores[0]
        try:
            average_drop_wrong_prediction[f'correct_{x}'].append(score.item())
        except:
            average_drop_wrong_prediction[f'correct_{x}'] = [score.item()]
        print(score.item())
        
        cam_metric = ROADLeastRelevantFirst(percentile=100-x)
        scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
        score = scores[0]
        try:
            average_drop_wrong_prediction[f'correct_small_{x}'].append(score.item())
        except:
            average_drop_wrong_prediction[f'correct_small_{x}'] = [score.item()]


    cam_metric = CamMultImageConfidenceChange()
    scores, visualizations = cam_metric(input_tensor, grayscale_cams, targets, model, return_visualization=True)
    
    score = scores[0]
    try:
        average_drop_wrong_prediction[f'mult'].append(score.item())
    except:
        average_drop_wrong_prediction[f'mult'] = [score.item()]

    score = scores[0]
    try:
        average_drop_wrong_prediction[f'avg'].append(grayscale_cams.mean().item())
    except:
        average_drop_wrong_prediction[f'avg'] = [grayscale_cams.mean().item()]
    try:
        average_drop_wrong_prediction[f'label'].append(dataset[idx]['label'])
    except:
        average_drop_wrong_prediction[f'label'] = [dataset[idx]['label']]
    try:
        average_drop_wrong_prediction[f'pred'].append(res.item())
    except:
        average_drop_wrong_prediction[f'pred'] = [res.item()]
import json
with open('deit_last.json', 'w') as f:
    json.dump(average_drop_wrong_prediction, f)