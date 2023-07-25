import argparse
import time
from pathlib import Path

from FileGenerator import FileGenerator
import threading
from ModelInferenceHandler import ModelInferenceHandler

import json
from PIL import Image

import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet

class EfficientNet(ModelInferenceHandler):
    def __init__(self, options:argparse.ArgumentParser):
        super().__init__(options)

    def Train(self):
        pass

    def LoadResources(self,save_img=False):
        self.model_name = 'efficientnet-b0'
        self.image_size = EfficientNet.get_image_size(self.model_name) # 224
        self.labels_map = json.load(open('labels_map.txt'))
        self.labels_map = [self.labels_map[str(i)] for i in range(1000)]

    def Preprocess(self, batch:list):
        img = Image.open('img.jpg')
        for x, img in enumerate(batch):
            tfms = transforms.Compose([transforms.Resize(self.image_size), transforms.CenterCrop(self.image_size), 
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
            img = tfms(img).unsqueeze(0)
            batch[x] = img
        return batch

    # Classify with EfficientNet
    def Predict(self, batch:list):
        for x, img in enumerate(batch):
            model = EfficientNet.from_pretrained(self.model_name)
            model.eval()
            with torch.no_grad():
                logits = model(img)
            preds = torch.topk(logits, k=5).indices.squeeze(0).tolist()
            batch[x] = preds, logits
        return batch

    def Postprocess(self, batch:list):
        for preds, logits in batch:
            print('-----')
            for idx in preds:
                label = self.labels_map[idx]
                prob = torch.softmax(logits, dim=1)[0, idx].item()
                print('{:<75} ({:.2f}%)'.format(label, prob*100))
        
        