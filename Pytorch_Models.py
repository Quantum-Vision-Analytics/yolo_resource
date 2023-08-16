from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import argparse
import time
from pathlib import Path
from os.path import splitext
from FileGenerator import FileGenerator
import threading
import os
import glob
from ModelInferenceHandler import ModelInferenceHandler
from torchvision.models.detection import(
            fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights,
            retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights,
            fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights,
            ssd300_vgg16, SSD300_VGG16_Weights)

class Pytorch_Models(ModelInferenceHandler):
    def __init__(self, options:argparse.ArgumentParser, model_name:str):
        super().__init__(options)
        self.model_name = model_name

    def LoadResources(self):
        self.source, self.view_img, self.save_txt = self.opt.source, self.opt.view_img, self.opt.save_txt
        self.save_img = not self.opt.nosave and not self.source.endswith('.txt')  # save inference images

        # Define a dictionary mapping model names to their functions and weights
        models_info = {
            "fasterrcnn": (fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
            "retinanet": (retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT),
            "fcos": (fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights.DEFAULT),
            "ssd300": (ssd300_vgg16, SSD300_VGG16_Weights.DEFAULT)
        }

        # Check if the given model name exists in the dictionary
        if self.model_name in models_info:
            model, self.weights = models_info[self.model_name]
            self.model = model(weights=self.weights, box_score_thresh = self.opt.conf_thres, score_thresh = self.opt.conf_thres)
            
        else:
            print("Invalid model name.")

        # Initialize model with the best available weights
        self.model.eval()

        # Initialize the inference transforms
        self.preprocess = self.weights.transforms()

    def Preprocess(self, batch:list):
        # Dimensions of the image, [width,height]
        dim_list = [[len(im[0][0]), len(im[0])] for im in batch]
        # Process images and returns the output with dimensions
        return (dim_list,[self.preprocess(im) for im in batch])

    def Predict(self, batch:list):
        return [self.model([im])[0] for im in batch]

    def Postprocess(self, batch:list, dimensions:list, img_names:list):
        for prediction, img_dims, img_name in zip(batch, dimensions, img_names):
            with open(splitext(img_name)[0] + ".txt", "w") as f:
                for label, bb in zip(list(prediction["labels"]), list(prediction["boxes"])):
                    org_bb = bb.detach().cpu().numpy()
                    output = "{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(label.item(), (org_bb[0] + org_bb[2])/2/img_dims[0], (org_bb[1] + org_bb[3])/2/img_dims[1], abs(org_bb[2] - org_bb[0])/img_dims[0], abs(org_bb[3] - org_bb[1])/img_dims[1])
                    #print(f"{label.item()} {org_bb[0]} {org_bb[1]} {org_bb[2]} {org_bb[3]}")
                    f.write(output)
                    
        with open("val2017\classes.txt", "w") as f:
            for x in self.weights.meta["categories"]:
                f.write(x + "\n")

    def Detect(self):
        self.LoadResources()

        # img_name = "val2017/000000000139.jpg"
        # img_name2 = "val2017/000000000632.jpg"
        # img_names = [img_name, img_name2]
        # img_names2 = [img_name]
        # img = read_image(img_name)
        # img2 = read_image(img_name2)
        # batch = [img, img2]
        # batch2 = [img]

        # dims, batch = self.Preprocess(batch)
        # batch = self.Predict(batch)
        # self.Postprocess(batch, dims, img_names)

        # Supported image types
        img_formats = ['jpg', 'jpeg', 'png']

        abs_pos = str(Path(self.source).absolute())
        if os.path.isdir(self.source):
            file_names = sorted(glob.glob(os.path.join(abs_pos, '*.*')))  # dir
        elif os.path.isfile(self.source):
            file_names = [self.source]  # files
        files = [x for x in file_names if x.split('.')[-1].lower() in img_formats]

        for img_name in files:
            batch = [read_image(img_name)]
            image_names = [img_name]

            dims, batch = self.Preprocess(batch)
            batch = self.Predict(batch)
            self.Postprocess(batch, dims, image_names)

        # Step 3: Apply inference preprocessing transforms
        #batch = [self.preprocess(img)]

        # Step 4: Use the model and visualize the prediction
        #prediction = model(batch)[0]

        #labels = [weights.meta["categories"][i] for i in prediction["labels"]]

        '''
        with open(splitext(img_name)[0] + ".txt", "w") as f:
            for label, bb in zip(list(prediction["labels"]), list(prediction["boxes"])):
                org_bb = bb.detach().cpu().numpy()
                output = "{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(label.item(), (org_bb[0] + org_bb[2])/2/img_dims[0], (org_bb[1] + org_bb[3])/2/img_dims[1], abs(org_bb[2] - org_bb[0])/img_dims[0], abs(org_bb[3] - org_bb[1])/img_dims[1])
                #print(output)
                print(f"{label.item()} {org_bb[0]} {org_bb[1]} {org_bb[2]} {org_bb[3]}")
                f.write(output)

        with open("classes.txt", "w") as f:
            for x in weights.meta["categories"]:
                if not x.startswith("_"):
                    f.write(x + "\n")
        '''
        

    def Train(self):
        pass
