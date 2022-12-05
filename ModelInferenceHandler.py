import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

from FileGenerator import FileGenerator

class ModelInferenceHandler:
    def __init__(self, options = None):
        self.fileGen = FileGenerator()
        if(options is not None):
            self.SetOptions(options)

    def SetOptions(self,options):
        self.opt = options

    def Train(self):
        pass

    def Preprocess(self,save_img=False):
        source, weights, self.view_img, self.save_txt, imgsz, trace = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size, not self.opt.no_trace
        self.save_img = save_img; self.save_img = not self.opt.nosave and not source.endswith('.txt')  # save inference images
        self.webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        # Get full system directory with relevant directory
        self.save_dir = Path(increment_path(Path(self.opt.project) / self.opt.name, exist_ok=self.opt.exist_ok))  # increment run
        (self.save_dir / 'labels' if self.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, self.opt.img_size)

        if self.half:
            self.model.half()  # to FP16

        self.classify = False
        if self.classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

        # Set Dataloader
        self.vid_path, self.vid_writer = None, None
        if self.webcam:
            self.view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            self.dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        self.old_img_w = self.old_img_h = imgsz
        self.old_img_b = 1

        for imgIndex, data in enumerate(self.dataset.data):
            path, img, im0s, vid_cap = data
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            self.dataset.data[imgIndex] = [path, img, im0s, vid_cap]

        inputClasses = self.opt.classes
        if(inputClasses is not None):
            self.filterClasses = []
            for i, cls in enumerate(self.names):
                if(cls in inputClasses):
                    self.filterClasses.append(i)
        else:
            self.filterClasses = None
                


    def Predict(self):
        self.preds = []
        self.timelaps = []
        for imgIndex, data in enumerate(self.dataset.data):
            path, img, im0s, vid_cap = data
            timelap = []
            # Warmup
            if self.device.type != 'cpu' and (self.old_img_b != img.shape[0] or self.old_img_h != img.shape[2] or self.old_img_w != img.shape[3]):
                self.old_img_b = img.shape[0]
                self.old_img_h = img.shape[2]
                self.old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.opt.augment)[0]

            # Inference
            timelap.append(time_synchronized())

            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.opt.augment)[0]
            timelap.append(time_synchronized())

            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.filterClasses, agnostic=self.opt.agnostic_nms, multi_label=self.opt.multi_label)
            timelap.append(time_synchronized())

            self.timelaps.append(timelap)

            # Apply Classifier
            if self.classify:
                pred = apply_classifier(pred, self.modelc, img, im0s)
            self.preds.append(pred)
        self.dataset.data[imgIndex] = [path, img, im0s, vid_cap]

    def Postprocess(self):
    # Process detections
        detRes = []
        #Will store all the detections for annotation verifier
        for imgIndex, data in enumerate(self.dataset.data):
            path, img, im0s, vid_cap = data
            pred = self.preds[imgIndex]
            detections = []
            for i, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), self.dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(self.save_dir / p.name)  # img.jpg
                txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if self.opt.save_conf else (cls, *xywh)  # label format
                        detections.append(('%g ' * len(line)).rstrip() % line)
                        if self.save_txt:  # Write to file
                            self.fileGen.Generate_Annotation(txt_path, line)
                            
                        if self.save_img or self.view_img:  # Add bbox to image
                            label = f'{self.names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (self.timelaps[imgIndex][1] - self.timelaps[imgIndex][0])):.1f}ms) Inference, ({(1E3 * (self.timelaps[imgIndex][2] - self.timelaps[imgIndex][1])):.1f}ms) NMS')

                # Stream results
                if self.view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if self.save_img:
                    if self.dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if self.vid_path != save_path:  # new video
                            self.vid_path = save_path
                            if isinstance(self.vid_writer, cv2.VideoWriter):
                                self.vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        self.vid_writer.write(im0)
            detRes.append(detections.reverse())
        #if self.save_txt or self.save_img:
        #s = f"\n{len(list(self.save_dir.glob('labels/*.txt')))} labels saved to {self.save_dir / 'labels'}" if self.save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

        #print(f'Done. ({time.time() - t0:.3f}s)')

        return detRes