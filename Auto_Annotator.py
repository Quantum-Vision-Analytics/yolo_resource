from AnnotationVerifier import AnnotationVerifier
import argparse
import torch
from utils.general import strip_optimizer

# Master class to call all sub-classes
class Auto_Annotator:
    def __init__(self,options):
        self.opt = options
        arc_opt = options.architecture.strip().lower()

        if arc_opt in ["yolov7","yolo","yolo7"]:
            from YOLOv7 import YOLOv7
            self.mih = YOLOv7(options)

        elif arc_opt in ["fasterrcnn","retinanet","fcos","ssd300"]:
            from Pytorch_Models import Pytorch_Models
            self.mih = Pytorch_Models(options,arc_opt)

        else:
            print(f"No architecture named {options.architecture} found.")
            exit()

        self.annotVer = AnnotationVerifier()
        print("Innit done")

    # Process, predict and save predictions systematically
    def RunModel(self):
        self.mih.StartDetection()

        if(not self.opt.no_verify):
            self.annotVer.annot_verifier(self.opt.source, str(self.mih.save_dir))

        return True


# Get user arguments/inputs
def Parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='imagesTest', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=str, help='filter by class: --class person, or --class person cat car')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='detections', help='save results to project/name')
    parser.add_argument('--name', default='result', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', default=True, help='don`t trace model')
    parser.add_argument('--multi-label', action='store_true', help='label with multiple classes')
    parser.add_argument('--no-verify', action='store_true', help='don`t verify images')
    parser.add_argument('--half-precision', action='store_true', help='use half precision')
    parser.add_argument('--show-details', action='store_true', help='show detection details')
    parser.add_argument('--batch-size', type=int, default=20, help='number of the images to work on per thread')
    # parser.add_argument('--thread-count', type=int, default=2, help='number of the threads to work with')
    parser.add_argument('--architecture', type=str, default='yolov7', help='architecture used')
    return parser#.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

# Auto launch program
if __name__ == "__main__":
    opt = Parsing()
    aa = Auto_Annotator(opt)
    with torch.no_grad():
        # Check if new pre-trained models released
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                aa.RunModel()
                strip_optimizer(opt.weights)
        else:
            aa.RunModel()