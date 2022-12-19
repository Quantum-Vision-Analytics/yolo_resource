from ModelInferenceHandler import ModelInferenceHandler
from AnnotationVerifier import AnnotationVerifier
import argparse
import torch
from utils.general import strip_optimizer

# Master class to call all sub-classes
class Auto_Annotator:
    def __init__(self,options):
        self.opt = options
        self.modelInfHandler = ModelInferenceHandler(options)
        self.annotVer = AnnotationVerifier()

    # Process, predict and save predictions systematically
    def Process(self):
        self.modelInfHandler.Preprocess()
        self.modelInfHandler.Predict()
        self.detectList = self.modelInfHandler.Postprocess()

        #if(self.modelInfHandler.opt.no_verify == False):
            #self.annotVer.annot_verifier(self.opt.source, str(self.modelInfHandler.save_dir))
            #self.annotVer.annot_verifier("labelImg/imagesTest","labelImg/det")
        

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
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--multi-label', action='store_true', help='label with multiple classes')
    parser.add_argument('--no-verify', action='store_true', help='don`t verify images')
    return parser.parse_args()
    #check_requirements(exclude=('pycocotools', 'thop'))

# Auto launch program
if __name__ == "__main__":
    opt = Parsing()
    aa = Auto_Annotator(opt)
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                aa.Process()
                strip_optimizer(opt.weights)
        else:
            aa.Process()