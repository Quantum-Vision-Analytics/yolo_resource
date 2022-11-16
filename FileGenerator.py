from pathlib import Path
import torch
from utils.general import xyxy2xywh

class FileGenerator:
    def SetGenerator(self, path, GN, save_cnf):
        self.txt_path = path
        self.gn = GN
        self.save_conf = save_cnf

    def Generate_Annotation(this, xyxy, conf, cls):
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / this.gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if this.save_conf else (cls, *xywh)  # label format
        with open(this.txt_path + '.txt', 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')