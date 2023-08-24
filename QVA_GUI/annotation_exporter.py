import os
import json
# import sys
# cwd = os.getcwd()
# sys.path.append(os.getcwd())
# sys.path.append(os.path.dirname(cwd))
from PyQt5.QtWidgets import QMessageBox
from pylabel.importer import ImportYoloV5
from pylabel.dataset import Dataset
from qt_gui_elements import QtGuiElements
from pathlib import Path
class AnnotationExporter():
    annot_export_func_dict = {}
    annot_export_dir_dict = {}
    dataset = Dataset
    gui_els = QtGuiElements
    project_directory = Path

    def __init__(self, project_directory:Path, gui_els:QtGuiElements):
        self.gui_els = gui_els
        self.project_directory = project_directory
        self.yolo_dir = self.project_directory / "exported" / "labels_yolo"
        self.voc_dir = self.project_directory / "exported" / "labels_voc"
        self.coco_dir = self.project_directory / "exported" / "labels_coco"
        self.annot_export_func_dict = {'PascalVoc': self.ExportVocLabels, 'Coco':self.ExportCocoLabels, 'Yolo':self.ExportYoloLabels}
        self.annot_export_dir_dict = {'PascalVoc': self.voc_dir, 'Coco': self.coco_dir, 'Yolo': self.yolo_dir}

        self.create_folder_if_not_exist()
    def create_folder_if_not_exist(self):
        self.yolo_dir.mkdir(parents=True, exist_ok=True)
        self.voc_dir.mkdir(parents=True, exist_ok=True)
        self.coco_dir.mkdir(parents=True, exist_ok=True)
    def ClassesTxtFileGenerator(self, exportPath:str):
       with open(self.anno_dir_path, "r") as fr:
           data = json.loads(fr.read())
           classes_dict = data['categories']

           with open(exportPath+"\\classes.txt","w") as fw:
               i = 0
               x = 0
               fw.writelines("names:\n")

               while i < classes_dict[-1]['id']:
                   if i +1 == classes_dict[x]['id']:
                       fw.writelines(classes_dict[x]['name']+"\n")
                       x += 1
                       i += 1
                   else:
                       fw.writelines("\n")
                       i += 1
           fw.close()
       fr.close()
    def create_export_file(self, sel_anno_dir_path, img_path):
        if sel_anno_dir_path is not None:
            self.dataset = ImportYoloV5(path=sel_anno_dir_path, path_to_images=img_path, cat_names= self.gui_els.yoloclasses, img_ext="jpg,jpeg,png,webp")
            exportValue = str(self.gui_els.comboBox_export.currentText())
            self.annot_export_func_dict[exportValue](self.annot_export_dir_dict[exportValue])
            QMessageBox.information(self.gui_els, 'Info', 'Export process in completed')
        else:
            sel_img_fpath = sel_anno_dir_path.name
            message = str(f"{str(sel_img_fpath)} file doesn't have annotation file")
            QMessageBox.warning(self.gui_els, "warning", message)

    def ExportYoloLabels(self, exportPath: str):
        if not os.path.isdir(exportPath):
            os.mkdir(exportPath)
        self.dataset.export.ExportToYoloV5(output_path=exportPath)
        # self.ClassesTxtFileGenerator(exportPath = exportPath)

    def ExportVocLabels(self, exportPath: str):
        if os.path.isdir(exportPath) == False:
            os.mkdir(exportPath)
        self.dataset.export.ExportToVoc(output_path=exportPath)
    def ExportCocoLabels(self, export_dir_path: str):
        if os.path.isdir(export_dir_path) == False:
            os.mkdir(export_dir_path)
        self.dataset.export.ExportToCoco(output_dir_path=export_dir_path)
