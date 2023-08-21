import cv2
import shutil
import json


from pylabel.importer import ImportYoloV5
import warnings
import torch
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLayout
from PyQt5.QtGui import QPixmap, QImage

from PyQt5.QtWidgets import QMessageBox, QFileDialog, QListWidgetItem, QInputDialog,QSpinBox, QDoubleSpinBox, QComboBox


import os
import sys
cwd = os.getcwd()
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(cwd))

from Auto_Annotator import Auto_Annotator
from quantum_auto_annot_arguments import Quantum_AA_Arguments
from utils.general import strip_optimizer
from qt_gui_elements import QtGuiElements
from pathlib import Path

valid_extensions = ["jpeg", "jpg", "png", "jpe", "bmp"]
def check_is_image_file(file_name):
    extension = file_name.rsplit(".", 1)[1].lower()
    return extension in valid_extensions

class AutoAnnotatorWindow():
    sel_img_fpath = str
    selected_annotation_fpath = Path
    selected_image_directory = Path
    sel_anno_dir_path = Path
    def __init__(self, project_directory, opening_window):


        self.project_directory = Path(project_directory)
        self.project_name = self.project_directory.name
        self.anno_dir_path = self.project_directory / "annotations"
        #Identify the path to get from the annotations to the images
        self.img_dir_path = self.project_directory / "images"

        self.yolo_dir = self.project_directory / "exported" / "labels_yolo"
        self.voc_dir = self.project_directory / "exported" / "labels_voc"
        self.coco_dir = self.project_directory / "exported" / "labels_coco"
        self.sel_imgs = [] # to hold img files
        self.sel_anns = [] # to hold ann files
        self.selected_annotation_fpath = None
        self.selected_image_directory = None
        self.sel_img_fpath = None
        self.sel_anno_dir_path = None

        self.gui_els = QtGuiElements()
        self.connect_gui_elements_to_functions()
        self.load_exist_images()
        self.load_exist_annotations()
        self.opening_window = opening_window
    def connect_gui_elements_to_functions(self):
        # Düğmelere işlevsellik eklemek
        self.gui_els.choose_image_button.clicked.connect(self.choose_image)
        # self.choose_label_button.clicked.connect(self.choose_label)
        self.gui_els.detect_button.clicked.connect(self.detect)
        # self.choose_model_button.clicked.connect(self.choose_model)
        self.gui_els.edit_button.clicked.connect(self.edit)
        self.gui_els.export_button.clicked.connect(self.export)
        self.gui_els.next_button.clicked.connect(self.next_image)
        self.gui_els.previous_button.clicked.connect(self.previous_image)
        self.gui_els.verify_button.clicked.connect(self.verify)

        self.gui_els.close_project_button.clicked.connect(self.close_project)

    def close_project(self):
        self.opening_window.show()
        self.gui_els.close()
    def list_images(self,directory):
        if directory:
            self.gui_els.image_list_widget.clear()
            for filename in os.listdir(directory):
                if check_is_image_file(filename):
                    item = QListWidgetItem(filename)
                    self.gui_els.image_list_widget.addItem(item)

    def choose_image(self):
        directory = QFileDialog.getExistingDirectory(self.gui_els, "Resim Klasörünü Seç","")
        if not os.listdir(self.img_dir_path):
            files = os.listdir(directory)
            for file in files :
                shutil.copy2(os.path.join(directory,file), self.img_dir_path)
            self.selected_image_directory = self.img_dir_path
            self.load_images_from_directory(self.img_dir_path)
            self.list_images(self.img_dir_path)

    def load_exist_images(self):
        if os.listdir(self.img_dir_path):
            self.selected_image_directory = self.img_dir_path
            self.load_images_from_directory(self.img_dir_path)
            self.list_images(self.img_dir_path)

    def find_annot_file(self, img_name):
        img_name = Path(img_name)
        name_to_search = img_name.stem
        for annot in self.sel_anns:
            if name_to_search in annot:
                return annot
        return None
    def load_exist_annotations(self):
        self.find_sel_annot_folder()
        class_txt_path = self.sel_anno_dir_path / 'classes.txt'
        if os.path.exists(class_txt_path):
            self.define_annotation_image()
            file_names = os.listdir(self.sel_anno_dir_path)
            file_names.remove("classes.txt")
            self.sel_anns = file_names
            self.find_annotation_file()
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_fpath)

        else:
            QMessageBox.information(self.gui_els, 'warning', 'there is classes.txt file to display label')

    def load_images_from_directory(self, directory):
        file_names = []
        # valid_extensions = ['.jpg', '.jpeg', '.png']
        self.current_image_index = 0

        for file_name in os.listdir(directory):
            if check_is_image_file(file_name):
                file_names.append(os.path.join(directory, file_name))

        if file_names:
            self.sel_imgs = file_names
            self.current_image_index = 0
            self.load_image(file_names[self.current_image_index])
    def create_qimage_from_opencv(self, opencv_img):
        scale_percent = 60  # percent of original size
        nchannel = opencv_img.shape[2]
        width = int(opencv_img.shape[1] * scale_percent / 100)
        height = int(opencv_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        bytes_per_line = width * nchannel
        # resize image
        resized = cv2.resize(opencv_img, dim, interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)
    def load_image(self, file_path):
        image = cv2.imread(file_path)
        pixmap = self.create_qimage_from_opencv(image)
        self.gui_els.org_img_display.setPixmap(pixmap)
    def display_annotated_image(self):
        self.define_annotation_image()
        self.find_annotation_file()
        self.draw_bounding_boxes(self.current_file, self.selected_annotation_fpath)
    def next_image(self):
        if self.current_image_index + 1 < len(self.sel_imgs):
            self.current_image_index += 1
            self.load_image(self.sel_imgs[self.current_image_index])
            self.display_annotated_image()

        else:
            QMessageBox.information(self.gui_els, 'Info',
                                    'you reached into the end of image files please use prev button')


    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.current_image_index -= 1
            self.load_image(self.sel_imgs[self.current_image_index])
            self.display_annotated_image()

        else:
            QMessageBox.information(self.gui_els, 'Info',
                                    'this the beginning of files please use next button to proceed')

    def find_annotation_file(self):
        self.sel_img_fpath = self.sel_imgs[self.current_image_index]
        annot_file = self.find_annot_file(self.sel_img_fpath)
        self.selected_annotation_fpath = self.sel_anno_dir_path / annot_file if annot_file is not None else None

    def draw_bounding_boxes(self, image_path, annotations_path):
        # Resmi yükle
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Bounding box verilerini oku ve çiz
        if annotations_path:
            with open(annotations_path, 'r') as file:
                for line in file:
                    data = line.split()
                    class_id = int(data[0])
                    x = int((float(data[1]) - (float(data[3]) / 2)) * width)
                    y = int((float(data[2]) - (float(data[4]) / 2)) * height)
                    w = int(float(data[3]) * width)
                    h = int(float(data[4]) * height)

                    # Bounding box çiz
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        pixmap = self.create_qimage_from_opencv(image)
        self.gui_els.detection_display.setPixmap(pixmap)
    def execute_auto_annotator(self, kwargs):
        qaaa = Quantum_AA_Arguments(kwargs)
        opt_cmd = qaaa.generate_arguments()
        aa = Auto_Annotator(opt_cmd)
        with torch.no_grad():
            if opt_cmd.update:  # update all models (to fix SourceChangeWarning)
                for opt_cmd.weights in ['yolov7.pt']:
                    result = aa.Process()
                    strip_optimizer(opt_cmd.weights)
            else:
                result = aa.Process()
        return result
    def find_sel_annot_folder(self):
        architecture = str(self.gui_els.comboBox_architecture.currentText())
        targetClasses = str(self.gui_els.comboBox_targetClasses.currentText())
        annot_folder_name = f"{architecture}_{targetClasses}" if targetClasses != "" else architecture
        annot_path_list = os.listdir(self.anno_dir_path)
        if annot_path_list:
            if annot_folder_name in annot_path_list:
                self.sel_anno_dir_path = self.anno_dir_path / annot_folder_name
            else:
                message = (f'there is no such "{annot_folder_name}" annotation folder. Please run detect first'
                           f' to create such annotation folder')
                QMessageBox.information(self.gui_els, 'warning', message)
                self.sel_anno_dir_path = self.anno_dir_path / annot_path_list[-1]
        else:
            self.sel_anno_dir_path = None
            QMessageBox.information(self.gui_els, 'warning', 'there is no annotation folder')

        


    def __create_detect_keywords(self):
        directory = self.selected_image_directory
        current_directory = os.getcwd()
        source = os.path.relpath(directory, current_directory)
        print(source)

        batch_size = str(self.gui_els.spinbox_batchsize.value())
        conf_threshold = str(self.gui_els.threshold_bar.value())
        imgsize = str(self.gui_els.comboBox_imgsize.currentText())
        architecture = str(self.gui_els.comboBox_architecture.currentText())
        targetClasses = str(self.gui_els.comboBox_targetClasses.currentText())
        target_classes_msg = '--classes ' + targetClasses if targetClasses != "" else ""
        deviceText = "0" if str(self.gui_els.comboBox_device.currentText()) == "GPU" else "cpu"

        annotations_dir = self.anno_dir_path.__str__()
        QMessageBox.information(self.gui_els, 'Info',
                                'detection is processing. you will see the results after finishing')

        detection_output_fname = f"{architecture}_{targetClasses}" if targetClasses != "" else architecture
        kwargs = (f'--project {annotations_dir}' + f' --architecture {architecture}' +  f' --device {deviceText}' +
                    f' --batch-size {batch_size}' + f' --conf-thres {conf_threshold}' + f' --name {detection_output_fname}' +
                    f' --img-size {imgsize}' + f' --source {source}' + f' --save-txt {target_classes_msg}')
        kwargs = kwargs + ' --no-trace --nosave --no-verify' + ' --exist-ok' + ' --iou-thres 0.4' + " --weights yolov7-e6e.pt"
        print(kwargs)
        return kwargs

    def detect(self):
        kwargs = self.__create_detect_keywords()
        if self.execute_auto_annotator(kwargs):
            QMessageBox.information(self.gui_els, 'Info', 'detection is finished.')

        self.load_exist_annotations()
        self.display_annotated_image()



    def edit(self):
        if self.sel_anno_dir_path:
            image_name = self.sel_imgs[self.current_image_index]
            self.find_annot_file(image_name)
            anno_file = self.selected_annotation_fpath
            if anno_file is None:
                anno_file = self.sel_anno_dir_path / "classes.txt"
                if os.path.isfile(anno_file):
                    os.system("python ..\labelImg\labelImg.py "+ image_name + " " + str(anno_file))
                    self.load_exist_annotations()
                else:
                    QMessageBox.warning(self.gui_els, "warning", "please insert classes.txt")
        else:
            QMessageBox.warning(self.gui_els, "warning", "please select or insert annotation folder")


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

    def ExportYoloLabels(self, exportPath:str):
        if not os.path.isdir(exportPath):
            os.mkdir(exportPath)

        self.dataset.export.ExportToYoloV5(output_path = exportPath)[0]
        # self.ClassesTxtFileGenerator(exportPath = exportPath)


    def ExportVocLabels(self, exportPath:str):

        if os.path.isdir(exportPath) == False:
            os.mkdir(exportPath)

        self.dataset.export.ExportToVoc(output_path = exportPath)[0]

    def ExportCocoLabels(self, exportPath:str):
        self.dataset.annot_path = self.coco_dir
        if os.path.isdir(exportPath) == False:
            os.mkdir(exportPath)

        self.dataset.export.ExportToCoco(output_path = None, cat_id_index=0)[0]
        self.dataset.annot_path = self.current_dir / "annotations"

    def export(self):
        #self.last_detections_path = self.annot_path / os.listdir(self.annot_path)[-1]
        if self.selected_annotation_fpath is not None:
            self.dataset = ImportYoloV5(path=self.selected_annotation_fpath, path_to_images=self.img_path, cat_names= self.gui_els.yoloclasses, img_ext="jpg,jpeg,png,webp")
            exportValue = str(self.gui_els.comboBox_export.currentText())
            if exportValue == "PascalVoc":
                self.ExportVocLabels(exportPath = self.voc_dir)
            elif exportValue == "Coco":
                self.ExportCocoLabels(exportPath = self.coco_dir)
            elif exportValue == "Yolo":
                self.ExportYoloLabels(exportPath = self.yolo_dir)
            QMessageBox.information(self.gui_els, 'Info', 'Export process in completed')
        else:
            message = str(f"{str(self.sel_img_fpath)} file doenst have annotation file")
            QMessageBox.warning(self.gui_els, "warning", message)

    def define_annotation_image(self):
        self.current_file = self.sel_imgs[self.current_image_index]
        self.current_file = self.current_file.replace("/","\\")

    def verify(self):
        verify_dir = self.project_directory / "verified"
        current_dir = os.getcwd()
        dst_dir= os.path.join(current_dir,verify_dir)
        self.define_annotation_image()
        self.find_annotation_file()
        if self.current_file and self.selected_annotation_fpath:
            shutil.copy(self.current_file, dst_dir)
            shutil.copy(self.selected_annotation_fpath, dst_dir)
            print(self.current_file)
            print(self.selected_annotation_fpath)
        else:
            warnings.warn("there is no annotations generated. you can only verify images with their annotations")

        
        
        

