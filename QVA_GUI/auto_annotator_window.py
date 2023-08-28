import cv2
import shutil
import json



import warnings
import torch

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
from annotation_exporter import AnnotationExporter

valid_extensions = ["jpeg", "jpg", "png", "jpe", "bmp"]
def check_is_image_file(file_name):
    extension = file_name.rsplit(".", 1)[1].lower()
    return extension in valid_extensions

class AutoAnnotatorWindow():
    sel_img_fpath = str
    selected_annotation_fpath = Path
    selected_image_directory = Path
    sel_anno_dir_path = Path
    default_classes_files = list
    target_class = str
    annotation_exporter = AnnotationExporter
    def __init__(self, project_directory, opening_window):

        self.opening_window = opening_window
        self.project_directory = Path(project_directory)
        self.project_name = self.project_directory.name
        self.init_folder_path()
        self.sel_imgs = [] # to hold img files
        self.sel_anns = [] # to hold ann files
        self.selected_annotation_fpath = None
        self.selected_image_directory = None
        self.sel_img_fpath = None
        self.sel_anno_dir_path = None
        self.target_class = None
        self.bImagesUploaded = False
        self.gui_els = QtGuiElements()
        self.annotation_exporter = AnnotationExporter(self.project_directory, self.gui_els)
        self.get_default_class_paths()
        self.connect_gui_elements_to_functions()
        self.load_exist_images()
        if self.sel_imgs:
            self.load_annotations()
    def init_folder_path(self):
        self.anno_dir_path = self.project_directory / "annotations"
        # Identify the path to get from the annotations to the images
        self.img_dir_path = self.project_directory / "images"

        self.create_folder_if_not_exist()
    def create_folder_if_not_exist(self):
        self.anno_dir_path.mkdir(parents=True, exist_ok=True)
        self.img_dir_path.mkdir(parents=True, exist_ok=True)

    def get_default_class_paths(self):
        def_class_dir_path = self.project_directory.parent.parent.parent / "classes_for_allmodels"
        classes_files = os.listdir(def_class_dir_path)
        self.default_classes_files = [def_class_dir_path/fname for fname in classes_files
                                      if fname.rsplit(".",1)[-1] == "txt"]
    def connect_gui_elements_to_functions(self):
        # Düğmelere işlevsellik eklemek
        self.gui_els.choose_image_button.clicked.connect(self.choose_image)
        # self.choose_label_button.clicked.connect(self.choose_label)
        self.gui_els.detect_button.clicked.connect(self.detect)
        # self.choose_model_button.clicked.connect(self.choose_model)
        self.gui_els.edit_button.clicked.connect(self.edit_annotation)
        self.gui_els.export_button.clicked.connect(self.export_annotation_file)
        self.gui_els.next_button.clicked.connect(self.next_image)
        self.gui_els.previous_button.clicked.connect(self.previous_image)
        self.gui_els.verify_button.clicked.connect(self.verify)

        self.gui_els.close_project_button.clicked.connect(self.close_project)
        self.gui_els.comboBox_targetClasses.currentTextChanged.connect(self.change_target_class)
        self.gui_els.comboBox_architecture.currentTextChanged.connect(self.change_architecture)
    def change_target_class(self):
        self.load_annotations()

    def change_architecture(self):
        self.load_annotations()

    def close_project(self):
        self.opening_window.show()
        self.gui_els.close()
    def insert_images_into_gui(self, directory):
        if directory:
            self.gui_els.image_list_widget.clear()
            for filename in os.listdir(directory):
                if check_is_image_file(filename):
                    item = QListWidgetItem(filename)
                    self.gui_els.image_list_widget.addItem(item)

    def choose_image(self):
        directory = QFileDialog.getExistingDirectory(self.gui_els, "Select Image Folder","")
        files = os.listdir(directory)
        if files:
            for file in files :
                shutil.copy2(os.path.join(directory,file), self.img_dir_path)
            self.selected_image_directory = self.img_dir_path
            self.load_images_from_directory(self.img_dir_path)
            if self.sel_imgs:
                self.insert_images_into_gui(self.img_dir_path)
                self.load_annotations()


    def load_exist_images(self):
        if os.listdir(self.img_dir_path):
            self.selected_image_directory = self.img_dir_path
            self.load_images_from_directory(self.img_dir_path)
            self.insert_images_into_gui(self.img_dir_path)
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
            self.bImagesUploaded = True
        else:
            QMessageBox.information(self.gui_els, 'warning',
                                    f'please upload images by clicking select image')
    def find_annot_file(self, img_name):
        img_name = Path(img_name)
        name_to_search = img_name.stem
        for annot in self.sel_anns:
            if name_to_search in annot:
                return annot
        return None
    def load_annotations(self):
        self.find_sel_annot_folder()
        class_txt_path = self.sel_anno_dir_path / 'classes.txt'
        if os.path.exists(class_txt_path):
            self.define_annotation_image()
            file_names = os.listdir(self.sel_anno_dir_path)
            file_names.remove("classes.txt")
            self.sel_anns = file_names
            self.find_annotation_file()
            self.draw_bounding_boxes(self.curr_img_fpath, self.selected_annotation_fpath)

        else:
            arch = self.create_classses_file_selected_annot_fold()
            QMessageBox.information(self.gui_els, 'warning', f'there is no classes.txt file for {arch} to display label')


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
        self.draw_bounding_boxes(self.curr_img_fpath, self.selected_annotation_fpath)
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
    def find_classes_file_name(self):
        architecture = str(self.gui_els.comboBox_architecture.currentText())
        for fclass_path in self.default_classes_files:
            if architecture.lower() in fclass_path.name:
                return fclass_path, architecture
        return None
    def create_classses_file_selected_annot_fold(self):
        fclass_path, architecture = self.find_classes_file_name()
        if fclass_path:
            shutil.copy(fclass_path, self.sel_anno_dir_path / "classes.txt")
            QMessageBox.information(self.gui_els, 'info', f'default classes files for {architecture} is inserted')
        else:
            QMessageBox.information(self.gui_els, 'error', f'please insert classes file for {architecture}')
        return architecture
    def create_annotation_folder(self):

        arch = str(self.gui_els.comboBox_architecture.currentText())
        tar_cls = str(self.gui_els.comboBox_targetClasses.currentText())
        sel_ann_dir_name = f"{arch}_{tar_cls}" if tar_cls != "" else arch
        self.sel_anno_dir_path = self.anno_dir_path / sel_ann_dir_name
        self.sel_anno_dir_path.mkdir(parents=True, exist_ok=True)
        self.create_classses_file_selected_annot_fold()

        return sel_ann_dir_name
    def find_sel_annot_folder(self):
        architecture = str(self.gui_els.comboBox_architecture.currentText())
        targetClasses = str(self.gui_els.comboBox_targetClasses.currentText())
        annot_folder_name = f"{architecture}_{targetClasses}" if targetClasses != "" else architecture
        annot_path_list = os.listdir(self.anno_dir_path)
        if not annot_path_list or annot_folder_name not in annot_path_list:
            QMessageBox.information(self.gui_els, 'warning', 'there is no existing annotation folder')
            sel_ann_dir_name = self.create_annotation_folder()
            QMessageBox.information(self.gui_els, 'info', f'{sel_ann_dir_name} folder is created for annotations')
        else:
            self.sel_anno_dir_path = self.anno_dir_path / annot_folder_name

        


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

        self.load_annotations()
        self.display_annotated_image()



    def edit_annotation(self):
        if self.sel_anno_dir_path:
            image_name = self.sel_imgs[self.current_image_index]
            os.system("python ..\labelImg\labelImg.py "+ image_name + " " + str(self.sel_anno_dir_path))
            self.load_annotations()






    def export_annotation_file(self):
        self.annotation_exporter.create_export_file(self.sel_anno_dir_path, self.selected_image_directory)


    def define_annotation_image(self):
        self.curr_img_fpath = self.sel_imgs[self.current_image_index]
        self.curr_img_fpath = self.curr_img_fpath.replace("/", "\\")

    def verify(self):
        verify_dir = self.project_directory / "verified"
        current_dir = os.getcwd()
        dst_dir= os.path.join(current_dir,verify_dir)
        self.define_annotation_image()
        self.find_annotation_file()
        if self.curr_img_fpath and self.selected_annotation_fpath:
            shutil.copy(self.curr_img_fpath, dst_dir)
            shutil.copy(self.selected_annotation_fpath, dst_dir)
            print(self.curr_img_fpath)
            print(self.selected_annotation_fpath)
        else:
            warnings.warn("there is no annotations generated. you can only verify images with their annotations")

        
        
        

