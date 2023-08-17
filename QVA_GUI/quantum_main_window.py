import cv2
import shutil
import json


from pylabel.importer import ImportYoloV5
import warnings
import torch
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLayout
from PyQt5.QtGui import QPixmap, QImage

from PyQt5.QtWidgets import QWidget, QLabel, QListWidget, QMessageBox, QPushButton, QFileDialog, QListWidgetItem, QInputDialog,QSpinBox, QDoubleSpinBox, QComboBox


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

class MainWindow():
    def __init__(self, project_directory):


        self.project_directory = Path(project_directory)
        self.current_dir = Path(os.getcwd())
        self.project_name = self.project_directory.name
        self.annot_path = self.current_dir / "annotations"
        #Identify the path to get from the annotations to the images
        self.img_path = self.current_dir / "images"

        self.yolo_dir = self.current_dir / "exported\\labels_yolo"
        self.voc_dir = self.current_dir / "exported\\labels_voc"
        self.coco_dir = self.current_dir / "exported\\labels_coco"
        self.annotationCheck = False
        self.sel_imgs = [] # to hold img files
        self.sel_anns = [] # to hold ann files

        self.gui_els = QtGuiElements()
        self.connect_gui_elements_to_functions()
        self.load_exist_images()
        self.load_exist_annotations()
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
        self.window = ProjectWindow()
        self.window.show()
        self.hide()
    def list_images(self,directory):
        if directory:
            self.gui_els.image_list_widget.clear()
            for filename in os.listdir(directory):
                if check_is_image_file(filename):
                    item = QListWidgetItem(filename)
                    self.gui_els.image_list_widget.addItem(item)

    def choose_image(self):
        directory = QFileDialog.getExistingDirectory(self.gui_els, "Resim Klasörünü Seç","")
        if not os.listdir(self.img_path):
            files = os.listdir(directory)
            for file in files :
                shutil.copy2(os.path.join(directory,file), self.img_path)
            self.selected_image_directory = self.img_path
            self.load_images_from_directory(self.img_path)
            self.list_images(self.img_path)

    def load_exist_images(self):
        if os.listdir(self.img_path):
            self.selected_image_directory = self.img_path
            self.load_images_from_directory(self.img_path)
            self.list_images(self.img_path)

    def find_annot_file(self, img_name):
        img_name = Path(img_name)
        name_to_search = img_name.stem
        for annot in self.sel_anns:
            if name_to_search in annot:
                return annot
        return None
    def load_exist_annotations(self):
        annot_path_list = os.listdir(self.annot_path)
        if annot_path_list:
            recent_proj_path = annot_path_list[-1]
            class_txt_path = self.annot_path / recent_proj_path / 'classes.txt'
            if os.path.exists(class_txt_path):
                self.define_annotation_image()
                directory = self.annot_path
                find_last_detections = os.listdir(directory)[-1]
                self.last_detections_folder = directory / find_last_detections
                file_names = os.listdir(self.last_detections_folder)
                file_names.remove("classes.txt")
                self.sel_anns = file_names
                image_name = self.sel_imgs[self.current_image_index]
                ann_fname = self.find_annot_file(image_name)
                if ann_fname is not None:
                    self.selected_annotation_file = self.last_detections_folder / ann_fname
                    self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)
                self.annotationCheck = True
        else:
            warnings.warn("there is no annotations generated. please run detection first to create annoations")

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
        self.gui_els.image_label.setPixmap(pixmap)

    def next_image(self):
        if self.current_image_index + 1 < len(self.sel_imgs):
            self.current_image_index += 1
            self.load_image(self.sel_imgs[self.current_image_index])

            if self.annotationCheck == True:
                self.define_annotation_image()
                self.load_annotation()
                if self.selected_annotation_file:
                    self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.current_image_index -= 1
            self.load_image(self.sel_imgs[self.current_image_index])
        if self.annotationCheck == True:
            self.define_annotation_image()
            self.load_annotation()
            if self.selected_annotation_file:
                self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)

    def load_annotation(self):

        image_name = self.sel_imgs[self.current_image_index]
        annot_file = self.find_annot_file(image_name)
        self.selected_annotation_file = self.last_detections_folder  / annot_file if annot_file is not None else None

    def draw_bounding_boxes(self, image_path, annotations_path):
        # Resmi yükle
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # Bounding box verilerini oku ve çiz
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
        self.gui_els.detection_result.setPixmap(pixmap)
    def execute_auto_annotator(self, kwargs):
        qaaa = Quantum_AA_Arguments(kwargs)
        opt_cmd = qaaa.generate_arguments()
        aa = Auto_Annotator(opt_cmd)
        with torch.no_grad():
            if opt_cmd.update:  # update all models (to fix SourceChangeWarning)
                for opt_cmd.weights in ['yolov7.pt']:
                    aa.Process()
                    strip_optimizer(opt_cmd.weights)
            else:
                aa.Process()
    def detect(self):

        directory = self.selected_image_directory
        current_directory = os.getcwd()
        source = os.path.relpath(directory, current_directory)
        print(source)
        thread_count = str(self.gui_els.spinbox_thread.value())
        batch_size = str(self.gui_els.spinbox_batchsize.value())
        conf_threshold = str(self.gui_els.threshold_bar.value())
        imgsize = str(self.gui_els.comboBox_imgsize.currentText())
        architecture = str(self.gui_els.comboBox_architecture.currentText())
        targetClasses = str(self.gui_els.comboBox_targetClasses.currentText())
        targetClassesText = '--classes ' + str(self.gui_els.comboBox_targetClasses.currentText()) if targetClasses != "" else ""
        deviceText = "0" if str(self.gui_els.comboBox_device.currentText()) == "GPU" else "cpu"

        annotations_dir = self.annot_path.__str__()
        print(targetClassesText)
        QMessageBox.information(self.gui_els, 'Bilgi', 'Detection işlemi yapılıyor. İşlem tamamlandığında sonuçları görebileceksiniz.')

        kwargs= ('--project ' + annotations_dir + ' --architecture '+architecture+' --thread-count '+thread_count+
                   ' --batch-size '+batch_size+' --weights yolov7-e6e.pt --conf-thres '+conf_threshold+
                   ' --iou-thres 0.4 --img-size '+imgsize+' --source '+source+' --save-txt '+targetClassesText+
                   ' --no-trace --nosave --no-verify --device '+deviceText)
        self.execute_auto_annotator(kwargs)
        # command = 'python ../Auto_Annotator.py --project ' + annotations_dir + ' --architecture ' + architecture + ' --thread-count ' + thread_count + ' --batch-size ' + batch_size + ' --weights yolov7-e6e.pt --conf-thres ' + conf_threshold + ' --iou-thres 0.4 --img-size ' + imgsize + ' --source ' + source + ' --save-txt ' + targetClassesText + ' --no-trace --nosave --no-verify --device ' + deviceText
        # process = subprocess.Popen(command, shell=True)
        # process.wait()
        # if process.returncode == 0:
        self.annotationCheck = True

        if self.annotationCheck == True:
            self.define_annotation_image()
            self.load_annotation()
            if self.selected_annotation_file:
                self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)



    def edit(self):

        directory = self.selected_image_directory
        current_directory = os.getcwd()
        images = os.path.relpath(directory, current_directory)

        directory = self.project_directory / "annotations"
        find_last_detections = os.listdir(directory)[-1]
        annotations = str(directory / find_last_detections)
        print(images)
        print(annotations)

        os.system("python labelImg\labelImg.py "+ images + " " + annotations)

    def ClassesTxtFileGenerator(self, exportPath:str):

       with open(self.annot_path, "r") as fr:
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
        self.dataset.annot_path = self.current_dir + "\\annotations"

    def export(self):
        self.last_detections_path = self.annot_path + "\\" + os.listdir(self.annot_path)[-1]
        self.dataset = ImportYoloV5(path=self.last_detections_path, path_to_images=self.img_path, cat_names= self.yoloclasses, img_ext="jpg,jpeg,png,webp")
        exportValue = str(self.comboBox_export.currentText())
        if exportValue == "PascalVoc":
            self.ExportVocLabels(exportPath = self.voc_dir)
        elif exportValue == "Coco":
            self.ExportCocoLabels(exportPath = self.coco_dir)
        elif exportValue == "Yolo":
            self.ExportYoloLabels(exportPath = self.yolo_dir)
        QMessageBox.information(self, 'Bilgi', 'Export işlemi tamamlandı.')

    def define_annotation_image(self):
        self.current_file = self.sel_imgs[self.current_image_index]
        self.current_file = self.current_file.replace("/","\\")

    def verify(self):
        verify_dir = self.project_directory / "verified"
        current_dir = os.getcwd()
        dst_dir= os.path.join(current_dir,verify_dir)
        self.define_annotation_image()
        self.load_annotation()
        if self.current_file and self.selected_annotation_file:
            shutil.copy(self.current_file, dst_dir)
            shutil.copy(self.selected_annotation_file, dst_dir)
            print(self.current_file)
            print(self.selected_annotation_file)
        else:
            warnings.warn("there is no annotations generated. you can only verify images with their annotations")

        
        
        

