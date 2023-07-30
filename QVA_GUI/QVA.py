import os 
import sys
import cv2
import shutil
import json
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QFileDialog, QListWidgetItem, QInputDialog,QSpinBox, QDoubleSpinBox, QComboBox
from PyQt5.QtGui import QPixmap, QImage
import subprocess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout,QLayout
sys.path.append(os.getcwd())
from pylabel.importer import ImportYoloV5

class ProjectWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setGeometry(100, 100, 1600, 800)
        # self.setMinimumSize(1600, 800)
        icon = QIcon('logo.png')
        self.setWindowIcon(icon)
        self.list_projects()

    def init_ui(self):
        self.setWindowTitle('İlk Ekran')
        self.projects_dir = "Projects"
        if not os.path.exists(self.projects_dir):
            os.makedirs(self.projects_dir)
        layout = QVBoxLayout()
        self.label = QLabel('Bu birinci ekran')
        layout.addWidget(self.label)

        self.label_project = QLabel("Project Name: ")
        layout.addWidget(self.label_project)

        self.project_name = QLineEdit(self)
        layout.addWidget(self.project_name)
        
        button1 = QPushButton('Proje Oluştur')
        button1.clicked.connect(self.create_project)
        layout.addWidget(button1)

        self.projects_list = QListWidget(self)
        layout.addWidget(self.projects_list)
        self.projects_list.itemDoubleClicked.connect(self.on_item_double_clicked)
        

        self.setLayout(layout)

    def create_project(self):
        
        if self.projects_dir:
            
            self.project_directory = os.path.join(self.projects_dir, self.project_name.text())
            os.makedirs(self.project_directory)
            verify_dir = self.project_directory+"\\verified"
            images_dir = self.project_directory+"\\images"
            annotations_dir = self.project_directory+"\\annotations"
            exported_dir = self.project_directory+"\\exported"
            os.mkdir(verify_dir)
            os.mkdir(images_dir)
            os.mkdir(annotations_dir)
            os.mkdir(exported_dir)

            os.chdir(self.project_directory)
            self.open_main_window(self.project_directory)
            os.chdir("../../")
        

    def list_projects(self):
        if self.projects_dir:
            self.projects_list.clear()
            for filename in os.listdir(self.projects_dir):
                    item = QListWidgetItem(filename)
                    self.projects_list.addItem(item)

    def on_item_double_clicked(self, item):
        # Çift tıklanan öğenin metnini alıyoruz
        self.project_name = item.text()
        
        # Mesaj kutusuyla çift tıklanan öğenin metnini gösteriyoruz
        # QMessageBox.information(self, "Çift Tıklama", f"Çift tıklanan öğe: {clicked_item_text}")
        self.project_directory = os.path.join(self.projects_dir, self.project_name)
        os.chdir(self.project_directory)
        self.open_main_window(self.project_directory)
        os.chdir("../../")

    def open_main_window(self, project_directory):
        self.main_window = MainWindow(project_directory)
        self.main_window.show()
        self.hide()

class MainWindow(QWidget):
    def __init__(self, project_directory):
        super().__init__()
        
        self.project_directory = project_directory
        self.current_dir = os.getcwd()
        self.project_name = self.project_directory.split("\\")[-1]
        self.path_to_annotations = self.current_dir+"\\annotations"
        #Identify the path to get from the annotations to the images 
        self.path_to_images = self.current_dir + "\\images"
        

        self.yolo_dir = self.current_dir +"\\exported\\labels_yolo"
        self.voc_dir = self.current_dir +"\\exported\\labels_voc"
        self.coco_dir = self.current_dir +"\\exported\\labels_coco"

        self.yoloclasses = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
        
        # Pencere boyutlandırma
        self.setGeometry(100, 100, 1600, 800)
        # self.setMinimumSize(1600, 800)

        # Resim seçimi için QLabel ve QPushButton
        self.image_label = QLabel(self)
        self.detection_result = QLabel(self)
        self.choose_image_button = QPushButton('Choose Image', self)
        # model seçimi için QLabel ve QLineEdit
        # self.model_label = QLabel('Model:', self)
        # self.model_text = QLineEdit(self)
       
        # Parametrelerini seçme ve Algılama 
        self.detect_button = QPushButton('Detection', self)
        self.annotationCheck = False

        self.label_batchsize = QLabel('Batch-Size: ')
        self.spinbox_batchsize = QSpinBox()
        self.spinbox_batchsize.setValue(20)
        self.spinbox_batchsize.setMaximum(5000)

        self.label_thread = QLabel('Thread: ')
        self.spinbox_thread = QSpinBox()
        self.spinbox_thread.setValue(2)

        self.label_threshold = QLabel('Conf-Threshold: ')
        self.threshold_bar = QDoubleSpinBox()
        self.threshold_bar.setMinimum(0)
        self.threshold_bar.setMaximum(1)
        self.threshold_bar.setValue(0.25)
        self.threshold_bar.setSingleStep(0.05)

        self.label_imgsize = QLabel('Image-Size: ')
        self.comboBox_imgsize = QComboBox()
        self.comboBox_imgsize.addItems(["384", "640", "1024"])
        self.comboBox_imgsize.setCurrentIndex(1)
        
        self.label_architecture = QLabel('Architecture: ')
        self.comboBox_architecture =  QComboBox()
        self.comboBox_architecture.addItems(["Yolo", "ResNet", "Centernet"])
        
        self.label_targetClasses = QLabel('Target Class: ')
        self.comboBox_targetClasses =  QComboBox()
        self.comboBox_targetClasses.addItems([""])
        self.comboBox_targetClasses.addItems(self.yoloclasses)
    
        self.label_device = QLabel("Device: ")
        self.comboBox_device = QComboBox()
        self.comboBox_device.addItems(["GPU","CPU"])

        self.label_export = QLabel('Export As: ')
        self.comboBox_export =  QComboBox()
        self.comboBox_export.addItems(["PascalVoc", "Coco", "Yolo"])

        # düzenleme için QPushButton
        self.edit_button = QPushButton('Edit by labelImg', self)
        # Veri ihracı için QPushButton
        self.export_button = QPushButton('Export', self)
        # Doğrulama için QPushButton
        self.verify_button = QPushButton('Verify', self)
        # Model seçimi için QPushButton
        # self.choose_model_button = QPushButton('Choose Model', self)
        # Resim ve etiket listeleri için QListWidget
        self.image_list_widget = QListWidget(self)
        # self.label_list_widget = QListWidget(self)
        self.next_button = QPushButton('Next', self)
        self.next_button.setGeometry(120, 460, 90, 30)
        self.previous_button = QPushButton('Previous', self)
        self.previous_button.setGeometry(10, 460, 90, 30)
        self.close_project_button = QPushButton('Projeyi kapat')

        icon = QIcon('logo.png')
        self.setWindowIcon(icon)
        self.load_exist_images()
        self.load_exist_annotations()
        self.initUI()
        

        
    def initUI(self):
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()

        # Resim seçimi için QLabel ve QPushButton
        hbox1.addWidget(self.image_label)
        hbox1.addWidget(self.detection_result)
        vbox1.addWidget(self.choose_image_button)
        vbox1.addWidget(self.close_project_button)

        hbox2.addWidget(self.previous_button)
        hbox2.addWidget(self.next_button)
        
        # Algılama için QPushButton
        hbox3.addWidget(self.label_architecture)
        hbox3.addWidget(self.comboBox_architecture)
        hbox3.addWidget(self.label_imgsize)
        hbox3.addWidget(self.comboBox_imgsize)
        hbox3.addWidget(self.label_threshold)
        hbox3.addWidget(self.threshold_bar)
        hbox3.addWidget(self.label_thread)
        hbox3.addWidget(self.spinbox_thread)
        hbox3.addWidget(self.label_batchsize)
        hbox3.addWidget(self.spinbox_batchsize)
        hbox3.addWidget(self.label_targetClasses)
        hbox3.addWidget(self.comboBox_targetClasses)
        hbox3.addWidget(self.label_device)
        hbox3.addWidget(self.comboBox_device)
        
        
        hbox3.addWidget(self.detect_button)

        # düzenleme için QPushButton
        hbox4.addWidget(self.edit_button)
        # Veri ihracı için QPushButton
        hbox4.addWidget(self.label_export)
        hbox4.addWidget(self.comboBox_export)
        hbox4.addWidget(self.export_button)

        # Doğrulama için QPushButton
        hbox2.addWidget(self.verify_button)
        
        # Model seçimi için QPushButton
        # vbox2.addWidget(self.choose_model_button)

        # Resim listeleri için QListWidget
        vbox1.addWidget(self.image_list_widget)
    

        # Widget'ların yerleştirilmesi
        vbox2.addLayout(hbox1)
        vbox2.addLayout(vbox1)
        vbox2.addLayout(hbox2)
        vbox2.addLayout(hbox3)
        vbox2.addLayout(hbox4)
        
        # Ana layout oluşturma
        self.setLayout(vbox2)

        # Düğmelere işlevsellik eklemek
        self.choose_image_button.clicked.connect(self.choose_image)
        # self.choose_label_button.clicked.connect(self.choose_label)
        self.detect_button.clicked.connect(self.detect)
        # self.choose_model_button.clicked.connect(self.choose_model)
        self.edit_button.clicked.connect(self.edit)
        self.export_button.clicked.connect(self.export)
        self.next_button.clicked.connect(self.next_image)
        self.previous_button.clicked.connect(self.previous_image)
        self.verify_button.clicked.connect(self.verify)
        
        self.close_project_button.clicked.connect(self.close_project)
        
        # # Pencere ayarları
        self.setWindowTitle('QVA-AutoAnnotator')
        self.show()
        
    def close_project(self):
        self.window = ProjectWindow()
        self.window.show()
        self.hide()
        
    def list_images(self,directory):
        
        if directory:
            self.image_list_widget.clear()
            for filename in os.listdir(directory):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    item = QListWidgetItem(filename)
                    self.image_list_widget.addItem(item)

    def choose_image(self):
        
        directory = QFileDialog.getExistingDirectory(self, "Resim Klasörünü Seç","")
        
        if not os.listdir(self.path_to_images):

            files = os.listdir(directory)
            for file in files :
                shutil.copy2(os.path.join(directory,file), self.path_to_images)

            self.selected_image_directory = self.path_to_images
            self.load_images_from_directory(self.path_to_images)
            self.list_images(self.path_to_images)

    def load_exist_images(self):
        if os.listdir(self.path_to_images):
            self.selected_image_directory = self.path_to_images
            self.load_images_from_directory(self.path_to_images)
            self.list_images(self.path_to_images)   
                
    
    def load_exist_annotations(self):
        if os.listdir(self.current_dir+"\\annotations") and os.path.exists('classes.txt'):
            self.define_annotation_image()
            directory = self.path_to_annotations
            find_last_detections = os.listdir(directory)[-1]
            self.last_detections_folder = directory+'\\'+find_last_detections
            file_names = os.listdir(self.last_detections_folder)
            file_names.remove("classes.txt")
            self.selected_annotation_file = self.last_detections_folder + '\\' +file_names[self.current_image_index]
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)
            self.annotationCheck = True 

    def load_images_from_directory(self, directory):
        file_names = []
        # valid_extensions = ['.jpg', '.jpeg', '.png']
        self.current_image_index = 0
        
        for file_name in os.listdir(directory):
            if os.path.splitext(file_name)[1].lower(): #in valid_extensions:
                file_names.append(os.path.join(directory, file_name))
        
        if file_names:
            self.selected_image_files = file_names
            self.current_image_index = 0
            self.load_image(file_names[self.current_image_index])

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        # height, width, _ = image.shape
        # label_width = 800  # Hedef genişlik
        # label_height = 600  # Hedef yükseklik
        #
        # aspect_ratio = width / height
        # if aspect_ratio > label_width / label_height:
        #     scaled_width = label_width
        #     scaled_height = int(label_width / aspect_ratio)
        # else:
        #     scaled_height = label_height
        #     scaled_width = int(label_height * aspect_ratio)
        # resized_image = cv2.resize(image, (scaled_width, scaled_height))
        #
        # rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        scale_percent = 60  # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    def next_image(self):
        if self.current_image_index + 1 < len(self.selected_image_files):
            self.current_image_index += 1
            self.load_image(self.selected_image_files[self.current_image_index])
        if self.annotationCheck == True:
            self.define_annotation_image()
            self.load_annotation()
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.current_image_index -= 1
            self.load_image(self.selected_image_files[self.current_image_index])
        if self.annotationCheck == True:
            self.define_annotation_image()
            self.load_annotation()
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)
    
    def load_annotation(self):

        directory = self.project_directory+"\\annotations"
        find_last_detections = os.listdir(directory)[-1]
        self.last_detections_folder = directory+'\\'+find_last_detections
        file_names = os.listdir(self.last_detections_folder)
        
        file_names.remove("classes.txt")
        self.selected_annotation_file = self.last_detections_folder + '\\' +file_names[self.current_image_index]

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

        # Resmi uygun boyuta yeniden boyutlandır
        label_width = 800  # Hedef genişlik
        label_height = 600  # Hedef yükseklik
        aspect_ratio = width / height
        if aspect_ratio > label_width / label_height:
            scaled_width = label_width
            scaled_height = int(label_width / aspect_ratio)
        else:
            scaled_height = label_height
            scaled_width = int(label_height * aspect_ratio)
        resized_image = cv2.resize(image, (scaled_width, scaled_height))

        # Resmi PyQt5 uyumlu bir formata dönüştür
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, scaled_width, scaled_height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.detection_result.setPixmap(pixmap)

    def detect(self):
        
        directory = self.selected_image_directory
        current_directory = os.getcwd()
        source = os.path.relpath(directory, current_directory)
        print(source)
        thread_count = str(self.spinbox_thread.value())
        batch_size = str(self.spinbox_batchsize.value())
        conf_threshold = str(self.threshold_bar.value())
        imgsize = str(self.comboBox_imgsize.currentText())
        architecture = str(self.comboBox_architecture.currentText())
        targetClasses = str(self.comboBox_targetClasses.currentText())
        targetClassesText = "--classes \"" + str(self.comboBox_targetClasses.currentText())+ "\"" if targetClasses != "" else ""
        deviceText = "0" if str(self.comboBox_device.currentText()) == "GPU" else "cpu"

        annotations_dir = self.project_directory +"\\annotations"
        print(targetClassesText)
        QMessageBox.information(self, 'Bilgi', 'Detection işlemi yapılıyor. İşlem tamamlandığında sonuçları görebileceksiniz.')
        
        command = 'python ../Auto_Annotator.py --project '+annotations_dir+' --architecture '+architecture+' --thread-count '+thread_count+' --batch-size '+batch_size+' --weights yolov7-e6e.pt --conf-thres '+conf_threshold+' --iou-thres 0.4 --img-size '+imgsize+' --source '+source+' --save-txt '+targetClassesText+' --no-trace --nosave --no-verify --device '+deviceText
        process = subprocess.Popen(command, shell=True)
        process.wait()
        if process.returncode == 0:
            self.annotationCheck = True 

        if self.annotationCheck == True:
            self.define_annotation_image()
            self.load_annotation()
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)
            
            
    
    def edit(self):
        
        directory = self.selected_image_directory
        current_directory = os.getcwd()
        images = os.path.relpath(directory, current_directory)

        directory = self.project_directory+"\\annotations"
        find_last_detections = os.listdir(directory)[-1]
        annotations = directory+'\\'+find_last_detections
        print(images)
        print(annotations)
        
        os.system("python labelImg\labelImg.py "+ images + " " + annotations)
    
    def ClassesTxtFileGenerator(self, exportPath:str):
       
       with open(self.path_to_annotations,"r") as fr:
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
        self.dataset.path_to_annotations = self.coco_dir
        if os.path.isdir(exportPath) == False:
            os.mkdir(exportPath)

        self.dataset.export.ExportToCoco(output_path = None, cat_id_index=0)[0]
        self.dataset.path_to_annotations = self.current_dir+"\\annotations"
    
    def export(self):
        self.last_detections_path = self.path_to_annotations +"\\"+ os.listdir(self.path_to_annotations)[-1]
        self.dataset = ImportYoloV5(path=self.last_detections_path, path_to_images=self.path_to_images, cat_names= self.yoloclasses, img_ext="jpg,jpeg,png,webp")
        exportValue = str(self.comboBox_export.currentText())
        if exportValue == "PascalVoc":
            self.ExportVocLabels(exportPath = self.voc_dir)
        elif exportValue == "Coco":
            self.ExportCocoLabels(exportPath = self.coco_dir)
        elif exportValue == "Yolo":
            self.ExportYoloLabels(exportPath = self.yolo_dir)
        QMessageBox.information(self, 'Bilgi', 'Export işlemi tamamlandı.')
        
    def define_annotation_image(self):
        self.current_file = self.selected_image_files[self.current_image_index]
        self.current_file = self.current_file.replace("/","\\")  

    def verify(self):
        verify_dir = self.project_directory + "\\verified"
        current_dir = os.getcwd()
        dst_dir= os.path.join(current_dir,verify_dir)
        self.define_annotation_image()
        self.load_annotation()
        shutil.copy(self.current_file, dst_dir)
        shutil.copy(self.selected_annotation_file, dst_dir)
        print(self.current_file)
        print(self.selected_annotation_file)
        
        
        
if __name__ == '__main__':
    app = QApplication([])
    window = ProjectWindow()
    window.show()
    app.exec_()
