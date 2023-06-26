import os 
import sys
import cv2
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QFileDialog, QListWidgetItem, QInputDialog,QSpinBox, QSlider
from PyQt5.QtGui import QPixmap, QImage
import subprocess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout,QLayout
class Window(QWidget):
    def __init__(self):
        super().__init__()
        # model seçimi için QLabel ve QLineEdit
        # self.model_label = QLabel('Model:', self)
        # self.model_text = QLineEdit(self)

        # Resim seçimi için QLabel ve QPushButton
        self.image_label = QLabel(self)
        self.detection_result = QLabel(self)
        self.choose_image_button = QPushButton('Choose Image', self)
     
       
        # Parametrelerini seçme ve Algılama 
        self.detect_button = QPushButton('Detection', self)
        self.detected = False
        self.label_batchsize = QLabel('Batch-Size: ')
        self.spinbox_batchsize = QSpinBox()
        self.spinbox_batchsize.setValue(20)
        self.label_thread = QLabel('Thread: ')
        self.spinbox_thread = QSpinBox()
        self.spinbox_thread.setValue(2)
        
        self.threshold_bar = QSlider()
        self.threshold_bar.setMinimum(0)
        self.threshold_bar.setMaximum(1)
        # self.setCentralWidget(self.threshold_bar)
        self.threshold_bar.setValue(int(0.5 * 1))
        # düzenleme için QPushButton
        self.edit_button = QPushButton('Edit by labelImg', self)

        # Veri ihracı için QPushButton
        self.export_button = QPushButton('Import from COCO', self)

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
        

        self.initUI()
        icon = QIcon('QVA_GUI\logo.png')
        self.setWindowIcon(icon)
        

        
    def initUI(self):
        vbox1 = QVBoxLayout()
        vbox2 = QVBoxLayout()
        hbox1 = QHBoxLayout()
        hbox2 = QHBoxLayout()
        hbox3 = QHBoxLayout()
        hbox4 = QHBoxLayout()
        
       
        # model seçimi için QLabel ve QLineEdit
        # hbox1.addWidget(self.model_label)
        # hbox1.addWidget(self.model_text)

        # Resim seçimi için QLabel ve QPushButton
        hbox1.addWidget(self.image_label)
        hbox1.addWidget(self.detection_result)
        vbox1.addWidget(self.choose_image_button)

        hbox2.addWidget(self.previous_button)
        hbox2.addWidget(self.next_button)
        
        # Algılama için QPushButton
        hbox3.addWidget(self.detect_button)
        hbox3.addWidget(self.label_batchsize)
        hbox3.addWidget(self.spinbox_batchsize)
        hbox3.addWidget(self.label_thread)
        hbox3.addWidget(self.spinbox_thread)
        # düzenleme için QPushButton
        hbox4.addWidget(self.edit_button)
        # Veri ihracı için QPushButton
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

        # Pencere boyutlandırma
        self.setGeometry(100, 100, 800, 600)
        self.setMinimumSize(1600, 800)

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
        # # Pencere ayarları
        # self.setGeometry(100, 100, 220, 500)
        self.setWindowTitle('QVA-AutoAnnotator')
        self.show()
        

    def list_images(self,directory):
        
        if directory:
            self.image_list_widget.clear()
            for filename in os.listdir(directory):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    item = QListWidgetItem(filename)
                    self.image_list_widget.addItem(item)

    def choose_image(self):
        
        self.folder_selected = False
        directory = QFileDialog.getExistingDirectory(self, "Resim Klasörünü Seç","")
        
        if directory:
            self.selected_image_directory = directory
            self.load_images_from_directory(directory)
            self.list_images(directory)
            self.folder_selected = True
            # if self.detected == True:    
            #     self.detection_result.clear()
            # self.detected = False
            # self.current_image_index = 0
            
            

    def load_images_from_directory(self, directory):
        file_names = []
        # valid_extensions = ['.jpg', '.jpeg', '.png']
        self.current_image_index = 0
        
        for file_name in os.listdir(directory):
            if os.path.splitext(file_name)[1].lower(): #in valid_extensions:
                file_names.append(os.path.join(directory, file_name))
        
        self.load_image(file_names[self.current_image_index])
        
        if file_names:
            self.selected_image_files = file_names
            self.current_image_index = 0
            self.load_image(file_names[self.current_image_index])

    def load_image(self, file_path):
        image = cv2.imread(file_path)
        height, width, _ = image.shape
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

        
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb_image.data, scaled_width, scaled_height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap)

    def next_image(self):
        if self.current_image_index + 1 < len(self.selected_image_files):
            self.current_image_index += 1
            self.load_image(self.selected_image_files[self.current_image_index])
        if self.detected == True:
            self.define_annotation_image()
            self.load_annotation()
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.current_image_index -= 1
            self.load_image(self.selected_image_files[self.current_image_index])
        if self.detected == True:
            self.define_annotation_image()
            self.load_annotation()
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)
    
    def load_annotation(self):

        directory = "detections"
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


    # def choose_model(self):
    #     items = ['yolo', 'resnet', 'centernet']  # Örnek etiketler
    #     ok = False
    #     while not ok:
    #         item, ok = QInputDialog.getItem(self, "Model Seç", "Model:", items, 0, False)
    #     self.selected_model = item
    #     self.model_text.setText(item)

    def detect(self):
        
        directory = self.selected_image_directory
        current_directory = os.getcwd()
        source = os.path.relpath(directory, current_directory)
        print(source)
        thread_count = str(self.spinbox_batchsize.value())
        batch_size = str(self.spinbox_batchsize.value())

        QMessageBox.information(self, 'Bilgi', 'Detection işlemi yapılıyor. İşlem tamamlandığında sonuçları görebileceksiniz.')
        
        command = 'python Auto_Annotator.py --architecture yolov7 --thread-count '+thread_count+' --batch-size '+batch_size+' --weights yolov7-e6e.pt --conf 0.25 --iou-thres 0.4 --img-size 384 --source '+source+' --save-txt --no-trace --nosave --device 0'
        process = subprocess.Popen(command, shell=True)
        process.wait()
        
        if process.returncode == 0:
            self.detected = True
        
        
        if self.detected == True:
            self.define_annotation_image()
            self.load_annotation()
            self.draw_bounding_boxes(self.current_file, self.selected_annotation_file)
    
    def edit(self):
        
        directory = self.selected_image_directory
        current_directory = os.getcwd()
        images = os.path.relpath(directory, current_directory)

        directory = "detections"
        find_last_detections = os.listdir(directory)[-1]
        annotations = directory+'\\'+find_last_detections
        print(images)
        print(annotations)
        
        os.system("python labelImg\labelImg.py "+ images + " " + annotations)
    
    def export(self):
        
        subprocess.run(['python', 'QVA_GUI\exportButton.py'])
        QMessageBox.information(self, 'Bilgi', 'Export işlemi tamamlandı. COCOval2017 verisetinin Coco formatlı instance dosyası yolo formatlı olarak exported klasörüne çıktı alınmıştır.')
        
    def define_annotation_image(self):
        self.current_file = self.selected_image_files[self.current_image_index]
        self.current_file = self.current_file.replace("/","\\")  

    def verify(self):
        if not os.path.isdir("verified"):
            os.mkdir("verified")
        else:
            current_dir = os.getcwd()
            dst_dir= os.path.join(current_dir,"verified")
            self.define_annotation_image()
            self.load_annotation()
            shutil.copy(self.current_file, dst_dir)
            shutil.copy(self.selected_annotation_file, dst_dir)
            print(self.current_file)
            print(self.selected_annotation_file)
        
        
        
if __name__ == '__main__':
    app = QApplication([])
    window = Window()
    app.exec_()
