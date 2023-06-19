import os 
import shutil
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QFileDialog, QListWidgetItem, QInputDialog,QSpinBox, QSlider
from PyQt5.QtGui import QPixmap
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
        self.choose_image_button = QPushButton('Choose Image', self)
     
       
        # Parametrelerini seçme ve Algılama 
        self.detect_button = QPushButton('Detection', self)
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
        vbox1.addWidget(self.image_label)
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

        # Resim ve etiket listeleri için QListWidget
        vbox1.addWidget(self.image_list_widget)
        # vbox2.addWidget(self.label_list_widget)

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
        self.setMinimumSize(800, 600)

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
        pixmap = QPixmap(file_path)
        self.image_label.setPixmap(pixmap)

    def next_image(self):
        if self.current_image_index + 1 < len(self.selected_image_files):
            self.current_image_index += 1
            self.load_image(self.selected_image_files[self.current_image_index])

    def previous_image(self):
        if self.current_image_index - 1 >= 0:
            self.current_image_index -= 1
            self.load_image(self.selected_image_files[self.current_image_index])

    # def choose_model(self):
    #     items = ['yolo', 'resnet', 'centernet']  # Örnek etiketler
    #     ok = False
    #     while not ok:
    #         item, ok = QInputDialog.getItem(self, "Model Seç", "Model:", items, 0, False)
    #     self.selected_model = item
    #     self.model_text.setText(item)

    def detect(self):
        
        source_dir = self.selected_image_directory
        substring = "C:/Python Projects/yolo_resource/"
        source = source_dir.replace(substring,"")
        print(source)
        thread_count = str(self.spinbox_batchsize.value())
        batch_size = str(self.spinbox_batchsize.value())
        
        command = 'python Auto_Annotator.py --architecture yolov7 --thread-count '+thread_count+' --batch-size '+batch_size+' --weights yolov7-e6e.pt --conf 0.25 --iou-thres 0.4 --img-size 384 --source '+source+' --save-txt --no-trace --nosave --device 0'
        subprocess.Popen(command, shell=True)

        QMessageBox.information(self, 'Bilgi', 'Detection işlemi yapılıyor. İşlem tamamlandığında sonuçları görebileceksiniz.')
    
    def edit(self):
        
        subprocess.run(['python', "labelImg/labelimg.py"])
        # subprocess.run(['python', "labelImg/labelimg.py " + self.img_path + " " + self.label_path])

    
    def export(self):
        
        subprocess.run(['python', 'QVA_GUI\exportButton.py'])
        QMessageBox.information(self, 'Bilgi', 'Export işlemi tamamlandı. COCOval2017 verisetinin Coco formatlı instance dosyası yolo formatlı olarak exported klasörüne çıktı alınmıştır.')
        
    
    def verify(self):
        if not os.path.isdir("verified"):
            os.mkdir("verified")
        else:
            current_dir = os.getcwd()
            dst_dir= os.path.join(current_dir,"verified")
            current_file = self.selected_image_files[self.current_image_index]
            current_file = current_file.replace("/","\\")  
        
            shutil.copy(current_file, dst_dir)
            print(current_file)
        
        
        
if __name__ == '__main__':
    app = QApplication([])
    window = Window()
    app.exec_()
