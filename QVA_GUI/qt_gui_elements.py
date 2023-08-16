from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QLayout
from PyQt5.QtGui import QPixmap, QImage

from PyQt5.QtWidgets import QWidget, QLabel, QListWidget, QMessageBox, QPushButton, QFileDialog, QListWidgetItem, QInputDialog,QSpinBox, QDoubleSpinBox, QComboBox

class QtGuiElements(QWidget):
    def __init__(self):
        super().__init__()
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
        self.create_and_initiate_gui_elements()
        # Parametrelerini seçme ve Algılama
        self.detect_button = QPushButton('Detection', self)
        self.initUI()

    def create_and_initiate_gui_elements(self):
        self.label_batchsize = QLabel('Batch-Size: ')
        self.spinbox_batchsize = QSpinBox()
        self.spinbox_batchsize.setValue(500)
        self.spinbox_batchsize.setMaximum(5000)

        self.label_thread = QLabel('Thread: ')
        self.spinbox_thread = QSpinBox()
        self.spinbox_thread.setValue(1)

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
        self.comboBox_architecture = QComboBox()
        self.comboBox_architecture.addItems(["Yolo", "ResNet", "Centernet"])

        self.label_targetClasses = QLabel('Target Class: ')
        self.comboBox_targetClasses = QComboBox()
        self.comboBox_targetClasses.addItems([""])
        self.comboBox_targetClasses.addItems(self.yoloclasses)

        self.label_device = QLabel("Device: ")
        self.comboBox_device = QComboBox()
        self.comboBox_device.addItems(["GPU", "CPU"])

        self.label_export = QLabel('Export As: ')
        self.comboBox_export = QComboBox()
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
