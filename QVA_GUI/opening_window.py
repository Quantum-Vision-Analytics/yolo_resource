
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QListWidget, QPushButton, QListWidgetItem
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QVBoxLayout

import os
from quantum_main_window import MainWindow
class OpeningWindow(QWidget):
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
            verify_dir = self.project_directory + "\\verified"
            images_dir = self.project_directory + "\\images"
            annotations_dir = self.project_directory + "\\annotations"
            exported_dir = self.project_directory + "\\exported"
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

