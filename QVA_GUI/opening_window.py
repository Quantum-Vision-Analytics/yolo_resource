
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QListWidget, QPushButton, QListWidgetItem
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QVBoxLayout
from pathlib import Path
import os
from quantum_main_window import MainWindow
from PyQt5.QtWidgets import QMessageBox
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

        self.projects_dir = Path(os.getcwd())/ "Projects"
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
    def create_folders(self):
        main_dir = self.sel_proj_dir
        main_dir.mkdir(parents=True, exist_ok=True)
        folders = ["verified","images","annotations","exported"]
        [(main_dir/x).mkdir(parents=True, exist_ok=True) for x in folders]
    def create_project(self):
        if self.projects_dir:
            project_name = self.project_name.text()
            if project_name:
                self.sel_proj_dir = self.projects_dir / project_name
            else:
                QMessageBox.warning(self, "warning","please enter project name")
                return False
            self.create_folders()
            self.open_main_window(self.sel_proj_dir)

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
        self.sel_proj_dir = os.path.join(self.projects_dir, self.project_name)
        self.open_main_window(self.sel_proj_dir)

    def open_main_window(self, project_directory):
        self.main_window = MainWindow(project_directory)
        self.main_window.gui_els.show()
        self.hide()

