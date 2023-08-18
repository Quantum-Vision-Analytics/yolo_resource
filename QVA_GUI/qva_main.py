from PyQt5.QtWidgets import QApplication
from opening_window import OpeningWindow
if __name__ == '__main__':
    app = QApplication([])
    window = OpeningWindow()
    window.show()
    app.exec_()