# Quantum Vision & Analytics Otomatik Görüntü Etiketleyici(Auto Annotator) yolo_resource

## 1 Giriş

Otomatik Görüntü Etiketleme tekniklerinin ana fikri, çok sayıda görüntü örneğinden anlamsal kavram modellerini otomatik
olarak öğrenmek ve kavram modellerini yeni görüntüleri etiketlemek için kullanmaktır. Resimlere anlamsal etiketler
eklendikten sonra, metin belgesi geri çağrımına benzer şekilde, resimler anahtar kelimelerle alınabilir. Bu sayede
duruma özgü veriler elde edilmiş olur. Otomatik Görüntü Etiketleyici bunu sağlamaktadır.

### 1.1	Bu Dökümanın Amacı

Otomatik Görüntü Etiketleyicinin çalışması için gerekli olan kütüphanelerin, sistemlerin kurulumunu ve olası
problemlerin çözümlerini bir rehber olarak sunmaktır.

## 2	Kurulum Öncesi Gereksinimler

### 2.1 Donanım Gereksinimleri

#### 2.1.1 Minimum Donanım Gereksinimleri (Kendi Pc min Özelliklerini yazdım)

İşlemci hızı:  2,30 GHz
Rastgele erişim belleği (RAM): 12 GB

#### 2.1.2 Önerilen Donanım Gereksinimleri (Buraya İstenilen bi otralama değer yazabiliriz.)

İşlemci hızı:
Rastgele erişim belleği (RAM):

### 2.2 Yazılım Gereksinimleri

#### 2.2.1 Gerekli IDE’ler

- Pycharm Comminity Edition veya VS Code
- Pylabel Image Editor

##### 2.2.2 Kuruluması Gereken Kütüphaneler

- pip install pyqt5-tools
- pip install lxlml
- Daha sonra terminelden labelımg’in dosya yoluna gelip  
  `pyrcc5 -o libs/resources.py resources.qrc` komutu execute edilir.
  Bu işlem yapılmadığı takdirde File `"C:\Users..\QVA_GUI\labelImg\labelImg.py", from libs.resources import *` \
ModuleNotFoundError: No module named 'libs.resources' hatası alınır.

## 3 Mimari Model Ağırlıkları
- YOLOv7 [yolov7-e6e.pt]
- Pytorch Modelleri [FasterRCNN, RetinaNet, Fcos, SSD300]

## 4 Class yapilari:

``` 
'''  
kapsami:  
Annotation file generation  
annotation file conversion  
new object adaptation  
'''  
class AutoAnnotator():  
    ModelInferenceHandler()  
    FileGenerator()
    AnnotationVerifier()
  
class ModelInferenceHandler:  
    traininig(img_data, annoation_file) -> newobject_detector '''tennis raketli oyuncular bulmak'''
    loadresources(args) -> Model and image loader
    preprocess(img_data) -> processed_img_data  
    predict(processed_img_data) -> detections  
    postproces(detections: list)
    detect(args) -> annotations
  
class AnnotationVerifier:
    def display(self, img, label, score, bbox):  
    def annot_verifier(self, annot_file_path):  
        detections, img_path = read_annot_file(annot_file_path)  
        img = read_img(self, img_path)  
        for detect in detections:  
            label, score, bbox = detect  
            display(img, label, score, bbox)  
    def coonvert(self, pascal2voc):  
        pascal -> voc  
    def object_modifier(self):  
  
class FileGenerator:  
    annot_type: str  
    json_file_handler:  
    def generate_annot(self, detections: list):  
        detect_dict = {object}  
        for detect in detections:  
            label, score, bbox = detect  
            json_file_handler.append( label, score, bbox)  
```
