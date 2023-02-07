# yolo_resource

## Ana Yapı:
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
