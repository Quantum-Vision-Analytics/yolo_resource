import os

class AnnotationVerifier():
    def display():
        pass

    #Order: Images directory, labels directory 
    def annot_verifier(self, img_path, label_path):
        """
        if img_path:
            ip = img_path if os.path.exists(os.path.dirname(img_path)) else '.'
        else:
            ip = '.'
        
        if label_path is not None and len(label_path) > 1:
            self.default_save_dir = label_path
        """
        os.system("python labelImg/labelimg.py " + img_path + " " + label_path)
        
    def convert():
        pass

    def object_modifier():
        pass

#av = AnnotationVerifier()
#av.annot_verifier("labelImg/imagesTest","labelImg/det")