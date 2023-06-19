import shutil
import json
import os
import sys

sys.path.append(os.getcwd())
from pylabel import importer

#Specify path to the coco.json file
path_to_annotations = 'C:\\Python Projects\\yolo_resource\\coco\\annotations\\instances_val2017.json'
#Specify the path to the images (if they are in a different folder than the annotations)
path_to_images = 'C:\\Python Projects\\yolo_resource\\coco\\images\\val2017'

#Import the dataset into the pylable schema 
dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name="BCCD_coco")
dataset.df.head(5)

def CreateExportedDir():
    if not os.path.isdir("exported"):
        os.mkdir("exported")
    if not os.path.isdir("exported\images"):
        os.mkdir("exported\images")


def SourceImagesToExported(src_dir,dst_dir):

    for file_name in os.listdir(src_dir):
        if os.path.isfile(src_dir+file_name):
            shutil.copy(src_dir+file_name, dst_dir)

# parametre olarak verilecek
def ClassesTxtFileGenerator(exportPath:str):
    
    with open(path_to_annotations,"r") as fr:
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

def ExportYoloLabels(exportPath:str):
    if not os.path.isdir(exportPath):
        os.mkdir(exportPath)
    
    dataset.export.ExportToYoloV5(output_path = exportPath)[0]
    ClassesTxtFileGenerator(exportPath = exportPath)
    

def ExportVocLabels(exportPath:str):
    
    if os.path.isdir(exportPath) == False:
        os.mkdir(exportPath)
    
    dataset.export.ExportToVoc(output_path = exportPath)[0]

# parametre olarak gelecek
src_dir="C:\\Python Projects\\yolo_resource\\coco\\images\\val2017\\"
dst_dir="C:\\Python Projects\\yolo_resource\\exported\\images"
# self vs ile gelecek
voc_dir = "exported\\labels_voc"
yolo_dir = "exported\\labels_yolo"

CreateExportedDir()
SourceImagesToExported(src_dir,dst_dir)

ExportYoloLabels(exportPath = yolo_dir)
# ExportVocLabels(exportPath = voc_dir)