import json
import mmcv
import xml.etree.ElementTree as ET
import os
import os.path as osp
import cv2
import numpy as np
import argparse

def saveCropImages(Classes,img_name,cv2image,xml_path,newPath):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    number=1
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in Classes:
            continue
        bnd_box = obj.find('bndbox')
        bbox = [
            int(float(bnd_box.find('xmin').text)),
            int(float(bnd_box.find('ymin').text)),
            int(float(bnd_box.find('xmax').text)),
            int(float(bnd_box.find('ymax').text))
        ]
        cropimage = cv2.UMat(cv2image,[bbox[1],bbox[3]],[bbox[0],bbox[2]])
        cv2.imwrite(osp.join(newPath,name,f'{img_name}_{number}.jpg'), cropimage)
        number+=1



def getImageObjects(img_prefix,newPath,pathsDataset):
    Classes=('aeroplane', 'bicycle', 'boat', 'car', 'cat',
             'chair', 'diningtable', 'dog', 'horse',
             'sheep', 'train', 'tvmonitor', 'bird',
             'bus', 'cow', 'motorbike', 'sofa')
    for classe in Classes:
        path = os.path.join(newPath, classe)
        os.mkdir(path)
  
    for vocdata in pathsDataset:
        img_names = mmcv.list_from_file(vocdata)
        for img_name in img_names:
            if 'VOC2007' in vocdata:
                dataset_year = 'VOC2007'
                filename = f'VOC2007/JPEGImages/{img_name}.jpg'
            elif 'VOC2012' in vocdata:
                dataset_year = 'VOC2012'
                filename = f'VOC2012/JPEGImages/{img_name}.jpg'
            jpg_path = osp.join(img_prefix,dataset_year, 'JPEGImages',f'{img_name}.jpg')
            xml_path = osp.join(img_prefix,dataset_year, 'Annotations',f'{img_name}.xml')
            cv2image = cv2.imread(jpg_path)
            saveCropImages(Classes,img_name,cv2image,xml_path,newPath)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("pathsDataset",help="Dirección de los datasets", nargs='+')
    parser.add_argument("--img_prefix",help="Dirección de origen")
    parser.add_argument("--newPath",help="Nueva dirección de guardado")
    args,unknown =parser.parse_known_args()
    img_prefix=str(args.img_prefix)
    newPath=(args.newPath)
    pathsDataset=args.pathsDataset
    getImageObjects(img_prefix,newPath,pathsDataset)
