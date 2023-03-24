# coding=utf-8
import cv2
import argparse
def showImages(path,nombre='Imagen'):
  image=cv2.imread(path)
  cv2.imshow(nombre,image)

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("path",help="Dirección de la imagen")
  parser.add_argument("--nombre",help="Nombre de la imagen")
  args,unknown =parser.parse_known_args()
  path=str(args.path)
  nombre=str(args.nombre)
  showImages(path,nombre)

