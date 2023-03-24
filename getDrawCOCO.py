# coding=utf-8
import csv
import os
import argparse

def get_Draw_COCO(csvfile,output):
  with open(csvfile) as csvfile:
    csvreader=csv.reader(csvfile,delimiter=';')
    for row in csvreader:
      if row[1]!="None":
        print(f'downloading "{row[0]}" class')
        os.system(f'wget {row[1]} -P {output}')

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("csvfile",help="Dirección del archivo csv")
  parser.add_argument("--output",help="Dirección de la carpeta donde se guarde")
  args,unknown =parser.parse_known_args()
  csvfile=str(args.csvfile)
  output=str(args.output)
  get_Draw_COCO(csvfile,output)

