# coding=utf-8
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import os


def countDrawCOCO(csvfile,pathIn):
  with open(csvfile) as csvfile:
    csvReader = csv.reader(csvfile, delimiter=';')
    for row in csvReader:
      if row[1]!="None":
        suma = sum(1 for line in open(f"{pathIn}/{row[2]}"))
        print(f"{row[0]} : {suma}")
if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("csvfile",help="Dirección del archivo csv")
  parser.add_argument("pathIn",help="Dirección de los archivos ndjson")
  args,unknown =parser.parse_known_args()
  csvfile=str(args.csvfile)
  pathin=str(args.pathIn)
  countDrawCOCO(csvfile,pathin)

