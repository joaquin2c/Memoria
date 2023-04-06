# coding=utf-8
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import os


def saveDrawCOCO(csvfile,amount,pathIn,pathOut):
  matplotlib.use('Agg')
  with open(csvfile) as csvfile:
    csvReader = csv.reader(csvfile, delimiter=';')
    for row in csvReader:
      if row[1]!="None":
        os.system(f"mkdir {pathOut}/'{row[0]}'")
        print("Getting data from source")
        df = pd.read_json(f"{pathIn}/{row[2]}",nrows=amount*2,lines=True)
        print("Getting images")
        dfrec=df.loc[df['recognized'] == True]
        number=0
        count=0
        for stroke in dfrec['drawing']:
          if count==amount:
            break
          name=str(number)
          x=[]
          y=[]
          for i in stroke:
            x.append(i[0])
            y.append(i[1])
          fig = plt.figure(frameon=False)
          ax2 = plt.Axes(fig, [0., 0., 1., 1.])
          ax2.set_axis_off()
          fig.add_axes(ax2)
          largo=len(x)
          for i in range(0,largo):
            plt.plot(x[i], y[i],'black',linewidth=3)
          ax = plt.axis()
          plt.axis((ax[0],ax[1],ax[3],ax[2]))
          plt.savefig(f"{pathOut}/{row[0]}/{name.zfill(5)}.jpg")
          plt.clf()
          plt.close(fig)
          number+=1
          count+=1

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("csvfile",help="Dirección del archivo csv")
  parser.add_argument("amount",help="Cantidad de imagenes a guardar")
  parser.add_argument("pathIn",help="Dirección de los archivos ndjson")
  parser.add_argument("pathOut",help="Dirección de la carpeta donde se guarde")
  args,unknown =parser.parse_known_args()
  csvfile=str(args.csvfile)
  amount=int(args.amount)
  pathIn=str(args.pathIn)
  pathOut=str(args.pathOut)
  saveDrawCOCO(csvfile,amount,pathIn,pathOut)

