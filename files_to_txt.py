import argparse
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

def files_to_txt(mypath,seed):
  '''
  Clases_draw=('bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'traffic light', 'fire hydrant',
                 'stop sign', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella',
                 'suitcase','baseball bat', 'skateboard',
                 'tennis racket', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'banana', 'apple',
                 'sandwich', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'toothbrush')
  '''
  Clases_draw=  ('dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'toothbrush')

  for i in Clases_draw:
    final_path=mypath+"/"+i
    fileslist = [f[:-4] for f in listdir(final_path)]
    files_trainval,files_test=train_test_split(fileslist,test_size=0.3, random_state=seed)
    files_trainval.sort()
    files_test.sort()
    txt_trainval_file= open(final_path+"/trainval.txt","w+")
    for files in files_trainval:
      txt_trainval_file.write(files+"\n") 
    txt_trainval_file.close()
    txt_test_file= open(final_path+"/test.txt","w+")
    for files in files_test:
      txt_test_file.write(files+"\n") 
    txt_test_file.close()

if __name__=="__main__":
  parser=argparse.ArgumentParser()
  parser.add_argument("--path",help="Dirección de la carpeta draw")
  parser.add_argument("--seed",help="Semilla de random")
  args,unknown =parser.parse_known_args()
  mypath=str(args.path)
  myseed=int(args.seed)
  files_to_txt(mypath,myseed)

