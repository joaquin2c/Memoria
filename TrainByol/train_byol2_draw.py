from models.BYOL2_model import BYOL2
from data.custom_transforms import BatchTransform, ListToTensor, PadToSquare, SelectFromTuple
from data.pairs_dataset import PairsDatasetDraw, pair_collate_fn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sys
import torch
from torch.utils.data import Subset
import torchvision.models as models
import torchvision.transforms as T
import warnings
import argparse

def trainByol(weightsPath,pathImages,pathSketch,epochs):
    train_dataset = PairsDatasetDraw(
        pathImages,
        pathSketch
    )

    transforms_1 = T.Compose([
        BatchTransform(SelectFromTuple(0)),
        BatchTransform(PadToSquare(255)),
        BatchTransform(T.Resize((224,224))),
        ListToTensor('cuda', torch.float),
    ])
    transforms_2 = T.Compose([
        BatchTransform(SelectFromTuple(1)),
        BatchTransform(PadToSquare(255)),
        BatchTransform(T.Resize((224,224))),
        ListToTensor('cuda', torch.float),
    ])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        collate_fn=pair_collate_fn,
        num_workers=4
    )
    encoder = models.resnet50(pretrained=False)
    empty_transform = T.Compose([])
    #epochs = 5
    epoch_size = len(train_loader)
    learner = BYOL2(
        encoder,
        image_size=224,
        hidden_layer='avgpool',
        augment_fn=empty_transform,
        cosine_ema_steps=epochs*epoch_size
    )
    learner.augment1 = transforms_1
    learner.augment2 = transforms_2
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    learner.load_state_dict(torch.load(weightsPath))
    learner = learner.to('cuda')
    learner.train()
    filehandler = open('../training_draw_byol.txt', 'w')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        running_loss = np.array([], dtype=np.float32)
        print('Beginning trainning')
        for epoch in range(epochs):
            i = 0
            for images in train_loader:
                loss = learner(images) #.to('cuda', dtype=torch.float))
                opt.zero_grad()
                loss.backward()
                opt.step()
                learner.update_moving_average()
                running_loss = np.append(running_loss, [loss.item()])
                sys.stdout.write('\rEpoch {}, batch {} - loss {:.4f}'.format(epoch+1, i+1, np.mean(running_loss)))
                filehandler.write('Epoch {}, batch {} - loss {:.4f}\n'.format(epoch+1, i+1, np.mean(running_loss)))
                filehandler.flush()
                i += 1
                if i%(epoch_size/2)==0:
                    torch.save(learner.state_dict(), 'path/self_bimodal_byol_Quick_Draw{}epochs.pt'.format(epochs))
            running_loss = np.array([], dtype=np.float32)
            sys.stdout.write('\n')
    filehandler.close()

    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--checkpoint",help="Dirección de los pesos del modelo")
    parser.add_argument("--pathImages",help="Dirección de las imagenes")
    parser.add_argument("--pathSketch",help="Dirección de los sketchs")
    parser.add_argument("--epochs",help="epocas")
    args,unknown =parser.parse_known_args()
    checkpoint=str(args.checkpoint)
    pathImages=str(args.pathImages)
    pathSketch=str(args.pathSketch)
    epochs=int(args.epochs)
    trainByol(checkpoint,pathImages,pathSketch,epochs)
