import pyarrow.parquet as pq
import requests
import os
import webdataset as wds
from PIL import Image
import io
import matplotlib.pyplot as plt
import json


from warnings import filterwarnings


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json

import clip


from PIL import Image, ImageFile


# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)



model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   
max_time=5
#parquet name is data1

parquet_file=pq.ParquetFile('data1.parquet')

table = parquet_file.read()
#add prompt here
km = ["木纹理","金属","毛玻璃","3D","大理石","瓷砖","地面","材质","厚涂","扁平风","描线风","2.5D","剪纸风","弥散光","国潮","肌理","涂鸦","照片写真","CG人物","扁平人物","数字人","metal","wooden","texture"]
count=0
now=0
for i in range(parquet_file.num_row_groups):
    row_group=parquet_file.read_row_group(i)
    row_group=row_group.to_pandas()
    for idx,row in row_group.iterrows():
        if (row["HEIGHT"])<600.00:
            continue
        if (row["WIDTH"])<600.00:
            continue
        text=row["TEXT"]
        for j in km:
            if text.find(j) != -1:
                now+=1
                url=row["URL"]
                # print(url)
                if url.find("jpg")!=-1:
                    suf=".jpg"
                elif url.find("png")!=-1:
                    suf=".png"
                elif url.find("jpeg")!=-1:
                    suf=".jpeg"
                elif url.find("webp")!=-1:
                    suf=".webp"
                else:
                    break
                # 写入图片
                image_name=str(now)+suf
                time = 3
                while(time<max_time):
                    try:
                        r = requests.get(url,timeout=7)
                        with open(image_name, "wb") as f:
                            f.write(r.content)                
                        img_path = image_name
                        time2=4
                        while(time2<max_time):
                            try:
                                pil_image = Image.open(img_path)
                                image = preprocess(pil_image).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    image_features = model2.encode_image(image)
                                im_emb_arr = normalized(image_features.cpu().detach().numpy() )
                                prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
                                # print( "Aesthetic score predicted by the model:")
                                # print(prediction)
                                score=prediction.item()
                                score=round(score,8)
                                count+=1
                                new_name=str(score)+"+"+str(image_name)
                                os.rename(image_name,new_name)
                                break
                            except:
                                # new_name="0.000000000+"+str(image_name)
                                os.remove(image_name)
                                time2+=1
                        break
                    except:
                        time+=1
                #####  This script will predict the aesthetic score for this image file:

                break
        if now==50:
            break
print("Hi")



