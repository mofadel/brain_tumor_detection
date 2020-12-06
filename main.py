import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use("dark_background")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import os
import glob
import random
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.models import resnext50_32x4d

#custom libraries
from model import UNet, RESNET_UNET

from dataTransformer import DatasetGenerator, dataAugmented, transforms
from train import plot_model_history, compute_iou, TRAINING_MODEL
from metrics import soft_dice_loss
from utils import plot_samples, plot_statistics, random_test_sample


#Manage the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#************************************Data processing*****************************#
# Path to all data
DATA_PATH = "/home/aissatou/Documents/AI4DEV/AI4DEV/Test/"

BASE_LEN = 89 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_ <-!!!43.tif)
END_IMG_LEN = 4 # len(/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->.tif)
END_MASK_LEN = 9 # (/kaggle/input/lgg-mri-segmentation/kaggle_3m/TCGA_DU_6404_19850629/TCGA_DU_6404_19850629_43 !!!->_mask.tif)

# img size
IMG_SIZE = 512

Data= []
for sub_dir_path in glob.glob(DATA_PATH+"*"):
    if os.path.isdir(sub_dir_path):
        dirname = sub_dir_path.split("/")[-1]
        for filename in os.listdir(sub_dir_path):
            image_path = sub_dir_path + "/" + filename
            Data.extend([dirname, image_path])
    else:
        print("This is not a dir:", sub_dir_path)
        
        
df = pd.DataFrame({"dirname" : Data[::2],
                  "path" : Data[1::2]})

# Masks/Not masks
df_imgs = df[~df['path'].str.contains("mask")]
df_masks = df[df['path'].str.contains("mask")]

# Data sorting
imgs  = sorted(df_imgs["path"].values, key=lambda x : (x[BASE_LEN:-END_IMG_LEN]))
masks = sorted(df_masks["path"].values, key=lambda x : (x[BASE_LEN:-END_MASK_LEN]))

from utils import positiv_negativ_diagnosis
# # General dataframe using masks and images
general_DF = pd.DataFrame({"patient_code": df_imgs.dirname.values,"img_path": imgs,"mask_path": masks})
general_DF["diagnosis_class"] = general_DF["mask_path"].apply(lambda x: positiv_negativ_diagnosis(x))

#*********************************Statistics**************************************#
plot_statistics(general_DF)

# *************************Plotting Data*********************************#

#DATA SIMPLES
Plot_samples_df = general_DF[general_DF["diagnosis_class"] == 1].sample(5).values
plot_samples(Plot_samples_df, IMG_SIZE)

#***********************************Split data on train val test*************************#

# Split df into train_df and val_df
train_df, val_df = train_test_split(general_DF, stratify=general_DF.diagnosis_class, test_size=0.15)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Split train_df into train_df and test_df
train_df, test_df = train_test_split(train_df, stratify=train_df.diagnosis_class, test_size=0.1)
train_df = train_df.reset_index(drop=True)

#train_df = train_df[:1000]
print(f"Train: {train_df.shape} \nVal: {val_df.shape} \nTest: {test_df.shape}")

# train
train_dataset = DatasetGenerator(general_DF=train_df, transforms=transforms)
train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

# val
val_dataset = DatasetGenerator(general_DF=val_df, transforms=transforms)
val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)

#test
test_dataset = DatasetGenerator(general_DF=test_df, transforms=transforms)
test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=True)

# # #******************************Augmentation Visualization*******************#    
images, masks = next(iter(train_dataloader))
print(images.shape, masks.shape)

dataAugmented(images)
dataAugmented(masks, image=False)


#***************************Unet Model***********************************#
unet = UNet(n_classes=1).to(device)
output = unet(torch.randn(1,3,256,256).to(device))
print("",output.shape)

## ***********************Unet_plus_ResNeXt50 Model Combunation********##
RESTNET_UNET_MODEL = RESNET_UNET(n_classes=1).to(device)
output = RESTNET_UNET_MODEL(torch.randn(1,3,256,256).to(device))
print(output.shape)

# ******************************Training******************************#
num_ep = 1                                                                                  # Optimizers
optimizer = torch.optim.Adam(RESTNET_UNET_MODEL.parameters(), lr=5e-4)
             
# Train RESTNET_UNET_MODEL
print('Model Training...')
RESTNET_UNET_MODEL_lh, RESTNET_UNET_MODEL_th, RESTNET_UNET_MODEL_vh = TRAINING_MODEL("RESTNET_UNET_MODEL",RESTNET_UNET_MODEL, train_dataloader, val_dataloader, soft_dice_loss, optimizer, False, num_ep)

print('Done!')
#plot_model_history
#plot_model_history("UNet with ResNeXt50 backbone", RESTNET_UNET_MODEL_th, RESTNET_UNET_MODEL_vh, num_ep)

#********************************Test Prediction**********************#

#Test iou
test_iou = compute_iou(RESTNET_UNET_MODEL, test_dataloader)
print(f"""RESTNET_UNET_MODEL\n Decie Coeff of the test images - {np.around(test_iou, 2)}""")

#random test sample
random_test_sample(test_df, RESTNET_UNET_MODEL)
