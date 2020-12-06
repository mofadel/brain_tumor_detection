
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid import ImageGrid

import os
import glob
import pandas as pd
import cv2
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def positiv_negativ_diagnosis(mask_path):
    value = np.max(cv2.imread(mask_path))
    if value > 0 : return 1
    else: return 0


def plot_samples(samples_df, IMG_SIZE):

    sample_img = []
    for i, data in enumerate(samples_df):
        img = cv2.resize(cv2.imread(data[1]), (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(cv2.imread(data[2]), (IMG_SIZE, IMG_SIZE))
        sample_img.extend([img, mask])

    sample_imgs_data = np.hstack(np.array(sample_img[::2]))
    sample_masks_data = np.hstack(np.array(sample_img[1::2]))

    # Plot
    fig = plt.figure(figsize=(25., 25.))
    grid = ImageGrid(fig, 111,nrows_ncols=(2, 1),axes_pad=0.1,)
    grid[0].imshow(sample_imgs_data)
    grid[0].set_title("Imgs", fontsize=15)
    grid[0].axis("off")
    grid[1].imshow(sample_masks_data)
    grid[1].set_title("Masks", fontsize=15, y=0.9)
    grid[1].axis("off")

    return plt.show()

def plot_statistics(df):
    ax = df.diagnosis_class.value_counts().plot(kind='bar',
                                        stacked=True,
                                        figsize=(10, 6),
                                        color=["yellow", "green"])


    ax.set_xticklabels(["Not", "Yes"], rotation=30, fontsize=14);
    ax.set_ylabel('Total Images', fontsize = 14)
    ax.set_title(" Statistic distribution  by diagnosis class",fontsize = 20, y=1.05)

    # Annotate
    for i, rows in enumerate(df.diagnosis_class.value_counts().values):
        ax.annotate(int(rows), xy=(i, rows-12), 
        rotation=0, color="white", 
        ha="center", verticalalignment='bottom',fontsize=15, fontweight="bold")
        
    ax.text(1.2, 2550, f"Total {len(df)} img", size=15,
    color="gray",ha="center", va="center",
    bbox=dict(boxstyle="round",fc=("lightblue"),ec=("gray"),));

# #random test sample
# # image

def random_test_sample(test_df, RESTNET_UNET_MODEL):
    test_sample = test_df[test_df["diagnosis_class"] == 1].sample(1).values[0]
    image = cv2.resize(cv2.imread(test_sample[1]), (128, 128))

    #mask
    mask = cv2.resize(cv2.imread(test_sample[2]), (128, 128))

    # pred
    pred = torch.tensor(image.astype(np.float32) / 255.).unsqueeze(0).permute(0,3,1,2)
    pred = RESTNET_UNET_MODEL(pred.to(device))
    pred = pred.detach().cpu().numpy()[0,0,:,:]

    # pred with tshd
    pred_t = np.copy(pred)
    pred_t[np.nonzero(pred_t < 0.8)] = 0.0
    pred_t[np.nonzero(pred_t >= 0.8)] = 255.#1.0
    pred_t = pred_t.astype("uint8")

    # plot
    fig, ax = plt.subplots(nrows=2,  ncols=2, figsize=(10, 10))

    ax[0, 0].imshow(image)
    ax[0, 0].set_title("image")
    ax[0, 1].imshow(mask)
    ax[0, 1].set_title("mask")
    ax[1, 0].imshow(pred)
    ax[1, 0].set_title("prediction")
    ax[1, 1].imshow(pred_t)
    ax[1, 1].set_title("prediction with threshold")

    return plt.show()