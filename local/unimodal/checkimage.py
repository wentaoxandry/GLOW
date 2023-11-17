import numpy as np
from emoji import demojize
from nltk.tokenize import TweetTokenizer
import torch
import random
import tqdm
import os, json
import skimage.util
import argparse
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../Dataset/en', type=str, help='Dir saves the datasource information')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasetdir = args.datasdir
    with open(os.path.join(datasetdir, "pan18-author-profiling-training-dataset-2018-02-27.json"),
              encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasetdir, "pan18-author-profiling-test-dataset-2018-03-20.json"),
              encoding="utf8") as json_file:
        valdict = json.load(json_file)

    for datadict in [traindict, valdict]:
        keys = datadict.keys()
        for key in tqdm.tqdm(list(keys)):
            imagelist = datadict[key]['image']
            imgdata = []
            for i in imagelist:
                try:
                    img = Image.open(i.replace('./', './../')).convert('RGB')
                except:
                    with open(os.path.join(datasetdir, 'errorimagelist.txt'), 'a') as f:
                        f.write(i + '\n')



