import os, sys
import json
import numpy as np
from tqdm import tqdm
import argparse
from model import *
from utils import *
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def find_best(filename):
    filedict = {}
    for i in filename:
        score = float(i.split('_')[-1].strip('.pkl'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return Keymax, filedict[Keymax]

def training(config, dataset=None, checkpoint_dir=None, data_dir=None):
    traindict = config["traindict"]
    testdict = config["valdict"]

    if config["imagetype"] == 'single':
        train_dataset = singleimagedatasetclass(datadict=traindict)
        test_dataset = singleimagedatasetclass(datadict=testdict)
        model = Imagesingle(config["cachedir"])
        padding = pad_image_single
    else:
        train_dataset = multiimagedatasetclass(datadict=traindict)
        test_dataset = multiimagedatasetclass(datadict=testdict)
        model = Imagemulti(config["cachedir"])
        padding = pad_image_multi


    print(len(train_dataset))
    print(len(test_dataset))

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    modellist = os.listdir(modeldir)
    modelname = find_best(modellist)
    model = torch.load(os.path.join(modeldir, modelname[0]))


    model = model.to(config["device"])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    train_examples_len = len(train_dataset)
    data_loader_train = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=padding)

    data_loader_dev = torch.utils.data.DataLoader(test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=padding)

    trainoutpre = {}
    model.eval()
    trainpredict = []
    trainlabel = []
    for i, data in enumerate(tqdm(data_loader_train), 0):
        image = data[0].to(config["device"])
        label = data[1].to(config["device"])
        label = label.squeeze(-1)
        filename = data[2]

        outputs, _ = model(image)
        prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
        predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
        trainpredict.extend(predicted.cpu().detach().tolist())
        trainlabel.extend(label.cpu().data.numpy().tolist())
        for i in range(len(filename)):
            trainoutpre.update({filename[i]: {}})
            trainoutpre[filename[i]].update({'label': int(label[i].cpu().detach())})
            trainoutpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
            trainoutpre[filename[i]].update({'prob': prob[i]})
    trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)

    with open(os.path.join(resultsdir, 'trainsetresults_' + str(trainallscore)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(trainoutpre, f, ensure_ascii=False, indent=4)

    correct = 0
    outpre = {}
    total = 0
    for i, data in enumerate(tqdm(data_loader_dev), 0):
        with torch.no_grad():
            image = data[0].to(config["device"])
            labels = data[1].to(config["device"])
            labels = labels.squeeze(-1)
            filename = data[2]
            outputs, _ = model(image)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filename[i]].update({'prob': prob[i]})

    allscore = correct / total
    with open(os.path.join(resultsdir, 'testsetresults_' + str(allscore)[:6] + ".json"), 'w',
              encoding='utf-8') as f:
        json.dump(outpre, f, ensure_ascii=False, indent=4)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../Dataset/en', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='image', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./../../trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--imagetype', default='single', type=str, help='single or multi images')
    parser.add_argument('--cachedir', default='./../../CACHE', type=str, help='Dir to save downloaded pretrained model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasetdir = args.datasdir
    modal = args.modal
    savedir = args.savedir
    cachedir = args.cachedir
    imagetype = args.imagetype

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    #device = "cpu"
    modal = modal + '_' + imagetype
    modeldir = os.path.join(savedir, modal, 'model')
    resultsdir = os.path.join(savedir, modal, 'results')

    max_len = 500
    overlap = 128
    max_num_epochs = 70

    for makedir in [modeldir, resultsdir, cachedir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    with open(os.path.join(datasetdir, "pan18-author-profiling-training-dataset-2018-02-27.json"), encoding="utf8") as json_file:
        traindict = json.load(json_file)
    with open(os.path.join(datasetdir, "pan18-author-profiling-test-dataset-2018-03-20.json"), encoding="utf8") as json_file:
        valdict = json.load(json_file)


    videofeatscpdir = os.path.join(args.datasdir, 'errorimagelist.txt')
    with open(videofeatscpdir) as f:
        srcdata = f.readlines()
    for j in srcdata:
        author = j.split('/')[-2]
        image = j.split('/')[-1].strip('\n')
        if 'train' in j:
            for i in traindict[author]['image']:
                if image in i:
                    traindict[author]['image'].remove(i)
        elif 'test' in j:
            for i in valdict[author]['image']:
                if image in i:
                    valdict[author]['image'].remove(i)

    del traindict['cbc0e7675ce123b7ca31f127dc7aeff5'] # this author missing 4 images

    #tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", cache_dir=cachedir)
    config = {
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "lr": 2e-5,
            "imagetype": imagetype,
            "batch_size": 2, #4, #8,
            "modeldir": modeldir,
            "resultsdir": resultsdir,
            "max_len": max_len,
            "overlap": overlap,
            "cachedir": cachedir,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "traindict": traindict,
            "valdict": valdict
        }
    training(config)


