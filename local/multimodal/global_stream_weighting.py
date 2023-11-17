import os, sys
import json
import numpy as np
from tqdm import tqdm
from model import *
import torch
from utils import *
import argparse
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def training(config, dataset=None, checkpoint_dir=None, data_dir=None):
    traindict = config["traindict"]
    testdict = config["valdict"]

    dataset = Multifeaturedatasetclass(train_file=traindict,
                               test_file=testdict)
    print(len(dataset.train_dataset))
    print(len(dataset.test_dataset))

    resultsdir = config["resultsdir"]

    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=pad_multi_feature)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=pad_multi_feature)

    #weights1 = torch.range(0, 1, 0.5)
    weights1 = torch.range(0, 1, 0.1)
    round1acc = {}
    for weight1 in weights1:
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            text_feat = data[0].to(config["device"])
            text_prob = data[1].to(config["device"])
            image_feat = data[2].to(config["device"])
            image_prob = data[3].to(config["device"])
            label = data[4].to(config["device"])
            label = label.squeeze(-1)
            filename = data[5]
            # zero the parameter gradients
            outputs = weight1 * torch.softmax(text_prob, dim=-1) + (1 - weight1) * torch.softmax(image_prob, dim=-1)

            predicted = torch.argmax(outputs, dim=-1)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
        trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)
        round1acc.update({weight1.tolist(): trainallscore})

    round1acc = {k: v for k, v in sorted(round1acc.items(), reverse=True, key=lambda item: item[1])}
    weights2 = torch.range(list(round1acc.keys())[0]- 0.1, list(round1acc.keys())[0] + 0.1, 0.01)
    round2acc = {}
    for weight2 in weights2:
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            text_feat = data[0].to(config["device"])
            text_prob = data[1].to(config["device"])
            image_feat = data[2].to(config["device"])
            image_prob = data[3].to(config["device"])
            label = data[4].to(config["device"])
            label = label.squeeze(-1)
            filename = data[5]
            # zero the parameter gradients
            outputs = weight2 * torch.softmax(text_prob, dim=-1) + (1 - weight2) * torch.softmax(image_prob, dim=-1)

            predicted = torch.argmax(outputs, dim=-1)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
        trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)
        round2acc.update({weight2.tolist(): trainallscore})
    round2acc = {k: v for k, v in sorted(round2acc.items(), reverse=True, key=lambda item: item[1])}
    weight = list(round2acc.keys())[0]
    outpre = {}
    for i, data in enumerate(tqdm(data_loader_dev), 0):
        with torch.no_grad():
            text_feat = data[0].to(config["device"])
            text_prob = data[1].to(config["device"])
            image_feat = data[2].to(config["device"])
            image_prob = data[3].to(config["device"])
            label = data[4].to(config["device"])
            labels = label.squeeze(-1)
            filename = data[5]
            outputs = weight * torch.softmax(text_prob, dim=-1) + (1 - weight) * torch.softmax(image_prob, dim=-1)

            predicted = torch.argmax(outputs, dim=-1)
            prob = outputs.cpu().detach().tolist()
            for i in range(len(filename)):
                outpre.update({filename[i]: {}})
                outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                outpre[filename[i]].update({'prob': prob[i]})

    out = {}
    for documentid in outpre.keys():
        author = documentid.split('_')[0]
        if out.get(author) is not None:
            pass
        else:
            out.update({author: {}})
        doc_id = documentid.split('_')[1]
        out[author].update({doc_id: {}})
        out[author][doc_id].update({'label': float(outpre[documentid]['label'])})
        out[author][doc_id].update({'predict': int(outpre[documentid]['predict'])})
        out[author][doc_id].update({'prob': outpre[documentid]['prob']})
    authors = out.keys()
    correct = 0
    outfinal = {}
    total = 0
    for author in authors:
        n_doc = len(out[author])
        lebel = int(out[author][list(out[author].keys())[0]]['label'])
        vote = 0
        for j in list(out[author].keys()):
            vote = vote + out[author][j]['predict']
        vote = vote / n_doc
        final_decision = np.round(vote)
        correct += (final_decision == lebel).sum()
        total = total + 1
        outfinal.update({author: {}})
        outfinal[author].update({'predict': int(final_decision)})
        outfinal[author].update({'label': int(lebel)})

    allscore = correct / total
    accs = {}
    accs.update({'round1acc': round1acc})
    accs.update({'round12acc': round2acc})
    with open(os.path.join(resultsdir, str(allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
        json.dump(accs, f, ensure_ascii=False, indent=4)



def find_bestmodel(modeldir):
    filename = os.listdir(modeldir)
    filedict = {}
    for i in filename:
        score = float(i.split('_')[-1].strip('.pkl'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return os.path.join(modeldir, Keymax)

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../Dataset/en', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='multi_global_stream_weight', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./../../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasetdir = args.datasdir
    modal = args.modal
    savedir = args.savedir
    featdir = os.path.join(datasetdir, 'data')

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    modeldir = os.path.join(savedir, modal, 'model')
    resultsdir = os.path.join(savedir, modal, 'results')

    max_len = 500
    overlap = 128
    max_num_epochs = 70

    for makedir in [modeldir, resultsdir]:
        if not os.path.exists(makedir):
            os.makedirs(makedir)


    traindict = {}
    valdict = {}
    for workdict, dset in zip([valdict, traindict], ["test", "train"]):
        textfeatscpdir = os.path.join(featdir, dset + 'textfeats.scp')
        textprobscpdir = os.path.join(featdir, dset + 'textprob.scp')
        imagefeatscpdir = os.path.join(featdir, dset + 'imagefeats.scp')
        imageprobscpdir = os.path.join(featdir, dset + 'imageprob.scp')
        labelscpdir = os.path.join(featdir, dset + 'label.scp')

        for featinfodir, name in zip([textfeatscpdir, textprobscpdir, imagefeatscpdir, imageprobscpdir, labelscpdir],
                                 ['textfeat', 'textprob', 'imagefeat', 'imageprob', 'label']):
            with open(featinfodir) as f:
                srcdata = f.readlines()
            for j in srcdata:
                if j.split(' ')[0] in workdict:
                    pass
                else:
                    workdict.update({j.split(' ')[0]: {}})
                workdict[j.split(' ')[0]].update({name: j.split(' ')[1]})


    config = {
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "featdir": featdir,
            "eps": 1e-8,
            "lr": 2e-3, #2e-3,
            "batch_size": 512, #8,
            "modeldir": modeldir,
            "resultsdir": resultsdir,
            "max_len": max_len,
            "overlap": overlap,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "traindict": traindict,
            "valdict": valdict
        }
    training(config)





