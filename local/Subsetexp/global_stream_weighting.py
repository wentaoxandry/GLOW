import os, sys
import json
import random
import torch
import numpy as np
from tqdm import tqdm
import argparse
from utils import *
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def training(config):
    testdict = config["valdict"]

    for id in range(10):
        traindatadir = os.path.join(config['cvsubfolder'], str(id) + '.json')
        with open(traindatadir, encoding="utf8") as json_file:
            traindict = json.load(json_file)
        newtraindict = {}
        for author in list(traindict.keys()):
            for sid in list(traindict[author].keys()):
                author_id = author + '_' + sid
                newtraindict.update({author_id: {}})
                newtraindict[author_id].update({'textfeat': traindict[author][sid]['textfeat']})
                newtraindict[author_id].update({'imagefeat': traindict[author][sid]['imagefeat']})
                newtraindict[author_id].update({'textprob': traindict[author][sid]['textprob']})
                newtraindict[author_id].update({'imageprob': traindict[author][sid]['imageprob']})
                newtraindict[author_id].update({'label': traindict[author][sid]['label']})
        traindict = newtraindict
        dataset = Multifeaturedatasetclass(train_file=traindict,
                                            test_file=testdict)
        resultsdir = os.path.join(config["savedir"], config["modal"], str(config["N"]) + '_samples', str(id), 'results')

        for makedir in [resultsdir]:
            if not os.path.exists(makedir):
                os.makedirs(makedir)

        data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=pad_multi_feature)

        data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=1024,
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=pad_multi_feature)

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
                outputs = weight1 * text_prob + (1 - weight1) * image_prob
                #outputs = weight1 * torch.softmax(text_prob, dim=-1) + (1 - weight1) * torch.softmax(image_prob, dim=-1)
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                trainpredict.extend(predicted.cpu().detach().tolist())
                trainlabel.extend(label.cpu().data.numpy().tolist())
            trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)
            round1acc.update({weight1.tolist(): trainallscore})

        round1accmax = max(list(round1acc.values()))
        selectedkeys = [k for k, v in round1acc.items() if v == round1accmax and k != 0.0]
        round1weight = random.sample(selectedkeys, 1)[0]
        weights2 = torch.range(round1weight - 0.1, round1weight + 0.1, 0.01)
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
                outputs = weight2 * text_prob + (1 - weight2) * image_prob
                #outputs = weight2 * torch.softmax(text_prob, dim=-1) + (1 - weight2) * torch.softmax(image_prob, dim=-1)

                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                trainpredict.extend(predicted.cpu().detach().tolist())
                trainlabel.extend(label.cpu().data.numpy().tolist())
            trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)
            round2acc.update({weight2.tolist(): trainallscore})

        round2accmax = max(list(round2acc.values()))
        selectedkeys2 = [k for k, v in round2acc.items() if v == round2accmax and k != 0.0]
        weight = random.sample(selectedkeys2, 1)[0]
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
                outputs = weight * text_prob + (1 - weight) * image_prob
                #outputs = weight * torch.softmax(text_prob, dim=-1) + (1 - weight) * torch.softmax(image_prob, dim=-1)
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
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
        with open(os.path.join(resultsdir, '_'.join(['weight', str(weight), 'score', str(allscore)[:6]]) + ".json"), 'w', encoding='utf-8') as f:
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
    parser.add_argument('--modal', default='multi_global_stream_weight_subset', type=str, help='which data stream')
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

    with open(os.path.join(datasetdir, "pan18-author-profiling-training-dataset-2018-02-27.json"),
              encoding="utf8") as json_file:
        trainolddict = json.load(json_file)

    traindict = {}
    valdict = {}
    for workdict, dset in zip([valdict, traindict], ["test", "train"]):
        textfeatscpdir = os.path.join(featdir, dset + 'textfeats.scp')
        textprobscpdir = os.path.join(featdir, dset + 'textprob.scp')
        imagefeatscpdir = os.path.join(featdir, dset + 'imagefeats.scp')
        imageprobscpdir = os.path.join(featdir, dset + 'imageprob.scp')
        labelscpdir = os.path.join(featdir, dset + 'label.scp')

        for featinfodir, name in zip(
                [textfeatscpdir, textprobscpdir, imagefeatscpdir, imageprobscpdir, labelscpdir],
                ['textfeat', 'textprob', 'imagefeat', 'imageprob', 'label']):
            with open(featinfodir) as f:
                srcdata = f.readlines()
            for j in srcdata:
                if j.split(' ')[0] in workdict:
                    pass
                else:
                    workdict.update({j.split(' ')[0]: {}})
                workdict[j.split(' ')[0]].update({name: j.split(' ')[1]})
        out = {}
        for documentid in workdict.keys():
            author = documentid.split('_')[0]
            if out.get(author) is not None:
                pass
            else:
                out.update({author: {}})
            doc_id = documentid.split('_')[1]
            out[author].update({doc_id: {}})
            out[author][doc_id].update({'textfeat': workdict[documentid]['textfeat']})
            out[author][doc_id].update({'imagefeat': workdict[documentid]['imagefeat']})
            out[author][doc_id].update({'textprob': workdict[documentid]['textprob']})
            out[author][doc_id].update({'imageprob': workdict[documentid]['imageprob']})
            out[author][doc_id].update({'label': workdict[documentid]['label']})


        if dset == 'train':
            traindict = out


    cvfolder = os.path.join(datasetdir, 'cvsplit')
    if not os.path.exists(cvfolder):
        os.makedirs(cvfolder)
    for N in [16, 32, 64, 128, 256]: #[4, 16, 64, 256, 1024, 2048]:
        cvsubfolder = os.path.join(cvfolder, str(N))
        if not os.path.exists(cvsubfolder):
            os.makedirs(cvsubfolder)
            i = 0
            while i < 10:
                trainsubdict = {k: traindict[k] for k in random.sample(list(traindict.keys()), N)}
                label0 = 0
                label1 = 0
                for id in list(trainsubdict.keys()):
                    if trainolddict[id.split('_')[0]]['label'] == 0:
                        label0 = label0 + 1
                    else:
                        label1 = label1 + 1
                if N == 2:
                    if label1 == 1 and label0 == 1:
                        with open(os.path.join(cvsubfolder, str(i) + ".json"), 'w', encoding='utf-8') as f:
                            json.dump(trainsubdict, f, ensure_ascii=False, indent=4)
                        i = i + 1
                    else:
                        pass
                else:
                    if label1 > 1 and label0 > 1:
                        with open(os.path.join(cvsubfolder, str(i) + ".json"), 'w', encoding='utf-8') as f:
                            json.dump(trainsubdict, f, ensure_ascii=False, indent=4)
                        i = i + 1
                    else:
                        pass
        else:
            pass

    #for N in [4, 16]:
    for N in [16, 32, 64, 128, 256]: #[4, 16, 64, 256, 1024, 2048]:
        batchsize = min(N, 512)
        cvsubfolder = os.path.join(cvfolder, str(N))
        max_len = 500
        overlap = 128
        max_num_epochs = 70

        config = {
            "NWORKER": 0,
            "device": device,
            "featdir": featdir,
            "N": N,
            "batch_size": batchsize,  # 8,
            "savedir": savedir,
            "modal": modal,
            "max_len": max_len,
            "overlap": overlap,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "cvsubfolder": cvsubfolder,
            "valdict": valdict
        }
        training(config)


    savemaindir = os.path.join(savedir, modal)
    splitlist = os.listdir(savemaindir)
    accdict = {}
    for split in splitlist:
        nsplit = split.split('_')[0]
        accdict.update({int(nsplit): []})
        cvid = os.listdir(os.path.join(savemaindir, split))
        for cv in cvid:
            # accdict[nsplit].update({cv: []})
            resultscvdir = os.path.join(savemaindir, split, cv, 'results')
            bestmodel = os.listdir(resultscvdir)[0]
            acc = float(bestmodel.split('_')[-1].strip('.json'))
            accdict[int(nsplit)].append(acc)
        a = sum(accdict[int(nsplit)]) / len(accdict[int(nsplit)])
        print(split, "acc is %s " % a)
    #accdict.update({'10': [0.8, 0.8, 0.8, 0.8, 0.8,0.8, 0.8, 0.8, 0.8, 0.8]})
    accdict = dict(sorted(accdict.items()))
    data = []
    labels = []
    for nsamples in list(accdict.keys()):
        data.append(accdict[nsamples])
        labels.append(nsamples)

    x = [i + 1 for i in range(len(labels))]

    plt.figure()
    plt.ylabel('Accuracy')
    plt.xlabel('Trainset size')

    plt.title('Accuracy changed based on trainset size')
    plt.boxplot(data)
    plt.xticks(x, labels)
    # show plot
    # plt.show()
    plt.savefig(os.path.join(savemaindir, 'Accuracy_trainsetsize.png'))







