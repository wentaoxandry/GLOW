import os, sys
import json
import numpy as np
from tqdm import tqdm
from model import *
from utils import *
import argparse
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
        modeldir = os.path.join(config["savedir"], config["modal"], str(config["N"]) + '_samples', str(id), 'model')
        resultsdir = os.path.join(config["savedir"], config["modal"], str(config["N"]) + '_samples', str(id), 'results')

        for makedir in [modeldir, resultsdir]:
            if not os.path.exists(makedir):
                os.makedirs(makedir)

        evalacc_best = 0
        early_wait = 5
        run_wait = 1
        continuescore = 0
        stop_counter = 0
        criterion = torch.nn.CrossEntropyLoss()
        model = DSW()


        model = model.to(config["device"])
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = torch.nn.DataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config["lr"],
                                    momentum=0.9)
        train_examples_len = len(dataset.train_dataset)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1,
                                                               min_lr=2e-8,
                                                               verbose=True)

        data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=pad_multi_feature)

        data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=pad_multi_feature)


        for epoch in range(config["epochs"]):  # loop over the dataset multiple times
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            model.train()
            train_loss = 0.0
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
                optimizer.zero_grad()
                outputs = model(text_prob, image_prob, text_feat, image_feat)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # print("\r%f" % loss, end='')

                # print statistics
                tr_loss += loss.item()
                nb_tr_steps += 1
                predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                trainpredict.extend(predicted.cpu().detach().tolist())
                trainlabel.extend(label.cpu().data.numpy().tolist())
            trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)
            train_loss = train_loss / (i + 1)


            # Validation loss
            torch.cuda.empty_cache()
            evallossvec = []
            evalacc = 0
            model.eval()
            valid_loss = 0.0
            correct = 0
            outpre = {}
            total = 0
            for i, data in enumerate(tqdm(data_loader_dev), 0):
                with torch.no_grad():
                    text_feat = data[0].to(config["device"])
                    text_prob = data[1].to(config["device"])
                    image_feat = data[2].to(config["device"])
                    image_prob = data[3].to(config["device"])
                    label = data[4].to(config["device"])
                    labels = label.squeeze(-1)
                    filename = data[5]
                    outputs = model(text_prob, image_prob, text_feat, image_feat)
                    dev_loss = criterion(outputs, labels)
                    valid_loss += dev_loss.item()
                    evallossvec.append(dev_loss.cpu().data.numpy())
                    predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
                    prob = torch.softmax(outputs, dim=-1).cpu().detach().tolist()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    for i in range(len(filename)):
                        outpre.update({filename[i]: {}})
                        outpre[filename[i]].update({'label': int(labels[i].cpu().detach())})
                        outpre[filename[i]].update({'predict': int(predicted[i].cpu().detach())})
                        outpre[filename[i]].update({'prob': prob[i]})
            valid_loss = valid_loss / (i + 1)
            scheduler.step(valid_loss)
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
            # evalacc = evalacc / len(evallabel)
            evallossmean = np.mean(np.array(evallossvec))
            for param_group in optimizer.param_groups:
                currentlr = param_group['lr']
            OUTPUT_DIR = os.path.join(modeldir,
                                      str(epoch) + '_' + str(
                                          currentlr) + '_' + str(train_loss)[:6] + '_' + str(trainallscore)[:6] + '_' +
                                      str(valid_loss)[:6] + '_' + str(
                                          allscore)[:6] + '.pkl')
            torch.save(model, OUTPUT_DIR)
            with open(os.path.join(resultsdir, str(epoch) + '_' + str(
                                      currentlr) + '_' + str(train_loss)[:6] + '_' + str(trainallscore)[:6] + '_' +
                                  str(valid_loss)[:6] + '_' + str(
                                      allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
                json.dump(outfinal, f, ensure_ascii=False, indent=4)
            with open(os.path.join(resultsdir, 'analyse_' + str(epoch) + '_' + str(
                                      currentlr) + '_' + str(train_loss)[:6] + '_' + str(trainallscore)[:6] + '_' +
                                  str(valid_loss)[:6] + '_' + str(
                                      allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
                json.dump(out, f, ensure_ascii=False, indent=4)

            torch.cuda.empty_cache()
            if allscore <= evalacc_best:
                stop_counter = stop_counter + 1
                print('no improvement')
                continuescore = 0
            else:
                print('new score')
                evalacc_best = allscore
                continuescore = continuescore + 1

            if continuescore >= run_wait:
                stop_counter = 0
            print(stop_counter)
            print(early_wait)
            if stop_counter < early_wait:
                pass
            else:
                break

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
    parser.add_argument('--modal', default='multi_stream_weight_subset', type=str, help='which data stream')
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
    #modal = modal + '_with_pretrain_frozen_CV_fest'

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

    for N in [16, 32, 64, 128, 256]: #[4, 16, 64, 256, 1024, 2048]:
        batchsize = min(N, 64)
        cvsubfolder = os.path.join(cvfolder, str(N))
        max_len = 500
        overlap = 128
        max_num_epochs = 70

        config = {
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "featdir": featdir,
            "eps": 1e-8,
            "lr": 2e-3,
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
            resultscvdir = os.path.join(savemaindir, split, cv, 'model')
            bestmodel = find_bestmodel(resultscvdir)
            acc = float(bestmodel.split('_')[-1].strip('.pkl'))
            accdict[int(nsplit)].append(acc)
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







