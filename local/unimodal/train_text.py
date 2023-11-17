import os, sys
import json
import numpy as np
from tqdm import tqdm
import argparse
import torch
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

def training(config, dataset=None, checkpoint_dir=None, data_dir=None):
    traindict = config["traindict"]
    testdict = config["valdict"]

    traindict = combine_text(traindict, shuffle=True)
    testdict = combine_text(testdict, shuffle=True)
    traindict = get_split(traindict, config["max_len"], config["overlap"])
    testdict = get_split(testdict, config["max_len"], config["overlap"])

    dataset = Textdatasetclass(train_file=traindict,
                               test_file=testdict,
                               tokenizer=config["tokenizer"],
                               device=config["device"],
                               max_len=config["max_len"])
    print(len(dataset.train_dataset))
    print(len(dataset.test_dataset))

    resultsdir = config["resultsdir"]
    modeldir = config["modeldir"]

    evalacc_best = 0
    early_wait = 4
    run_wait = 1
    continuescore = 0
    stop_counter = 0
    criterion = torch.nn.CrossEntropyLoss()
    model = Textmodel(config["cachedir"])

    model = model.to(config["device"])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["lr"],
                                  eps=config["eps"], weight_decay=config["weight_decay"]
                                  )
    train_examples_len = len(dataset.train_dataset)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(
                                                    train_examples_len / config["batch_size"]) * 5,
                                                num_training_steps=int(
                                                    train_examples_len / config["batch_size"]) * config["epochs"])

    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=pad_text)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=pad_text)

    for epoch in range(config["epochs"]):  # loop over the dataset multiple times
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        trainpredict = []
        trainlabel = []
        for i, data in enumerate(tqdm(data_loader_train), 0):
            node_sets = data[0]
            mask = data[1]
            label = data[2].to(config["device"])
            label = label.squeeze(-1)
            filename = data[3]
            optimizer.zero_grad()
            outputs, _ = model(node_sets, mask)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            #print("\r%f" % loss, end='')

            tr_loss += loss.item()
            nb_tr_steps += 1
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            trainpredict.extend(predicted.cpu().detach().tolist())
            trainlabel.extend(label.cpu().data.numpy().tolist())
        trainallscore = np.sum((np.array(trainpredict) == np.array(trainlabel)), axis=-1) / len(trainlabel)

        # Validation loss
        torch.cuda.empty_cache()
        evallossvec = []
        evalacc = 0
        model.eval()
        correct = 0
        outpre = {}
        total = 0
        for i, data in enumerate(tqdm(data_loader_dev), 0):
            with torch.no_grad():
                node_sets = data[0]
                mask = data[1]
                labels = data[2].to(config["device"])
                labels = labels.squeeze(-1)
                filename = data[3]
                outputs, _ = model(node_sets, mask)
                dev_loss = criterion(outputs, labels)
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
        evallossmean = np.mean(np.array(evallossvec))
        for param_group in optimizer.param_groups:
            currentlr = param_group['lr']
        OUTPUT_DIR = os.path.join(modeldir,
                          str(epoch) + '_' + str(evallossmean) + '_' + str(
                              currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                              allscore)[:6] + '.pkl')
        torch.save(model, OUTPUT_DIR)
        with open(os.path.join(resultsdir, str(epoch) + '_' + str(evallossmean) + '_' + str(
                                        currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(outfinal, f, ensure_ascii=False, indent=4)
        with open(os.path.join(resultsdir, 'analyse_' + str(epoch) + '_' + str(evallossmean) + '_' + str(
                                        currentlr) + '_' + str(trainallscore)[:6] + '_' + str(
                                        allscore)[:6] + ".json"), 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=4)

        torch.cuda.empty_cache()
        if allscore < evalacc_best:
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

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../Dataset/en', type=str, help='Dir saves the datasource information')
    parser.add_argument('--modal', default='text', type=str, help='which data stream')
    parser.add_argument('--savedir', default='./../../trained', type=str, help='Dir to save trained model and results')
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

    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'
    #device = "cpu"
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

    ## for test ##
    #traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 20))}
    #valdict = {k: valdict[k] for k in list(random.sample(list(valdict.keys()), 20))}


    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", cache_dir=cachedir)
    config = {
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "lr": 2e-5,
            "batch_size": 8,
            "modeldir": modeldir,
            "resultsdir": resultsdir,
            "tokenizer": tokenizer,
            "max_len": max_len,
            "overlap": overlap,
            "cachedir": cachedir,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "traindict": traindict,
            "valdict": valdict
        }
    training(config)


