import os, sys
import json
import numpy as np
import torch
from tqdm import tqdm
from model import *
import argparse
from utils import *
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from kaldiio import WriteHelper

SEED=666
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Gaussian_loss(torch.nn.Module):
    def __init__(self, device):
        super(Gaussian_loss, self).__init__()
        self.device = device
        self.mse = torch.nn.MSELoss()

    def forward(self, text_logits, image_logits, pretrain_text_prob, pretrain_image_prob , alpha, label):
        beta1 = torch.log(pretrain_text_prob[:, 0] / pretrain_text_prob[:, 1])
        beta2 = torch.log(pretrain_image_prob[:, 0] / pretrain_image_prob[:, 1])
        betaold = torch.stack((beta1, beta2), dim=-1)


        beta1new = torch.log(text_logits[:, 0] / text_logits[:, 1])
        beta2new = torch.log(image_logits[:, 0] / image_logits[:, 1])
        betanew = torch.stack((beta1new, beta2new), dim=-1)

        BS = label.size()[0]
        beta_1_class_0 = []
        beta_2_class_0 = []
        beta_1_class_1 = []
        beta_2_class_1 = []
        labelbeta = []
        for i in range(BS):
            if label[i] == 0:
                beta_1_class_0.append(torch.log(text_logits[i][0] / text_logits[i][1]))
                beta_2_class_0.append(torch.log(image_logits[i][0] / image_logits[i][1]))
                labelbeta.append(betaold[i] + torch.FloatTensor([3.0, 3.0]).to(self.device))
            else:
                beta_1_class_1.append(torch.log(text_logits[i][0] / text_logits[i][1]))
                beta_2_class_1.append(torch.log(image_logits[i][0] / image_logits[i][1]))
                labelbeta.append(betaold[i] - torch.FloatTensor([3.0, 3.0]).to(self.device))
        beta_1_class_0 = torch.stack(beta_1_class_0)
        beta_2_class_0 = torch.stack(beta_2_class_0)
        beta_class0 = torch.stack([beta_1_class_0, beta_2_class_0], dim=1)
        beta_1_class_1 = torch.stack(beta_1_class_1)
        beta_2_class_1 = torch.stack(beta_2_class_1)
        beta_class1 = torch.stack([beta_1_class_1, beta_2_class_1], dim=1)
        labelbeta = torch.stack(labelbeta)

        class0_conv = torch.cov(beta_class0)
        class0_mean = torch.mean(beta_class0, dim=0)
        class1_conv = torch.cov(beta_class1)
        class1_mean = torch.mean(beta_class1, dim=0)

        class0fact = (alpha * class0_mean[0] + (1 - alpha) * class0_mean[1]) / \
                     (torch.sqrt(torch.FloatTensor([2]).to(self.device)) * torch.sqrt(
                         torch.pow(alpha, 2) * class0_conv[0][0] + torch.pow((1 - alpha), 2) * class0_conv[1][1] + \
                         2 * alpha * (1 - alpha) * class0_conv[0][1]))
        score1 = (0.5 + 0.5 * torch.erf(class0fact)) * 0.5  # P-class = 0.5, dataset is balanced
        class1fact = (alpha * class1_mean[0] + (1 - alpha) * class1_mean[1]) / \
                     (torch.sqrt(torch.FloatTensor([2]).to(self.device)) * torch.sqrt(
                         torch.pow(alpha, 2) * class1_conv[0][0] + torch.pow((1 - alpha), 2) * class1_conv[1][1] + \
                         2 * alpha * (1 - alpha) * class1_conv[0][1]))
        score2 = (0.5 - 0.5 * torch.erf(class1fact)) * 0.5  # P-class = 0.5, dataset is balanced

        mseloss = self.mse(betanew, labelbeta)
        Gaussianloss = -(score1 + score2)
        loss = 0.5 * mseloss + 0.5 * Gaussianloss
        #print(betanew)
        #print(labelbeta)
        return loss

def eval_output(textprob, imageprob, alpha):
    text_beta = torch.log(textprob[:, 0] / textprob[:, 1])
    image_beta = torch.log(imageprob[:, 0] / imageprob[:, 1])

    prob = torch.sigmoid(alpha * text_beta + (1 - alpha) * image_beta)
    #estimate_prob = torch.stack([prob, 1 - prob], dim=-1)
    return prob

def training(config, dataset=None, checkpoint_dir=None, data_dir=None):
    traindict = config["traindict"]
    testdict = config["valdict"]

    traindict = combine_text(traindict, shuffle=True)
    testdict = combine_text(testdict, shuffle=True)
    traindict = get_split(traindict, config["max_len"], config["overlap"])
    testdict = get_split(testdict, config["max_len"], config["overlap"])

    dataset = multidatasetclass(train_file=traindict,
                               test_file=testdict,
                               tokenizer=config["tokenizer"],
                               device=config["device"],
                               max_len=config["max_len"])
    print(len(dataset.train_dataset))
    print(len(dataset.test_dataset))

    model = feature_extraction(config["cachedir"])

    for param in model.textmodel.BERT.roberta.parameters():
        param.requires_grad = False
    for param in model.imagemodel.ViT.parameters():
        param.requires_grad = False

    if config["ifpretrain"] == 'false':
        pass
    else:
        loaded_text_state = torch.load(config["textmodeldir"], map_location='cpu').state_dict()
        self_state = model.state_dict()
        loaded_text_state = {'textmodel.' + k: v for k, v in loaded_text_state.items() if 'textmodel.' + k in self_state}
        self_state.update(loaded_text_state)
        loaded_image_state = torch.load(config["imagemodeldir"], map_location='cpu').state_dict()
        loaded_image_state = {'imagemodel.' + k: v for k, v in loaded_image_state.items() if 'imagemodel.' + k in self_state}
        self_state.update(loaded_image_state)
        model.load_state_dict(self_state)

    model = model.to(config["device"])
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)

    train_examples_len = len(dataset.train_dataset)


    data_loader_train = torch.utils.data.DataLoader(dataset.train_dataset, shuffle=True,
                                                        batch_size=config["batch_size"],
                                                        num_workers=config["NWORKER"],
                                                        collate_fn=pad_multi)

    data_loader_dev = torch.utils.data.DataLoader(dataset.test_dataset, shuffle=True,
                                                      batch_size=config["batch_size"],
                                                      num_workers=config["NWORKER"],
                                                      collate_fn=pad_multi)

    #### Feature extraction ###
    traintextprobarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'traintextprob.ark') + ',' + os.path.join(config['datadir'],
        'traintextprob.scp')
    traintextfeatsarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'traintextfeats.ark') + ',' + os.path.join(config['datadir'],
        'traintextfeats.scp')
    trainimageprobarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'trainimageprob.ark') + ',' + os.path.join(config['datadir'],
        'trainimageprob.scp')
    trainimagefeatsarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'trainimagefeats.ark') + ',' + os.path.join(config['datadir'],
        'trainimagefeats.scp')
    trainlabelarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                                          'trainlabel.ark') + ',' + os.path.join(config['datadir'],
                                                                                                      'trainlabel.scp')
    model.train()
    traindict = {}
    for i, data in enumerate(tqdm(data_loader_train), 0):
        node_sets = data[0].to(config["device"])
        mask = data[1].to(config["device"])
        image = data[2].to(config["device"])
        labels = data[3].to(config["device"])
        filename = data[4]
        # zero the parameter gradients
        text_logits, text_feat, image_logits, image_feat = model(node_sets, mask, image)
        for i in range(len(filename)):
            textprob = np.expand_dims(text_logits[i, :].cpu().data.numpy(), axis=0)
            textfeats = np.expand_dims(text_feat[i, :].cpu().data.numpy(), axis=0)
            imageprob = np.expand_dims(image_logits[i, :].cpu().data.numpy(), axis=0)
            imagefeats = np.expand_dims(image_feat[i, :].cpu().data.numpy(), axis=0)
            label = np.expand_dims(labels[i, :].cpu().data.numpy(), axis=0)
            traindict.update({filename[i]: {}})
            traindict[filename[i]].update({'textprob': textprob})
            traindict[filename[i]].update({'textfeats': textfeats})
            traindict[filename[i]].update({'imageprob': imageprob})
            traindict[filename[i]].update({'imagefeats': imagefeats})
            traindict[filename[i]].update({'label': label})

    with WriteHelper(traintextprobarksavedir, compression_method=2) as writer1:
        with WriteHelper(traintextfeatsarksavedir, compression_method=2) as writer2:
            with WriteHelper(trainimageprobarksavedir, compression_method=2) as writer3:
                with WriteHelper(trainimagefeatsarksavedir, compression_method=2) as writer4:
                    with WriteHelper(trainlabelarksavedir, compression_method=2) as writer5:
                        for ids in tqdm(list(traindict.keys())):
                            writer1(ids, traindict[ids]['textprob'])
                            writer2(ids, traindict[ids]['textfeats'])
                            writer3(ids, traindict[ids]['imageprob'])
                            writer4(ids, traindict[ids]['imagefeats'])
                            writer5(ids, traindict[ids]['label'])

    testtextprobarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'testtextprob.ark') + ',' + os.path.join(config['datadir'],
        'testtextprob.scp')
    testtextfeatsarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'testtextfeats.ark') + ',' + os.path.join(config['datadir'],
        'testtextfeats.scp')
    testimageprobarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'testimageprob.ark') + ',' + os.path.join(config['datadir'],
        'testimageprob.scp')
    testimagefeatsarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                           'testimagefeats.ark') + ',' + os.path.join(config['datadir'],
        'testimagefeats.scp')
    testlabelarksavedir = 'ark,scp:' + os.path.join(config['datadir'],
                                                          'testlabel.ark') + ',' + os.path.join(config['datadir'],
                                                                                                      'testlabel.scp')
    testdict = {}
    for i, data in enumerate(tqdm(data_loader_dev), 0):
        with torch.no_grad():
            node_sets = data[0].to(config["device"])
            mask = data[1].to(config["device"])
            image = data[2].to(config["device"])
            labels = data[3].to(config["device"])
            filename = data[4]
            text_logits, text_feat, image_logits, image_feat = model(node_sets, mask, image)
            for i in range(len(filename)):
                textprob = np.expand_dims(text_logits[i, :].cpu().data.numpy(), axis=0)
                textfeats = np.expand_dims(text_feat[i, :].cpu().data.numpy(), axis=0)
                imageprob = np.expand_dims(image_logits[i, :].cpu().data.numpy(), axis=0)
                imagefeats = np.expand_dims(image_feat[i, :].cpu().data.numpy(), axis=0)
                label = np.expand_dims(labels[i, :].cpu().data.numpy(), axis=0)
                testdict.update({filename[i]: {}})
                testdict[filename[i]].update({'textprob': textprob})
                testdict[filename[i]].update({'textfeats': textfeats})
                testdict[filename[i]].update({'imageprob': imageprob})
                testdict[filename[i]].update({'imagefeats': imagefeats})
                testdict[filename[i]].update({'label': label})

    with WriteHelper(testtextprobarksavedir, compression_method=2) as writer1:
        with WriteHelper(testtextfeatsarksavedir, compression_method=2) as writer2:
            with WriteHelper(testimageprobarksavedir, compression_method=2) as writer3:
                with WriteHelper(testimagefeatsarksavedir, compression_method=2) as writer4:
                    with WriteHelper(testlabelarksavedir, compression_method=2) as writer5:
                        for ids in tqdm(list(testdict.keys())):
                            writer1(ids, testdict[ids]['textprob'])
                            writer2(ids, testdict[ids]['textfeats'])
                            writer3(ids, testdict[ids]['imageprob'])
                            writer4(ids, testdict[ids]['imagefeats'])
                            writer5(ids, testdict[ids]['label'])

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
    parser.add_argument('--savedir', default='./../../trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--ifpretrain', default='true', type=str, help='single or multi images')
    parser.add_argument('--cachedir', default='./../../CACHE', type=str, help='Dir to save downloaded pretrained model')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasetdir = args.datasdir
    ifpretrain = args.ifpretrain
    savedir = args.savedir
    cachedir = args.cachedir


    if torch.cuda.is_available() == True:
        device = 'cuda'
    else:
        device = 'cpu'

    textmodeldir = os.path.join(savedir, 'text', 'model')
    textmodeldir = find_bestmodel(textmodeldir)
    imagemodeldir = os.path.join(savedir, 'image_single', 'model')
    imagemodeldir = find_bestmodel(imagemodeldir)

    datadir = os.path.join(datasetdir, 'data')

    max_len = 500
    overlap = 128
    max_num_epochs = 70


    if not os.path.exists(datadir):
        os.makedirs(datadir)


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

    ## for test ##
    #traindict = {k: traindict[k] for k in list(random.sample(list(traindict.keys()), 20))}
    #valdict = {k: valdict[k] for k in list(random.sample(list(valdict.keys()), 20))}


    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-large", cache_dir=cachedir)
    config = {
            "NWORKER": 0,
            "device": device,
            "weight_decay": 0,
            "eps": 1e-8,
            "lr": 2e-2, #2e-3,
            "batch_size": 64, #8,
            "ifpretrain": ifpretrain,
            "datadir": datadir,
            "textmodeldir": textmodeldir,
            "imagemodeldir": imagemodeldir,
            "tokenizer": tokenizer,
            "max_len": max_len,
            "overlap": overlap,
            "cachedir": cachedir,
            "epochs": max_num_epochs,  # tune.choice([3, 5, 10, 15])
            "traindict": traindict,
            "valdict": valdict
        }
    training(config)



