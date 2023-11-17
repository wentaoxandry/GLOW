import os, sys
import json
import math
import numpy as np
import argparse
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture

def calculate_beta(dsetdict, savedir, type):
    newdsetdict = {}
    beta_1_class_0 = []
    beta_2_class_0 = []
    beta_1_class_1 = []
    beta_2_class_1 = []
    for i in list(dsetdict.keys()):
        newdsetdict.update({i: {}})
        newdsetdict[i].update({'label': dsetdict[i]['label']})
        newdsetdict[i].update({'beta_1': math.log(dsetdict[i]['prob'][0] / dsetdict[i]['prob'][1])})
        newdsetdict[i].update({'beta_2': math.log(dsetdict[i]['imageprob'][0] / dsetdict[i]['imageprob'][1])})
        if newdsetdict[i]['label'] == 0.0:
            beta_1_class_0.append(newdsetdict[i]['beta_1'])
            beta_2_class_0.append(newdsetdict[i]['beta_2'])
        else:
            beta_1_class_1.append(newdsetdict[i]['beta_1'])
            beta_2_class_1.append(newdsetdict[i]['beta_2'])

    return newdsetdict


def calculate_alpha(traindict, nmodel):
    beta_1_class = []
    beta_2_class = []
    label = []
    for i in list(traindict.keys()):
        beta_1_class.append(traindict[i]['beta_1'])
        beta_2_class.append(traindict[i]['beta_2'])
        label.append(traindict[i]['label'])

    scoredict = {}
    alphas = np.linspace(0.0, 1.0, num=1000)
    for alpha in alphas:
        score = targetfunction(beta_1_class, beta_2_class, label, alpha)
        scoredict.update({alpha: score})
    Keymax = max(zip(scoredict.values(), scoredict.keys()))[1]
    return Keymax

def targetfunction(beta_1_class, beta_2_class, label, alpha):
    beta = alpha * np.asarray(beta_1_class) + (1-alpha) * np.asarray(beta_2_class)
    newprob = 1/(1 + np.exp(-beta))
    pred = np.round(1 - newprob)
    acc = (np.asarray(label) == pred).sum() / len(label)

    return acc

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def eval_alpha(testdict, alpha):
    for i in list(testdict.keys()):
        beta = alpha * testdict[i]['beta_1'] + (1 - alpha) * testdict[i]['beta_2']
        newprob = sigmoid(beta)
        testdict[i].update({'prob': newprob})
        testdict[i].update({'predict': round(1 - newprob)})


    authors = testdict.keys()
    allpredict = []
    alllabel = []
    correct = 0
    for author in authors:
        alllabel.append(int(testdict[author]['label']))
        allpredict.append(testdict[author]['predict'])
        if testdict[author]['predict'] == int(testdict[author]['label']):
            correct = correct + 1

    testallscore = correct / len(alllabel)
    return testallscore, testdict

def find_best(filelists):
    for i in filelists:
        if "trainsetresults_" in i:
            traindir = i
        elif "testsetresults_" in i:
            testdir = i
    return traindir, testdir

def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--sourcedir', default='./../../pretrained/', type=str, help='Dir saves the datasource information')
    parser.add_argument('--textmodal', default='text', type=str, help='which data stream')
    parser.add_argument('--imagemodal', default='image_single', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    sourcedir = args.sourcedir
    textmodal = args.textmodal
    imagemodal = args.imagemodal
    savedir = args.savedir
    savedir = os.path.join(savedir, 'NLSO')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    results2dir = os.path.join(sourcedir, imagemodal, 'results')
    results1dir = os.path.join(sourcedir, textmodal, 'results')
    results2listdir = os.listdir(results2dir)
    results1listdir = os.listdir(results1dir)
    imagetrainfile, imagetestfile = find_best(results2listdir)
    texttrainfile, texttestfile = find_best(results1listdir)

    for dicts in [[texttrainfile, imagetrainfile], [texttestfile, imagetestfile]]:
        newtextdict = {}
        with open(os.path.join(results1dir, dicts[0]), encoding="utf8") as json_file:
            textdict = json.load(json_file)
        with open(os.path.join(results2dir, dicts[1]), encoding="utf8") as json_file:
            imagedict = json.load(json_file)

        for i in list(textdict.keys()):
            newtextdict.update({i: {}})
            newtextdict[i].update({'label': textdict[i]['0']['label']})
            probs = []
            for j in list(textdict[i].keys()):
                probs.append(textdict[i][j]['prob'])
            prob = np.mean(np.asarray(probs), axis=0)
            newtextdict[i].update({'prob': prob})
        textdict = newtextdict

        for i in list(textdict.keys()):
            textdict[i].update({'imageprob': imagedict[i]['prob']})

        if dicts == [texttrainfile, imagetrainfile]:
            traindict = textdict
        else:
            testdict = textdict

    for dsetdict in [traindict, testdict]:
        if dsetdict == traindict:
            type = 'train'
            trainbetadict = calculate_beta(dsetdict, savedir, type)
        else:
            type = 'test'
            testbetadict = calculate_beta(dsetdict, savedir, type)

    nmodel = 1
    labelpads=0.1
    rotations = 0
    print('GMM with ' + str(nmodel) + ' models')
    alpha = calculate_alpha(trainbetadict, nmodel)
    for dset in ['train', 'test']:
        if dset == 'train':
            betadict = trainbetadict
        else:
            betadict = testbetadict

        beta_1_class_0 = []
        beta_2_class_0 = []
        beta_1_class_1 = []
        beta_2_class_1 = []

        for i in list(betadict.keys()):
            if betadict[i]['label'] == 0.0:
                beta_1_class_0.append(betadict[i]['beta_1'])
                beta_2_class_0.append(betadict[i]['beta_2'])
            else:
                beta_1_class_1.append(betadict[i]['beta_1'])
                beta_2_class_1.append(betadict[i]['beta_2'])
        fontsize = 15
        #if not os.path.exists(os.path.join(savedir, dset + 'beta_verteilung.pdf')):
        plt.figure()
        plt.ylabel(r'$\beta$2', labelpad=labelpads, rotation=rotations, fontsize=fontsize)
        plt.xlabel(r'$\beta$1', fontsize=fontsize)
        # plt.title(dset + 'beta_verteilung')
        plt.scatter(beta_1_class_0, beta_2_class_0, color='red', alpha=0.1)
        plt.scatter(beta_1_class_1, beta_2_class_1, color='blue', alpha=0.1)
        plt.axline((0, 0), slope=alpha / (alpha - 1.0), color='green', label='by slope')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.locator_params(axis='y', nbins=3) 
        plt.locator_params(axis='x', nbins=3) 
        plt.savefig(os.path.join(savedir, dset + 'beta_verteilung.pdf'), bbox_inches='tight')
        plt.figure()
        plt.ylabel(r'$\beta$2', labelpad=labelpads, rotation=rotations, fontsize=fontsize)
        plt.xlabel(r'$\beta$1', fontsize=fontsize)
        # plt.title(dset + 'beta_verteilung_class0.png')
        plt.scatter(beta_1_class_0, beta_2_class_0, color='red', alpha=0.1)
        plt.axline((0, 0), slope=alpha / (alpha - 1.0), color='green', label='by slope')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.savefig(os.path.join(savedir, dset + 'beta_verteilung_class0.pdf'))
        plt.figure()
        plt.ylabel(r'$\beta$2', labelpad=labelpads, rotation=rotations, fontsize=fontsize)
        plt.xlabel(r'$\beta$1', fontsize=fontsize)
        # plt.title(dset + 'beta_verteilung_class1.png')
        plt.scatter(beta_1_class_1, beta_2_class_1, color='blue', alpha=0.1)
        plt.axline((0, 0), slope=alpha / (alpha - 1.0), color='green', label='by slope')
        plt.axhline(0, color='black', linestyle='--')
        plt.axvline(0, color='black', linestyle='--')
        plt.yticks(fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
       
        plt.savefig(os.path.join(savedir, dset + 'beta_verteilung_class1.pdf'), bbox_inches='tight')

    combines, testdict = eval_alpha(testbetadict, alpha)
    with open(os.path.join(savedir, 'GLGSW_test.json'), 'w', encoding='utf-8') as f:
            json.dump(testdict, f, ensure_ascii=False, indent=4)
    text_only_score = texttestfile.split('_')[-1].strip('.json')
    image_only_score = imagetestfile.split('_')[-1].strip('.json')
    print('text-only model accuracy is: ' + text_only_score + '\n')
    print('image-only model accuracy is: ' + image_only_score + '\n')
    print('fusion model accuracy is: ' + str(combines) + '\n')










