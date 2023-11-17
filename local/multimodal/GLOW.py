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
    beta_1_class_0 = []
    beta_2_class_0 = []
    beta_1_class_1 = []
    beta_2_class_1 = []
    for i in list(traindict.keys()):
        if traindict[i]['label'] == 0.0:
            beta_1_class_0.append(traindict[i]['beta_1'])
            beta_2_class_0.append(traindict[i]['beta_2'])
        else:
            beta_1_class_1.append(traindict[i]['beta_1'])
            beta_2_class_1.append(traindict[i]['beta_2'])


    GMM_0 = GaussianMixture(n_components=nmodel, random_state=0).fit(np.column_stack((beta_1_class_0, beta_2_class_0)))
    GMM_1 = GaussianMixture(n_components=nmodel, random_state=0).fit(np.column_stack((beta_1_class_1, beta_2_class_1)))

    Conv_0 = GMM_0.covariances_
    mean_0 = GMM_0.means_
    weights_0 = GMM_0.weights_
    Conv_1 = GMM_1.covariances_
    mean_1 = GMM_1.means_
    weights_1 = GMM_1.weights_
    scoredict = {}
    alphas = np.linspace(0.0, 1.0, num=1000)
    for alpha in alphas:
        score = targetfunction(alpha, Conv_0, mean_0, weights_0, Conv_1, mean_1, weights_1)
        scoredict.update({alpha: score})
    Keymax = max(zip(scoredict.values(), scoredict.keys()))[1]
    return Keymax

def targetfunction(alpha, class0_conv, class0_mean, weights_0, class1_conv, class1_mean, weights_1):
    n_models = len(weights_0)
    score1 = 0
    score2 = 0
    for i in range(n_models):
        class0fact = (alpha * class0_mean[i][0] + (1 - alpha) * class0_mean[i][1]) /  \
                (math.sqrt(2) * math.sqrt(math.pow(alpha, 2) * class0_conv[i][0][0] + math.pow((1 - alpha), 2) * class0_conv[i][1][1] + \
                                          2 * alpha * (1 - alpha) * class0_conv[i][0][1]))
        score1 = score1 + weights_0[i] * (0.5 + 0.5 * math.erf(class0fact)) * 0.5 # P-class = 0.5, dataset is balanced
        class1fact = (alpha * class1_mean[i][0] + (1 - alpha) * class1_mean[i][1]) / \
                 (math.sqrt(2) * math.sqrt(
                     math.pow(alpha, 2) * class1_conv[i][0][0] + math.pow((1 - alpha), 2) * class1_conv[i][1][1] + \
                     2 * alpha * (1 - alpha) * class1_conv[i][0][1]))
        score2 = score2 + weights_1[i] * (0.5 - 0.5 * math.erf(class1fact)) * 0.5 # P-class = 0.5, dataset is balanced
    return score1 + score2

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
    return testallscore

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
    parser.add_argument('--sourcedir', default='./../../pretrained_results', type=str, help='Dir saves the datasource information')
    parser.add_argument('--textmodal', default='text', type=str, help='which data stream')
    parser.add_argument('--imagemodal', default='image_single', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../Image', type=str, help='Dir to save trained model and results')
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
    savedir = os.path.join(savedir, 'GLOW')
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
        fontsize = 10
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
        plt.savefig(os.path.join(savedir, dset + 'beta_verteilung.pdf'))
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
        plt.savefig(os.path.join(savedir, dset + 'beta_verteilung_class1.pdf'))

    combines = eval_alpha(testbetadict, alpha)
    text_only_score = texttestfile.split('_')[-1].strip('.json')
    image_only_score = imagetestfile.split('_')[-1].strip('.json')
    print('text-only model accuracy is: ' + text_only_score + '\n')
    print('image-only model accuracy is: ' + image_only_score + '\n')
    print('fusion model accuracy is: ' + str(combines) + '\n')










