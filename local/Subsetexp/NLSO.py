import os, sys
import json
import math
import numpy as np
import random
import argparse
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import kaldiio
def _get_from_loader(filepath, filetype):

    if filetype in ['mat', 'vec']:
        # e.g.
        #    {"input": [{"feat": "some/path.ark:123",
        #                "filetype": "mat"}]},
        # In this case, "123" indicates the starting points of the matrix
        # load_mat can load both matrix and vector
        #filepath = filepath.replace('/home/wentao', '.')
        return kaldiio.load_mat(filepath)
    elif filetype == 'scp':
        # e.g.
        #    {"input": [{"feat": "some/path.scp:F01_050C0101_PED_REAL",
        #                "filetype": "scp",
        filepath, key = filepath.split(':', 1)
        loader = self._loaders.get(filepath)
        if loader is None:
            # To avoid disk access, create loader only for the first time
            loader = kaldiio.load_scp(filepath)
            self._loaders[filepath] = loader
        return loader[key]
    else:
        raise NotImplementedError(
            'Not supported: loader_type={}'.format(filetype))

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
    return testallscore

def find_best(filelists):
    '''devdict = {}
    evaldict = {}
    searchdict = {}
    for i in filelists:
        if 'results' in i:
            pass
        else:
            epoch = i#.split('_')[0]
            score = i.split('_')[-1].strip('.json')
            searchdict.update({epoch: float(score)})
            if 'dev' in i:
                devdict.update({epoch: i})
            else:
                evaldict.update({epoch: i})


    Keymax = max(zip(searchdict.values(), searchdict.keys()))[1]
    traindir = devdict[Keymax]
    testdir = evaldict[Keymax]'''
    for i in filelists:
        if "trainsetresults_" in i:
            traindir = i
        elif "testsetresults_" in i:
            testdir = i
    return traindir, testdir


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


def get_args():
    parser = argparse.ArgumentParser()

    # get arguments from outside
    parser.add_argument('--datasdir', default='./../../pretrained', type=str, help='Dir saves the datasource information')
    parser.add_argument('--textmodal', default='text', type=str, help='which data stream')
    parser.add_argument('--imagemodal', default='image_single', type=str, help='single or multi images')
    parser.add_argument('--modal', default='Logits_Space_opt_subset', type=str, help='single or multi images')
    parser.add_argument('--savedir', default='./../../trained', type=str, help='Dir to save trained model and results')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    datasetdir = args.datasdir
    textmodal = args.textmodal
    imagemodal = args.imagemodal
    modal = args.modal
    savedir = args.savedir
    featdir = os.path.join(datasetdir, 'data')

    globaldir = os.path.join(savedir, 'multi_global_stream_weight_subset')
    with open(os.path.join(datasetdir, "pan18-author-profiling-training-dataset-2018-02-27.json"),
              encoding="utf8") as json_file:
        trainolddict = json.load(json_file)
    traindict = {}
    testdict = {}
    for workdict, dset in zip([testdict, traindict], ["test", "train"]):
        imageprobscpdir = os.path.join(featdir, dset + 'imageprob.scp')
        textprobscpdir = os.path.join(featdir, dset + 'textprob.scp')
        labelscpdir = os.path.join(featdir, dset + 'label.scp')

        for featinfodir, name in zip(
                [textprobscpdir, imageprobscpdir, labelscpdir],
                ['prob', 'imageprob', 'label']):
            with open(featinfodir) as f:
                srcdata = f.readlines()
            for j in srcdata:
                if j.split(' ')[0] in workdict:
                    pass
                else:
                    workdict.update({j.split(' ')[0]: {}})
                data = _get_from_loader(
            filepath=j.split(' ')[1],
            filetype='mat')
                if 'prob' in name:
                    data = softmax(data)
                workdict[j.split(' ')[0]].update({name: list(data[0])})

    for workdict, dset in zip([testdict, traindict], ["test", "train"]):
        out = {}
        for documentid in workdict.keys():
            author = documentid.split('_')[0]
            if out.get(author) is not None:
                pass
            else:
                out.update({author: {}})
            doc_id = documentid.split('_')[1]
            out[author].update({doc_id: {}})
            out[author][doc_id].update({'label': float(workdict[documentid]['label'][0])})
            out[author][doc_id].update({'prob': workdict[documentid]['prob']})
            out[author][doc_id].update({'imageprob': workdict[documentid]['imageprob']})
        outfinal = {}
        for author in list(out.keys()):
            imageprobs = []
            probs = []
            for id in list(out[author].keys()):
                imageprobs.append(out[author][id]['imageprob'])
                probs.append(out[author][id]['prob'])
            label = out[author][id]['label']
            outfinal.update({author: {}})
            outfinal[author].update({'imageprob': np.mean(np.asarray(imageprobs), axis=0)})
            outfinal[author].update({'prob': np.mean(np.asarray(probs), axis=0)})
            outfinal[author].update({'label': label})
        if dset == 'train':
            traindict = outfinal
        else:
            testdict = outfinal



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
        cvsubfolder = os.path.join(cvfolder, str(N))
        globalsubdir = os.path.join(globaldir, str(N) + '_samples')
        for id in range(10):
            globalsubcvdir = os.path.join(globalsubdir, str(id), 'results')

            globalfiledir = [i for i in os.listdir(globalsubcvdir) if i.endswith('.json')]
            globalname = os.path.join(globalsubcvdir, globalfiledir[0])
            with open(globalname, encoding="utf8") as json_file:
                globaldict = json.load(json_file)
            globalweight = float(list(globaldict['round12acc'].keys())[0])
            traindatadir = os.path.join(cvsubfolder, str(id) + '.json')
            with open(traindatadir, encoding="utf8") as json_file:
                traincvdict = json.load(json_file)
            newtrainedict = {}
            for trainkey in list(traincvdict.keys()):
                newtrainedict.update({trainkey: traindict[trainkey]})

            resultsdir = os.path.join(savedir, modal, str(N) + '_samples',
                                      str(id), 'results')
            if not os.path.exists(resultsdir):
                os.makedirs(resultsdir)





            for dsetdict in [newtrainedict, testdict]:
                if dsetdict == newtrainedict:
                    type = 'train'
                    trainbetadict = calculate_beta(dsetdict, savedir, type)
                else:
                    type = 'test'
                    testbetadict = calculate_beta(dsetdict, savedir, type)

            nmodel = 1
            labelpads = 0.1
            rotations = 0
            #print('GMM with ' + str(nmodel) + ' models')
            alpha = calculate_alpha(trainbetadict, nmodel)
            print(str(N) + ' ' + str(id) + ' ' + str(alpha)[:4])
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
                if not os.path.exists(os.path.join(resultsdir, dset + 'beta_verteilung.pdf')):
                    plt.figure()
                    plt.ylabel(r'$\beta$2', labelpad=labelpads, rotation=rotations, fontsize=fontsize)
                    plt.xlabel(r'$\beta$1', fontsize=fontsize)
                    #plt.title(dset + 'beta_verteilung')
                    plt.scatter(beta_1_class_0, beta_2_class_0, color='red', alpha=0.1)
                    plt.scatter(beta_1_class_1, beta_2_class_1, color='blue', alpha=0.1)
                    plt.axline((0, 0), slope=alpha / (alpha - 1.0), color='green', label='GLGSW slope')
                    plt.axline((0, 0), slope=globalweight / (globalweight - 1.0), color='fuchsia', label='GSW slope')
                    plt.axhline(0, color='black', linestyle='--')
                    plt.axvline(0, color='black', linestyle='--')
                    plt.legend(loc="lower right", fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    plt.xticks(fontsize=fontsize)
                    plt.locator_params(axis='y', nbins=3) 
                    plt.locator_params(axis='x', nbins=3) 
                    plt.savefig(os.path.join(resultsdir, dset + 'beta_verteilung.pdf'), bbox_inches='tight')
                    plt.figure()
                    plt.ylabel(r'$\beta$2', labelpad=labelpads, rotation=rotations, fontsize=fontsize)
                    plt.xlabel(r'$\beta$1', fontsize=fontsize)
                    #plt.title(dset + 'beta_verteilung_class0.png')
                    plt.scatter(beta_1_class_0, beta_2_class_0, color='red', alpha=0.1)
                    plt.axline((0, 0), slope=alpha / (alpha - 1.0), color='green', label='GLGSW slope')
                    plt.axline((0, 0), slope=globalweight / (globalweight - 1.0), color='yellow', label='GSW slope')
                    plt.axhline(0, color='black', linestyle='--')
                    plt.axvline(0, color='black', linestyle='--')
                    plt.legend(loc="lower right", fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    plt.xticks(fontsize=fontsize)
                    plt.savefig(os.path.join(resultsdir, dset + 'beta_verteilung_class0.pdf'))
                    plt.figure()
                    plt.ylabel(r'$\beta$2', labelpad=labelpads, rotation=rotations, fontsize=fontsize)
                    plt.xlabel(r'$\beta$1', fontsize=fontsize)
                    #plt.title(dset + 'beta_verteilung_class1.png')
                    plt.scatter(beta_1_class_1, beta_2_class_1, color='blue', alpha=0.1)
                    plt.axline((0, 0), slope=alpha / (alpha - 1.0), color='green', label='GLGSW slope')
                    plt.axline((0, 0), slope=globalweight / (globalweight - 1.0), color='yellow', label='GSW slope')
                    plt.axhline(0, color='black', linestyle='--')
                    plt.axvline(0, color='black', linestyle='--')
                    plt.legend(loc="lower right", fontsize=fontsize)
                    plt.yticks(fontsize=fontsize)
                    plt.xticks(fontsize=fontsize)
                    plt.savefig(os.path.join(resultsdir, dset + 'beta_verteilung_class1.pdf'))




            combines = eval_alpha(testbetadict, alpha)
            #text_only_score = texttestfile.split('_')[-1].strip('.json')
            #image_only_score = imagetestfile.split('_')[-1].strip('.json')
            file = open(os.path.join(resultsdir, 'results.txt'), 'a')  # Open a file in append mode
            #file.write('text-only model accuracy is: ' + text_only_score + '\n')
            #file.write('image-only model accuracy is: ' + image_only_score + '\n')
            file.write('fusion model accuracy is: ' + str(combines) + '\n')  # Write some text
            file.close()  # Close the file


    savemaindir = os.path.join(savedir, modal)
    splitlist = os.listdir(savemaindir)
    accdict = {}
    for split in splitlist:
        nsplit = split.split('_')[0]
        accdict.update({int(nsplit): []})
        cvid = os.listdir(os.path.join(savemaindir, split))
        for cv in cvid:
            #accdict[nsplit].update({cv: []})
            resultscvdir = os.path.join(savemaindir, split, cv, 'results',  'results.txt')
            with open(resultscvdir) as f:
                lines = f.readlines()
            for i in lines:
                if 'fusion' in i:
                    accdict[int(nsplit)].append(float(i.split(' ')[-1].strip('\n')))
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
    #plt.show()
    plt.savefig(os.path.join(savemaindir, 'Accuracy_trainsetsize.png'))







