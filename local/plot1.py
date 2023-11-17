import os, sys
import numpy as np
import argparse
import matplotlib.pyplot as plt


def draw_plot(data, offset,edge_color, fill_color, ax):
    pos = np.arange(data.shape[1])+offset 
    bp = ax.boxplot(data, positions= pos, widths=0.3, patch_artist=True, manage_ticks=False, sym='')
    for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
def find_file(modeldir):
    filename = os.listdir(modeldir)
    filedict = {}
    for i in filename:
        if i.endswith('.txt'):
            with open(os.path.join(modeldir, i)) as f:
                lines = f.readlines()
            for i in lines:
                if 'fusion' in i:
                    acc = float(i.split(' ')[-1].strip('\n'))

    return acc

def find_bestmodel(modeldir):
    filename = os.listdir(modeldir)
    filedict = {}
    for i in filename:
        score = float(i.split('_')[-1].strip('.json'))
        filedict.update({i: score})
    Keymax = max(zip(filedict.values(), filedict.keys()))[1]
    return os.path.join(modeldir, Keymax)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traindir', default='./../trained', type=str, help='Dir to save trained model and results')
    parser.add_argument('--savedir', default='./../Image', type=str, help='single or multi images')
    parser.add_argument('--modal1', default='GLGSW_subset', type=str, help='single or multi images')
    parser.add_argument('--modal2', default='GSW-GLGSW_subset', type=str, help='single or multi images')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print("Arguments:")
    [print(key, val) for key, val in vars(args).items()]

    traindir = args.traindir
    savedir = args.savedir
    modal1 = args.modal1
    modal2 = args.modal2
    if modal2 == 'multi_global_stream_weight_subset':
        typeref = 'GSW'
    elif modal2 == 'dynamic_stream_weight_subset':
        typeref = 'DSW'
    elif modal2 == 'representation_fusion_subset':
        typeref = 'RF'
    elif modal2 == 'NLSO_subset':
        typeref = 'NLSO'

    type = 'GLOW'

    savemaindir = os.path.join(traindir, modal2)
    splitlist = os.listdir(savemaindir)
    splitlist.remove('Accuracy_trainsetsize.png')
    accdict = {}
    maxacc = 0
    for split in splitlist:
        nsplit = split.split('_')[0]
        accdict.update({nsplit: []})
        cvid = os.listdir(os.path.join(savemaindir, split))
        for cv in cvid:
            resultscvdir = os.path.join(savemaindir, split, cv, 'results', 'results.txt')
            with open(resultscvdir) as f:
                lines = f.readlines()
            for i in lines:
                if 'fusion' in i:
                    acc = float(i.split(' ')[-1].strip('\n'))
                    accdict[nsplit].append(acc)
                    if acc > maxacc:
                        maxacc = acc

    savemaindir = os.path.join(traindir, modal1)
    splitlist = os.listdir(savemaindir)
    splitlist.remove('Accuracy_trainsetsize.png')
    accdictgaussian = {}
    maxgaussianacc = 0
    for split in splitlist:
        nsplit = split.split('_')[0]
        accdictgaussian.update({nsplit: []})
        cvid = os.listdir(os.path.join(savemaindir, split))
        #print(cvid)
        for cv in cvid:
            # accdict[nsplit].update({cv: []})
            resultscvdir = os.path.join(savemaindir, split, cv, 'results', 'results.txt')
            with open(resultscvdir) as f:
                lines = f.readlines()
            for i in lines:
                if 'fusion' in i:
                    acc = float(i.split(' ')[-1].strip('\n'))
                    accdictgaussian[nsplit].append(acc)
                    if acc > maxgaussianacc:
                        maxgaussianacc = acc
    fontsize = 15
    ticks = ['16', '32', '64', '128', '256']#, '1024', '2048']
    data = []
    labels = []
    x = [0, 1, 2, 3, 4]#, 8, 9]
    for nsplit in ticks:
        data.append(accdict[nsplit])
        labels.append(int(nsplit))

    datagaussian = []
    for nsplit in ['16', '32', '64', '128', '256']:#, '1024', '2048']:
        datagaussian.append(accdictgaussian[nsplit])
    plt.figure(figsize=(4,3))
    plt.ylabel('Accuracy')
    plt.xlabel('Trainsubset size')

    plt.title('Accuracy changed based on trainsubset size')
    fig, ax = plt.subplots()
    draw_plot(np.transpose(np.array(data)), -0.2, "tomato", "white", ax)
    draw_plot(np.transpose(np.array(datagaussian)), +0.2,"skyblue", "white", ax)
    plt.xticks(x)
    #plt.savefig(__file__+'.png', bbox_inches='tight')

    #datasw = plt.boxplot(data)
    #datagaussian = plt.boxplot(datagaussian)
    #set_box_color(datasw, '#D7191C')  # colors are from http://colorbrewer2.org/
    #set_box_color(datagaussian, '#2C7BB6')

    plt.plot([], c='tomato', label=typeref)
    plt.plot([], c='skyblue', label=type)
    plt.legend(fontsize=fontsize)
    plt.xlabel('The size of sub-training set', fontsize=fontsize)
    plt.ylabel('Accuracy',fontsize=fontsize)

    plt.xticks(range(len(labels)), labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.xlim(-2, len(ticks))
    plt.tight_layout()

    plt.savefig(os.path.join(savedir, type + '-' + typeref + '.pdf'))
    print('max stream weighting acc is ' + str(maxacc))
    print('max gaussian distribution acc is ' + str(maxgaussianacc))




