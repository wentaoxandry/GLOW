from xml.dom import minidom

import json
import os, sys
import random
from PIL import Image
import multiprocessing as mp

def processtext(setsdir, LN, name):
    xmldir = os.path.join(setsdir,LN, "text", name + ".xml")
    mydoc = minidom.parse(xmldir)
    items = mydoc.getElementsByTagName('document')
    Utts = []
    for i in range(len(items)):
        rowutt = items[i].childNodes[0].data
        Utts.append(rowutt)

    templist = []
    for k in range(len(Utts)):
        templist.append({name + "_" + str(k).zfill(3): Utts[k]})

    return templist



def product_helper(args):
    return processtext(*args)


def main(Datadir, LN, Savedir):
    subdirs = os.listdir(Datadir)
    if not os.path.exists(os.path.join(Savedir, LN)):
        os.makedirs(os.path.join(Savedir, LN))
    for subdir in subdirs:
        imagedir = os.path.join(Datadir, subdir, LN, 'photo')
        setsdir = os.path.join(Datadir, subdir)
        with open(os.path.join(setsdir, LN, LN + ".txt")) as f:
            lines = f.readlines()
        labeldict = {}
        for i in range(len(lines)):
            index = lines[i].split(":::")[0]
            gender = lines[i].split(":::")[1].strip('\n')
            labeldict.update({index: {}})
            if gender == "male":
                label = 0
            else:
                label = 1
            labeldict[index].update({"label": label})

        idkey = labeldict.keys()
        results = []
        pool = mp.Pool()
        job_args = [(setsdir, LN, i) for i in list(idkey)]
        results.extend(pool.map(product_helper, job_args))
        for i in range(len(results)):
            authortext = []

            for j in range(len(results[i])):
                name = list(results[i][j].keys())[0]
                labelname = name.split('_')[0]
                text = results[i][j][name]
                authortext.append(text)
            imagelist = os.listdir(os.path.join(imagedir, labelname))
            authorimage = [os.path.join(os.path.join(imagedir, labelname, m)) for m in imagelist]
            labeldict[labelname].update({'text': authortext})
            labeldict[labelname].update({'image': authorimage})

        with open(os.path.join(Savedir, LN, subdir + ".json"), 'w', encoding='utf-8') as f:
            json.dump(labeldict, f, ensure_ascii=False, indent=4)

    with open(os.path.join(Savedir, LN, "pan18-author-profiling-training-dataset-2018-02-27.json")) as json_file:
        trainsrcdata = json.load(json_file)
    with open(os.path.join(Savedir, LN, "pan18-author-profiling-test-dataset-2018-03-20.json")) as json_file:
        testsrcdata = json.load(json_file)
    '''trainkey = trainsrcdata.keys()
    devsetlen = int(0.2 * len(trainkey))
    devdict = {}
    devkeys = random.sample(list(trainkey), devsetlen)
    for devkey in devkeys:
        devdict.update({devkey: {}})
        devdict[devkey].update(trainsrcdata[devkey])
        del trainsrcdata[devkey]
    with open(os.path.join(Savedir, LN, "train_a.json"), 'w', encoding='utf-8') as f:
        json.dump(trainsrcdata, f, ensure_ascii=False, indent=4)
    with open(os.path.join(Savedir, LN, "dev_a.json"), 'w', encoding='utf-8') as f:
        json.dump(devdict, f, ensure_ascii=False, indent=4)
    with open(os.path.join(Savedir, LN, "test_a.json"), 'w', encoding='utf-8') as f:
        json.dump(testsrcdata, f, ensure_ascii=False, indent=4)'''











'''Datadir = "./../pan18"
LN = 'en'
Savedir = "./../Dataset"
main(Datadir, LN, Savedir)'''


main(sys.argv[1],sys.argv[2],sys.argv[3])
