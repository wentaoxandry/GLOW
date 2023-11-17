import random
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import torch
from torch.utils.data import Dataset
from transformers import ViTFeatureExtractor
from PIL import Image, ImageFile
import kaldiio
ImageFile.LOAD_TRUNCATED_IMAGES = True

tokenizer = TweetTokenizer()


def pad_multi_feature(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    textfeat_sequence = []
    textprob_sequence = []
    imagefeat_sequence = []
    imageprob_sequence = []
    label_sequence = []
    filename_sequence = []
    #imagelist_sequence = []
    for textfeat, textprob, imagefeat, imageprob, label, filename in sequences:
        textfeat_sequence.append(textfeat.squeeze(0))
        textprob_sequence.append(textprob.squeeze(0))
        imagefeat_sequence.append(imagefeat.squeeze(0))
        imageprob_sequence.append(imageprob.squeeze(0))
        label_sequence.append(label.squeeze(0))
        filename_sequence.append(filename)
    textfeat_sequence = torch.nn.utils.rnn.pad_sequence(textfeat_sequence, batch_first=True)
    textprob_sequence = torch.nn.utils.rnn.pad_sequence(textprob_sequence, batch_first=True)
    imagefeat_sequence = torch.nn.utils.rnn.pad_sequence(imagefeat_sequence, batch_first=True)
    imageprob_sequence = torch.nn.utils.rnn.pad_sequence(imageprob_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return textfeat_sequence, textprob_sequence, imagefeat_sequence, imageprob_sequence, label_sequence, filename_sequence#, imagelist_sequence
def pad_multi(sequences):
    '''
    To pad different sequences into a padded tensor for training. The main purpose of this function is to separate different sequence, pad them in different ways and return padded sequences.
    Input:
        sequences <list>: A sequence with a length of 4, representing the node sets sequence in index 0, neighbor sets sequence in index 1, public edge mask sequence in index 2 and label sequence in index 3.
                          And the length of each sequences are same as the batch size.
                          sequences: [node_sets_sequence, neighbor_sets_sequence, public_edge_mask_sequence, label_sequence]
    Return:
        node_sets_sequence <torch.LongTensor>: The padded node sets sequence (works with batch_size >= 1).
        neighbor_sets_sequence <torch.LongTensor>: The padded neighbor sets sequence (works with batch_size >= 1).
        public_edge_mask_sequence <torch.BoolTensor>: The padded public edge mask sequence (works with batch_size >= 1).
        label_sequence <torch.FloatTensor>: The padded label sequence (works with batch_size >= 1).
    '''
    image_sequence = []
    node_sets_sequence = []
    mask_sequence = []
    label_sequence = []
    filename_sequence = []
    #imagelist_sequence = []
    for node_sets, mask, imagearray, label, filename in sequences:
        node_sets_sequence.append(node_sets.squeeze(0))
        mask_sequence.append(mask.squeeze(0))
        image_sequence.append(imagearray)
        label_sequence.append(label)
        filename_sequence.append(filename)
        #imagelist_sequence.append(imagelist)
    node_sets_sequence = torch.nn.utils.rnn.pad_sequence(node_sets_sequence, batch_first=True, padding_value=1)
    mask_sequence = torch.nn.utils.rnn.pad_sequence(mask_sequence, batch_first=True, padding_value=0)
    image_sequence = torch.nn.utils.rnn.pad_sequence(image_sequence, batch_first=True)
    label_sequence = torch.nn.utils.rnn.pad_sequence(label_sequence, batch_first=True)
    return node_sets_sequence, mask_sequence, image_sequence, label_sequence, filename_sequence#, imagelist_sequence

def combine_text(dictionary, shuffle=False):
    authors = dictionary.keys()
    for author in authors:
        texts = dictionary[author]['text']
        random.shuffle(texts)
        if shuffle is True:
            random.shuffle(texts)
        texts = ' '.join(texts)
        dictionary[author]['text'] = texts
    return dictionary


def _get_from_loader(filepath, filetype):
    """Return ndarray

    In order to make the fds to be opened only at the first referring,
    the loader are stored in self._loaders

    >>> ndarray = loader.get_from_loader(
    ...     'some/path.h5:F01_050C0101_PED_REAL', filetype='hdf5')

    :param: str filepath:
    :param: str filetype:
    :return:
    :rtype: np.ndarray
    """
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
def get_split(dictionary, max_len, overlap, onedict=False):
    outdict = {}
    window_shift = max_len - overlap
    authors = dictionary.keys()
    for author in authors:
        texts = dictionary[author]['text']
        l_total = []
        if len(texts.split()) // window_shift > 0:
            n = len(texts.split()) // window_shift
        else:
            n = 1
        for w in range(n):
            if w == 0:
                l_parcial = texts.split()[:max_len]
                l_total.append(" ".join(l_parcial))
            else:
                l_parcial = texts.split()[w * window_shift:w * window_shift + max_len]
                l_total.append(" ".join(l_parcial))
        if onedict is False:
            for i in range(len(l_total)):
                textid = author + '_' + str(i)
                outdict.update({textid: {}})
                outdict[textid].update({'text': l_total[i]})
                outdict[textid].update({'label': dictionary[author]['label']})
                outdict[textid].update({'image': dictionary[author]['image']})
        else:
            outdict.update({author: {}})
            for i in range(len(l_total)):
                outdict[author].update({i: {}})
                outdict[author][i].update({'text': l_total[i]})
                outdict[author][i].update({'label': dictionary[author]['label']})
                outdict[author][i].update({'image': dictionary[author]['image']})

    return outdict

def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token
def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


class multidatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, test_file, tokenizer, device, max_len):
        self.train_file = train_file
        self.test_file = test_file
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len
        self.train_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        [self.train_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.train_file[id]['text']), return_tensors='pt', truncation=True, max_length=self.max_len)})
            for id in list(self.train_file.keys())]
        [self.test_file[id].update(
            {'encode': self.tokenizer(normalizeTweet(self.test_file[id]['text']), return_tensors='pt', truncation=True, max_length=self.max_len)})
            for id in list(self.test_file.keys())]
        train_dataset = multidatasetloader(self.train_file)
        test_dataset = multidatasetloader(self.test_file)
        return train_dataset, test_dataset

class multidatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict):
        super(multidatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224')

    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        id = self.datadict[self.datakeys[index]]['encode'].data['input_ids']
        mask = self.datadict[self.datakeys[index]]['encode'].data['attention_mask']

        filename = self.datakeys[index]

        imagelist = self.datadict[self.datakeys[index]]['image']
        imgdata = []
        for i in imagelist:
            #img = Image.open(i.replace('./', './../')).convert('RGB')
            img = Image.open(i).convert('RGB')
            img = img.resize((256, 256))
            inputs = self.feature_extractor(images=img, return_tensors="pt")
            # image = np.array((img))
            imgdata.append(inputs.data['pixel_values'])
        imgdata = random.sample(imgdata, 9)
        random.shuffle(imgdata)
        imgdata = torch.cat(imgdata, dim=0)



        label = self.datadict[self.datakeys[index]]['label']
        label = torch.LongTensor([label])
        #imagelist = self.datadict[self.datakeys[index]]['image']


        return id, mask, imgdata, label, filename#, imagelist  # twtfsingdata.squeeze(0), filename


class Multifeaturedatasetclass:  # This class is used to achieve parameters sharing among datasets
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_dataset, self.test_dataset = self.prepare_dataset()

    def prepare_dataset(self):  # will also build self.edge_stat and self.public_edge_mask
        # preparing self.train_dataset
        train_dataset = Multifeatdatasetloader(self.train_file)
        test_dataset = Multifeatdatasetloader(self.test_file)
        return train_dataset, test_dataset

class Multifeatdatasetloader(Dataset):  # For instantiating train, validation and test dataset
    def __init__(self, datadict):
        super(Multifeatdatasetloader).__init__()
        self.datakeys = self._get_keys(datadict)
        self.datadict = datadict
    def _get_keys(self, datadict):
        """Return absolute paths to all utterances, transcriptions and phoneme labels in the required subset."""
        keys = list(datadict.keys())
        return keys

    def __len__(self):
        return len(self.datakeys)

    def __getitem__(self, index):
        filename = self.datakeys[index]

        textfeatdir = self.datadict[filename]['textfeat']
        textprobdir = self.datadict[filename]['textprob']
        imagefeatdir = self.datadict[filename]['imagefeat']
        imageprobdir = self.datadict[filename]['imageprob']
        labeldir = self.datadict[filename]['label']

        textfeat = _get_from_loader(
            filepath=textfeatdir,
            filetype='mat')
        textfeat = torch.FloatTensor(textfeat)

        textprob = _get_from_loader(
            filepath=textprobdir,
            filetype='mat')
        textprob = torch.FloatTensor(textprob)

        imagefeat = _get_from_loader(
            filepath=imagefeatdir,
            filetype='mat')
        imagefeat = torch.FloatTensor(imagefeat)

        imageprob = _get_from_loader(
            filepath=imageprobdir,
            filetype='mat')
        imageprob = torch.FloatTensor(imageprob)

        label = _get_from_loader(
            filepath=labeldir,
            filetype='mat')
        label = torch.LongTensor(label)


        return textfeat, textprob, imagefeat, imageprob, label, filename#, imagelist  # twtfsingdata.squeeze(0), filename