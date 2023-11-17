import torch
from transformers import AutoModelForSequenceClassification, ViTForImageClassification

class Textmodel(torch.nn.Module):
    def __init__(self, cachedir):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.BERT = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-large", cache_dir=cachedir,
                                                                       output_hidden_states=True,
                                                                       ignore_mismatched_sizes=True)

    def forward(self, nodes, mask):
        x = self.BERT(nodes, mask)
        feats = x.hidden_states[-1][:, 0, :]
        return x.logits, feats

class Imagesingle(torch.nn.Module):
    def __init__(self, cachedir):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.ViT = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
        self.classifier = torch.nn.Sequential(torch.nn.Linear(1000, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, 2, bias=True))

    def forward(self, image):
        BS, N, C, H, W = image.size()
        image = image.view(-1, C, H, W)
        feats = self.ViT(image).logits
        feats = feats.view(BS, N, -1)
        feats = torch.mean(feats, dim=1)
        x = self.classifier(feats)
        return x, feats

class Imagemulti(torch.nn.Module):
    def __init__(self, cachedir):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.ViT = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
        self.classifier = torch.nn.Sequential(torch.nn.Linear(1000, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, 2, bias=True))

    def forward(self, image):
        feats = self.ViT(image).logits
        x = self.classifier(feats)
        return x, feats