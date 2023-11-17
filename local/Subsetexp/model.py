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

class feature_extraction(torch.nn.Module):
    def __init__(self, cachedir):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.textmodel = Textmodel(cachedir)
        self.imagemodel = Imagesingle(cachedir)

    def forward(self, nodes, mask, image):
        text_logits, text_feat = self.textmodel(nodes, mask)
        image_logits, image_feat = self.imagemodel(image)


        return text_logits, text_feat, image_logits, image_feat

class DSW(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.classifier = torch.nn.Sequential(torch.nn.Linear(2024, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, 2, bias=True),
                                              torch.nn.Sigmoid())

    def forward(self, text_logits, image_logits, text_feat, image_feat):
        feats = torch.cat([text_feat, image_feat], dim=-1)
        weights = self.classifier(feats)
        x = weights[:, 0].unsqueeze(1) * text_logits + weights[:, 1].unsqueeze(1) * image_logits
        return x

class RFmodel(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.classifier = torch.nn.Sequential(torch.nn.Linear(2024, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, 2, bias=True))

    def forward(self, textfeats, imagefeats):
        feats = torch.cat([textfeats, imagefeats], dim=-1)
        x = self.classifier(feats)
        return x

class MultiPooling(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # loaded_model \
        self.textmodel = torch.nn.Sequential(torch.nn.Linear(1024, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, 2, bias=True))
        self.imagemodel = torch.nn.Sequential(torch.nn.Linear(1000, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, 2, bias=True))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(2024, 1000, bias=True),
                                              torch.nn.Dropout(p=0.1, inplace=True),
                                              torch.nn.Linear(1000, 2, bias=True))

    def forward(self, text_feat, image_feat):
        G_mateix = torch.transpose(text_feat.unsqueeze(1), 1, 2) * image_feat.unsqueeze(1)

        text_pool_feat, _ = torch.max(G_mateix, dim=2)
        image_pool_feat, _ = torch.max(G_mateix, dim=1)

        pool_feat = torch.cat([text_pool_feat, image_pool_feat], dim=-1)
        x = self.classifier(pool_feat)
        return x
