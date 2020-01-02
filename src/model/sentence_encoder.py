import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Encoder(nn.Module):
    def __init__(self,
                 embeddings_model = 'bert-base-uncased',
                 model_type = 'bert'
                 ):
        super(Encoder, self).__init__()

        self.model_type = model_type

        if model_type == "bert":
            model = BertModel.from_pretrained(embeddings_model)
            self.embeddings_encoder = model

    def forward(self, batch, cls_pos):
        if self.model_type == "bert":
            #Word embeddings
            embeddings = self.embeddings_encoder(batch)[0]

            #Sentence
            sentence_embeddings = embeddings[cls_pos]
        return sentence_embeddings
