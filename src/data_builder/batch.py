import itertools

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

def local_collate_fn_bert(batch, tokenizer):
    for i in range(0, len(batch)):
        article  = batch[i]['src']
        abstract = batch[i]['tgt']

        for j in range(0, len(article)):
            article[j] = tokenizer.encode(article[j])

        for j in range(0, len(abstract)):
            abstract[j] = tokenizer.encode(abstract[j])

        article = list( itertools.chain.from_iterable(article) )
        abstract = list( itertools.chain.from_iterable(abstract) )

        cls_pos_article = [i for i in range(0, len(article)) if article[i] == tokenizer.cls_token]
        cls_pos_abstract = [i for i in range(0, len(abstract)) if abstract[i] == tokenizer.cls_token]

        batch[i]['src'] = torch.LongTensor(article)
        batch[i]['tgt'] = torch.LongTensor(abstract)
        batch[i]['cls_pos_article'] = cls_pos_article
        batch[i]['cls_pos_abstract'] = cls_pos_abstract

    return batch

class Batch():
    def __init__(self,
                 dataset,
                 batchsize,
                 model_type = "bert", #bert/elmo/etc tokenizer type
                 num_workers = 1,
                 tokenizer_model = 'bert-base-uncased',
                 max_article_len = 5000
                 ):
        self.dataset = dataset
        self.batchsize = batchsize
        self.num_workers = num_workers
        self.tokenizer_model = tokenizer_model
        self.max_article_len = max_article_len

        if model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_model)
            local_collate_fn = lambda x: local_collate_fn_bert(x, self.tokenizer)

        self.dataloader = DataLoader(
                                        dataset,
                                        batch_size = batchsize,
                                        shuffle = True,
                                        num_workers = num_workers,
                                        collate_fn = local_collate_fn
                                        )

    def nextbatch(self):
        batch = next(iter(self.dataloader))
        article = torch.zeros(self.batchsize, self.max_article_len).long()

        for i in range(0, len(batch)):
            sample          = batch[i]
            n               = len(sample['src'])
            article[i, :n]  = sample['src']

        return article
