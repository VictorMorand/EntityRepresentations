import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


def align_tags_with_tokens(tokens, tags):
    """ align CoNLL tags with tokens from tokenizer, WITHOUT <bos> token
    Args:
        tokens: (seq) str tokens from tokenizer WITHOUT <bos> token
        tags: (#words) word-level CoNLL tags 
    Return
        token_tags (seq) token-level ConLL tags
    """
    token_tags = [tags[0]]
    word_index = 0
    
    for token in tokens[1:]:
        if token.startswith(' ') :#or token.strip() in special_tokens:
            #new word
            word_index += 1
            if word_index >= len(tags):
                print("end")
                word_index = len(tags) - 1
            token_tags.append(tags[word_index])
        else:
            #it is a subword, take previous ner tag
            tag = token_tags[-1]
            tag += tag % 2
            #for CoNLL, following tags are +1, so add one to keep only first token even.
            token_tags.append(tag)
    return token_tags 

def get_attn_pattern(item):
    """compute target attention pattern from data item"""
    token_tags = item["token_tags"] #there is an additional bos token not in str tokens    
    seq = len(token_tags)
    pattern = torch.zeros(seq, seq) # pattern[i][j] is attn i -> j
    head = 0
    for i, tag in enumerate(token_tags):
        if tag == 0: #nothing
            head = 0
            pattern[i][0] = 1
        if tag%2 == 1: #if tag is even, this token is the first of an entity
            pattern[i][i] = 1
            head = i # TODO should the first token of a multi token mention attend to himself anyways ? 
        else: #tag is odd, we are continuing an entity
            pattern[i][head] = 1
    return pattern

class CoNLLDataset():
    def __init__(self, CoNLLdata, max_ent_length=40, max_length=512):
        """CoNLL dataset class
        Args:
            data: list of CoNLL dataset items
        """
        self.data = []
        self.max_ent_length = max_ent_length
        self.max_length = max_length

        for item in CoNLLdata:
            self.data += self.extract_from_item(item)

        #reindex
        for i, item in enumerate(self.data):
            item['id'] = i
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pattern = get_attn_pattern(self.data[idx]).unsqueeze(0)
        # add pattern to the item
        return self.data[idx] | {"pattern": pattern}

        
    def extract_from_item(self, item):
        ###  CoNLL 2003 tags
        # beg_tags = [1, 3, 5, 7]
        # i_tags   = [2, 4, 6, 8]

        words = item["tokens"]    
        item["text"] = " ".join(words)
        
        #filters
        if ((len(item["text"]) > self.max_length) or  # max context length
            all(np.unique(item["ner_tags"]) == [0]) # if no entities in context
            ):
            return []            
        else:
            return [item]
        
    def tokenize_and_augment(self, model: HookedTransformer, verbose:bool = True):
        """ Tokenize the texts and compute token-level NER tags from a CoNLL item
        Args:
            model: model that will be used, will use its tokenizer
        """
        for item in tqdm(self.data, disable = not verbose ):
            # Tokenize the sentence
            # item["tokens"] = model.to_tokens(item["text"], prepend_bos=True, move_to_device=False)
            item["str_tokens"] = model.to_str_tokens(item["text"], prepend_bos=True) # include bos token
            
            # Align the tags
            item["token_tags"] = [0] + align_tags_with_tokens(
                                            item["str_tokens"][1:], # the function expects tokens without bos token
                                            item["ner_tags"])
    
    def get_loader(self, batch_size:int = 16):
        batched_dataset = BatchedDataset(self, batch_size=batch_size)
        return DataLoader(batched_dataset, batch_size=1, collate_fn=lambda x: x[0])

# Batching is delicate here because of the broad distribution of context lengths
# it is therefore managed manually here and we use a Dataloader with batch size 1 on top of it
class BatchedDataset():
    def __init__(self, dataset:CoNLLDataset, batch_size:int = 16):
        """Wrapper for a dataset to create batches with proper dynamic tokenization"""
        self.dataset = dataset
        self.batch_size = batch_size
        #sort by length
        self.dataset = sorted(self.dataset, key=lambda x: len(x["str_tokens"]))
        #create batches
        self.batches = [self.dataset[i:i+self.batch_size] for i in range(0, len(self.dataset), self.batch_size)]
        #randomize batches
        np.random.shuffle(self.batches)

    def __len__(self):
        return len(self.batches)
    
    def __getitem__(self, idx):
        items = self.batches[idx]
        # Collate the batch
        batch = {key : [item[key] for item in items] for key in items[0].keys()}
        return batch

# Main Function that is exposed 
def load_datasets(dataset_name:str = "conll2003"):
    """Train, Val and Test Dataset loaders for this Experiment"""
    if dataset_name.lower() == "conll2003":
        ds = load_dataset("eriktks/conll2003", trust_remote_code=True)
        max_ent_length = 60
        max_length = 400
        train_dataset = CoNLLDataset(ds["train"], max_ent_length=max_ent_length,max_length=max_length)
        val_dataset = CoNLLDataset(ds["validation"], max_ent_length=max_ent_length,max_length=max_length)
        test_dataset = CoNLLDataset(ds["test"], max_ent_length=max_ent_length,max_length=max_length)

    else:
        raise NotImplementedError("Unknown dataset, can be: 'CoNLL2003' ")
    
    return train_dataset, val_dataset, test_dataset
