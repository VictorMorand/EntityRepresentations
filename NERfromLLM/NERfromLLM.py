import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformer_lens as tl
from transformer_lens import HookedTransformer
from tqdm import tqdm
import numpy as np
from jaxtyping import Float

from NERfromLLM.models import *

###### NER Utils

def NER_inference(text, model, attn, layer, max_ent_length = 5, threshold = 0.5):
    """Infer NER tags from text using attention scores
    Args:
        text: text to infer NER tags from
        model: HookedTransformer form TransformerLens to extract representations from
        attn: attention model to use, should be a nn.Module with forward method
    Returns:
        ner_tags: NER tags for each token
    """
    tokens = model.to_tokens(text).cuda()
    scores = Cross_Attn(model, tokens, layer, attn, 
                    apply_softmax=False,
                    )
    ner_tags = NER_tags_from_scores(scores, max_ent_length = max_ent_length, threshold = threshold)
    return ner_tags

def NER_tags_from_scores(scores, max_ent_length = 5, threshold = None):
    """Build NER tags from attention scores
    Args:
        scores (batch, seq, seq): attention scores between all pairs of tokens
    Returns:
        ner_tags (seq): NER tags for each token
    """
    # print(scores.shape)
    seq = scores.size(-1)
    ner_tags = torch.zeros(seq, dtype=torch.int)
    
    for i in range(seq):
        matches = scores[0,i,:]
        sorted_idx = torch.argsort(matches, descending=True)
        if sorted_idx[0] == 0: #biggest attention to bos token, no entity
            ner_tags[i] = 0
        else :
            first_idx = 0
            while 1:
                # print(i, sorted_idx[first_idx], matches[sorted_idx[first_idx]])
                if threshold is not None and matches[sorted_idx[first_idx]] < threshold: # no match above threshold
                    ner_tags[i] = 0
                    break
                elif i - sorted_idx[first_idx] > max_ent_length: #entity too long, ignore match and continue
                    first_idx += 1
                else: #entity found
                    if i == sorted_idx[first_idx]: #first detected token
                        ner_tags[i] = 1
                    else: #token points to first detected token
                        ner_tags[i] = 2
                    break
    return ner_tags
    # print(scores)
    # print(scores.shape)
    # print(scores

def get_entities(tags):
    """Extract entities from NER tags, first token of entity is tagged 1, following tokens are tagged 2, lone 2 are ignored
    Args:
        tags: NER tags
    Returns:
        entities (List (tuples)): list of extracted entities
    """
    entities = []
    entity = None

    for i, tag in enumerate(tags):
        if tag % 2 == 1: #we start a new entity
            if entity is not None:
                entities.append(entity)
            entity = (i,i)
        elif tag >= 2 and entity is not None:
            entity = (entity[0], i)
        else: #tag is 0
            if entity is not None:
                entities.append(entity)
                entity = None

    if entity is not None:
        entities.append(entity)
    return entities

def count_perf( tags, target):
    """Compares tags with target and returns number of correct entities and total number of entities
    Args:
        tags: tags from NER_inference
        target: target tags
    Returns:
        true_pos: number of correct entities
        false_pos: number of incorrect entities
        total: total number of entities in target
    """
    inferred = get_entities(tags)
    targets = get_entities(target)

    total = len(targets)
    true_pos = 0
    false_pos = 0

    for entity in inferred:
        if entity in targets:
            true_pos += 1
        else:
            false_pos += 1

    return true_pos, false_pos, total

def compute_metrics(dataloader, model, attn, layer:int, max_ent_length = 5, threshold = None):
    """Compute metrics for a dataset
    Args:
        data: dataset to compute metrics on
        model: HookedTransformer form TransformerLens to extract representations from
        attn: attention model to use, should be a nn.Module with forward method
        layer: layer at which to retreive the representations
    Returns:
        true_pos: number of correct entities
        false_pos: number of incorrect entities
        total: total number of entities in target
    """
    true_pos = 0
    false_pos = 0
    total = 0

    for batch in tqdm(dataloader):
        texts = batch["text"]

        tokens = model.to_tokens(texts, padding_side='right', move_to_device=True)
        attn_patterns = batch["pattern"]
        
        #we batch the forward pass of representations and attention scores 
        reps = utils.compute_to_layer(model, layer, tokens).cuda() # shape (batch, seq, dim)
        b_scores = attn(reps,     # we compute all attention scores even if we only use the diagonal, and the pad tokens ... 
                    apply_softmax=False,
                    )
        for j, target_pattern in enumerate(attn_patterns):
            seq = target_pattern.size(-1)
            target = batch["token_tags"][j]
            scores = b_scores[j,:seq,:seq].unsqueeze(0)
            ner_tags = NER_tags_from_scores(scores, max_ent_length = max_ent_length, threshold = threshold)
            tp, fp, tot = count_perf(ner_tags, target)
            true_pos += tp
            false_pos += fp
            total += tot
    
    #compute metrics
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / total
    f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}