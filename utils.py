from transformer_lens import HookedTransformer, utils
from datasets import load_dataset
import torch, gc
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from importlib import reload
from typing import List, Dict, Any



# place a hook to replace representation of "_" 
def get_replace_with_rep_hook(reps, ind): 
    
    def replace_with_rep(tensor, reps, ind):
        """replace first token with representation
        Args:
            tensor: the act cache to modify
            ind, the token index that we want to overwrite 

        """
        # print(hook.name)
        # print("got tensor of shape", tensor.shape)
        #replace the token a ind by the given representations
        tensor[:,ind,:] = reps
        return tensor
    
    return lambda tensor, hook: replace_with_rep(tensor, reps, ind)


def get_representation(model: HookedTransformer, tokens, token_inds, layer:int, verbose:bool=False ):
    """extract model representation of token [token_inds] at layer [layer]
    Args:
        model: HookedTransformer form TransformerLens to extract representations from
        tokens: (batch, N) tokenized texts to process
        token_inds: (batch) index of tokens where to extract representation
        layer: layer at which to retreive the representations
    """ 

    dim = model.QK.shape[-1]
    b_size = tokens.shape[0]
    assert len(token_inds) == b_size
    buffer = torch.zeros(b_size, dim)       #create buffer where to store representations
    hook_name = utils.get_act_name('resid_post', layer=layer)   # get hook name for the layer output 
    if verbose: print(f"extract representation of tokens {token_inds} at hook {hook_name}")
    
    def save_activation(tensor, buffer, inds):
        """Save wanted activation in buffer
        Args:
            tensor: the act cache to modify
            buffer (Tensor): the buffer where to store the wanted activations
            inds (List): the token index that we want to overwrite 
        """
        #just store the wanted activations in the buffer
        buffer[:] = torch.vstack([tensor[i,inds[i],:] for i in range(len(inds))])
        return tensor
    
    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            return_type=None,
            fwd_hooks=[(hook_name, lambda tensor, hook: save_activation(tensor, buffer, token_inds) )],
        )
    return buffer

def generate_from_repr( model, 
                        repr, 
                        taskVector = None, 
                        retr_prompt:str ="_ named",
                        max_tokens = 10,
                        do_sample = False,
                        prepend_bos=True,
                        ):
        """
        Args:
                repr: The representation vectors
                Model: the GPT model to use
                taskVector: the task vector to use. If None, will use default prompt
                max_tokens: num of tokens to generate
                do_sample: if True, sample from logits, else take argmax
                prepend_bos: if True, prepend '<BOS>' token to the input

        """
        inp_toks = model.to_tokens(retr_prompt, prepend_bos=prepend_bos)
        rep_idx = 1 if prepend_bos else 0
        replace_hook_name = utils.get_act_name('embed')
        taskVec_idx = rep_idx + 1
        
        if taskVector is not None:
            taskVector = taskVector.view(1,-1)
            b_taskVec = taskVector.repeat(repr.shape[0],1)
        # print(b_taskVec.shape)
        # print(repr.shape)
        
        for i in range(max_tokens):

                # print(inputs, targets)
                if taskVector is not None:
                    logits = model.run_with_hooks(
                            inp_toks,
                            return_type = "logits",
                            fwd_hooks=[
                            (replace_hook_name, get_replace_with_rep_hook(repr, rep_idx)), # replace '_' by the subject Representation
                            (replace_hook_name, get_replace_with_rep_hook(b_taskVec, taskVec_idx)) # replace 'called' by TaskVec Representation
                            ])
                else:
                    logits = model.run_with_hooks(
                            inp_toks,
                            return_type = "logits",
                            fwd_hooks=[
                            (replace_hook_name, get_replace_with_rep_hook(repr, rep_idx)), # replace '_' by the subject Representation
                            ])

                final_logits =  logits[0,-1,:] #extract logits for last token 

                if do_sample:
                        new_tok = utils.sample_logits(
                        final_logits,
                        top_k=None,
                        top_p=None,
                        temperature=0,
                        freq_penalty=0,
                        tokens=inp_toks,
                        ).view(1,-1)
                else:       #greedy generation
                        new_tok = final_logits.argmax(-1).view(1,-1) 
                
                inp_toks = torch.hstack((inp_toks,new_tok))
                # stop if EOS token
                if new_tok == model.tokenizer.eos_token_id: break
        print(model.to_string(inp_toks))


############################## DATASETS ##############################

class WebNLGDataset(Dataset):
    def __init__(self, data, max_ent_length=40, max_length=512):
        """WebNLG dataset class
        Args:
            data: list of WebNLG items
            max_ent_length: filter entities whose span is bigger than this.
            max_length: maximum length of the text
        """
        self.max_length = max_length
        self.max_ent_length = max_ent_length
        self.data = []
        for it in data:
            self.data += self.extract_from_item(it)
        # finally add index
        for i, item in enumerate(self.data):
            item["id"] = i
        
        #sanity checks
        assert not None in self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
    def extract_from_item(self, item):
        """ Extracts the entity and text from a WebNLG item
        Args:
            item: a dictionary with keys 'modified_triple_sets' and 'lex'
        returns:
        a list of dictionaries with keys 'entity' and 'text'
        """
        # try :
        def clean_entity(entity):
            return entity.replace('"','').replace('_',' ').replace('  ',' ').split(' (')[0]
        # Extract entities
        rels = item["modified_triple_sets"]["mtriple_set"][0]
        entities = set()
        for rel in rels:
            ents = rel.split(' | ')
            ents.pop(1)
            for ent in ents:
                entities.update([clean_entity(ent)])
        # filter too big entities
        entities = [ent for ent in entities if 
                    (len(ent) <= self.max_ent_length and 
                     any(c.isalpha() for c in ent))]
        res = []
        texts = item["lex"]["text"]
        if not len(texts) or not len(entities): return []

        for i, ent in enumerate(entities):
            res.append({
                "entity" : ent,
                "text" :  texts[i%len(texts)] + ' ' + ent,
            })

        return res
        
    def augment_with_repr(self, model, layer, batch_size):
        """
        augment the dataset with extracted representations in given model at given hook
        Args:
            model: HookedTransformer model to extract representations from
            layer: layer at which to retreive the representations
            batch_size: batch size for inference
        """
        new_data = {}
        prepend_bos = True
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader):
                texts = batch["text"]
                ids = batch["id"].detach().cpu().numpy()
                #batched GPU inference
                str_tokens = model.to_str_tokens(texts, prepend_bos=prepend_bos)
                subj_inds = [len(toks) - 1 for toks in str_tokens]
                tokens = model.to_tokens(texts, prepend_bos=prepend_bos, padding_side='right')
                
                reps = get_representation(model, tokens=tokens, token_inds=subj_inds, layer=layer)

                #Augment dataset with representation and retrieval prompt
                for i in range(len(ids)):
                    new_data[ids[i]] = {"representation" : reps[i,:],}
        
        #GPU cleanup
        del dataloader
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        self.data =  [item | new_data[item["id"]] for item in self.data]
        