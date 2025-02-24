""" author: Victor Morand
This script gathers all the common functions that I use throughout my various projects, 
"""
from transformer_lens import HookedTransformer, utils
from datasets import load_dataset
import torch, os, gc, re
os.environ['HF_EVALUATE_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from tqdm import tqdm
import numpy as np


########################################################################
############################## Parameters ##############################

checkpoints_folder = ""
data_folder = ""

########################################################################

############################## FUNCTIONS ##############################

def load_model(model_name, dtype = torch.float32, token=None):
    """Use fixed parameters to load models from Tlens"""
    try :
        model = HookedTransformer.from_pretrained(
                                model_name, 
                                trust_remote_code=True, 
                                low_cpu_mem_usage = True, 
                                fold_ln=False,
                                fold_value_biases=False,
                                device_map='auto',
                                dtype=dtype,
                                local_files_only=True,
                                )
    except Exception as e:
        print(f"Error loading model {model_name}, trying to download it ...")
        #pass hugginface in online mode
        os.environ['HF_HUB_OFFLINE'] = '0'
        if token : os.environ['HF_TOKEN'] = token #pass token to huggingface 

        #pass token to huggingface
        model = HookedTransformer.from_pretrained(
                                model_name, 
                                trust_remote_code=True, 
                                low_cpu_mem_usage = True, 
                                fold_ln=False,
                                fold_value_biases=False,
                                device_map='auto',
                                dtype=dtype,
                                huggingface_token=token,
                                )
    return model

# we use the following function to truncate the computation of the transformer to a given layer
def compute_to_layer(model, layer, tokens, verbose:bool = False):
    """Compute the transformer up to a given layer, return the hidden states
    Args:
        model: HookedTransformer from TransformerLens
        layer: layer to compute the transformer up to
        tokens (tensor 'batch, seq'): tokens to compute the transformer on
    Returns:
        hidden_states: tensor of shape (batch, len, dim) with the hidden states at the given layer
    """

    dim = model.QK.shape[-1]
    b_size = tokens.shape[0]
    seq = tokens.shape[-1]
    dtype = model.W_U.dtype
    buffer = torch.zeros(b_size, seq, dim, dtype = dtype)       #create buffer where to store representations
    
    if layer < 0:
        #baseline, extract embedding only
        hook_name = utils.get_act_name('embed')
    else:        
        hook_name = utils.get_act_name('resid_post', layer=layer)   # get hook name for the layer output 
        
    if verbose: print(f"compute hidden states at hook {hook_name}")
    
    def save_activation(tensor, buffer):
        """Save wanted activation in buffer
        Args:
            tensor: the act cache to modify
            buffer (Tensor): the buffer where to store the wanted activations
            inds (List): the token index that we want to overwrite 
        """
        #just store the wanted activations in the buffer
        buffer[:] = tensor
        # stop the forward pass
        raise ValueError("Stopping the forward pass")
    
    with torch.no_grad():
        #run the model with the hook
        try: 
            model.run_with_hooks(
                tokens,
                return_type=None,
                fwd_hooks = [(hook_name, lambda tensor, hook: save_activation(tensor, buffer) )],
            )
        except ValueError as e:
            if verbose: print(f"Caught exception {e}")
    return buffer
    #run the model with the hook

# place a hook to replace representation of "_" 
def get_replace_with_rep_hook(reps, inds): 
    
    def replace_with_rep(tensor, reps, inds):
        """replace first token with representation
        Args:
            tensor: (batch, sent_len, H) the act cache to modify
            reps tensor(batch, H) the representations to overwrite at hook
            inds tensor(batch), the indexes that we want to overwrite in each sentence of batch 
        """
        #replace the token a ind by the given representations
        # tensor[torch.arange(tensor.shape[0]),inds,:] = reps # slower
        inds_expanded = inds.unsqueeze(1).expand(-1, tensor.shape[-1]).unsqueeze(1).to(tensor.device)
        tensor.scatter_(1, inds_expanded, reps.unsqueeze(1))
        return tensor
    
    return lambda tensor, hook: replace_with_rep(tensor, reps, inds)

def get_representation(model: HookedTransformer, tokens, token_inds, layer:int, hooks:list = [], verbose:bool=False ):
    """Extract model representation of token `token_inds` at layer `layer` from batch
    Args:
        model: HookedTransformer form TransformerLens to extract representations from
        tokens: tensor(batch, N) tokenized texts to process
        token_inds: (batch) index of tokens where to extract representation
        hooks: (Optionnal) List [(hook_name, hook_fx)] hooks that will also be placed on the model during computation
        layer: layer at which to retreive the representations
    """ 

    dim = model.QK.shape[-1]
    b_size = tokens.shape[0]
    dtype = model.W_U.dtype
    assert len(token_inds) == b_size
    buffer = torch.zeros(b_size, dim, dtype = dtype)       #create buffer where to store representations
    
    if layer < 0:
        #baseline, extract embedding only
        hook_name = utils.get_act_name('embed')
    else:        
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
            fwd_hooks= hooks + [(hook_name, lambda tensor, hook: save_activation(tensor, buffer, token_inds) )],
        )
    return buffer

def get_avg_representation(model: HookedTransformer, prompts, entities, layer:int, verbose:bool=False ):
    """extract average model representation of entities at given layer 
    representations will be extracted at entity tokens after reading f"{prompt}{entity}" 
    Args:
        model: HookedTransformer form TransformerLens to extract representations from
        prompts: list of prompts to use for extraction
        entities: list of entities to extract representations from
        layer: layer at which to retreive the representations
    """ 

    dim = model.QK.shape[-1]
    b_size = len(prompts)
    dtype = model.W_U.dtype
    assert len(entities) == b_size
    
    if layer < 0:
        #baseline, extract embedding only
        hook_name = utils.get_act_name('embed')
    else:        
        hook_name = utils.get_act_name('resid_post', layer=layer)   # get hook name for the layer output 
    
    if verbose: print(f"extract average representation of {entities} at hook {hook_name}")
    
    #tokenize prompts
    prompt_tokens = model.to_tokens(prompts, padding_side='left')
    entity_tokens = model.to_tokens(entities, padding_side='right', prepend_bos=False)

    #concatenate prompts and entities
    tokens = torch.hstack([prompt_tokens, entity_tokens])
    
    #check that decoder string is correct
    # print(model.tokenizer.decode(tokens.view(-1).cpu().numpy())) # OK !

    ind_min = prompt_tokens.shape[-1]
    ind_max = prompt_tokens.shape[-1] + entity_tokens.shape[-1] - 1

    #mask with zeros where there is an eos
    mask = torch.ones_like(entity_tokens)
    mask[entity_tokens == model.tokenizer.eos_token_id] = 0

    buffer = torch.zeros(b_size, entity_tokens.shape[-1], dim, dtype = dtype)       #create buffer where to store representations
    buffer.requires_grad = False
    buffer = buffer.cuda()

    def save_activations(tensor, buffer, min_ind, max_ind):
        """Save wanted activations in buffer
        Args:
            tensor: the act cache to modify
            buffer (Tensor): the buffer where to store the wanted activations
            min_ind: the first token index that we want to save
            max_ind: the last token index that we want to save
        """
        #just store the wanted activations in the buffer
        buffer[:] = tensor[:,min_ind:max_ind+1,:]
        return tensor
    
    with torch.no_grad():
        model.run_with_hooks(
            tokens,
            return_type=None,
            fwd_hooks= [(hook_name, lambda tensor, hook: save_activations(tensor, buffer, ind_min, ind_max) )],
        )
    
    buffer = buffer * mask.unsqueeze(-1)        #apply mask to zero out eos tokens
    buffer = buffer.sum(dim=1) / mask.sum(dim=1).unsqueeze(-1) #sum over tokens and divide by number of non-zero tokens
    return buffer       #return obtained average representation

def project_on_vocab(model, rep, k=10):
    """ project a representation on the vocabulary of the model
    Args:
        model: the model to use
        rep: the representation to project
        k: number of top tokens to return
    """
    logits = torch.tensor(rep).float().cuda() @ model.W_U
    topk_inds = torch.topk(logits, k).indices
    topk_tokens = model.tokenizer.convert_ids_to_tokens(topk_inds.cpu().numpy())
    return topk_tokens

def generate_from_repr( model, 
                        repr, 
                        taskVector = None, 
                        retr_prompt:str ="_ named",
                        max_tokens = 10,
                        do_sample = False,
                        prepend_bos=True,
                        return_type:str = "str"
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
            b_taskVec = taskVector.repeat(repr.shape[0],1).cuda()
        
        repr = repr.cuda()
        # print(b_taskVec.shape)
        # print(repr.shape)
        
        for i in range(max_tokens):
                # print(inputs, targets)
                if taskVector is not None:
                    logits = model.run_with_hooks(
                            inp_toks,
                            return_type = "logits",
                            fwd_hooks=[
                            (replace_hook_name, get_replace_with_rep_hook(repr, torch.tensor([rep_idx]))), # replace '_' by the subject Representation
                            (replace_hook_name, get_replace_with_rep_hook(b_taskVec, torch.tensor([taskVec_idx]))) # replace 'called' by TaskVec Representation
                            ])
                else:
                    logits = model.run_with_hooks(
                            inp_toks,
                            return_type = "logits",
                            fwd_hooks=[
                            (replace_hook_name, get_replace_with_rep_hook(repr, torch.tensor([rep_idx]))), # replace '_' by the subject Representation
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
        if return_type == "tokens": 
            return model.to_str_tokens(inp_toks)
        else: 
            return model.tokenizer.decode(inp_toks.view(-1).tolist()[1:])

def sample_random_entities(model, dataset, n=3):
    """BASELINE: Creates a new dataset with random sampled spans as entities. 
    for each row, Sample n successive tokens in the given context
    Args:
        dataset : dataset object
        n : number of tokens to extract
    """
    new_dataset = []
    for row in tqdm(dataset):
        
        tokens = model.to_tokens(row["text"]).view(-1)
        
        #if not enough tokens, remove the row
        if len(tokens) <= n+1: continue

        end = np.random.randint(n+1, len(tokens)) # don't sample eos token
        toks = tokens[end-n:end]
        entity = model.tokenizer.decode(toks).strip()
        
        new_dataset.append( { 
            "text": row["text"],
            "entity": entity,
            "entity_tokens": toks.cpu(), # directly store tokens instead of string
            })
    
    return EntityReprDataset(new_dataset)


############################## DATASETS ##############################
class EntityReprDataset(Dataset):
    def __init__(self, data,  max_ent_length=40, max_length=512):
        """Entity Representation dataset class
        Args:
            data: list of dicts containing at least "text" and "entity" keys. 
            max_ent_length: filter entities whose span is bigger than this.
            max_length: maximum length of the text (in chars)
        """

        #sanity checks
        assert not None in data
        for item in data:
             assert "text" in item.keys()
             assert "entity" in item.keys()

        self.max_length = max_length
        self.max_ent_length = max_ent_length
        self.data = data

        # filter data
        def filtered(item):
            return (len(item["entity"]) > self.max_ent_length or 
                    len(item["text"]) > self.max_length or 
                    not any(c.isalpha() for c in item["entity"]))
        
        # filter data
        self.data = [item for item in self.data if not filtered(item)]

        # index data
        for i, item in enumerate(self.data):
            item["id"] = i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        
    def augment_with_repr(self, model, layer, batch_size, method="after_context", verbose=True):
        """
        augment the dataset with extracted representations in given model at given hook
        Args:
            model: HookedTransformer model to extract representations from
            layer: layer at which to retreive the representations
            batch_size: batch size for inference
            method: method to use for retrieve representation, can be :
            - 'raw_entity',  :  
            - 'in_context'   : 
            - 'after_context': 
        """
        new_data = {}
        prepend_bos = True
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, disable = not verbose):
                texts = batch["text"]
                entities = batch["entity"]
                ids = batch["id"].detach().cpu().numpy()

                if method == "raw_entity":
                    prompts = [ent for ent in entities]
                elif method == "in_context":
                    prompts = [txt.split(ent)[0] + ent for txt, ent in zip(texts, entities)]
                elif method == "after_context":
                    prompts = [txt + ' ' + ent for txt, ent in zip(texts, entities)]
                else:
                    raise NotImplementedError(f"extraction method '{method}' not implemented!\n Can be 'raw_entity', 'in_context' or 'after_context'")

                #batched GPU inference
                str_tokens = model.to_str_tokens(prompts, prepend_bos=prepend_bos)
                subj_inds = [len(toks) - 1 for toks in str_tokens] #always take the last token -> no need to compute the rest anyway.
                tokens = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side='right')
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

    def augment_with_avg_repr(self, model, layer, batch_size, method="in_context", verbose:bool = True):
        """
        (/!\ Baseline) Augment the dataset with average representations of entities in given model at given layer
        Args:
            model: HookedTransformer model to extract representations from
            layer: layer at which to retreive the AVERAGE representations
            batch_size: batch size for inference
            method: method to use for retrieve representation, can be :
            - 'raw_entity',  :  
            - 'in_context'   : 
            - 'after_context': 
        """
        new_data = {}
        dataloader = DataLoader(self.data, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for batch in tqdm(dataloader, disable = not verbose):
                texts = batch["text"]
                entities = batch["entity"]
                ids = batch["id"].detach().cpu().numpy()

                if method == "raw_entity":
                    prompts = ["" for ent in entities]
                elif method == "in_context":
                    prompts = [txt.split(ent)[0] for txt, ent in zip(texts, entities)]
                elif method == "after_context":
                    prompts = [txt + ' ' for txt, ent in zip(texts, entities)]
                else:
                    raise NotImplementedError(f"extraction method '{method}' not implemented!\n Can be 'raw_entity', 'in_context' or 'after_context'")

                reps = get_avg_representation(model, prompts, entities, layer)

                for i in range(len(ids)):
                    new_data[ids[i]] = {"representation" : reps[i,:],}
        
        #GPU cleanup
        del dataloader
        del batch
        gc.collect()
        torch.cuda.empty_cache()
        self.data =  [item | new_data[item["id"]] for item in self.data]


## Implementation for WebNLG
class WebNLGDataset(EntityReprDataset):
    def __init__(self, WebNLGdata, max_ent_length=40, max_length=512):
        """WebNLG dataset class
        Args:
            data: list of WebNLG items
            max_ent_length: filter entities whose span is bigger than this.
            max_length: maximum length of the text
        """
        data = [] 

        for item in WebNLGdata:
            data += self.extract_from_item(item)
        
        super().__init__(
            data=data,
            max_ent_length=max_ent_length,
            max_length=max_length)
        
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
        
        res = []
        texts = item["lex"]["text"]
        if not len(texts) or not len(entities): return []

        for i, ent in enumerate(entities):
            res.append({
                "entity" : ent,
                "text" :  texts[i%len(texts)],
            })

        return res

## Implementation for TACRED   
class TacredDataset(EntityReprDataset):
    def __init__(self, WebNLGdata, max_ent_length=40, max_length=400, remove_pronouns:bool =True):
        """WebNLG dataset class
        Args:
            data: list of WebNLG items
            max_ent_length: filter entities whose span is bigger than this.
            max_length: maximum length of the text
        """
        data = [] 

        for item in WebNLGdata:
            data += self.extract_from_item(item, remove_pronouns=remove_pronouns)
        
        super().__init__(
            data=data,
            max_ent_length=max_ent_length,
            max_length=max_length)
                    
    def extract_from_item(self, tacred_item: dict, remove_pronouns: bool):
        """Extracts the subject and label from a TACRED item.
        Args:
            item: a dictionary with keys 'modified_triple_sets' and 'lex'
        returns:
        a list of dictionaries with keys 'entity' and 'text'
        """
        def clean(s):
            s = s.replace("[subject_start]", "").replace("[subject_end]", "").replace("[object_start]", "").replace("[object_end]", "").strip()
            return re.sub(r'\s+', ' ', s) #max one space
        
        #extract subject 
        subj = (" " + tacred_item["text"]).split("[subject_start]")[1].split("[subject_end]")[0].strip()
        subj = clean(subj)
        #extract label
        obj = (" " + tacred_item["text"]).split("[object_start]")[1].split("[object_end]")[0].strip()
        obj = clean(obj)

        # remove tags from text 
        # remove 2nd sentence
        text = clean(tacred_item["text"]).replace(" , ", ", ")
        text = text.split(" . ")[0].strip() + "."
        res = []
        for ent in [subj, obj]:
            if remove_pronouns and ent.lower() in ["he", "she", "his", "her", "him", "they", "their", "them"]:
                continue
            res.append({
                "text": text,
                "entity": ent,
            })
        return res
    
## Implementation for CoNLL
class CoNLLDataset(EntityReprDataset):
    def __init__(self, CoNLLdata, max_ent_length=40, max_length=512):
        """CoNLL dataset class
        Args:
            data: list of CoNLL dataset items
        """
        data = [] 
        for item in CoNLLdata:
            data += self.extract_from_item(item)
        
        super().__init__(
            data=data,
            max_ent_length=max_ent_length,
            max_length=max_length)
        
    def extract_from_item(self, item):
        """ Extracts the entity and text from a CoNLL item
        Args:
            item: a dictionary with keys 'modified_triple_sets' and 'lex'
        Returns:
            dataset_items: a list of dictionaries with keys 'entity' and 'text'
        """
        
        def clean(text):    
            # remove space before punctuation
            text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" :", ":").replace(" ;", ";")
            text = text.replace(" )", ")").replace("( ", "(").replace(" '", "'").replace(" %", "%")
            return text
        
        if all(np.unique(item["ner_tags"]) == [0]):
            return []
        
        entities = []
        res = []
        text= " ".join(item["tokens"])

        # CoNLL 2003 tags
        beg_tags = [1, 3, 5, 7]
        i_tags   = [2, 4, 6, 8]

        entity = None
        for tag, token in zip(item["ner_tags"], item["tokens"]):
            if tag in beg_tags:
                if entity:
                    entities.append(entity)
                entity = token
            elif tag in i_tags:
                entity += " " + token
            else:
                if entity:
                    entities.append(entity)
                entity = None

        for ent in entities:
            res.append({
                "entity" : clean( ent),
                "text" :   clean(text),
            })

        return res
    

def load_datasets(dataset_name, max_ent_length=20):
    """
    Args: 
        dataset_name (str): the name of the dataset to load.
        max_ent_length (int): the maximum length for entities to be included in the dataset
    Returns: 
        train_dataset, test_dataset, val_dataset"""
    
    if dataset_name.lower() == "conll2003":
        ds = load_dataset("eriktks/conll2003", trust_remote_code=True)
        max_ent_length = 60
        max_length = 300
        train_dataset = CoNLLDataset(ds["train"], max_ent_length=max_ent_length,max_length=max_length)
        val_dataset = CoNLLDataset(ds["validation"], max_ent_length=max_ent_length,max_length=max_length)
        test_dataset = CoNLLDataset(ds["test"], max_ent_length=max_ent_length,max_length=max_length)

    elif dataset_name.lower() == "webnlg":
        dataset = load_dataset("web_nlg", "release_v3.0_en", trust_remote_code=True)

        #optionnal, filter categories from datset
        # cat = ['Food'] #WebNLG Categories to remove because too specific
        cat = None

        if cat :
            dataset["train"] = [item for item in dataset["train"] if item["category"] not in cat]
            dataset["dev"] = [item for item in dataset["dev"] if item["category"] not in cat]
            dataset["test"] = [item for item in dataset["test"] if item["category"] not in cat]

        # Create dataset instances
        train_dataset = WebNLGDataset(dataset['train'], max_ent_length=max_ent_length)
        val_dataset = WebNLGDataset(dataset['dev'], max_ent_length=max_ent_length)
        test_dataset = WebNLGDataset(dataset['test'], max_ent_length=max_ent_length)

    elif dataset_name.lower() == "tacred":
        dataset = load_dataset("AmirLayegh/tacred_text_label")
        train_dataset = TacredDataset(dataset["train"], max_ent_length=max_ent_length, max_length=200)
        val_dataset = TacredDataset(dataset["test"], max_ent_length=max_ent_length, max_length=200)
        test_dataset = TacredDataset(dataset["validation"], max_ent_length=max_ent_length, max_length=200)

    else: 
        # unknown Dataset 
        raise NotImplementedError("Unknown dataset, can be: 'webnlg' 'tacred' or 'CoNLL2003' ")

    return train_dataset, test_dataset, val_dataset