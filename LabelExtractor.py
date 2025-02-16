### Author: Victor Morand
### this experiment script

from typing import List, Optional
import logging
import torch, gc, json, os
from experimaestro import Config, Task, Param, Constant
#remove HF networks calls that takes an eternity to timeout... We are loading them offline.
os.environ['HF_EVALUATE_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
from datasets import load_dataset
import evaluate
#compute chrf with HF evaluate package -> https://huggingface.co/spaces/evaluate-metric/chrf 
chrf = evaluate.load("chrf")  
import transformer_lens as tl 
from transformer_lens import HookedTransformer
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path

############# utils #############  
import utils 

def eval_model(model, 
    TaskVec, 
    test_loader, 
    transform: nn.Module = None,
    first_token_only: bool =False,
    with_context: bool =False,
    prepend_bos: bool =True,
    metric:str = "acc"):
    """
    Evaluate the model on a given test_loader augmented with representations.
    Args:
    model : HookedTransformer : the model to use
    TaskVec : torch.Tensor : the task vector to evaluate
    test_loader : DataLoader : the test loader to use
    transform : nn.Module : optionnal transformation done to the representation before inserting it at embed hook.
    first_token_only : bool : if True, only the first token of the entity is considered
    with_context : bool : if True, the context is prepended to the entity
    prepend_bos : bool : if True, the entity is prepended with the bos token
    metric: metric to use, 'acc' or 'loss'
    """
    b_count = 0
    m_acc = 0
    replace_hook_name = tl.utils.get_act_name('embed') #pos_embed ?
    eos_tok_str = model.tokenizer.eos_token
    rep_idx = 1 if prepend_bos else 0
    taskVec_idx = rep_idx + 1

    with torch.no_grad():
        for batch in test_loader:
            b_count += 1
            entities = batch["entity"]
            texts = batch["text"]
            
            if first_token_only:
                entities = [toks[0] for toks in model.to_str_tokens(entities,prepend_bos=False)]
            else:    
                entities = [ent + eos_tok_str for ent in entities] # take whole label add eos token

            reps = batch["representation"].squeeze(1).cuda()
            b_size = reps.shape[0]
            b_taskVec = TaskVec.repeat(b_size,1).cuda()

            # Optionnal transformation of representations:
            if transform is not None:
                reps = transform(reps)

            # depending on context:
            if with_context:
                prompts = [txt + "_ >" for txt in texts]
                context_toks = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side="left") 
                entities_toks = model.to_tokens(entities, prepend_bos=False, padding_side="right")
                rep_idx = context_toks.shape[1] - 2
                # logging.info(rep_idx)
                rep_idxs = torch.tensor(b_size * [rep_idx])
                taskVec_idxs = torch.tensor(b_size * [rep_idx + 1])
                inputs = torch.cat([context_toks, entities_toks], dim=1)
                # print(inputs.shape)
            else:
                prompts = ["_ > " + ent for ent in entities]
                rep_idxs = torch.tensor(b_size * [rep_idx])
                taskVec_idxs = torch.tensor(b_size * [taskVec_idx])
                tokens = model.to_tokens(prompts, prepend_bos=prepend_bos,)
                inputs = tokens[:,:]

            targets = inputs[:,rep_idx+1:] #don't take the '<eos>', <context>, '_' tokens into account
            # print(tokens)

            # print(inputs, targets)
            logits = model.run_with_hooks(
                    inputs,
                    return_type = "logits",
                    fwd_hooks=[
                        (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idxs)), # replace '_' by the subject Representation
                        (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idxs)) # replace '>' by TaskVec Representation
                        ]
                ,)

            # LM Loss and optimization
            if metric == 'acc':
                acc = tl.utils.lm_accuracy(logits[:, rep_idx+1:,:], targets) # implementation here
            elif metric == 'loss':
                acc = model.loss_fn(logits[:, rep_idx+1:,:], targets)
            else:
                raise NotImplementedError(f"{metric} not implemented, can be 'acc' or 'loss'.")
            m_acc += acc.item()

    return m_acc / b_count

@torch.no_grad()
def infer_entities(model,
                   taskVector, 
                   dataset, 
                   with_context: bool = False, 
                   return_attn_pattern: bool = False,
                   max_tokens = 20, 
                   b_size = 10, 
                   prepend_bos:bool = True,
                   verbose: bool = True,
    ):
    """
    Infer entities from a given dataset augmented with representations.
    Will write the inferred entities back to the dataset in the 'inferred' field.
    Args:
        model: the model to use
        TaskVec: the task vector to use
        dataset: the dataset to infer entities from, 
            must have a 'representation' key that stores the entity representation to generate from. 
            must contain 'text' key for context
        with_context : bool : if True, the context is prepended to the entity
        return_attn_pattern : bool : if True, will return the attention pattern
        max_tokens: the maximum number of tokens to generate
        b_size: the batch size to use when inferring on a bug dataset
        prepend_bos: whether to prepend the BOS token to the input
    Returns:
        None, will write the inferred entities (and optionnally attn patterns) back to the dataset
    """
    assert taskVector is not None
    
    # inp_toks = model.to_tokens(, prepend_bos=prepend_bos)
    replace_hook_name = tl.utils.get_act_name('embed')
    #get the attention pattern hooks names
    pattern_hooks_names = [ tl.utils.get_act_name("pattern", l, "attn") for l in range(model.cfg.n_layers)]
    
    taskVector = taskVector.view(1,-1)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)
    eos_tok = model.tokenizer.eos_token_id
    eos_tok_str = model.tokenizer.eos_token
    generated = []
        
    #inference loop    
    for batch in tqdm(dataloader, disable= not verbose):

        ids = batch["id"].detach().cpu().numpy()
        reps = batch["representation"].squeeze(1).cuda()
        b_size = reps.shape[0]
        b_taskVec = taskVector.repeat(b_size,1).cuda()
        
        if with_context:
            texts = batch["text"]
            prompts = [txt + "_ >" for txt in texts]
        else:
            prompts = ["_ >" for r in reps]

        inp_toks = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side="left") 
        rep_idx = inp_toks.shape[1] - 2
        taskVec_idx = rep_idx + 1
        rep_idxs = torch.tensor(b_size * [rep_idx])
        taskVec_idxs = torch.tensor(b_size * [taskVec_idx])
        fwd_hooks = [   
                        (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idxs)), # replace '_' by the subject Representation
                        (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idxs)) # replace 'called' by TaskVec Representation
                    ]
        #inference
        for i in range(max_tokens):
                # print(inputs, targets)
                logits = model.run_with_hooks(
                    inp_toks,
                    return_type = "logits",
                    fwd_hooks=fwd_hooks
                ,)

                final_logits =  logits[:,-1,:] #extract logits for last token only
                new_toks = final_logits.argmax(-1).view(-1,1)
                inp_toks = torch.hstack((inp_toks,new_toks))

                #check if we have reached the end of the sequence
                if all(new_toks == eos_tok): break


        if return_attn_pattern:
            n_toks = inp_toks.shape[1]
            #instanciate the attention pattern
            attn_patterns = torch.zeros((   b_size,
                                            model.cfg.n_layers,
                                            model.cfg.n_heads,
                                            n_toks, 
                                            n_toks))
            def get_attn(
                pattern: torch.Tensor, # batch head_index dest_pos source_pos
                hook,
                ):
                """Hook function that stores the attention pattern"""
                l = int(hook.name.split(".")[1])
                attn_patterns[:,l,:,:,:] = pattern.cpu().detach()

            fwd_hooks += [ (hook, get_attn) for hook in pattern_hooks_names]
            #get the attention patterns for the batch
            model.run_with_hooks(
                    inp_toks,
                    return_type = None,
                    fwd_hooks=fwd_hooks
                ,)
            #store the attention patterns in the dataset
            for i in range(b_size):
                dataset[ids[i]]["attn_pattern"] = attn_patterns[i]

        for i in range(b_size) :
            gen = model.tokenizer.decode(
                inp_toks[i,taskVec_idxs[i]+1:].view(-1))
            # gen = "".join(gen).split(eos_tok_str)[0].strip()
            gen = gen.split(eos_tok_str)[0].strip()
            #store the inferred entity
            dataset[ids[i]]["inferred"] = gen
    return
 
METRIC_VERSION = 1.2
evalFileName = f"Evaluation_{METRIC_VERSION}.json"

def compute_metrics(model, TaskVec, test_dataset, max_tokens=10, b_size = 5, with_context=True, prepend_bos=True, force_recompute=True, verbose=True):
    """
    Evaluate the model on a given test_set augmented with representations.
    """
    perfect_acc = 0
    partial_acc = 0

    if force_recompute or (not "inferred" in test_dataset[0]):
        infer_entities(model, TaskVec, test_dataset, max_tokens=max_tokens, b_size=b_size, with_context=with_context, prepend_bos=prepend_bos)
    
    for item in tqdm(test_dataset, disable= not verbose):
        # print(st(item).replace("', ", "'\n"))
        # prompts = item["prompt"]
        target = item["entity"].strip()
        gen_entity = item["inferred"].strip()
        # print("gen_entity:", gen)
        # print("target:", target)
        if gen_entity == target:
            perfect_acc += 1
        if gen_entity in target or target in gen_entity:
            partial_acc += 1
      
    chrf_score = chrf.compute(predictions = [ item['inferred'] for item in test_dataset],
                               references = [ item['entity'] for item in test_dataset])

    return {
        "Partial Match": partial_acc / len(test_dataset),
        "Exact Match": perfect_acc / len(test_dataset),
        "Chr-F": chrf_score["score"], 
        "Version": METRIC_VERSION,
    }

def save_inferences(model_name, layer, test_dataset, fileName=None):
    """ Write inferred entities save in dataset under 'inferrred' key to a json file for further examination.
    """
    if fileName is None:
        fileName = f'Inference_{model_name.split("/")[-1]}_l{layer}.json'
    generation = [ {
                "entity": item['entity'], 
                "generation": item['inferred'],
                    } for item in test_dataset ]
    with open(fileName, 'w') as fp:
        json.dump(generation, fp)

############# Main Task #############  

LEARNER_VERSION = '1.1'

class LearnLabelExtractor(Task):

    model_name: Param[str]
    dataset_name: Param[str]
    layer: Param[int]
    with_context: Param[bool] = False
    extraction_method: Param[str]       # Can be either 'in_context' 'after_context' 'raw_entity' OR 'average' for baseline
    first_token_only: bool = False
    max_ent_length: Param[int] = 20 
    max_length: Param[int] = 200
    epochs: Param[int] = 5
    logs_per_epoch: Param[int] = 3
    lr: Param[float] = 1e-2
    batch_size: Param[int] = 64
    run: Param[int] = 0
    version: Constant[str] = LEARNER_VERSION      # Can change if code has been updated and need to recompute

    def execute(self):
        """Called when this task is run"""
        
        dtype = torch.bfloat16 if "12b" in self.model_name.lower() else torch.float32

        ################ Model ################
        logging.info(f"Loading model {self.model_name} ...")
        model = HookedTransformer.from_pretrained(
                                    self.model_name, 
                                    trust_remote_code=True, 
                                    low_cpu_mem_usage = True, 
                                    fold_ln=False,
                                    fold_value_biases=False,
                                    device_map='auto',
                                    dtype=dtype,
                                    local_files_only=True,
                                    )
        model.eval()
        dim = model.QK.shape[-1]

        ################ DATA  ################
        logging.info(f"loading data from {self.dataset_name} ...")
        
        max_dev_length = 2000

        if self.dataset_name.lower() == "webnlg":
            dataset = load_dataset("web_nlg", "release_v3.0_en", trust_remote_code=True)

            #optionnal, filter categories from datset
            cat = ['Food'] #WebNLG Categories to remove
            # cat = None
            if cat :
                dataset["train"] = [item for item in dataset["train"] if item["category"] not in cat]
                dataset["dev"] = [item for item in dataset["dev"] if item["category"] not in cat]
                dataset["test"] = [item for item in dataset["test"] if item["category"] not in cat]

            # Create dataset instances
            train_dataset = utils.WebNLGDataset(dataset['train'], max_ent_length=self.max_ent_length)
            dev_dataset = utils.WebNLGDataset(dataset['dev'], max_ent_length=self.max_ent_length)
            test_dataset = utils.WebNLGDataset(dataset['test'], max_ent_length=self.max_ent_length)

        elif self.dataset_name.lower() == "tacred":
            dataset = load_dataset("AmirLayegh/tacred_text_label")
            train_dataset = utils.TacredDataset(dataset["train"], max_ent_length=self.max_ent_length, max_length=200)
            dev_dataset = utils.TacredDataset(dataset["test"], max_ent_length=self.max_ent_length, max_length=200)
            test_dataset = utils.TacredDataset(dataset["validation"], max_ent_length=self.max_ent_length, max_length=200)

        elif self.dataset_name.lower() == "conll2003":
            ds = load_dataset("eriktks/conll2003", trust_remote_code=True)
            max_ent_length = 60
            max_length = 300
            train_dataset = utils.CoNLLDataset(ds["train"], max_ent_length=max_ent_length,max_length=max_length)
            dev_dataset = utils.CoNLLDataset(ds["validation"], max_ent_length=max_ent_length,max_length=max_length)
            test_dataset = utils.CoNLLDataset(ds["test"], max_ent_length=max_ent_length,max_length=max_length)

        else: 
            # unknown Dataset 
            raise NotImplementedError("Unknown dataset, can be: 'webnlg' 'tacred' or 'CoNLL2003' ")

        #limit the number of samples for testing
        logging.info(f"initial dev dataset size: {len(dev_dataset.data)}, truncating to {max_dev_length}")
        dev_dataset.data = list(np.random.choice(dev_dataset.data, max_dev_length, replace=False))

        logging.info("loading dataset done !")
        logging.info(f"train length: {len(train_dataset)}")
        logging.info(f"test length: {len(test_dataset)}")
        logging.debug(f"ex sample: {train_dataset[np.random.randint(len(train_dataset))]}")

        if self.extraction_method == 'average':
            #we are doing a baseline average extraction
            extraction_method = "in_context" #consider only this method for the moment
            logging.info(f"(BASELINE): Augmenting Train set with AVERAGE subject representations with method {extraction_method}... ")
            train_dataset.augment_with_avg_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)
            logging.info("(BASELINE): Augmenting Test set with AVERAGE subject representations ... ")
            test_dataset.augment_with_avg_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)
            logging.info("(BASELINE): Augmenting dev set with AVERAGE subject representations ... ")
            dev_dataset.augment_with_avg_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)
            logging.info("Extraction of subjects representatons Done !\n")

        elif 'random_sample' in self.extraction_method: # param is like 'random_sample_10', hacky..
            try: 
                n = int(self.extraction_method.split("_")[-1])
            except:
                n = 3 #default value
            extraction_method = "in_context" #consider only this method for the moment
            logging.info(f"BASELINE: Sampling random spans of {n} tokens in texts and extracting reps with method {extraction_method}... ")
            train_dataset = utils.sample_random_entities(model, train_dataset, n=n)
            logging.info(f"Augmenting Train set with subject representations with method {self.extraction_method}... ")
            train_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)

            test_dataset = utils.sample_random_entities(model, test_dataset, n=n)
            logging.info("Augmenting Test set with subject representations ... ")
            test_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)

            dev_dataset = utils.sample_random_entities(model, dev_dataset, n=n)
            logging.info("Augmenting dev set with subject representations ... ")
            dev_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)
            logging.info("Extraction of subjects representatons Done !\n")

        else: 
            logging.info(f"Augmenting Train set with subject representations with method {self.extraction_method}... ")
            train_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=self.extraction_method)
            logging.info("Augmenting Test set with subject representations ... ")
            test_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=self.extraction_method)
            logging.info("Augmenting dev set with subject representations ... ")
            dev_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=self.extraction_method)
            logging.info("Extraction of subjects representatons Done !\n")

        ################ TRAINING ################
        dim = model.QK.shape[-1]
        prepend_bos = True
        # create Task Vector
        TaskVec = torch.normal(mean=0, std=1.0, size=(1,dim), requires_grad=True, dtype=dtype)
        best_TaskVec = torch.zeros_like(TaskVec)
        hist = []

        for param in model.parameters():
            param.requires_grad = False

        logging.info(f"Begining Task Vector Training ...")
        eos_tok_str = model.tokenizer.eos_token
        eos_tok = model.tokenizer.eos_token_id
        replace_hook_name = tl.utils.get_act_name('embed') #pos_embed for gpt2 ... 
        logging.info(f"will insert representation at hook '{replace_hook_name}'")
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=self.batch_size, shuffle=True)
        n_log = len(train_dataloader) // self.logs_per_epoch
        
        len_loader = len(train_dataloader)
        optim = torch.optim.Adam([TaskVec], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, len_loader, eta_min=self.lr/10)

        gc.collect()
        torch.cuda.empty_cache()

        logging.info(f"Starting training TaskVec {'with context' if self.with_context else ''} for layer {self.layer} of {self.model_name} with lr {self.lr}...")
        for epoch in range(self.epochs):
            b_count = 0
            m_loss = 0
            for batch in tqdm(train_dataloader):
                b_count += 1
                
                entities = batch["entity"]
                entity_toks = batch.get("entity_tokens", None)
                texts = batch["text"]
                reps = batch["representation"].squeeze(1).cuda()
                b_size = reps.shape[0]
                b_taskVec = TaskVec.repeat(b_size,1).cuda()
                
                if self.first_token_only:
                    entities = [toks[0] for toks in model.to_str_tokens(entities,prepend_bos=False)]
                else:    
                    entities = [ent + eos_tok_str for ent in entities] # take whole label add eos token
                
                if entity_toks is not None:
                    eos = torch.tensor([eos_tok]).repeat(b_size,1)
                    entity_toks = torch.cat([entity_toks, eos], dim=1).cuda()

                if self.with_context:
                    prompts = [txt + "_ >" for txt in texts]
                    context_toks = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side="left") 
                    if entity_toks is None:
                        entity_toks = model.to_tokens(entities, prepend_bos=False, padding_side="right")
                    inputs = torch.cat([context_toks, entity_toks], dim=1)
                    rep_idx = context_toks.shape[1] - 2
                else:
                    rep_idx = 1 if prepend_bos else 0 
                    
                    if entity_toks is not None:
                        prompts = ["_ >" for ent in entities]
                        inputs = model.to_tokens(prompts, prepend_bos=prepend_bos,)
                        inputs = torch.cat([inputs, entity_toks.cuda()], dim=1)
                    else:
                        prompts = ["_ > " + ent for ent in entities]
                        inputs = model.to_tokens(prompts, prepend_bos=prepend_bos,)
                
                rep_idxs = torch.tensor(b_size * [rep_idx])
                taskVec_idxs = torch.tensor(b_size * [rep_idx + 1])            
                targets = inputs[:,rep_idx+1:] #don't take the '<eos>', <context>, '_' tokens into account

                logits = model.run_with_hooks(
                        inputs,
                        return_type = "logits",
                        fwd_hooks=[
                            (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idxs)), # replace '_' by the subject Representation
                            (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idxs)) # replace 'called' by TaskVec Representation
                            ]
                    ,)
    
                # LM Loss and optimization 
                loss = model.loss_fn(logits[:, rep_idx+1:,:], targets) # take only the loss on the entity tokens
                loss.backward()
                optim.step()
                optim.zero_grad()
                m_loss += loss.item()
                
                if epoch > 2: scheduler.step()

                if b_count % n_log == 0:
                    m_loss = m_loss / (len(train_dataloader) / self.logs_per_epoch)
                    acc = (eval_model(
                            model,
                            TaskVec,
                            test_loader=dev_dataloader, 
                            first_token_only= self.first_token_only, 
                            with_context= self.with_context,
                            prepend_bos=prepend_bos))
                    e = epoch + b_count/len(train_dataloader)
                    lr = scheduler.get_last_lr()[0]
                    #save best taskVec according to test accuracy
                    if hist and acc >= max([h["test accuracy"] for h in hist]):
                        best_TaskVec[:] = TaskVec[:]

                    hist.append({"epoch":e, "loss": m_loss, "test accuracy": acc, "lr":lr})
                    logging.info(f"\nEpoch {e:.1f}, LM loss: {m_loss:.3f}, Test Acc: {acc:.3f}, lr:{lr:.4f}")
                    m_loss = 0 # reset
        logging.info("Training Done !\n")
        
        TaskVec = best_TaskVec # retrieve best TaskVec
        
        # Save TaskVector and train history
        fileName = f"TaskVec_{self.model_name.split('/')[-1]}_l{self.layer}_e{hist[-1]['epoch']:.1f}.pth"
        torch.save(TaskVec, fileName) 
        
        #Evaluation stage
        logging.info(f"Evaluation on Test Set...")
        metrics = compute_metrics(
                                model,
                                TaskVec,
                                test_dataset,
                                b_size=5,
                                with_context=self.with_context,
                                prepend_bos=prepend_bos)
        logging.info("Done !\n")
        logging.info(metrics)

        #add metrics to last logging
        hist[-1].update(metrics)
        
        #save history
        with open('history.json', 'w') as fp:
            json.dump(hist, fp)

        #save generation
        save_inferences(self.model_name,self.layer, test_dataset)

        # save computed metrics
        with open(evalFileName, 'w') as fp:
            json.dump(metrics, fp) 


class LearnLinearFilter(Task):

    job_path: Param[str]
    TaskVec_path: Param[str]
    model_name: Param[str]
    dataset_name: Param[str]
    layer: Param[int]
    batch_size: Param[int] = 64
    epochs: Param[int] = 5
    logs_per_epoch: Param[int] = 3
    lr: Param[float] = 1e-2
    with_context: Param[bool] = False
    extraction_method: Param[str]
    max_ent_length: Param[int] = 20
    max_length: Param[int] = 200
    run: Param[int] = 0
    version: Constant[str] = LEARNER_VERSION      # Can change if code has been updated and need to recompute

    def execute(self):
        """Learns a linear layer on top of previously trained TaskVec"""
        
        global evalFileName
        
        #Load Model
        logging.info(f"loading model {self.model_name} ...")
        model = HookedTransformer.from_pretrained(
                                            self.model_name, 
                                            trust_remote_code = True, 
                                            low_cpu_mem_usage = True, 
                                            device_map='auto',
                                            move_to_device=False,
                                            fold_ln=False,
                                            fold_value_biases=False,
                                            center_writing_weights=False,
                                            center_unembed=False,
                                            )
        model.eval()
        model = model.cuda()

        #Load Task Vector
        logging.info(f"loading TaskVec from {self.TaskVec_path} ...")
        TaskVec = torch.load(self.TaskVec_path)

        # Load Data
        logging.info(f"loading data from {self.dataset_name} ...")
        train_dataset , test_dataset, val_dataset = utils.load_datasets(self.dataset_name)
        
        #augment datasets with representations
        logging.info(f"augmenting datasets with representations from layer {self.layer} with method {self.extraction_method}")
        if self.extraction_method == 'average':
            #we are doing a baseline average extraction
            extraction_method = "in_context" #consider only this method for the moment
            train_dataset.augment_with_avg_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)
            test_dataset.augment_with_avg_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)
            val_dataset.augment_with_avg_repr(model, self.layer, batch_size=self.batch_size, method=extraction_method)
        else: 
            train_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=self.extraction_method)
            test_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=self.extraction_method)
            val_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size, method=self.extraction_method)
        logging.info("Extraction of subjects representatons Done !\n")

        dtype = model.W_U.dtype
        prepend_bos = True
        dim = model.QK.shape[-1]
        eos_tok_str = model.tokenizer.eos_token
        replace_hook_name = tl.utils.get_act_name('embed') #pos_embed for gpt2 ... 
        hist = []

        #freeze the model and the task vector
        for param in model.parameters():
            param.requires_grad = False
        TaskVec.requires_grad_(False)
        
        ## instanciate linear model
        linear_model = torch.nn.Linear(dim, dim, bias=True)
        
        #initialize the linear model with identity
        linear_model.weight.data = torch.eye(dim).type(dtype)
        linear_model.bias.data = torch.zeros(dim, dtype=dtype)
        linear_model = linear_model.cuda()

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        len_loader = len(train_dataloader)
        n_log = n_log = len_loader // self.logs_per_epoch
        val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=True)

        optim = torch.optim.Adam(linear_model.parameters() , lr=self.lr)

        gc.collect()
        torch.cuda.empty_cache()
        logging.info(f"Starting training Linear layer on {self.dataset_name} for layer {self.layer} of {self.model_name} with{'out' if not self.with_context else ''} context...")
        b_count = -1
        for epoch in range(self.epochs):
            m_loss = 0
            for batch in tqdm(train_dataloader):
                        
                b_count += 1
                entities = batch["entity"]
                texts = batch["text"]
                reps = batch["representation"].squeeze(1).cuda()
                b_size = reps.shape[0]
                b_taskVec = TaskVec.repeat(b_size,1).cuda()
                
                entities = [ent + eos_tok_str for ent in entities] # take whole label and add eos token

                if self.with_context:
                    prompts = [txt + "_ >" for txt in texts]
                    context_toks = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side="left") 
                    entities_toks = model.to_tokens(entities, prepend_bos=False, padding_side="right")
                    inputs = torch.cat([context_toks, entities_toks], dim=1)
                    rep_idx = context_toks.shape[1] - 2
                else:
                    rep_idx = 1 if prepend_bos else 0 
                    prompts = ["_ > " + ent for ent in entities]
                    inputs = model.to_tokens(prompts, prepend_bos=prepend_bos,)

                rep_idxs = torch.tensor(b_size * [rep_idx])
                taskVec_idxs = torch.tensor(b_size * [rep_idx + 1])
                targets = inputs[:,rep_idx+1:] #don't take the '<eos>', <context>, '_' tokens into account

                #transform the representations with linear model
                reps = linear_model(reps)

                # run Model with replacements hooks
                logits = model.run_with_hooks(
                        inputs,
                        return_type = "logits",
                        fwd_hooks=[
                            (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idxs)), # replace '_' by the subject Representation
                            (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idxs)) # replace 'called' by TaskVec Representation
                            ]
                    ,)

                # LM Loss and optimization 
                loss = model.loss_fn(logits[:, rep_idx+1:,:], targets) # take only the loss on the entity tokens
                loss.backward()
                optim.step()
                optim.zero_grad()
                m_loss += loss.item()
                
                if b_count % n_log == 0:
                    m_loss = m_loss / n_log
                    e = epoch + b_count/len_loader
                    acc = eval_model(
                                    model,
                                    TaskVec,
                                    test_loader=val_dataloader, 
                                    with_context=self.with_context,
                                    prepend_bos=prepend_bos,
                                    transform=linear_model,
                                    # metric='loss'
                                    )
                    logging.info(f"Epoch {e:.1f},  Batch {b_count}/{len_loader}, Loss: {m_loss:.3f}, Test Acc: {acc:3f}, lr:{self.lr:.4f}")
                    
                    #save best taskVec according to test accuracy
                    if len(hist) == 0 or acc >= max([h["test accuracy"] for h in hist]):
                        best_ckpt = linear_model.state_dict()

                    hist.append({"epoch":e, "loss": m_loss, "test accuracy": acc, "lr":self.lr})
                    m_loss = 0
            b_count = 0        

        fileName = f"LinearFilter_{self.model_name}_l{self.layer}_e{hist[-1]['epoch']:.1f}{'' if self.with_context else 'no'}Context.pth"

        #retreive best weights
        linear_model.load_state_dict(best_ckpt)
        # Save linear model
        torch.save(linear_model, fileName)

        #save history
        with open('history.json', 'w') as fp:
            json.dump(hist, fp)

        ## Clean reps from test set 
        for item in test_dataset:
            item["representation"] = linear_model(item["representation"].cuda()).cpu().detach()

        #Compute metricss
        metrics = compute_metrics(
                                model,
                                TaskVec,
                                test_dataset,
                                b_size=self.batch_size,
                                with_context=self.with_context,
                                )
        logging.info(f"metrics on inference with trainsformed representations: {metrics}")

        # save inferences
        fileName = f"Cleaned_Inference_{self.model_name.split('/')[-1]}_l{self.layer}.json"
        save_inferences(self.model_name,self.layer, test_dataset, fileName=fileName)

        # Save metrics
        evalFileName = evalFileName.replace(".json", "_LinearTransform.json")
        with open(evalFileName, 'w') as fp:
            json.dump(metrics, fp) 


############# Evaluation Task #############  

class EvalLabelExtractor(Task):

    job_path: Param[str]
    TaskVec_path: Param[str]
    model_name: Param[str]
    dataset_name: Param[str]
    layer: Param[int]
    with_context: Param[bool] = False
    extraction_method: Param[str]
    max_ent_length: Param[int] = 20
    max_length: Param[int] = 200
    batch_size: Param[int] = 64
    metrics_v: Constant[float] = METRIC_VERSION

    def execute(self):
        """Perform Evaluation of a previously trained TaskVec"""
        
        #change working dir 
        os.chdir(self.job_path)

        #check if evaluation has already been done or not:
        if (Path(self.job_path) / evalFileName).exists():
            logging.info(f"{evalFileName} already exixts in {self.job_path}, returing...")
            return
        else :
            logging.info(f"Results will be written in {evalFileName}")

        #Load Model
        logging.info(f"loading model {self.model_name} ...")
        model = HookedTransformer.from_pretrained(
                                            self.model_name, 
                                            trust_remote_code = True, 
                                            low_cpu_mem_usage = True, 
                                            device_map='auto',
                                            move_to_device=False,
                                            fold_ln=False,
                                            fold_value_biases=False,
                                            center_writing_weights=False,
                                            center_unembed=False,
                                            )
        model.eval()
        model = model.cuda()

        #Load Task Vector
        logging.info(f"loading TaskVec from {self.TaskVec_path} ...")
        TaskVec = torch.load(self.TaskVec_path)

        # Load Data
        logging.info(f"loading data from {self.dataset_name} ...")
        if self.dataset_name.lower() == "webnlg":
            dataset = load_dataset("web_nlg", "release_v3.0_en", trust_remote_code=True)

            #optionnal, filter categories from datset
            cat = ['Food'] #WebNLG Categories to remove
            # cat = None
            if cat :
                dataset["test"] = [item for item in dataset["test"] if item["category"] not in cat]

            # Create dataset instances
            test_dataset = utils.WebNLGDataset(dataset['test'], 
                                               max_ent_length=self.max_ent_length,
                                               max_length=self.max_length)

        elif self.dataset_name.lower() == "tacred":
            dataset = load_dataset("AmirLayegh/tacred_text_label")
            test_dataset = utils.TacredDataset(dataset["validation"], 
                                               max_ent_length=self.max_ent_length, 
                                               max_length=self.max_length)
        else: 
            # unknown Dataset 
            raise NotImplementedError("dataset Name must be either 'webnlg' or 'tacred'")
        
        
        #augment test set with representations
        logging.info(f"augmenting test dataset with representations from layer {self.layer}")
        test_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size)

        #Compute metricss
        metrics = compute_metrics(
                                model,
                                TaskVec,
                                test_dataset,
                                b_size=self.batch_size,
                                with_context=self.with_context,
                                )
        logging.info(metrics)

        # save inferences
        save_inferences(self.model_name,self.layer, test_dataset)

        # save computed metrics
        with open(evalFileName, 'w') as fp:
            json.dump(metrics, fp) 


