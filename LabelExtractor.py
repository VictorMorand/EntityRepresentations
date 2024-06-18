### Author: Victor Morand
### this experiment script

from typing import List, Optional
import logging
import transformer_lens as tl 
from transformer_lens import HookedTransformer
from experimaestro import Config, Task, Param
from datasets import load_dataset
import torch, gc, json
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
    first_token_only=False,
    with_context=False,
    prepend_bos=True):
    """
    Evaluate the model on a given test_loader augmented with representations.
    Args:
    model : HookedTransformer : the model to use
    TaskVec : torch.Tensor : the task vector to evaluate
    test_loader : DataLoader : the test loader to use
    first_token_only : bool : if True, only the first token of the entity is considered
    with_context : bool : if True, the context is prepended to the entity
    prepend_bos : bool : if True, the entity is prepended with the bos token
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
            # print(st(batch).replace("', ", "'\n"))
            # prompts = batch["prompt"]
            reps = batch["representation"].squeeze(1).cuda()
            entities = batch["entity"]
            texts = batch["text"]
            
            if first_token_only:
                entities = [toks[0] for toks in model.to_str_tokens(entities,prepend_bos=False)]
            else:    
                entities = [ent + eos_tok_str for ent in entities] # take whole label add eos token

            reps = batch["representation"].squeeze(1).cuda()
            b_size = reps.shape[0]
            b_taskVec = TaskVec.repeat(b_size,1).cuda()

            if with_context:
                prompts = [txt + "_ >" for txt in texts]
                context_toks = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side="left") 
                entities_toks = model.to_tokens(entities, prepend_bos=False, padding_side="right")
                rep_idx = context_toks.shape[1] - 2
                # print(rep_idx)
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
                        (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idxs)) # replace 'called' by TaskVec Representation
                        ]
                ,)

            # LM Loss and optimization
            acc = tl.utils.lm_accuracy(logits[:, rep_idx+1:,:], targets) # implementation here
            m_acc += acc.item()

    return m_acc / b_count

def infer_entities(model, taskVector, dataset, with_context=False, max_tokens = 20, b_size = 10, prepend_bos=True):
    """
    Infer entities from a given dataset augmented with representations.
    Will write the inferred entities back to the dataset in the 'generated' field.
    Args:
        model: the model to use
        TaskVec: the task vector to use
        dataset: the dataset to infer entities from, must have been augmented with representations and contain prompt
        with_context : bool : if True, the context is prepended to the entity
        max_tokens: the maximum number of tokens to generate
        b_size: the batch size to use
        prepend_bos: whether to prepend the BOS token to the input
    """
    assert taskVector is not None
    # assert isinstance(dataset, utils.WebNLGDataset)
    assert "text" in dataset[0]
    
    # inp_toks = model.to_tokens(, prepend_bos=prepend_bos)
    replace_hook_name = tl.utils.get_act_name('embed')
    
    taskVector = taskVector.view(1,-1)
    dataloader = DataLoader(dataset, batch_size=b_size, shuffle=True)
    eos_tok = model.tokenizer.eos_token_id
    eos_tok_str = model.tokenizer.eos_token
    generated = []
    # print(b_taskVec.shape)
    # print(repr.shape)
    
    with torch.no_grad():
        for batch in tqdm(dataloader):

            texts = batch["text"]
            ids = batch["id"].detach().cpu().numpy()
            reps = batch["representation"].squeeze(1).cuda()
            b_size = reps.shape[0]
            b_taskVec = taskVector.repeat(b_size,1).cuda()
           
            if with_context:
                prompts = [txt + "_ >" for txt in texts]
            else:
                prompts = ["_ >" for txt in texts]

            inp_toks = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side="left") 
            rep_idx = inp_toks.shape[1] - 2
            taskVec_idx = rep_idx + 1
            rep_idxs = torch.tensor(b_size * [rep_idx])
            taskVec_idxs = torch.tensor(b_size * [taskVec_idx])

            #inference
            for i in range(max_tokens):
                    # print(inputs, targets)
                    logits = model.run_with_hooks(
                        inp_toks,
                        return_type = "logits",
                        fwd_hooks=[
                            (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idxs)), # replace '_' by the subject Representation
                            (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idxs)) # replace 'called' by TaskVec Representation
                            ]
                    ,)

                    final_logits =  logits[:,-1,:] #extract logits for last token only
                    new_toks = final_logits.argmax(-1).view(-1,1)
                    inp_toks = torch.hstack((inp_toks,new_toks))

                    #check if we have reached the end of the sequence
                    if all(new_toks == eos_tok): break
            
            for i in range(b_size) :
                gen = model.to_str_tokens(
                    inp_toks[i,taskVec_idxs[i]+1:])
                gen = "".join(gen).split(eos_tok_str)[0].strip()
                generated.append( {"id":ids[i] , "inferred": gen})
                # print(generated)
        #fuse with original dataset
        for gen in generated:
             dataset[gen["id"]].update(gen)
        return 
 

def compute_metrics(model, TaskVec, test_dataset, max_tokens=10, b_size = 50, with_context=True, prepend_bos=True):
    """
    Evaluate the model on a given test_loader augmented with representations.
    """
    perfect_acc = 0
    partial_acc = 0

    if not "inferred" in test_dataset[0]:
        infer_entities(model, TaskVec, test_dataset, max_tokens=max_tokens, b_size=b_size, with_context=with_context, prepend_bos=prepend_bos)
    
    for item in tqdm(test_dataset):
        # print(st(item).replace("', ", "'\n"))
        # prompts = item["prompt"]
        target = item["entity"]
        gen = item["inferred"]
        gen_entity = gen.strip()
        # print("gen_entity:", gen)
        # print("target:", target)
        if gen_entity == target:
            perfect_acc += 1
        if gen_entity in target or target in gen_entity:
            partial_acc += 1

    return {
        "Partial Match": partial_acc / len(test_dataset),
        "Perfect Match Acc": perfect_acc / len(test_dataset),
    }

############# Main Task #############  

class LearnLabelExtractor(Task):

    model_name: Param[str]
    layer: Param[int]
    with_context: Param[bool] = False
    first_token_only: bool = False
    max_ent_length: Param[int] = 20
    epochs: Param[int] = 5
    logs_per_epoch: Param[int] = 3
    lr: Param[float] = 1e-2
    batch_size: Param[int] = 64

    def execute(self):
        """Called when this task is run"""
        
        ################ Model ################
        logging.info(f"Loading model {self.model_name} ...")
        model = HookedTransformer.from_pretrained(
                                    self.model_name, 
                                    trust_remote_code=True, 
                                    low_cpu_mem_usage = True, 
                                    fold_ln=False,
                                    fold_value_biases=False,
                                    device_map='auto',
                                    )
        model.eval()
        dim = model.QK.shape[-1]

        ################ DATA  ################
        dataset = load_dataset("web_nlg", "release_v3.0_en", trust_remote_code=True)

        #optionnal, filter categories from datset
        cat = ['Food'] #WebNLG Categories to remove
        # cat = None
        if cat :
            dataset["train"] = [item for item in dataset["train"] if item["category"] not in cat]
            dataset["test"] = [item for item in dataset["test"] if item["category"] not in cat]

        # Create dataset instances
        logging.info("loading dataset ...")
        train_dataset = utils.WebNLGDataset(dataset['train'], max_ent_length=self.max_ent_length)
        test_dataset = utils.WebNLGDataset(dataset['test'], max_ent_length=self.max_ent_length)
        
        logging.info("loading dataset done !")
        logging.info(f"train length: {len(train_dataset)}")
        logging.info(f"test length: {len(test_dataset)}")
        logging.debug(f"ex sample: {train_dataset[np.random.randint(len(train_dataset))]}")

        logging.info("Augmenting Train set with subject representations ... ")
        train_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size)
        logging.info("Augmenting Test set with subject representations ... ")
        test_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size)
        logging.info("Extraction of subjects representatons Done !\n")

        ################ TRAINING ################

        dim = model.QK.shape[-1]
        prepend_bos = True
        # create Task Vector
        TaskVec = torch.normal(mean=0, std=1.0, size=(1,dim), requires_grad=True)
        hist = []
        # TaskVec = torch.ones((1,d), requires_grad=True)

        for param in model.parameters():
            param.requires_grad = False


        logging.info(f"Beging Label extractor Training ...")
        
        eos_tok_str = model.tokenizer.eos_token
        replace_hook_name = tl.utils.get_act_name('embed') #pos_embed for gpt2 ... 
        logging.info(f"will insert representation at hook '{replace_hook_name}'")
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=True)
        n_log = len(train_dataloader) // self.logs_per_epoch
        

        optim = torch.optim.Adam([TaskVec], lr=self.lr)
        gc.collect()
        torch.cuda.empty_cache()

        logging.info(f"Starting training TaskVec {'with context' if self.with_context else ''} for layer {self.layer} of {self.model_name} with lr {self.lr}...")
        for epoch in range(self.epochs):
            b_count = 0
            m_loss = 0
            for batch in tqdm(train_dataloader):
                b_count += 1
                
                entities = batch["entity"]
                texts = batch["text"]
                reps = batch["representation"].squeeze(1).cuda()
                b_size = reps.shape[0]
                b_taskVec = TaskVec.repeat(b_size,1).cuda()
                
                if self.first_token_only:
                    entities = [toks[0] for toks in model.to_str_tokens(entities,prepend_bos=False)]
                else:    
                    entities = [ent + eos_tok_str for ent in entities] # take whole label add eos token

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
                    m_loss = m_loss / (len(train_dataloader) / self.logs_per_epoch)
                    acc = (eval_model(
                            model,
                            TaskVec,
                            test_loader=test_dataloader, 
                            first_token_only= self.first_token_only, 
                            with_context= self.with_context,
                            prepend_bos=prepend_bos))
                    e = epoch + b_count/len(train_dataloader)
                    hist.append({"epoch":e, "loss": m_loss, "test accuracy": acc})
                    logging.info(f" Epoch {e:.1f}, Language modeling loss: {m_loss:.3f}, Test Acc: {acc:.3f}")
                    m_loss = 0 # reset
        
        metrics = compute_metrics(
                                model,
                                TaskVec,
                                test_dataset,
                                b_size=100)
        logging.info(metrics)

        #add metrics to last logging
        hist[-1].update(metrics)
        
        # Save TaskVector and train history
        fileName = f'TaskVec_{self.model_name}_l{self.layer}_e{len(hist)}.pth'
        torch.save(TaskVec, fileName) 
        
        #save history
        with open('history.json', 'w') as fp:
            json.dump(hist, fp)

        #save generation
        fileName = f'Inference_{self.model_name}_l{self.layer}_e{len(hist)}.json'
        generation = [ {
                    "entity": item['entity'], 
                    "generation": item['inferred'],
                        } for item in test_dataset ]
        with open(fileName, 'w') as fp:
            json.dump(generation, fp)