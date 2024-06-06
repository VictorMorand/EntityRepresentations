### Author: Victor Morand
### this experiment script

from typing import List, Optional
import logging
import transformer_lens as tl 
from transformer_lens import HookedTransformer
from experimaestro import Config, Task, Param
from datasets import load_dataset
import torch, gc
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path

############# utils #############  
import utils 


def eval_model(model, TaskVec, test_loader, first_token_only=False, prepend_bos=True):
    """
    Evaluate the model on a given test_loader augmented with representations.
    """
    b_count = 0
    m_acc = 0
    replace_hook_name = tl.utils.get_act_name('embed') #pos_embed ?
    eos_tok_str = model.tokenizer.eos_token
    rep_idx = 1 if prepend_bos else 0
    taskVec_idx = rep_idx + 1

    with torch.no_grad():
        for batch in tqdm(test_loader):
            b_count += 1
            # print(st(batch).replace("', ", "'\n"))
            # prompts = batch["prompt"]
            reps = batch["representation"].squeeze(1)
            entities = batch["entity"]
            if first_token_only:
                entities = [toks[0] for toks in model.to_str_tokens(entities,prepend_bos=False)]
            prompts = ["_ > " + ent for ent in entities]

            b_taskVec = TaskVec.repeat(len(prompts),1)
            tokens = model.to_tokens(prompts, prepend_bos=prepend_bos,)
            # print(tokens)
            # break
            inputs = tokens[:,:]
            targets = tokens[:,2:] #don't take the '<eos>','_' ,'>'' tokens into account

            # print(inputs, targets)
            logits = model.run_with_hooks(
                    inputs,
                    return_type = "logits",
                    fwd_hooks=[
                        (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idx)), # replace '_' by the subject Representation
                        (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idx)) # replace 'called' by TaskVec Representation
                        ]
                ,)

            # LM Loss and optimization 
            acc = tl.utils.lm_accuracy(logits[:,2:,:], targets) # implementation here
            
            m_acc += acc.item()

    return m_acc / b_count


############# Main Task #############  

class LearnLabelExtractor(Task):

    model_name: Param[str]
    layer: Param[int]

    first_token_only: bool = False
    max_ent_length: Param[int] = 20
    epochs: Param[int] = 5
    lr: Param[float] = 1e-2
    batch_size: Param[int] = 32

    def execute(self):
        """Called when this task is run"""
        
        ################ Model ################
        logging.info(f"Loading model {self.model_name} ...")
        model = HookedTransformer.from_pretrained(self.model_name)
        model.eval()
        dim = model.QK.shape[-1]

        ################ DATA  ################
        dataset = load_dataset("web_nlg", "release_v3.0_en")

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
        losses = []
        accs = []
        # TaskVec = torch.ones((1,d), requires_grad=True)

        for param in model.parameters():
            param.requires_grad = False


        rep_idx = 1 if prepend_bos else 0 
        taskVec_idx = rep_idx + 1
        padding_tok = model.tokenizer.eos_token_id
        eos_tok_str = model.tokenizer.eos_token
        replace_hook_name = tl.utils.get_act_name('embed') #pos_embed for gpt2 ... 
        logging.debug(f"will insert representation at hook '{replace_hook_name}'")
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=True)
        optim = torch.optim.Adam([TaskVec], lr=self.lr)

        gc.collect()
        torch.cuda.empty_cache()

        for epoch in range(self.epochs):
            b_count = 0
            m_loss = 0
            for batch in tqdm(train_dataloader):
                b_count += 1
                
                entities = batch["entity"]
                if self.first_token_only:
                    entities = [toks[0] for toks in model.to_str_tokens(entities,prepend_bos=False)]
                else:    
                    entities = [ent + eos_tok_str for ent in entities] # take whole label add eos token

                prompts = ["_ > " + ent for ent in entities]

                reps = batch["representation"].squeeze(1)
                b_taskVec = TaskVec.repeat(len(prompts),1)
                tokens = model.to_tokens(prompts, prepend_bos=prepend_bos,)
                # print(tokens)
                # break

                inputs = tokens[:,:]
                targets = tokens[:,2:] #don't take the '<eos>','_' ,'called'' tokens into account

                # print(inputs, targets)
                logits = model.run_with_hooks(
                        inputs,
                        return_type = "logits",
                        fwd_hooks=[
                            (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idx)), # replace '_' by the subject Representation
                            (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idx)) # replace 'called' by TaskVec Representation
                            ]
                    ,)

                # LM Loss and optimization 
                loss = model.loss_fn(logits[:,2:,:], targets)
                loss.backward()
                optim.step()
                optim.zero_grad()
                m_loss += loss.item()

            losses.append(m_loss / b_count)
            accs.append(eval_model(model,TaskVec,test_loader=test_dataloader, first_token_only=self.first_token_only, prepend_bos=prepend_bos))
            logging.info(f" Epoch {len(losses)}, Language modeling loss: {losses[-1]:.3f}, Test Acc: {accs[-1]:.3f}")

        fileName = f'TaskVec_{self.model_name}_l{self.layer}_e{len(losses)}.pth'
        torch.save(TaskVec, fileName) 
                
