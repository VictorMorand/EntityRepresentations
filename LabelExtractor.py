### Author: Victor Morand
### this script 

import transformer_lens as tl 
from transformer_lens import HookedTransformer
from experimaestro import Config, Task, Param, experiment
from datasets import load_dataset
import torch, gc
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import click 
from pathlib import Path

#our own module
import utils 

############# utils #############  
from utils import get_replace_with_rep_hook

def eval_model(model, TaskVec, test_loader, prepend_bos=True):
    """
    Evaluate the model on a given test_loader
    """
    b_count = 0
    m_acc = 0
    replace_hook_name = tl.utils.get_act_name('embed') #pos_embed ?
    rep_idx = 1 if prepend_bos else 0
    taskVec_idx = rep_idx + 1

    with torch.no_grad():
        for batch in tqdm(test_loader):
            b_count += 1
            # print(str(batch).replace("', ", "'\n"))
            prompts = batch["prompt"]
            reps = batch["representation"].squeeze(1)
            b_taskVec = TaskVec.repeat(len(prompts),1)

            tokens = model.to_tokens(prompts, prepend_bos=prepend_bos,)
            inputs = tokens[:,:]
            targets = tokens[:,2:] #don't take the '<eos>','_' ,'called'' tokens into account

            # print(inputs, targets)
            logits = model.run_with_hooks(
                    inputs,
                    return_type = "logits",
                    fwd_hooks=[
                        (replace_hook_name, get_replace_with_rep_hook(reps, rep_idx)), # replace '_' by the subject Representation
                        (replace_hook_name, get_replace_with_rep_hook(b_taskVec, taskVec_idx)) # replace 'called' by TaskVec Representation
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

    max_ent_length: Param[int] = 20
    epochs: Param[int] = 50
    lr: Param[float] = 1e-2
    batch_size: Param[int] = 32

    def execute(self):
        """Called when this task is run"""
        
        ################ Model ################
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
        train_dataset = utils.WebNLGDataset(dataset['train'], max_ent_length=self.max_ent_length)
        test_dataset = utils.WebNLGDataset(dataset['test'], max_ent_length=self.max_ent_length)
        
        print("loading dataset done !")
        print("train length:", len(train_dataset))
        print("test length:", len(test_dataset))
        print("ex sample:", train_dataset[np.random.randint(len(train_dataset))])

        print("Augmenting Train set with subject representations ... ")
        train_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size)
        print("Augmenting Test set with subject representations ... ")
        test_dataset.augment_with_repr(model, self.layer, batch_size=self.batch_size)
        print("Extraction of subjects representatons Done !\n")

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
        eos_tok = model.tokenizer.eos_token_id
        padding_tok = model.tokenizer.eos_token_id
        replace_hook_name = tl.utils.get_act_name('embed') #pos_embed for gpt2 ... 

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
                # print(str(batch).replace("', ", "'\n"))
                prompts = batch["prompt"]
                reps = batch["representation"].squeeze(1)
                b_taskVec = TaskVec.repeat(len(prompts),1)

                tokens = model.to_tokens(prompts, prepend_bos=prepend_bos,)
                inputs = tokens[:,:]
                targets = tokens[:,2:] #don't take the '<eos>','_' ,'called'' tokens into account

                # print(inputs, targets)
                logits = model.run_with_hooks(
                        inputs,
                        return_type = "logits",
                        fwd_hooks=[
                            (replace_hook_name, get_replace_with_rep_hook(reps, rep_idx)), # replace '_' by the subject Representation
                            (replace_hook_name, get_replace_with_rep_hook(b_taskVec, taskVec_idx)) # replace 'called' by TaskVec Representation
                            ]
                    ,)

                # LM Loss and optimization 
                loss = model.loss_fn(logits[:,2:,:], targets)
                loss.backward()
                optim.step()
                optim.zero_grad()
                m_loss += loss.item()

            losses.append(m_loss / b_count)
            accs.append(eval_model(model,TaskVec,test_loader=test_dataloader))
            print(f" Epoch {len(losses)}, Language modeling loss: {losses[-1]:.3f}, Test Acc: {accs[-1]:.3f}")

        fileName = f'TaskVec_{self.model_name}_l{self.layer}_e{len(losses)}.pth'
        torch.save(TaskVec, fileName) 
                
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.option("--sleeptime", type=float, default=2, help="Sleep time")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, sleeptime):
    """Runs an experiment"""
    # Sets the working directory and the name of the xp
    with experiment(workdir, "helloworld", port=port) as xp:
        # Submit the tasks
        LearnLabelExtractor(model_name="gpt2-small", layer=10).submit()

if __name__ == "__main__":
    cli()