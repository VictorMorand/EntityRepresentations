"""
In this script we train a taskVector for relation extraction

"""
import torch, os, sys, pathlib, json, gc
sys.path.append(str(pathlib.Path.cwd().parent)) #WD
os.chdir("..")
import transformer_lens as tl 
from transformer_lens import HookedTransformer, patching
from importlib import reload
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

#our own code
import utils
from utils import EntityReprDataset
from LabelExtractor import eval_model, infer_entities
from processResults import *
from processResults import get_taskVec
from LabelExtractor import compute_metrics

################################################################### 
###############    UTILS   ######################################## 
################################################################### 

def augment_with_reprs(joint_df, model, layer=0, method='after_context', batch_size=32, verbose=False):
    """
    Augment the df with representations of the entities in the text
    """
    for key in ['subject', 'object']:
        if verbose: print(f"Extracting representations for {key} with method {method}")
        joint_df["entity"] = joint_df[key] # get key into entity col 
        #drop the key column
        new_key = f"{key}_repr"
        if new_key in joint_df.columns:
            joint_df.drop(columns=[new_key], inplace=True)
        data = joint_df.to_dict(orient='records')
        repr_df = EntityReprDataset(
            data=data,
            max_length=500,
            max_ent_length=500)
        if method == 'average':
            repr_df.augment_with_avg_repr(model, layer=layer, batch_size=32, method='in_context', verbose=verbose)
        else:
            repr_df.augment_with_repr(model, layer=layer, batch_size=32, method=method, verbose=verbose)
        #back to pandas dataframe
        joint_df = pd.DataFrame(repr_df.data)
        # rename the columns
        joint_df.rename(columns={
            'representation': new_key,
        }, inplace=True)
        joint_df.drop(columns=['entity'], inplace=True)

    return joint_df

def filter_known_facts(joint_dataset):
    """Filter out known facts from the joint dataset
    """
    if type(joint_dataset) == pd.DataFrame: joint_dataset = joint_dataset.to_dict(orient='records')
    for itm in tqdm(joint_dataset):
        obj = itm['object']
        subj = itm['subject']
        prompt = obj.join(itm['text'].split(obj)[:-1]).strip()
        #infer max five tokens
        gen = model.generate(prompt, max_new_tokens=5, temperature=0, return_type="str", verbose=False)
        gen = gen.split(prompt)[1].strip()
        itm['generated'] = gen
        itm['is_correct'] = gen.startswith(obj)

    joint_dataset = pd.DataFrame(joint_dataset)

    ## filter out the correct ones
    known_dataset = joint_dataset[joint_dataset['is_correct']]
    print(f"Found {len(known_dataset)} correct samples on {len(joint_dataset)} samples in total")

    return known_dataset

def train_TaskVec(model, 
        TaskVec, 
        with_context:bool,
        train_dataset, 
        test_dataset, 
        epochs=10, 
        lr=1e-2, 
        batch_size=5, 
        log_per_epoch=2, 
        hist=None,
        prepend_bos=True, 
        first_token_only = False,
        verbose = True
        ):
    """ Train the TaskVec on the train_dataset and test on the test_dataset. 
    Args:
    - model: the model to use for training
    - TaskVec: the task vector to train
    - with_context: if True, the model has access to the context to generate the entity mention
    - train_dataset: the training dataset MUST HAVE ['representation', 'subject', 'entity', 'text'] keys
    """
    if hist and type(hist) == list: 
        last_epoch = hist[-1]["epoch"]
    else :
        hist = []
        last_epoch = 0

    # sanity check for datasets
    for key in ['representation', 'subject', 'entity', 'text']:
        if key not in train_dataset[0].keys():
            raise ValueError(f"train_dataset must have a '{key}' key")
        if key not in test_dataset[0].keys():
            raise ValueError(f"test_dataset must have a '{key}' key")
        
    dtype = model.W_U.dtype
    rep_idx = 1 if prepend_bos else 0 
    eos_tok = model.tokenizer.eos_token_id
    eos_tok_str = model.tokenizer.eos_token
    replace_hook_name = tl.utils.get_act_name('embed') #pos_embed for gpt2 ... 

    train_dataloader =   DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  =   DataLoader(test_dataset, batch_size=10, shuffle=True)
    len_loader = len(train_dataloader)
    n_log = len_loader // log_per_epoch
    optim = torch.optim.Adam([TaskVec], lr=lr)
    # optim = torch.optim.SGD([TaskVec], lr=lr)
    # criterion = nn.CrossEntropyLoss()
    
    #Cosine annealing scheduler
    # buffer to store best taskVec
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, len_loader//2, eta_min=lr/10)
    best_TaskVec = torch.zeros_like(TaskVec)

    gc.collect()
    torch.cuda.empty_cache()
    print(f"{'Resuming' if hist else 'Starting'} training TaskVec on {dataset_name} for layer {layer} of {model_name} with{'out' if not with_context else ''} context...")
    for epoch in range(epochs):
        epoch += last_epoch
        b_count = 0
        m_loss = 0
        for batch in tqdm(train_dataloader, disable=not verbose):
            # print( batch)
            b_count += 1
            entities = batch["entity"]
            texts = batch["text"]
            reps = batch["representation"].squeeze(1).cuda()
            b_size = reps.shape[0]
            b_taskVec = TaskVec.repeat(b_size,1).cuda()
            
            if first_token_only:
                entities = [toks[0] for toks in model.to_str_tokens(entities,prepend_bos=False)]
            else:    
                entities = [ent + eos_tok_str for ent in entities] # take whole label add eos token

            if with_context:
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

            # print(prompts, entities,subjects)
            # break
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
            # scheduler.step()
            
            if b_count % n_log == 0:
                m_loss = m_loss / (len_loader // log_per_epoch)
                e = epoch + b_count/len_loader
                acc = eval_model(
                                model,
                                TaskVec,
                                test_loader=test_dataloader, 
                                first_token_only=first_token_only, 
                                with_context=with_context,
                                prepend_bos=prepend_bos)
                lr = scheduler.get_last_lr()[0]
                print(f"Epoch {e:.1f},  Batch {b_count}/{len_loader}, Loss: {m_loss:.3f}, Test Acc: {acc:.3f}, lr:{lr:.3f}")
                
                #save best taskVec according to test accuracy
                if hist and acc >= max([h["test accuracy"] for h in hist]):
                    print(f" {acc:.3f} is the best accuracy so far, saving TaskVec ...")
                    best_TaskVec[:] = TaskVec[:]

                hist.append({"epoch":e, "loss": m_loss, "test accuracy": acc, "lr":lr})
                m_loss = 0
    
    #save the best taskVec
    print(f"Training done, best accuracy: {max([h['test accuracy'] for h in hist])}, saving best TaskVec ...")
    TaskVec = best_TaskVec.detach().cpu()
    return hist

################################################################### 
####################   PARAMS    ##########
################################################################### 

# MODEL
model_name = "phi-2" # 2,5B ok 12cpus nope, gpu 24cpus ok
# dataset on which the taskVec was trained
dataset_name= "CoNLL2003"
dtype = torch.float32
with_context = False

# Dataset 
json_file = "product_by_company.json"
json_file = "task_done_by_tool.json" 
json_file = "country_largest_city.json"
json_file = "person_mother.json"
json_file = "person_father.json"
json_file = "person_occupation.json"
json_file = "object_superclass.json"
json_file = "star_constellation.json" 
json_file = "person_native_language.json" 
json_file = "landmark_in_country.json"

# Number of training samples
N_train = 50

#Exctaction method
method = 'in_context'
method = 'average'

# Use the pretrained TaskVec as starting point for relation extraction taskVec
use_pretrained = True       #converges faster
use_pretrained = False

# OPTIONALLY Filter Subject representations
filter_subject_reps = False
filter_subject_reps = True

# Training Params
lr = 5e-3
batch_size = 5
epochs = 30
log_per_epoch = 2

################################################################### 
################################################################### 

data_name = json_file.split('.json')[0]
### Load previously tained taskVecs results
xp_path = pathlib.Path.home() / "experiments_JeanZay" # load results
xp_paths= os.listdir(xp_path / "jobs")      #get all experiments directories
jobs_path = xp_path / "jobs" / xp_paths[0]

print(f"Loading results...")
results = loadResults(jobs_path)            #get jobs
results = results[(results["model_name"]==model_name) &
                  (results["dataset_name"]==dataset_name) &
                  (results["with_context"]==with_context) ]
print(f"Loaded {len(results)} results")

#get jobs for linear filters
xp_lin = 'labelextractor.learnlinearfilter'
print(f"Loading linear filters results...")
results_linear = loadResults(xp_path / "jobs" / xp_lin)

#load Model
print(f"Loading {model_name} ...")
model = HookedTransformer.from_pretrained(
                                    model_name, 
                                    trust_remote_code = True, 
                                    low_cpu_mem_usage = True, 
                                    device_map='auto',
                                    move_to_device=False,
                                    fold_ln=False,
                                    fold_value_biases=False,
                                    center_writing_weights=False,
                                    center_unembed=False,
                                    dtype=dtype,
                                    )
dim = model.QK.shape[-1]
model.eval()
model = model.cuda()

#freeze the model 
for param in model.parameters():
    param.requires_grad = False


################################################################### 
###################### DATA ########################################
################################################################### 

# Load the JSON file
data_path = Path("datasets")


with open(data_path / json_file, 'r') as file:
    data = json.load(file)
samples = data['samples']
print(f"Loaded {len(samples)} samples")
print(samples[0])
prompts = [ prompt.replace("{}","{s}") + " {o}" for prompt in data['prompt_templates']]

joint_df = []
for it in samples:
    prompt = prompts[np.random.randint(0, len(prompts))]
    s = it['subject']
    o = it['object']
    joint_df.append({
        "text": prompt.format(s=s, o=o),
        "subject": s,
        "object": o,
        })

print(f"Created {len(joint_df)} samples")
joint_df = pd.DataFrame(joint_df)
# joint_df.head()

################################################################### 
# ## Filter items for which the model has no factual knowledge
known_df = filter_known_facts(joint_df)
# known_df = joint_df


#separate the df into train and test
test_size = (len(known_df) - N_train) / len(known_df)
train_df, test_df = train_test_split(known_df, test_size= test_size, random_state=42)

#test if some landmarks are in the test set
train_subjects = set([it for it in train_df['subject'] ])
test_subjects = set([it for it in test_df['subject'] ])

print("Train subjects: ", len(train_subjects))
print("Test subjects: ", len(test_subjects))
print("Common subjects: ", len(train_subjects.intersection(test_subjects)))

################################################################### 
###################### TRAINING ########################################
################################################################### 

perfs = []
### TRAINING FOR EACH LAYER
for layer in range(model.cfg.n_layers):
    print('\n' + f"   layer {layer}, method {method}   ".center(100,'#') + '\n')
    # Augment with representations 
    print(f"- Augmenting train set with method {method}...")
    train_df = augment_with_reprs(train_df, model, layer=layer, method=method, batch_size=32, verbose=False)
    print(f"- Augmenting test set with method {method}...")
    test_df = augment_with_reprs(test_df, model, layer=layer, method=method, batch_size=32, verbose=False)

    if filter_subject_reps:
        print("- Cleaning representation of objects")
        path = get_taskVec(results_linear, model_name, layer=layer, with_context=with_context, verbose=False)
        linear_filter = torch.load(path)
        print(f"\t- Loaded model from {path}")
        linear_filter = linear_filter.type(dtype).cuda() #cast the model to the correct type
        rep_cleaner = lambda z: linear_filter(z.cuda()).detach().cpu()

        with torch.no_grad():
            # Apply linear filter ONLY ON OBJECT representations (USELESS as we don't use the object representation here)
            ## we only infer the object mention directly from the subject, with a custom Task Vector
            # train_df['object_repr'] = train_df['object_repr'].apply(rep_cleaner)
            # test_df['object_repr'] = test_df['object_repr'].apply(rep_cleaner)
            #Apply linear filter on SUBJECT representations
            train_df['subject_repr'] = train_df['subject_repr'].apply(rep_cleaner)
            test_df['subject_repr'] = test_df['subject_repr'].apply(rep_cleaner)
    else:
        print("- Didn't clean representation of objects")

    entity_key = 'subject'
    target_key = 'object'
    repr_key = 'subject_repr'

    dtype = model.W_U.dtype
    hist = []

    # to list 
    train_dataset = train_df.to_dict(orient='records')
    test_dataset = test_df.to_dict(orient='records')

    #test if some landmarks are in the test set
    train_subjects = set([it for it in train_df['subject'] ])
    test_subjects = set([it for it in test_df['subject'] ])

    print("Train subjects: ", len(train_subjects))
    print("Test subjects: ", len(test_subjects))
    print("Common subjects: ", len(train_subjects.intersection(test_subjects)))
    
    #rename the keys
    train_dataset = [{
        "representation": it[repr_key],
        "subject": it[entity_key],
        "object": it[target_key], 
        "entity": it[target_key], 
        "text" : it['text'],
        "id" : i,
        } for i, it in enumerate(train_dataset)]

    test_dataset = [{
        "representation": it[repr_key],
        "subject": it[entity_key],
        "object": it[target_key], 
        "entity": it[target_key], 
        "text" : it['text'],
        "id" : i,
        } for i, it in enumerate(test_dataset)]



    # create Task Vector
    if use_pretrained:
        fileName = get_taskVec(results, 
                        model_name, 
                        layer=layer, 
                        dataset_name=dataset_name, 
                        with_context = with_context)
        TaskVec = torch.load(fileName)
        #cast to dtype and make it leaf tensor 
        TaskVec = TaskVec.to(dtype).detach().requires_grad_()
        print("TaskVec loaded from ", fileName)
    else:
        TaskVec = torch.normal(mean=0, std=1.0, size=(1,dim), requires_grad=True, dtype=dtype)
        print("Created new TaskVec")

    #train 
    hist = train_TaskVec(model, 
            TaskVec, 
            with_context=with_context,
            train_dataset=train_dataset, 
            test_dataset=test_dataset, 
            epochs=epochs, 
            lr=lr, 
            batch_size=batch_size, 
            log_per_epoch=log_per_epoch, 
            hist=hist,
            prepend_bos=True, 
            first_token_only = False
            )

    task = "RelationExtractor"
    fileName = f'{task}_{model_name}_l{layer}_e{hist[-1]["epoch"]}.pth'

    #plot the loss
    plt.clf()
    plt.plot([it['loss'] for it in hist])
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.yscale('log')
    #add athoer y axis for lr   
    plt.twinx()
    plt.ylabel("Accuracy")
    # plt.plot([it['lr'] for it in hist], color='r')
    plt.plot([it['test accuracy'] for it in hist], color='r')


    plt.grid()
    plt.title(f"Training loss for {model_name}")
    plt.savefig( Path('plots') / f"training_l{layer}_{method}.png")

    ################################################################################
    ################################# EVAL ########################
    ################################################################################

    
    metrics = compute_metrics(model, TaskVec,
                       test_dataset,
                       with_context=with_context, 
                       max_tokens=20, b_size=5)

    for item in test_dataset:
        item.pop('representation', None)

    #save metrics
    data = {
        "model_name": model_name,
        "layer": layer,
        "N_train": N_train,
        "dataset_name": dataset_name,
        'taskVec': str(fileName),
        "with_context": with_context,
        "method": method,
        "cleaned": filter_subject_reps,
        "subject_cleaned": filter_subject_reps,
        "metrics": metrics,
        "inference": test_dataset
    }
    print("Done training for layer", layer, metrics)
    
    perfs.append(data)

save_file = lambda i: f"RelExtraction_{data_name}_{model_name}_{method}_TV{'_cleaned' if filter_subject_reps else ''}_{i}.json"
save_path = lambda i: Path("scripts") / "RelExtractionResults" / save_file(i)
i = 0 
while os.path.exists(save_path(i)):
    i += 1
save_path = save_path(i)

with open(save_path, 'w') as file:
    json.dump(perfs, file)
print(f"Done ! Saved results in {save_path} !")


#print some examples for last training:
print(json_file)
print(f"{'Subject'.center(45)}|{'Object'.center(25)}|{'Generated'.center(25)}")
print("="*100)

for i in range(25):
    item = test_dataset[np.random.randint(len(test_dataset))]
    s = item['subject']
    o = item['object']
    print(f"{s.center(45)}|{item['object'].center(25)}|{item['inferred'].center(25)}")
    if o != item['inferred']:
        print("\t -> Fail ! ", item['text'])

