
import torch, os, sys, pathlib, json
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

################################################################### 
####################   PARAMS    ##########
################################################################### 

#dataset 
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

data_name = json_file.split('.json')[0]
#number 
N_train = 50

#Exctaction method
method = 'in_context'
method = 'average'

# OPTIONALLY Filter Object representations
filter_reps = True   
filter_reps = False

filter_subject_reps = True   

#model Name
model_name = "phi-2" # 2,5B ok 12cpus nope, gpu 24cpus ok
# dataset on which the taskVec was trained
dataset_name= "CoNLL2003"
dtype = torch.float32
with_context = False

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

    if filter_reps:
        print("- Cleaning representation of objects")
        path = get_taskVec(results_linear, model_name, layer=layer, with_context=with_context, verbose=False)
        linear_filter = torch.load(path)
        print(f"\t- Loaded model from {path}")
        linear_filter = linear_filter.type(dtype).cuda() #cast the model to the correct type
        rep_cleaner = lambda z: linear_filter(z.cuda()).detach().cpu()

        with torch.no_grad():
            #Apply linear filter ONLY ON OBJECT representations
            train_df['object_repr'] = train_df['object_repr'].apply(rep_cleaner)
            test_df['object_repr'] = test_df['object_repr'].apply(rep_cleaner)
            
            if filter_subject_reps:
                #Apply linear filter on SUBJECT representations
                train_df['subject_repr'] = train_df['subject_repr'].apply(rep_cleaner)
                test_df['subject_repr'] = test_df['subject_repr'].apply(rep_cleaner)
    else:
        print("- Didn't clean representation of objects")

    train_dataset = train_df.to_dict(orient='records')
    test_dataset = test_df.to_dict(orient='records')

    linear_model = torch.nn.Linear(dim, dim, bias=True)
    linear_model.weight.data = torch.eye(dim).type(dtype)       # we initialize the linear model with identity
    linear_model.bias.data = torch.zeros(dim, dtype=dtype)
    linear_model = linear_model.cuda()
    
    hist = []
    lr = 1e-1
    epochs = 4000
    batch_per_epoch = 2
    batch_size = min(N_train // batch_per_epoch, 50)
    log_every = 1000
    grad_clip = 1

    # optimizer
    # optim = torch.optim.Adam(linear_model.parameters(), lr=lr)
    optim = torch.optim.SGD(linear_model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 
                                                step_size = epochs * batch_per_epoch // 3,
                                                gamma = 0.5)

    #dataloader from datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Training on {len(train_loader)} batches of size {batch_size}")
    # print(train_loader.dataset[0])

    # TRAIN LOOP
    for epoch in range(epochs):
        for batch in train_loader:
            # Zero the gradients
            optim.zero_grad()
            
            subjects_reps = batch['subject_repr'].cuda()
            objects_reps = batch['object_repr'].cuda()
            y = torch.ones(objects_reps.shape[0], device="cuda")
            # print(subjects_reps.shape, objects_reps.shape)
            # Project augmented_batch representations using matrix P
            projected = linear_model(subjects_reps)
            
            # Compute the loss between projected_batch and objects_reps
            # loss = criterion(projected, objects_reps, y)
            loss = criterion(projected, objects_reps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(linear_model.parameters(), grad_clip)

            optim.step()
            scheduler.step()

            # Log the loss
            hist.append(
                {
                    "epoch": epoch,
                    "batch": len(hist),
                    "loss": loss.item(),
                    "lr": optim.param_groups[0]['lr']
                }
            )
        if epoch % log_every == 0:
            print(f"Epoch: {epoch+1}, Batch: {len(hist)}, Loss: {loss.item()}, lr: {optim.param_groups[0]['lr']}")

    #plot the loss
    plt.clf()
    plt.plot([it['loss'] for it in hist])
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.yscale('log')
    #add athoer y axis for lr   
    plt.twinx()
    plt.ylabel("Learning rate")
    plt.plot([it['lr'] for it in hist], color='r')

    plt.grid()
    plt.title(f"Training loss for {model_name}")
    plt.savefig( Path('plots') / f"training_l{layer}_{method}.png")

    #### EVAL 
    fileName = str(get_taskVec(results, 
                        model_name, 
                        layer=layer, 
                        dataset_name=dataset_name, 
                        with_context = with_context))

    TaskVec = torch.load(fileName)
    #cast to dtype
    TaskVec = TaskVec.to(dtype)
    print("TaskVec loaded from ", fileName)

    with torch.no_grad():
        obj_from_subj = [
            {   'entity': item['object'],
                'subject': item['subject'],
                'object': item['object'],
                'representation': linear_model(item['subject_repr'].cuda()).detach().cpu(),
                'text': item['text'],
                'id': i,
            } for i, item in enumerate(test_dataset)#+ train_dataset)
            ]

    metrics = compute_metrics(model, 
                            TaskVec, 
                            obj_from_subj,
                            with_context=with_context,
                            b_size=64)

    for item in obj_from_subj:
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
        "cleaned": filter_reps,
        "subject_cleaned": filter_subject_reps,
        "metrics": metrics,
        "inference": obj_from_subj
    }
    print("Done training for layer", layer, metrics)
    
    perfs.append(data)


save_path = lambda i: f"./RelExtraction_{data_name}_{model_name}_{method}{'_cleaned' if filter_reps else ''}_{i}.json"
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
    item = obj_from_subj[np.random.randint(len(obj_from_subj))]
    s = item['subject']
    o = item['object']
    print(f"{s.center(45)}|{item['object'].center(25)}|{item['inferred'].center(25)}")
    if o != item['inferred']:
        print("\t -> Fail ! ", item['text'])

