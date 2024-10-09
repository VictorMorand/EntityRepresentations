import logging
logging.basicConfig(level=logging.DEBUG)
logging.info("loading libs...")
import torch, gc, sys, os, pathlib, pickle
import torch.nn as nn
os.environ['HF_EVALUATE_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
import transformer_lens as tl 
from transformer_lens import HookedTransformer
from tqdm import tqdm

#our own code
import utils
from LabelExtractor import eval_model, compute_metrics, infer_entities
from processResults import *

# Load a model (eg GPT-2 Small)
model_name = "bloom-3b" #crashed
model_name = "meta-llama/Meta-Llama-3-8B" # ?
model_name = "gpt2-small" # 117M ok
model_name = "gpt2-xl" # 1.5B ok
model_name = "gpt2-large" # 774M ok
model_name = "mistralai/Mistral-7B-v0.1" # ok JZ a100 8cpus | 
model_name = "gpt2-medium" # 302M ok
model_name = "phi-3" # 3,6B -> OK ?
model_name = "phi-2" # 2,5B ok
model_name = "phi-1_5"  # 1.5B ok
model_name = "EleutherAI/pythia-6.9b"#? CRASHED
model_name = "EleutherAI/pythia-410m"# ok !
model_name = "EleutherAI/pythia-2.8b"#ok !
model_name = "EleutherAI/pythia-1b"# ok ! (6Gb)

dataset_name= "TACRED"
dataset_name= "WebNLG"
dataset_name= "CoNLL2003"

layer = 10

with_context = True
with_context = False

dtype = torch.bfloat16
dtype = torch.float32

# load results
xp_path = pathlib.Path(os.environ.get("WORK")) / "experiments"
#get all experiments directories
xp_paths= os.listdir(xp_path / "jobs")
xp = [xps for xps in xp_paths if "learn" in xps][0]
jobs_path = xp_path / "jobs" / xp
logging.info(f"will look for jobs into :{jobs_path}")

#get jobs
results = loadResults(jobs_path)
logging.info(results.describe())


#filter results by model
results = results[(results["model_name"]==model_name) &
                  (results["dataset_name"]==dataset_name) &
                  (results["with_context"]==with_context) ]

logging.info(f"Loaded {len(results)} results")

model = HookedTransformer.from_pretrained(
                                    model_name,
                                    trust_remote_code=True, 
                                    low_cpu_mem_usage = True, 
                                    fold_ln=False,
                                    fold_value_biases=False,
                                    device_map='auto', 
                                    local_files_only=True, 
                                    )
model.eval()
model = model.cuda()

#load Taskvec
fileName = get_taskVec(results, model_name, layer=layer, dataset_name=dataset_name, with_context = with_context)
TaskVec = torch.load(fileName)
logging.info(f"TaskVec loaded from {fileName}")

##DATSET 
ds = load_dataset("eriktks/conll2003", trust_remote_code=True)
dataset_name = "CoNLL2003"
logging.info(f"train : {len(ds['train'])}")
logging.info(f"test : {len(ds['test'])}")
logging.info(f"validation : {len(ds['validation'])}")

# from utils import CoNLLDataset
max_ent_length = 60
max_length = 300
train_dataset = utils.CoNLLDataset(ds["train"], max_ent_length=max_ent_length,max_length=max_length)
test_dataset = utils.CoNLLDataset(ds["test"], max_ent_length=max_ent_length,max_length=max_length)
val_dataset = utils.CoNLLDataset(ds["validation"], max_ent_length=max_ent_length,max_length=max_length)
val_dataset.data = val_dataset.data[:1000] # limit validation set to 1000 samples

def unique_entities(dataset):
    entities = set()
    entities.update([d["entity"] for d in dataset])
    return [{"entity": ent} for ent in entities]

#ex sample from final dataset
logging.info(f"raw train length:{len(train_dataset)}")
logging.info(f"raw test length:{len(test_dataset)}")

#transform to unique entities
train_dataset = unique_entities(train_dataset)
test_dataset = unique_entities(test_dataset)
val_dataset = unique_entities(val_dataset)

#ex sample from final dataset
logging.info("\nAfter filtering:")
logging.info(f"train length: {len(train_dataset)}")
logging.info(f"test length: {len(test_dataset)}")
logging.info(f"ex sample: {train_dataset[np.random.randint(len(train_dataset))]}")


def train_reps(
            data:list,
            with_context:bool = False,
            N_trials:int = 40,
            l2_reg = 0.0,
            prepend_bos:bool = True,
            lr:float = 2e-1,
            N_steps:int = 150,
            log_every_N:int = 20,
          ):
    """
    Train `N_trials` representations from gaussian Noise to regenerate `entities` with the model and the given taskVec.
    """
    eos_tok_str = model.tokenizer.eos_token
    replace_hook_name = tl.utils.get_act_name('embed') #pos_embed for gpt2 ... 
    num_reps = len(data)
    dim = TaskVec.shape[-1]
    hist = []

    #instanciate all tensors 
    reps = torch.normal(mean=0, std=1.0, 
                        size=(num_reps*N_trials,dim), 
                        requires_grad=True, 
                        dtype=dtype, 
                        device="cuda")
    b_size = reps.shape[0]
    b_taskVec = TaskVec.repeat(b_size,1).cuda()

    # optimizer
    optim = torch.optim.Adam([reps], lr=lr)
    # optim = torch.optim.SGD([TaskVec], lr=lr)

    gc.collect()
    torch.cuda.empty_cache()

    # logging.info(f"Training reps for entities: {[d['entity'] for d in data]}, with {N_trials} trials")
    entities = []
    for d in data: entities += [d["entity"]] * N_trials
    entities = [ent + eos_tok_str for ent in entities] # take whole label add eos token

    if with_context:
        texts = []
        for d in data: texts += [d["text"]] * N_trials
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

    # logging.info(prompts)
    # logging.info(f"{'Resuming' if hist else 'Starting'} training TaskVec on {dataset_name} for layer {layer} of {model_name} with{'out' if not with_context else ''} context...")
    for step in range(N_steps):
            
        logits = model.run_with_hooks(
                inputs,
                return_type = "logits",
                fwd_hooks=[
                    (replace_hook_name, utils.get_replace_with_rep_hook(reps, rep_idxs)), # replace '_' by the subject Representation
                    (replace_hook_name, utils.get_replace_with_rep_hook(b_taskVec, taskVec_idxs)) # replace 'called' by TaskVec Representation
                    ]
            ,)

        # LM Loss and optimization # take only the loss on the entity tokens
        loss = model.loss_fn(logits[:, rep_idx+1:,:], targets)
        loss.backward()
        optim.step()
        optim.zero_grad()
        m_loss = loss.item()
        # scheduler.step()

        hist.append({"step":step, "loss": m_loss, "lr":lr})

    # logging.info(f"Step {step:.1f}, Loss: {m_loss:.5f}")
    reps = reps.detach().cpu()
    return reps, hist

N0 = 2 * len(train_dataset)//4 
N = len(train_dataset)//2 # number of entities to train
num_reps = 4
N_trials = 30
lr = 1
N_steps = 60
log_every_N = 100
prepend_bos = True

dim = model.QK.shape[-1]
dtype = model.W_U.dtype

for param in model.parameters():
    param.requires_grad = False
TaskVec.requires_grad = False

# split of whole dataset
train_data = train_dataset[N0:N0+N]

# load train data by batch of num_reps
for i in (pbar := tqdm(range(0, len(train_data), num_reps))):
    data = train_data[i:i+num_reps]
    # logging.info(f"\t[{i}/{len(train_data)}] -------------------")
    reps, hist = train_reps( 
                data,
                with_context=False, 
                N_trials=N_trials, 
                prepend_bos=prepend_bos, 
                lr=lr, 
                N_steps=N_steps, 
                # l2_reg=1e-4,
                log_every_N=log_every_N, 
                )
    last_loss = hist[-1]["loss"]
    pbar.set_description(f"LM loss: {last_loss:.3f}")

    for j, d in enumerate(data):
        reps_ent = reps[N_trials*j: N_trials*(j+1), :]
        d["representation"] = reps_ent.mean(dim=0)
        d["hist"] = hist

with open(f"Cleared_reps_{model_name.split('/')[-1]}_{dataset_name}_{N0}-{N0+N}_layer{layer}_avg{N_trials}.pkl", "wb") as f:
    pickle.dump(train_data, f)