import torch, logging
import torch.nn as nn
import torch.nn.functional as F
import transformer_lens as tl
from transformer_lens import HookedTransformer
from tqdm import tqdm
import numpy as np
from jaxtyping import Float
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import compute_to_layer
import matplotlib.pyplot as plt #maybe separate this ? 

######################  MODEL DEFINITION ###################### 

class SelfAttention(nn.Module):
    def __init__(self, dim: int, k: int = 0, scale: float = None, init_identity: bool = False):
        """One head of self attention mechanism, with custom query and key rank
        Args: 
            dim: dimension of the representations
            k: rank of the query and key matrices, if 0, will be equal to dim
            scale: scaling factor for the scores, if None, will be set to 1/sqrt(dim)
            init_identity: whether to initialize query and key matrices as identity matrices    
        """
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.scale = scale if scale else 1 / np.sqrt(dim)
        self.k = k if k else dim

        if init_identity:
            self.Q = nn.Parameter(torch.eye(self.dim, self.k))
            self.K = nn.Parameter(torch.eye(self.dim, self.k))
        else:
            self.Q = nn.Parameter(torch.randn(self.dim, self.k))
            self.K = nn.Parameter(torch.randn(self.dim, self.k))

    def scores(self, query: Float[torch.Tensor, "batch seq dim"], key: Float[torch.Tensor, "batch seq dim"]) -> Float[torch.Tensor, "batch seq"]:
        """Compute match between two batches of representations q and k,
        M(q, k) = q Q K^T k^T
        Args:
            query: tensor (batch, seq, dim) batch of query representations
            key: tensor (batch, seq, dim) batch of key representations
        """
        return self.scale * torch.einsum('...bd,...bd->...b', (query @ self.Q), (key @ self.K))
    
    def forward(self, reps: Float[torch.Tensor, "batch seq dim"], causal_mask = True, apply_softmax:bool = True) -> Float[torch.Tensor, "batch seq seq"]:
        """Compute self attention for given set of representations or hidden states
        Args:
            reps: tensor (batch, seq, dim) batches of representations 
        Returns:
            scores: tensor (batch, seq, seq) attention scores between all pairs of tokens
        """
        #prepare all pairs of representations
        seq = reps.size(1)
        query = reps.repeat(1, seq, 1)
        key = reps.repeat_interleave(seq, dim=-2)
        
        # print(query)
        # print(key)
        #compute all scores 
        scores = self.scores(query, key)
        # print(scores)
        
        #reshape 
        scores = scores.reshape(-1, seq, seq)
        
        #causal mask
        if causal_mask:
            mask = torch.triu(torch.ones(seq, seq), diagonal=1).bool().to(scores.device)
            scores.masked_fill_(mask, -1e5)

        # apply softmax
        if apply_softmax: scores = F.softmax(scores, dim=-1)
        
        return scores


def Cross_Attn(model, tokens, layer, attn: SelfAttention, apply_softmax:bool = True):
    """Compute cross attention between representations of tokens at given layer
    Args:
        model: HookedTransformer form TransformerLens to extract representations from
        text: text to compute attention on
        layer: layer at which to retreive the representations
        attn: attention model to use, should be a nn.Module with forward method
        apply_softmax: whether to apply softmax to the scores, if False, scores will be returned as is for further processing
    """
    hook_name = tl.utils.get_act_name('resid_post', layer=layer)   # get hook name for the layer output 
    # print(len(tokens))
    _ , cache = model.run_with_cache(tokens)
    reps = cache[hook_name] #shape (batch, len, dim)
    
    #compute all scores 
    scores = attn(reps, apply_softmax=apply_softmax)
    return scores

######################  TRAINING ###################### 

@torch.no_grad()
def validate_attn(model: HookedTransformer, layer:int, attn: nn.Module, val_loader, verbose:bool = False):
    """Run validation metric on given loader"""
    batch_size = val_loader.dataset.batch_size
    criterion = nn.BCEWithLogitsLoss(
                    pos_weight = torch.tensor(1), #no positive bias in validation
                    reduction="mean",
                    )
    val_loss = 0

    for batch in tqdm(val_loader, disable = not verbose):
        texts = batch["text"]
        tokens = model.to_tokens(texts, padding_side='right', move_to_device=True)
        attn_patterns = [ pattern.cuda() for pattern in  batch["pattern"]]
        
        #we batch the forward pass of representations and attention scores 
        reps = compute_to_layer(model, layer, tokens).cuda() # shape (batch, seq, dim)
        scores = attn(reps,     # we compute all attention scores even if we only use the diagonal, and the pad tokens ... 
                    apply_softmax=False,
                    )
        
        for j, pattern in enumerate(attn_patterns):
            seq = pattern.size(-1)
            val_loss += criterion(scores[j,:seq,:seq].unsqueeze(0), pattern).item()
        
    val_loss /= (len(val_loader) * batch_size)
    
    if verbose: print(f"Validation loss: {val_loss}")
    return val_loss

def train_attn(
    model: HookedTransformer,
    layer:int,
    attn: nn.Module,
    train_loader,
    val_loader,
    hist = [],
    epochs = 2,
    lr = 1e-4,
    grad_clip = 1.,
    pos_weight = 3,
    accumulation_steps = 1,
    n_log = 100):
    """Train attention model on a dataset
    Args:
        model: HookedTransformer form TransformerLens to extract representations from
        attn: attention model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: number of epochs to train
        lr: learning rate
        grad_clip (None, float): gradient clipping value, if None, no clipping
        pos_weight: weight for positive examples in BCE loss
        accumulation_steps: number of steps to accumulate gradients before updating weights
        batch_size: batch size for grouped LLM and self attn inference
        n_log: number of steps between validation and logging
    """
    batch_size = train_loader.dataset.batch_size #batch size is stored in the dataset, loader batch size is 1
    optimizer = torch.optim.Adam(attn.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(attn.parameters(), lr=lr)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss(
                    pos_weight = torch.tensor(pos_weight),
                    reduction="mean",
                    )

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    #special loader for batch size 1, batching is done in the batched dataset wrapper
    logging.info(f"Training Self Attention layer for {epochs} epochs with batch size {batch_size}... ")
    if len(hist):
        prev_samples = hist[-1]["samples"]
        prev_epoch = hist[-1]["epoch"]
    else:
        prev_samples = 0
        prev_epoch = 0

    #train loop
    i = 0 #step counter
    for epoch in range(prev_epoch, prev_epoch + epochs):
        optimizer.zero_grad()
        for batch in tqdm(train_loader):
            i += 1
            # print(tokens)
            # print(texts)
            texts = batch["text"]
            tokens = model.to_tokens(texts, padding_side='right', move_to_device=True)
            attn_patterns = [ pattern.cuda() for pattern in  batch["pattern"]]  # attn_patterns is a list of batch tensor of shape (1, seq, seq)
            
            #we batch the computation of representations and attention scores 
            reps = compute_to_layer(model, layer, tokens).cuda() # shape (batch, seq, dim)
            # print(reps.shape)

            #forward pass, we compute all attention scores even if we only use the diagonal, and the pad tokens ... 
            scores = attn(reps,
                        # causal_mask=False, 
                        apply_softmax=False,
                        )
            
            # print(attn_patterns.shape)
            loss = 0
            # print(scores.shape)
            for j, pattern in enumerate(attn_patterns):
                seq = pattern.size(-1)
                loss += criterion(scores[j,:seq,:seq].unsqueeze(0), pattern)
                
            loss /= batch_size * accumulation_steps
            loss.backward() #backward pass

            # Accumulate gradients and update weights every n_log steps
            if (i + 1) % accumulation_steps == 0:
                if grad_clip: nn.utils.clip_grad_norm_(attn.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                
            h = {
                "epoch": epoch,
                "samples": i*batch_size + prev_samples,
                "loss": loss.item() * batch_size,
                "lr": lr,
            }
            hist.append(h)

            if len(hist) - 1 % n_log == 0:
                #validate and log
                val_loss = validate_attn(model, layer, attn, val_loader, verbose=False)
                hist[-1]["val_loss"] = val_loss
                #scheduler step
                scheduler.step(val_loss)
                print(f"Epoch {epoch}, Step {i} mean loss: {np.mean([h['loss'] for h in hist[-n_log:]]):.4f}, val loss: {val_loss:.4f}, lr: {lr:.2e}")
            
        # Update weights at the end of each epoch if not already updated
        if (i + 1) % n_log != 0:
            optimizer.step()
            optimizer.zero_grad()

    return hist


######################  Plotting ###################### 

def plot_hist(hist, n_smooth = 20):
    """Plot history of training"""
    #compute smoothed loss

    keys = ["loss", "val_loss", "lr", "smooth_loss"]
    plots = {}
    for h in hist :
        for key in keys:
            if key in h:
                plots[key] = plots.get(key, [])
                plots[key].append( (h["samples"], h[key]) )
    
    #smooothing 'loss' and add in hist 
    plots["smooth_loss"] = [ (hist[i]["samples"], np.mean([h["loss"] for h in hist[i:i+n_smooth]])) for i in range(len(hist)-n_smooth)]

    # loss_smooth = np.convolve(loss, np.ones(n_smooth)/n_smooth, mode='same')

    plt.figure(figsize=(10,5))
    for key in ["smooth_loss", "val_loss"]:
        plt.plot(*zip(*plots[key]), label=key)
    #separate axis for learning rate
    plt.yscale("log")
    plt.ylabel("BCE Loss")
    plt.twinx()
    plt.plot(*zip(*plots["lr"]),"g--", label="lr",)
    plt.ylabel("Learning rate")
    plt.yscale("log")
    plt.xlabel("Step")
    #log scale for better visualization
    plt.legend()
    plt.show()

