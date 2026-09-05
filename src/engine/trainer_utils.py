import os, torch
import cv2
import numpy as np
# import torch_fidelity
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import importlib
from torch.optim import AdamW
from src.utils.lr_scheduler import build_scheduler


def get_obj_from_str(string, reload=False):
    """Get object from string path."""
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    """Instantiate an object from a config dictionary."""
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def is_dist_avail_and_initialized():
    """Check if distributed training is available and initialized."""
    if not torch.distributed.is_initialized():
        return False
    return True


def is_main_process():
    """Check if the current process is the main process."""
    return not is_dist_avail_and_initialized() or torch.distributed.get_rank() == 0


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def requires_grad(model, flag=True):
    """Set requires_grad flag for all model parameters."""
    for p in model.parameters():
        p.requires_grad = flag


def save_img(img, save_path):
    """Save a single image to disk."""
    img = np.clip(img.float().numpy().transpose([1, 2, 0]) * 255, 0, 255)
    img = img.astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(save_path, img)


def save_img_batch(imgs, save_paths):
    """Process and save multiple images at once using a thread pool."""
    # Convert to numpy and prepare all images in one go
    imgs = np.clip(imgs.float().numpy().transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)
    imgs = imgs[:, :, :, ::-1]  # RGB to BGR for all images at once
    
    with ThreadPoolExecutor(max_workers=32) as pool:
        # Submit all tasks at once
        futures = [pool.submit(cv2.imwrite, path, img) 
                  for path, img in zip(save_paths, imgs)]
        # Wait for all tasks to complete
        for future in futures:
            future.result()  # This will raise any exceptions that occurred


def get_fid_stats(real_dir, rec_dir, fid_stats):
    """Calculate FID statistics between real and reconstructed images."""
    stats = torch_fidelity.calculate_metrics(
        input1=rec_dir,
        input2=real_dir,
        fid_statistics_file=fid_stats,
        cuda=True,
        isc=True,
        fid=True,
        kid=False,
        prc=False,
        verbose=False,
    )
    return stats


def create_scheduler(optimizer, num_epoch, steps_per_epoch, lr_min, warmup_steps, 
                    warmup_lr_init, decay_steps, cosine_lr):
    """Create a learning rate scheduler."""
    scheduler = build_scheduler(
        optimizer,
        num_epoch,
        steps_per_epoch,
        lr_min,
        warmup_steps,
        warmup_lr_init,
        decay_steps,
        cosine_lr,
    )
    return scheduler


def load_state_dict(state_dict, model):
    """Helper to load a state dict with proper prefix handling."""
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    # Remove '_orig_mod' prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(
        state_dict, strict=False
    )
    if is_main_process():
        print(f"Loaded model. Missing: {missing}, Unexpected: {unexpected}")


def load_safetensors(path, model):
    """Helper to load a safetensors checkpoint."""
    from safetensors.torch import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        state_dict = {k: f.get_tensor(k) for k in f.keys()}
    load_state_dict(state_dict, model)


def setup_result_folders(result_folder):
    """Setup result folders for saving models and images."""
    model_saved_dir = os.path.join(result_folder, "models")
    os.makedirs(model_saved_dir, exist_ok=True)

    image_saved_dir = os.path.join(result_folder, "images")
    os.makedirs(image_saved_dir, exist_ok=True)
    
    return model_saved_dir, image_saved_dir


def create_optimizer(model, weight_decay, learning_rate, betas=(0.9, 0.95)):
    """Create an AdamW optimizer with weight decay for 2D parameters only."""
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    if is_main_process():
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer


class EMAModel:
    """Model Exponential Moving Average."""
    def __init__(self, model, device, decay=0.999):
        self.device = device
        self.decay = decay
        self.ema_params = OrderedDict(
            (name, param.clone().detach().to(device))
            for name, param in model.named_parameters()
            if param.requires_grad
        )

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in self.ema_params:
                    self.ema_params[name].lerp_(param.data, 1 - self.decay)
                else:
                    self.ema_params[name] = param.data.clone().detach()

    def state_dict(self):
        return self.ema_params

    def load_state_dict(self, params):
        self.ema_params = OrderedDict(
            (name, param.clone().detach().to(self.device))
            for name, param in params.items()
        )


class PaddedDataset(torch.utils.data.Dataset):
    """Dataset wrapper that pads a dataset to ensure even distribution across processes."""
    def __init__(self, dataset, padding_size):
        self.dataset = dataset
        self.padding_size = padding_size
        
    def __len__(self):
        return len(self.dataset) + self.padding_size
        
    def __getitem__(self, idx):
        if idx < len(self.dataset):
            return self.dataset[idx]
        return self.dataset[0]

class CacheDataLoader:
    """DataLoader-like interface for cached data with epoch-based shuffling."""
    def __init__(self, slots, targets=None, batch_size=32, num_augs=1, seed=None):
        self.slots = slots
        self.targets = targets
        self.batch_size = batch_size
        self.num_augs = num_augs
        self.seed = seed
        self.epoch = 0
        self.num_samples = len(slots) // num_augs
    
    def set_epoch(self, epoch):
        """Set epoch for deterministic shuffling."""
        self.epoch = epoch
    
    def __len__(self):
        """Return number of batches based on original dataset size."""
        return self.num_samples // self.batch_size
    
    def __iter__(self):
        """Return random indices for current epoch."""
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch if self.seed is not None else self.epoch)
        
        # Randomly sample indices from the entire augmented dataset
        indices = torch.randint(
            0, len(self.slots), 
            (self.num_samples,), 
            generator=g
        ).numpy()
        
        # Yield batches of indices
        for start in range(0, self.num_samples, self.batch_size):
            end = min(start + self.batch_size, self.num_samples)
            batch_indices = indices[start:end]
            yield (
                torch.from_numpy(self.slots[batch_indices]),
                torch.from_numpy(self.targets[batch_indices])
            )
            