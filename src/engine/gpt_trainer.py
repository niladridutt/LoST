import os, torch
import os.path as osp
import shutil
import numpy as np
import copy
import torch.nn as nn
from tqdm.auto import tqdm
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, DistributedSampler
from src.utils.logger import SmoothedValue, MetricLogger, empty_cache
from accelerate.utils import DistributedDataParallelKwargs
from src.stage2.gpt import GPT_models
from src.stage2.generate import generate
from pathlib import Path
import time
import torch.nn.functional as F
import wandb

from src.engine.trainer_utils import (
    instantiate_from_config, concat_all_gather,
    save_img_batch, get_fid_stats,
    EMAModel, create_scheduler, load_state_dict, load_safetensors,
    setup_result_folders, create_optimizer,
    CacheDataLoader
)

class GPTTrainer(nn.Module):
    def __init__(
        self,
        ae_model,
        gpt_model,
        dataset,
        test_dataset=None,
        test_only=False,
        num_test_images=50000,
        num_epoch=400,
        blr=1e-4,
        cosine_lr=False,
        lr_min=0,
        warmup_epochs=100,
        warmup_steps=None,
        warmup_lr_init=0,
        decay_steps=None,
        batch_size=32,
        eval_bs=32,
        cache_bs=8,
        test_bs=100,
        num_workers=8,
        pin_memory=False,
        max_grad_norm=None,
        grad_accum_steps=1,
        precision='bf16',
        save_every=10000,
        sample_every=1000,
        fid_every=50000,
        result_folder=None,
        log_dir="./log",
        ae_cfg=1.0,
        cfg=6.0,
        cfg_schedule="linear",
        temperature=0.0,
        train_num_slots=None,
        test_num_slots=None,
        eval_fid=False,
        fid_stats=None,
        enable_ema=False,
        compile=False,
        enable_cache_latents=False,
        cache_dir='/dev/shm/slot_cache'
    ):
        super().__init__()

        enable_cache_latents=False

        temperature = 0

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="wandb",
            project_dir=log_dir,
        )

        run_name = f"gpt_clip_{train_num_slots}"

        self.run_name = run_name
        project_name = f'semanticist3d-gpt'

        self.accelerator.init_trackers(
            project_name=project_name,
            init_kwargs={"wandb": {"name": run_name}}
        )


        self.model_name = gpt_model.target
        self.num_slots = gpt_model.params.num_slots
        self.slot_dim = gpt_model.params.slot_dim
        
        if 'GPT' in gpt_model.target:
            self.gpt_model = GPT_models[gpt_model.target](**gpt_model.params)
        else:
            raise ValueError(f"Unknown model type: {gpt_model.target}")
        # self.num_slots = 128 #ae_model.params.num_slots
        # self.slot_dim = 32 #ae_model.params.slot_dim

        self.test_only = test_only
        self.test_bs = test_bs
        self.eval_bs = eval_bs
        self.num_test_images = num_test_images
        self.num_classes = gpt_model.params.num_classes
        self.batch_size = batch_size
        if not test_only:
            self.train_ds = instantiate_from_config(dataset)
            test_dataset = instantiate_from_config(test_dataset)

            train_size = len(self.train_ds)
            self.valid_ds = test_dataset
            valid_size = len(test_dataset)

            if self.accelerator.is_main_process:
                print(f"train dataset size: {train_size}")

            sampler = DistributedSampler(
                self.train_ds,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
            )
            self.train_dl = DataLoader(
                self.train_ds,
                batch_size=batch_size if not enable_cache_latents else cache_bs,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )

            valid_sampler = DistributedSampler(
                self.valid_ds,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=False,  # No need to shuffle validation data
            )

            eval_bs = valid_size // 8
            self.valid_dl = DataLoader(
                self.valid_ds,
                batch_size=eval_bs,
                sampler=valid_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

            effective_bs = batch_size * grad_accum_steps * self.accelerator.num_processes
            lr = blr * effective_bs / 256
            if self.accelerator.is_main_process:
                print(f"Effective batch size is {effective_bs}")
                print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")

            self.g_optim = create_optimizer(self.gpt_model, weight_decay=0.05, learning_rate=lr)
            
            if warmup_epochs is not None:
                warmup_steps = warmup_epochs * len(self.train_dl)
                
            self.g_sched = create_scheduler(
                self.g_optim,
                num_epoch,
                len(self.train_dl),
                lr_min,
                warmup_steps,
                warmup_lr_init,
                decay_steps,
                cosine_lr
            )
            self.accelerator.register_for_checkpointing(self.g_sched)
            self.gpt_model, self.g_optim, self.g_sched = self.accelerator.prepare(self.gpt_model, self.g_optim, self.g_sched)
        else:
            self.gpt_model = self.accelerator.prepare(self.gpt_model)

        self.steps = 0
        self.loaded_steps = -1


        self.enable_ema = enable_ema
        if self.enable_ema and not self.test_only: 
            self.ema_model = EMAModel(self.accelerator.unwrap_model(self.gpt_model), self.device)
            self.accelerator.register_for_checkpointing(self.ema_model)

        self._load_checkpoint(gpt_model.params.ckpt_path)
        if self.test_only:
            self.steps = self.loaded_steps

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.sample_every = sample_every
        self.fid_every = fid_every
        self.max_grad_norm = max_grad_norm
        self.cfg = cfg
        self.ae_cfg = ae_cfg
        self.cfg_schedule = cfg_schedule
        self.temperature = temperature
        self.train_num_slots = train_num_slots
        self.test_num_slots = test_num_slots
        if self.train_num_slots is not None:
            self.train_num_slots = min(self.train_num_slots, self.num_slots)
        else:
            self.train_num_slots = self.num_slots
        if self.test_num_slots is not None:
            self.num_slots_to_gen = min(self.test_num_slots, self.train_num_slots)
        else:
            self.num_slots_to_gen = self.train_num_slots
        self.eval_fid = eval_fid
        if eval_fid:
            assert fid_stats is not None
        self.fid_stats = fid_stats

        self.result_folder = result_folder
        self.model_saved_dir, self.image_saved_dir = setup_result_folders(result_folder)

        self.cache_dir = Path(cache_dir)
        self.enable_cache_latents = enable_cache_latents
        self.cache_loader = None

    @property
    def device(self):
        return self.accelerator.device

    def _load_checkpoint(self, ckpt_path=None, model=None):
        if ckpt_path is None or not osp.exists(ckpt_path):
            return

        if model is None:
            model = self.accelerator.unwrap_model(self.gpt_model)

        if osp.isdir(ckpt_path):
            self.loaded_steps = int(
                ckpt_path.split("step")[-1].split("/")[0]
            )
            if not self.test_only:
                self.accelerator.load_state(ckpt_path)
            else:
                if self.enable_ema:
                    model_path = osp.join(ckpt_path, "custom_checkpoint_1.pkl")
                    if osp.exists(model_path):
                        state_dict = torch.load(model_path, map_location="cpu")
                        load_state_dict(state_dict, model)
                        if self.accelerator.is_main_process:
                            print(f"Loaded ema model from {model_path}")
                else:
                    model_path = osp.join(ckpt_path, "model.safetensors")
                    if osp.exists(model_path):
                        load_safetensors(model_path, model)
        else:
            if ckpt_path.endswith(".safetensors"):
                load_safetensors(ckpt_path, model)
            else:
                state_dict = torch.load(ckpt_path, map_location="cpu")
                load_state_dict(state_dict, model)
        if self.accelerator.is_main_process:
            print(f"Loaded checkpoint from {ckpt_path}")

    def _build_cache(self):
        """Build cache for slots and targets."""
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes
        
        slots_file = self.cache_dir / f"slots_rank{rank}_of_{world_size}.mmap"
        targets_file = self.cache_dir / f"targets_rank{rank}_of_{world_size}.mmap"
        
        if slots_file.exists():
            os.remove(slots_file)
        if targets_file.exists():
            os.remove(targets_file)
        
        dataset_size = len(self.train_dl.dataset)
        shard_size = dataset_size // world_size
        
        with torch.no_grad():
            sample_batch = next(iter(self.train_dl))
            img, _ = sample_batch
            num_augs = img.shape[1] if len(img.shape) == 5 else 1
        
        print(f"Rank {rank}: Creating new cache with {num_augs} augmentations per image...")
        os.makedirs(self.cache_dir, exist_ok=True)
        slots_file = self.cache_dir / f"slots_rank{rank}_of_{world_size}.mmap"
        targets_file = self.cache_dir / f"targets_rank{rank}_of_{world_size}.mmap"
        
        slots_mmap = np.memmap(
            slots_file,
            dtype='float32',
            mode='w+',
            shape=(shard_size * num_augs, self.train_num_slots, self.slot_dim)
        )
        
        targets_mmap = np.memmap(
            targets_file,
            dtype='int64',
            mode='w+',
            shape=(shard_size * num_augs,)
        )
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(
                self.train_dl, 
                desc=f"Rank {rank}: Caching data",
                disable=not self.accelerator.is_local_main_process
            )):
                imgs, targets = batch
                if len(imgs.shape) == 5:  # [B, num_augs, C, H, W]
                    B, A, C, H, W = imgs.shape
                    imgs = imgs.view(-1, C, H, W)  # [B*num_augs, C, H, W]
                    targets = targets.unsqueeze(1).expand(-1, A).reshape(-1)  # [B*num_augs]
                
                num_splits = num_augs
                split_size = imgs.shape[0] // num_splits
                imgs_splits = torch.split(imgs, split_size)
                targets_splits = torch.split(targets, split_size)
                
                start_idx = i * self.train_dl.batch_size * num_augs
                
                for split_idx, (img_split, targets_split) in enumerate(zip(imgs_splits, targets_splits)):
                    img_split = img_split.to(self.device, non_blocking=True)
                    
                    split_start = start_idx + (split_idx * split_size)
                    split_end = split_start + img_split.shape[0]
                    
                    slots_mmap[split_start:split_end] = slots_split.cpu().numpy()
                    targets_mmap[split_start:split_end] = targets_split.numpy()
        
        del slots_mmap
        del targets_mmap
        
        self.cached_latents = np.memmap(
            slots_file,
            dtype='float32',
            mode='r',
            shape=(shard_size * num_augs, self.train_num_slots, self.slot_dim)
        )
        
        self.cached_targets = np.memmap(
            targets_file,
            dtype='int64',
            mode='r',
            shape=(shard_size * num_augs,)
        )
        
        self.num_augs = num_augs

    def _setup_cache(self):
        """Setup cache if enabled."""
        self._build_cache()
        self.accelerator.wait_for_everyone()

        # Initialize cache loader if cache exists
        if self.cached_latents is not None:
            self.cache_loader = CacheDataLoader(
                slots=self.cached_latents,
                targets=self.cached_targets,
                batch_size=self.batch_size,
                num_augs=self.num_augs,
                seed=42 + self.accelerator.process_index
            )

    def __del__(self):
        """Cleanup cache files."""
        if self.enable_cache_latents:
            rank = self.accelerator.process_index
            world_size = self.accelerator.num_processes
            
            # Clean up slots cache
            slots_file = self.cache_dir / f"slots_rank{rank}_of_{world_size}.mmap"
            if slots_file.exists():
                os.remove(slots_file)
            
            # Clean up targets cache
            targets_file = self.cache_dir / f"targets_rank{rank}_of_{world_size}.mmap"
            if targets_file.exists():
                os.remove(targets_file)

    def _train_step(self, slots, targets=None):
        """Execute single training step."""
        
        with self.accelerator.accumulate(self.gpt_model):
            with self.accelerator.autocast():
                loss = self.gpt_model(slots, targets)
            
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients and self.max_grad_norm is not None:
                self.accelerator.clip_grad_norm_(self.gpt_model.parameters(), self.max_grad_norm)
            self.g_optim.step()
            if self.g_sched is not None:
                self.g_sched.step_update(self.steps)
            self.g_optim.zero_grad()

        # Update EMA model if enabled
        if self.enable_ema:
            self.ema_model.update(self.accelerator.unwrap_model(self.gpt_model))
        
        return loss

    def _train_epoch_cached(self, epoch, logger):
        """Train one epoch using cached data."""
        self.cache_loader.set_epoch(epoch)
        header = f'Epoch: [{epoch}/{self.num_epoch}]'
        
        for batch in logger.log_every(self.cache_loader, 20, header):
            slots, targets = (b.to(self.device, non_blocking=True) for b in batch)
            
            self.steps += 1

            if self.steps == 1:
                print(f"Training batch size: {len(slots)}")
                print(f"Hello from index {self.accelerator.local_process_index}")
            
            loss = self._train_step(slots, targets)
            self._handle_periodic_ops(loss, logger)

    def _train_epoch_uncached(self, epoch, logger):
        """Train one epoch using raw data."""
        header = f'Epoch: [{epoch}/{self.num_epoch}]'
        
        for batch in logger.log_every(self.train_dl, 20, header):
            # slots, img, sha256 = (b.to(self.device, non_blocking=True) for b in batch)

            slots, img, sha256 = batch
            slots = slots.to(self.device, non_blocking=True)
            img = img.to(self.device, non_blocking=True)
            device_batch = (slots, img, sha256)

            self.steps += 1
            
            if self.steps == 1:
                print(f"Training batch size: {img.size(0)}")
                print(f"Hello from index {self.accelerator.local_process_index}")

            # slots = self.ae_model.encode_slots(img)[:, :self.train_num_slots, :]
            slots = slots[:, :self.train_num_slots, :]
            loss = self._train_step(slots, img)
            self._handle_periodic_ops(loss, logger)

    def _handle_periodic_ops(self, loss, logger):
        """Handle periodic operations and logging."""
        # Get scalar values for logging
        loss_val = loss.item()
        current_lr = self.g_optim.param_groups[0]["lr"]

        # Update the console logger
        logger.update(loss=loss_val)
        logger.update(lr=current_lr)
        

        if self.accelerator.is_main_process:
            self.accelerator.log(
                {
                    "train/loss": loss_val,
                    "train/lr": current_lr
                },
                step=self.steps
            )
        
        if self.steps % self.save_every == 0:
            self.save()
        
        if (self.steps % self.sample_every == 0) or (self.eval_fid and self.steps % self.fid_every == 0):
            empty_cache()
            self.evaluate()
            self.accelerator.wait_for_everyone()
            empty_cache()

    def _save_config(self, config):
        """Save configuration file."""
        if config is not None and self.accelerator.is_main_process:
            import shutil
            from omegaconf import OmegaConf

            if isinstance(config, str) and osp.exists(config):
                shutil.copy(config, osp.join(self.result_folder, "config.yaml"))
            else:
                config_save_path = osp.join(self.result_folder, "config.yaml")
                OmegaConf.save(config, config_save_path)

    def _should_skip_epoch(self, epoch):
        """Check if epoch should be skipped due to loaded checkpoint."""
        loader = self.train_dl if not self.enable_cache_latents else self.cache_loader
        if ((epoch + 1) * len(loader)) <= self.loaded_steps:
            if self.accelerator.is_main_process:
                print(f"Epoch {epoch} is skipped because it is loaded from ckpt")
            self.steps += len(loader)
            return True
        
        if self.steps < self.loaded_steps:
            for _ in loader:
                self.steps += 1
                if self.steps >= self.loaded_steps:
                    break
        return False

    def train(self, config=None):
        """Main training loop."""
        # Initial setup
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            print(f"number of learnable parameters: {n_parameters//1e6}M")
        
        self._save_config(config)
        self.accelerator.init_trackers("gpt")

        # Handle test-only mode
        if self.test_only:
            empty_cache()
            self.evaluate()
            self.accelerator.wait_for_everyone()
            empty_cache()
            return

        # Setup cache if enabled
        if self.enable_cache_latents:
            self._setup_cache()

        # Training loop
        for epoch in range(self.num_epoch):
            if self._should_skip_epoch(epoch):
                continue

            self.gpt_model.train()
            logger = MetricLogger(delimiter="  ")
            logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

            # Choose training path based on cache availability
            if self.enable_cache_latents:
                self._train_epoch_cached(epoch, logger)
            else:
                self._train_epoch_uncached(epoch, logger)

            # Synchronize and log epoch stats
            logger.synchronize_between_processes()
            if self.accelerator.is_main_process:
                print("Averaged stats:", logger)

        # Finish training
        self.accelerator.end_training()
        self.save()
        if self.accelerator.is_main_process:
            print("Train finished!")

    # def save(self):
    #     self.accelerator.wait_for_everyone()
    #     self.accelerator.save_state(
    #         os.path.join(self.model_saved_dir, f"step{self.steps}")
    #     )
    def save(self):
        self.accelerator.wait_for_everyone()

        # 1. Define the local save path
        save_path = os.path.join(self.model_saved_dir, f"step{self.steps}")

        # 2. Save the full accelerator state to disk
        self.accelerator.save_state(save_path)

        # 3. Log the saved directory as a wandb artifact
        if self.accelerator.is_main_process:
            try:
                # Get the wandb tracker from the accelerator
                wandb_tracker = self.accelerator.get_tracker("wandb")
                
                # Create a new artifact
                artifact_name = f"{self.run_name}-step-{self.steps}"
                artifact = wandb.Artifact(name=artifact_name, type="model")
                
                # Add the entire checkpoint directory to the artifact
                artifact.add_dir(save_path)
                
                # Log the artifact to wandb
                wandb_tracker.run.log_artifact(artifact) 
                # wandb_tracker.log_artifact(artifact)
                
                if self.accelerator.is_main_process:
                    print(f"\nSuccessfully logged artifact {artifact_name} to wandb.")

            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"\nWarning: Failed to log artifact to wandb: {e}")

        self.accelerator.wait_for_everyone()


    @torch.no_grad()
    def evaluate(self):
        self.gpt_model.eval()

        # --- Check if validation dataloader exists ---
        if self.valid_dl is None:
            if self.accelerator.is_main_process:
                print(f"Step {self.steps}: No validation dataloader provided, skipping MSE evaluation.")
            self.gpt_model.train()
            return

        # --- Setup evaluation parameters ---
        eval_length = self.num_slots_to_gen
        
        # Determine CFG values to test
        cfg_values = [1.0, self.cfg] if self.cfg != 1.0 else [1.0]

        # Gating for saving qualitative results
        should_save_examples = (
            self.valid_dl is not None
            and (
                self.test_only
                or (
                    getattr(self, "sample_every", 0) > 0
                    and (self.steps % self.sample_every == 0)
                )
            )
        )
        num_batches_to_save = 8 if should_save_examples else 0
        
        # Token split logic (for logging only)
        token_split_size = 32
        num_splits = (eval_length + token_split_size - 1) // token_split_size

        # --- Main Evaluation Loop (per-CFG) ---
        for cfg_val in cfg_values:
            
            # --- Initialize accumulators for this CFG value ---
            mse_vals_full = []
            # Create a list of lists to hold MSEs for each split
            mse_vals_splits = [[] for _ in range(num_splits)]
            saved_count = 0
            
            with tqdm(
                self.valid_dl,
                desc=f"MSE Eval (cfg={cfg_val}, slots={eval_length})",
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            ) as loader:
                
                for batch_i, batch in enumerate(loader):
                    slots, targets, sha256 = batch
                    slots = slots.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    # Get ground truth latents, truncated to the length we are generating
                    gt_latents = slots[:, :eval_length, :]

                    with self.accelerator.autocast():
                        unwrapped_model = self.accelerator.unwrap_model(self.gpt_model)
                        # Generate the single, fixed-length sequence
                        pred_latents = generate(
                            unwrapped_model, # Use the model directly
                            targets,
                            eval_length,
                            cfg_scale=cfg_val,
                            cfg_schedule=self.cfg_schedule,
                            temperature=0.0 # <-- Set to 0.0 for deterministic MSE
                        )

                    # --- Calculate MSE (full and splits) ---
                    
                    # 1. Full sequence MSE
                    loss_full = F.mse_loss(pred_latents, gt_latents, reduction="mean")
                    loss_full_reduced = self.accelerator.reduce(loss_full, reduction="mean")
                    mse_vals_full.append(loss_full_reduced.item())

                    # 2. Split sequence MSE (for logging)
                    for i in range(num_splits):
                        start_idx = i * token_split_size
                        end_idx = min((i + 1) * token_split_size, eval_length)
                        
                        pred_split = pred_latents[:, start_idx:end_idx, :]
                        gt_split = gt_latents[:, start_idx:end_idx, :]
                        
                        loss_split = F.mse_loss(pred_split, gt_split, reduction="mean")
                        loss_split_reduced = self.accelerator.reduce(loss_split, reduction="mean")
                        mse_vals_splits[i].append(loss_split_reduced.item())
                        
                    # --- Save qualitative examples ---
                    if should_save_examples and saved_count < num_batches_to_save:
                        
                        os.makedirs(self.image_saved_dir, exist_ok=True)
                        
                        cfg_tag = "" if cfg_val == 1.0 else f"_cfg{cfg_val}"
                        # Include the process_index (rank) in the filename
                        fn = f"step_{self.steps}{cfg_tag}_slots{eval_length}_batch{batch_i}_rank{self.accelerator.process_index}.pt"
                        
                        # Each process saves its own local (non-gathered) data
                        torch.save(
                            {
                                "original_latents": gt_latents.detach().cpu(),
                                "reconstructed_latents": pred_latents.detach().cpu(),
                                "sha256": sha256,
                            },
                            os.path.join(self.image_saved_dir, fn),
                        )
                        
                        saved_count += 1
                        
            
            # --- Aggregate and Log Metrics for this CFG ---
            
            # 1. Log Full MSE
            avg_mse_full = (sum(mse_vals_full) / len(mse_vals_full)) if mse_vals_full else 0.0
            
            if cfg_val == 1.0:
                log_key = "eval/latent_mse"
                print_msg = f"Step {self.steps} | Slots {eval_length} | MSE: {avg_mse_full:.6f}"
            else:
                log_key = "eval/latent_mse_cfg"
                print_msg = f"Step {self.steps} | CFG {cfg_val} | Slots {eval_length} | MSE: {avg_mse_full:.6f}"
            
            if self.accelerator.is_main_process:
                print(print_msg)
            self.accelerator.log({log_key: avg_mse_full}, step=self.steps)

            # 2. Log Split MSEs
            for i in range(num_splits):
                start_idx = i * token_split_size
                end_idx = min((i + 1) * token_split_size, eval_length)
                key_name = f"tokens_{start_idx}-{end_idx}"
                
                avg_mse_split = (sum(mse_vals_splits[i]) / len(mse_vals_splits[i])) if mse_vals_splits[i] else 0.0
                
                log_key_split = f"eval/latent_mse_{key_name}"
                if cfg_val != 1.0:
                    log_key_split += "_cfg" 
                
                if self.accelerator.is_main_process:
                    print(f"  ... {key_name} MSE: {avg_mse_split:.6f}")
                
                self.accelerator.log({log_key_split: avg_mse_split}, step=self.steps)

        self.gpt_model.train()
