import os, torch
import os.path as osp
import shutil
from tqdm.auto import tqdm
from einops import rearrange
from accelerate import Accelerator
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, random_split, DistributedSampler
from src.utils.logger import SmoothedValue, MetricLogger, empty_cache
from accelerate.utils import DistributedDataParallelKwargs
from torchmetrics.functional.image import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim
)
import torch.nn.functional as F
from src.engine.trainer_utils import (
    instantiate_from_config, concat_all_gather,
    save_img_batch, get_fid_stats,
    EMAModel, PaddedDataset, create_scheduler, load_state_dict,
    load_safetensors, setup_result_folders, create_optimizer
)

from itertools import product
import torch.distributed as dist
from tqdm import tqdm
import wandb


class DiffusionTrainer:
    def __init__(
        self,
        model,
        dataset,
        test_dataset=None,   
        test_only=False,
        num_epoch=400,
        valid_size=32,
        blr=1e-4,
        cosine_lr=True,
        lr_min=0,
        warmup_epochs=100,
        warmup_steps=None,
        warmup_lr_init=0,
        decay_steps=None,
        batch_size=32,
        eval_bs=32,
        test_bs=64,         
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
        cfg=3.0,
        test_num_slots=None,
        eval_fid=False,
        fid_stats=None,
        enable_ema=False,
        compile=False,
        save_results_every=None,  
    ):
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="wandb",
            project_dir=log_dir,
        )
        project_name = 'LoST'
        run_name = "tokenizer"

        self.run_name = run_name

        self.accelerator.init_trackers(
            project_name=project_name,
            init_kwargs={"wandb": {"name": run_name}}
        )
        
        self.model = instantiate_from_config(model)
        self.num_slots = model.params.num_slots
        self.test_only = test_only
        self.save_results_every = save_results_every if save_results_every is not None else sample_every

        self.nest_after_epoch = model.params.enable_nest_after
        
        if not test_only:
            dataset = instantiate_from_config(dataset)
            test_dataset = instantiate_from_config(test_dataset)
            train_size = len(dataset)
            self.train_ds = dataset
            self.valid_ds = test_dataset
            valid_size = len(test_dataset)

            if self.accelerator.is_main_process:
                print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")

            sampler = DistributedSampler(
                self.train_ds,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
            )
            self.train_dl = DataLoader(
                self.train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=True,
            )
            

            eval_bs = valid_size // 8
            self.valid_dl = DataLoader(
                self.valid_ds,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )

            effective_bs = batch_size * grad_accum_steps * self.accelerator.num_processes
            lr = blr * effective_bs / 256
            if self.accelerator.is_main_process:
                print(f"Effective batch size is {effective_bs}")

            self.g_optim = create_optimizer(self.model, weight_decay=0.05, learning_rate=lr,)
            
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

            self.model, self.g_optim, self.g_sched = self.accelerator.prepare(self.model, self.g_optim, self.g_sched)

        else:
            dataset = instantiate_from_config(dataset)
            self.valid_ds = dataset
            self.valid_dl = DataLoader(
                self.valid_ds,
                batch_size=eval_bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=False,
            )
            self.model, self.valid_dl = self.accelerator.prepare(self.model, self.valid_dl)

        self.test_dl = self.valid_dl
        self.test_dataset_size = len(self.valid_ds)

        self.steps = 0
        self.loaded_steps = -1

        if compile:
            _model = self.accelerator.unwrap_model(self.model)
            _model.dit = torch.compile(_model.dit, mode="reduce-overhead")
            _model.encoder2slot = torch.compile(_model.encoder2slot, mode="reduce-overhead")

        self.enable_ema = enable_ema
        if self.enable_ema and not self.test_only:
            self.ema_model = EMAModel(self.accelerator.unwrap_model(self.model), self.device)
            self.accelerator.register_for_checkpointing(self.ema_model)

        self._load_checkpoint(model.params.ckpt_path)
        if self.test_only:
            self.steps = self.loaded_steps

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.sample_every = sample_every
        self.fid_every = fid_every
        self.max_grad_norm = 1 

        self.cfg = cfg
        self.test_num_slots = test_num_slots
        if self.test_num_slots is not None:
            self.test_num_slots = min(self.test_num_slots, self.num_slots)
        else:
            self.test_num_slots = self.num_slots

        eval_fid = eval_fid or model.params.eval_fid  
        self.eval_fid = eval_fid
        if eval_fid:
            if fid_stats is None:
                fid_stats = model.params.fid_stats 
            assert fid_stats is not None
            assert self.valid_dl is not None
        self.fid_stats = fid_stats

        self.result_folder = result_folder
        self.model_saved_dir, self.image_saved_dir = setup_result_folders(result_folder)
        self.current_epoch = 0



    @property
    def device(self):
        return self.accelerator.device

    def _load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None or not osp.exists(ckpt_path):
            return
        
        model = self.accelerator.unwrap_model(self.model)

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

    def train(self, config=None):
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            print(f"number of learnable parameters: {n_parameters//1e6}M")
        if config is not None:
            from omegaconf import OmegaConf
            if isinstance(config, str) and osp.exists(config):
                shutil.copy(config, osp.join(self.result_folder, "config.yaml"))
            else:
                config_save_path = osp.join(self.result_folder, "config.yaml")
                OmegaConf.save(config, config_save_path)

        if self.test_only:
            empty_cache()
            self.evaluate()
            self.accelerator.wait_for_everyone()
            empty_cache()
            return

        for epoch in range(self.num_epoch):
            self.current_epoch = epoch
            if ((epoch + 1) * len(self.train_dl)) <= self.loaded_steps:
                if self.accelerator.is_main_process:
                    print(f"Epoch {epoch} is skipped because it is loaded from ckpt")
                self.steps += len(self.train_dl)
                continue

            if self.steps < self.loaded_steps:
                for _ in self.train_dl:
                    self.steps += 1
                    if self.steps >= self.loaded_steps:
                        break
            
            
            self.accelerator.unwrap_model(self.model).current_epoch = epoch
            self.model.train() 

            logger = MetricLogger(delimiter="  ")
            logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}/{}]'.format(epoch, self.num_epoch)
            print_freq = 20
            for data_iter_step, batch in enumerate(logger.log_every(self.train_dl, print_freq, header)):
                latents, img, _ = batch
                latents = latents.to(self.device, non_blocking=True)
                img = img.to(self.device, non_blocking=True)
                device_batch = (latents, img, _)

                self.steps += 1

                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        if self.steps == 1:
                            print(f"Training batch size: {latents.size(0)}")
                            print(f"Hello from index {self.accelerator.local_process_index}")
                        losses = self.model(device_batch, epoch=epoch)
                        loss = sum([v for _, v in losses.items()])

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients and self.max_grad_norm is not None:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.g_optim.step()
                    if self.g_sched is not None:
                        self.g_sched.step_update(self.steps)
                    self.g_optim.zero_grad()

                self.accelerator.wait_for_everyone()

                if self.enable_ema:
                    self.ema_model.update(self.accelerator.unwrap_model(self.model))

                for key, value in losses.items():
                    logger.update(**{key: value.item()})
                logger.update(lr=self.g_optim.param_groups[0]["lr"])

                if self.steps % self.save_every == 0:
                    self.save()

                if (self.steps % self.sample_every == 0) or (self.steps % self.fid_every == 0):
                    empty_cache()
                    self.evaluate()
                    self.accelerator.wait_for_everyone()
                    empty_cache()

                write_dict = dict(epoch=epoch)
                for key, value in losses.items():
                    write_dict.update(**{key: value.item()})
                write_dict.update(lr=self.g_optim.param_groups[0]["lr"])
                self.accelerator.log(write_dict, step=self.steps)

            logger.synchronize_between_processes()
            if self.accelerator.is_main_process:
                print("Averaged stats:", logger)

        self.accelerator.end_training()
        self.save()
        if self.accelerator.is_main_process:
            print("Train finished!")


    def save(self):
        self.accelerator.wait_for_everyone()

        save_path = os.path.join(self.model_saved_dir, f"step{self.steps}")
        self.accelerator.save_state(save_path)

        if self.accelerator.is_main_process:
            try:
                wandb_tracker = self.accelerator.get_tracker("wandb")
                
                artifact_name = f"{self.run_name}-step-{self.steps}"
                artifact = wandb.Artifact(name=artifact_name, type="model")
                artifact.add_dir(save_path)
                wandb_tracker.run.log_artifact(artifact) 
                
                if self.accelerator.is_main_process:
                    print(f"\nSuccessfully logged artifact {artifact_name} to wandb.")

            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"\nWarning: Failed to log artifact to wandb: {e}")

        self.accelerator.wait_for_everyone()

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        cur_epoch = getattr(self, "current_epoch", 0)

        slots_to_eval = [self.test_num_slots]
        extra_slots_candidates = [1, 8, 64]
        if cur_epoch >= getattr(self, "nest_after_epoch", 999999):
            slots_to_eval += [s for s in extra_slots_candidates if s <= self.num_slots]

        cfg_values = [1.0]
        if getattr(self, "cfg", 1.0) != 1.0:
            cfg_values.append(self.cfg)

        should_save_examples = (
            self.valid_dl is not None
            and (
                self.test_only
                or (
                    getattr(self, "save_results_every", 0) > 0
                    and (self.steps % self.save_results_every == 0)
                )
            )
        )

        if self.valid_dl is None:
            self.model.train()
            return
        num_batches_to_save = 2 if should_save_examples else 0

        for slots in slots_to_eval:
            for cfg_val in cfg_values:

                mse_vals = []
                saved_count = 0

                with tqdm(
                    self.valid_dl,
                    desc=f"MSE (cfg={cfg_val}) slots={slots}",
                    dynamic_ncols=True,
                    disable=not self.accelerator.is_main_process,
                ) as loader:

                    for batch_i, batch in enumerate(loader):
                        latents, img, sha256 = batch
                        latents = latents.to(self.device, non_blocking=True)
                        img = img.to(self.device, non_blocking=True)
                        device_batch = (latents, img, sha256)

                        with self.accelerator.autocast():
                            rec = self.model(
                                device_batch,
                                sample=True,
                                inference_with_n_slots=slots,
                                cfg=cfg_val,
                            )

                        loss = F.mse_loss(rec, latents, reduction="mean")
                        loss = self.accelerator.reduce(loss, reduction="mean")
                        mse_vals.append(loss.item())
                        if should_save_examples and saved_count < num_batches_to_save:
                            if self.accelerator.is_main_process:
                                os.makedirs(self.image_saved_dir, exist_ok=True)
                                cfg_tag = "" if cfg_val == 1.0 else f"_cfg{cfg_val}"
                                fn = f"step_{self.steps}{cfg_tag}_slots{slots}_batch{batch_i}.pt"
                                torch.save(
                                    {
                                        "original_latents": latents.detach().cpu(),
                                        "reconstructed_latents": rec.detach().cpu(),
                                        "sha256": sha256,
                                    },
                                    os.path.join(self.image_saved_dir, fn),
                                )
                            saved_count += 1

                avg_mse = (sum(mse_vals) / len(mse_vals)) if mse_vals else 0.0

                if self.accelerator.is_main_process:
                    if slots == self.test_num_slots and cfg_val == 1.0:
                        print(f"Step {self.steps} | Slots {slots} | MSE: {avg_mse:.6f}")
                    elif slots == self.test_num_slots and cfg_val != 1.0:
                        print(f"Step {self.steps} | CFG {cfg_val} | Slots {slots} | MSE: {avg_mse:.6f}")
                    elif slots != self.test_num_slots and cfg_val == 1.0:
                        print(f"Step {self.steps} | Slots {slots} | MSE: {avg_mse:.6f}")
                    else:
                        print(f"Step {self.steps} | CFG {cfg_val} | Slots {slots} | MSE: {avg_mse:.6f}")

                if slots == self.test_num_slots and cfg_val == 1.0:
                    self.accelerator.log({"eval/latent_mse": avg_mse}, step=self.steps)
                elif slots == self.test_num_slots and cfg_val != 1.0:
                    self.accelerator.log({"eval/latent_mse_cfg": avg_mse}, step=self.steps)
                elif cfg_val == 1.0:
                    self.accelerator.log({f"eval/latent_mse_slots{slots}": avg_mse}, step=self.steps)
                else:
                    self.accelerator.log({f"eval/latent_mse_cfg_slots{slots}": avg_mse}, step=self.steps)

        self.model.train()
