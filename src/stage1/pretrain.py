# pretrain3.py
# Minimal updates: GroupNorm, correct InfoNCE temperature, epoch-level global KPI.

import os, os.path as osp
import math
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


PREDICTOR_MODEL_PATH = "./pretrain_semantic_predictor/h7fhhm12/checkpoints/last.ckpt"  # Path to your trained model

# --- Model Hyperparameters (Must match the trained model) ---
D_TOKEN = 768
DEPTH = 2
HEADS = 1
# NEW: Must match StudentTriFormer parameters from pretrain3.py
GLOBAL_POOL = "attn"  # Must match trained model ('attn' or 'mean')
GLOBAL_DIM = 768  # Must match trained model
SPATIAL_DIM = 256  # Must match trained model

@torch.no_grad()
def extract_student_features(model, dataloader, device):
    """
    Runs all latents through the StudentTriFormer model and collects global features (z).
    """
    all_features = []
    all_filepaths = []
    all_sha256s = []

    for batch in tqdm(dataloader, desc="Extracting Student Features"):
        latents, filepaths, sha256s = batch
        latents = latents.to(device).float()

        # Get the spatial grid (S_grid) and global feature (z)
        # z is already (B, global_dim) and L2-normalized from the model's forward pass
        _S_grid, global_features = model(latents)

        # Handle case where batch size is 1
        if global_features.dim() == 1:
            global_features = global_features.unsqueeze(0)

        # L2-normalize is NOT needed; StudentTriFormer.forward() already does it.

        all_features.append(global_features.cpu())
        all_filepaths.extend(filepaths)
        all_sha256s.extend(sha256s)

    return torch.cat(all_features, dim=0), all_filepaths, all_sha256s


def load_predictor_model():
    """
    Instantiates a StudentTriFormer model and loads the trained weights
    from a PyTorch Lightning checkpoint.
    """
    print(
        f"Loading StudentTriFormer model with d_token={D_TOKEN}, depth={DEPTH}, heads={HEADS}..."
    )

    # Instantiate the model with the same architecture as pretrain3.py
    model = StudentTriFormer(
        d_token=D_TOKEN,
        depth=DEPTH,
        heads=HEADS,
        mlp_ratio=2.0,
        dropout=0.0,
        # Add new parameters from StudentTriFormer
        global_pool=GLOBAL_POOL,
        global_dim=GLOBAL_DIM,
        spatial_dim=SPATIAL_DIM,
    )

    # # 1. Load the full checkpoint dictionary
    # checkpoint = torch.load(PREDICTOR_MODEL_PATH, map_location="cpu")

    # loaded_state_dict = checkpoint["state_dict"]

    # # 3. Create a new, clean state_dict to load into StudentTriFormer
    # new_state_dict = {}
    # prefix = "student."  # This is the prefix from LitMetricStudent

    # # Check if prefixes are even needed
    # has_prefix = any(key.startswith(prefix) for key in loaded_state_dict.keys())

    # for key, value in loaded_state_dict.items():
    #     if has_prefix:
    #         if key.startswith(prefix):
    #             # 4. Remove the "student." prefix
    #             new_key = key[len(prefix) :]
    #             new_state_dict[new_key] = value
    #     else:
    #         # No prefix, just copy
    #         new_state_dict[key] = value

    # # 5. Load the *new*, *clean* state_dict
    # model.load_state_dict(new_state_dict)

    for param in model.parameters():
        param.requires_grad = False

    # model.to(device)
    model.eval()
    return model


# =========================
# Dataset
# =========================
class LatentsNet(torch.utils.data.Dataset):
    def __init__(self, root, split='train'):
        self.dir = osp.join(root, split)
        self.pt_files = [osp.join(self.dir, f) for f in os.listdir(self.dir) if f.endswith('.pt')]
        self.pt_files.sort()
        if split != "train":
            self.pt_files = self.pt_files[:768]

        self.transform = TF.Compose([
            TF.Resize((224, 224), interpolation=TF.InterpolationMode.BICUBIC, antialias=True),
            TF.ToTensor(),
        ])

    def __len__(self): return len(self.pt_files)

    def __getitem__(self, idx):
        pt_path = self.pt_files[idx]
        tri = torch.load(pt_path).squeeze()  # (16,32,96)
        sha256 = os.path.basename(pt_path).split('.')[0]
        img = Image.open(f"/mnt/localssd/flux_images/{sha256}.png").convert('RGB')
        img = self.transform(img)  # (3,224,224) in [0,1]
        return tri, img, sha256


# =========================
# Utils
# =========================
def get_2d_sincos_pos_embed(h: int, w: int, dim: int, device):
    assert dim % 4 == 0
    def _pe(n, d_half):
        omega = torch.arange(d_half, device=device) / d_half
        omega = 1.0 / (10000 ** omega)
        pos = torch.arange(n, device=device, dtype=torch.float32)
        out = torch.einsum('n,d->nd', pos, omega)
        return torch.sin(out), torch.cos(out)
    siny, cosy = _pe(h, dim//4)
    sinx, cosx = _pe(w, dim//4)
    pos_y = torch.cat([siny, cosy], dim=1)
    pos_x = torch.cat([sinx, cosx], dim=1)
    pos = torch.cat([
        pos_y[:, None, :].repeat(1, w, 1),
        pos_x[None, :, :].repeat(h, 1, 1)
    ], dim=-1).view(1, h*w, dim)
    return pos

def pairwise_cosine(x):  # x: (B,D)
    x = F.normalize(x, dim=-1)
    return x @ x.t()

def upper_triangular_flat(M):
    B = M.shape[0]
    iu = torch.triu_indices(B, B, offset=1, device=M.device)
    return M[iu[0], iu[1]]

def row_softmax_affinity(tokens, tau=0.1):  # tokens: (B,N,D)
    T = F.normalize(tokens, dim=-1)
    G = torch.matmul(T, T.transpose(1, 2)) / tau   # (B,N,N)
    return F.softmax(G, dim=-1)

def kl_rowwise(p, q, eps=1e-8):  # p,q: (B,N,N)
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    return (p * (p.log() - q.log())).sum(dim=-1).mean()

def off_diagonal(M):
    return M - torch.diag(torch.diag(M))


# =========================
# Tokenizer: conv stem -> 16x16 tokens
# =========================
class DWConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, padding=1, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1)
        self.act = nn.GELU()
        # ---- GroupNorm (DDP-stable) ----
        num_groups = min(32, c_out)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)

    def forward(self, x):
        x = self.dw(x); x = self.pw(x)
        x = self.act(x); x = self.norm(x)
        return x

class TriTokenizer(nn.Module):
    def __init__(self, plane_c=16, d=128, D=768):
        super().__init__()
        self.stem = nn.Sequential(
            DWConvBlock(plane_c, 64),
            DWConvBlock(64, 96),
            DWConvBlock(96, d),
        )
        self.fuse = nn.Conv2d(3*d, D, 1)
        self.down = nn.Conv2d(D, D, 3, stride=2, padding=1)  # 32->16

    def forward(self, x):  # (B,16,32,96)
        assert x.shape[1:] == (16,32,96), f"Bad latent shape {x.shape}"
        w3 = x.shape[-1] // 3
        p1, p2, p3 = torch.split(x, w3, dim=-1)  # (B,16,32,32)*3
        f = torch.cat([self.stem(p1), self.stem(p2), self.stem(p3)], dim=1)  # (B,3d,32,32)
        f = self.fuse(f)        # (B,D,32,32)
        f = self.down(f)        # (B,D,16,16)
        B, D, H, W = f.shape
        tokens = f.flatten(2).transpose(1, 2)  # (B,256,D)
        return tokens, (B, D, H, W)


# =========================
# Tiny MLP and Attn Pool
# =========================
class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=2, out_dim=None):
        super().__init__()
        hidden = int(dim * hidden_mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim or dim),
        )
    def forward(self, x): return self.net(x)

class AttnPool1D(nn.Module):
    """Parameter-light attention pooling over tokens (no registers)."""
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, 1)
    def forward(self, tokens):  # (B,N,D)
        h = torch.tanh(self.w1(tokens))     # (B,N,D)
        a = self.w2(h).squeeze(-1)          # (B,N)
        a = torch.softmax(a, dim=-1)
        g = (a.unsqueeze(-1) * tokens).sum(dim=1)  # (B,D)
        return g


# =========================
# Student Encoder (no registers)
# =========================
class StudentTriFormer(nn.Module):
    def __init__(self, d_token=768, depth=12, heads=8, mlp_ratio=2.0, dropout=0.05,
                 global_pool='attn', global_dim=768, spatial_dim=256):
        super().__init__()
        self.tokenizer = TriTokenizer(plane_c=16, d=128, D=d_token)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_token, nhead=heads,
            dim_feedforward=int(d_token*mlp_ratio),
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(d_token)

        # heads
        self.spatial_head = nn.Linear(d_token, spatial_dim)  # student spatial token dim
        self.global_pool = global_pool
        if global_pool == 'attn':
            self.pooler = AttnPool1D(d_token)
        else:
            self.pooler = None  # mean pool
        self.global_mlp = MLP(d_token, hidden_mult=2, out_dim=global_dim)

        # pos embed buffer
        self.register_buffer("pos_embed", None, persistent=False)

    #  def forward(self, tri):  # tri: (B,16,32,96)
    #     tokens, (B, D, H, W) = self.tokenizer(tri)   # (B,256,D)
    #     if self.pos_embed is None or self.pos_embed.shape[-1] != D:
    #         self.pos_embed = get_2d_sincos_pos_embed(H, W, D, device=tokens.device)
    #     x = tokens + self.pos_embed                  # (B,256,D)
    #     x = self.encoder(x)
    #     x = self.norm(x)                             # (B,256,D)
    #     return x

    def forward(self, tri):  # tri: (B,16,32,96)
        tokens, (B, D, H, W) = self.tokenizer(tri)   # (B,256,D)
        if self.pos_embed is None or self.pos_embed.shape[-1] != D:
            self.pos_embed = get_2d_sincos_pos_embed(H, W, D, device=tokens.device)
        x = tokens + self.pos_embed                  # (B,256,D)
        x = self.encoder(x)
        x = self.norm(x)                             # (B,256,D)

        # spatial tokens
        S = self.spatial_head(x)                     # (B,256,spatial_dim)
        S = F.layer_norm(S, (S.shape[-1],))          # token LN
        S = F.normalize(S, dim=-1)                   # token L2-norm
        S_grid = S.transpose(1, 2).reshape(B, S.shape[-1], H, W)  # (B,spatial_dim,16,16)

        # global embedding
        if self.global_pool == 'attn':
            pooled = self.pooler(x)                 # (B,D)
        else:
            pooled = x.mean(dim=1)                  # (B,D)
        z = self.global_mlp(pooled)                 # (B,global_dim)
        z = F.layer_norm(z, (z.shape[-1],))
        z = F.normalize(z, dim=-1)
        return S_grid, z  # spatial grid, global


# =========================
# Frozen DINOv2 Teacher (patch + CLS)
# =========================
class DINOTeacher(nn.Module):
    """
    Returns teacher patch tokens (B,256,768) and CLS (B,768) from dinov2_vitb14.
    """
    def __init__(self, model_name='dinov2_vitb14', enc_img_size=224):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        if hasattr(self.model, "image_size"):
            self.model.image_size = enc_img_size
        self.enc_img_size = enc_img_size

        for p in self.model.parameters(): p.requires_grad = False
        self.model.eval()

        self.register_buffer("mean", torch.tensor(IMAGENET_DEFAULT_MEAN).view(1,3,1,1), persistent=False)
        self.register_buffer("std",  torch.tensor(IMAGENET_DEFAULT_STD).view(1,3,1,1), persistent=False)

    @torch.no_grad()
    def forward(self, x):  # (B,3,H,W)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        if x.shape[-2:] != (self.enc_img_size, self.enc_img_size):
            x = F.interpolate(x, size=(self.enc_img_size, self.enc_img_size), mode='bicubic', align_corners=False)

        out = self.model.forward_features(x)
        patch = out['x_norm_patchtokens']      # (B,257 or 256, 768)
        cls   = out.get('x_norm_clstoken', None)  # (B,768)
        if patch.shape[1] == 257:
            cls_from_patch = patch[:, 0, :]
            patch = patch[:, 1:, :]
            if cls is None: cls = cls_from_patch
        assert cls is not None, "x_norm_clstoken should be present for dinov2_vitb14."

        B, N, C = patch.shape
        assert N == 256 and C == 768, f"Expected (B,256,768) tokens, got {patch.shape}"

        tokens = F.normalize(patch, dim=-1)        # (B,256,768)
        mean_patch = F.normalize(tokens.mean(dim=1), dim=-1)  # (B,768)
        alpha = 0.7
        cls = F.normalize(alpha * cls + (1 - alpha) * mean_patch, dim=-1)

        return tokens, cls


# =========================
# Lightning Module
# =========================
class LitMetricStudent(pl.LightningModule):
    def __init__(self, lr=2e-4, weight_decay=0.05,
                 depth=12, heads=8, d_token=768, dropout=0.05,
                 global_pool='attn', global_dim=768, spatial_dim=256,
                 teacher_model='dinov2_vitb14', teacher_img_size=224,
                 contrastive_w=1.0, temperature=0.07, learn_temp=False,
                 pos_thresh=0.83, ign_thresh=0.65,
                 hardneg_w=0.3, hardneg_topk=10, hardneg_margin=0.30,
                 spatial_self_w=0.5, spatial_cross_w=0.25, spatial_tokens=128,
                 spatial_tau_self=0.1, spatial_tau_cross=0.1,
                 pair_align_w=1.0, pair_midrange_weight=False,
                 var_w=1.0, cov_w=0.01, var_gamma=0.5,
                 tri_noise_std=0.0, tri_drop_channel_prob=0.0,
                 abs_align_w=0.2,              # weight for absolute similarity calibration
                 abs_pos_thresh=0.75,          # only calibrate strong positives
                 abs_neg_thresh=-0.10,         # and strong negatives (optional)
                 row_mustd_w=0.10,             # weight to match per-row mean/std

                 ):
        super().__init__()
        self.save_hyperparameters()

        self.student = StudentTriFormer(
            d_token=d_token, depth=depth, heads=heads, mlp_ratio=2.0, dropout=dropout,
            global_pool=global_pool, global_dim=global_dim, spatial_dim=spatial_dim
        )
        self.teacher = DINOTeacher(model_name=teacher_model, enc_img_size=teacher_img_size)

        # temperature (scale = exp(logit_scale) = 1/tau)
        self.learn_temp = learn_temp
        init_scale = 1.0 / max(1e-6, temperature)
        if self.learn_temp:
            self.logit_scale = nn.Parameter(torch.tensor(math.log(init_scale)))
        else:
            self.register_buffer("logit_scale", torch.tensor(math.log(init_scale)), persistent=False)

        # epoch KPI buffers
        self._val_z = []
        self._val_t = []

    # ----- Optimizer + epoch-based warmup->cosine -----
    def configure_optimizers(self):
        betas = (0.9, 0.95)
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, betas=betas)

        max_epochs = int(self.trainer.max_epochs)
        warmup_epochs = max(1, int(0.05 * max_epochs))
        cosine_epochs = max(1, max_epochs - warmup_epochs)

        warmup = torch.optim.lr_scheduler.LambdaLR(
            opt, lr_lambda=lambda epoch: (epoch + 1) / float(warmup_epochs) if epoch < warmup_epochs else 1.0
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cosine_epochs, eta_min=0.0)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_epochs])

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}

    # ----- DDP helper -----
    def _gather(self, x, gather=True):  # x: (B, D)
        if not gather:
            return x
        out = self.all_gather(x)  # (world, B, D)
        return out.reshape(-1, x.shape[-1]).detach()

    # ----- Relational InfoNCE -----
    def _info_nce_rel(self, z_local, t_local, gather=True):
        z = F.normalize(F.layer_norm(z_local, (z_local.shape[-1],)), dim=-1)  # (B,D)
        with torch.no_grad():
            t = t_local  # normalized
            t_all = self._gather(t, gather=gather)   # (N_all,768)
            z_all = self._gather(z, gather=gather)   # (N_all,D)
            B = z.shape[0]
            rank = getattr(self, "global_rank", 0)
            offset = rank * B if gather else 0

            sim_tt = t @ t_all.t()                   # (B,N_all)
            pos_mask = sim_tt >= self.hparams.pos_thresh
            ar = torch.arange(B, device=z.device)
            pos_mask[ar, offset + ar] = True
            ign_mask = (sim_tt >= self.hparams.ign_thresh) & (~pos_mask)

        # scale = 1/tau
        scale = torch.exp(self.logit_scale)
        if self.learn_temp:
            with torch.no_grad():
                # clamp tau in [0.03, 0.2]
                self.logit_scale.data.clamp_(math.log(1/0.2), math.log(1/0.03))

        logits = (z @ z_all.t()) * scale            # (B,N_all)
        logits_den = logits.masked_fill(ign_mask, float('-inf'))
        lse_den = torch.logsumexp(logits_den, dim=1)

        logits_pos = logits.masked_fill(~pos_mask, float('-inf'))
        lse_num = torch.logsumexp(logits_pos, dim=1)

        loss = -(lse_num - lse_den).mean()
        avg_pos = pos_mask.float().sum(dim=1).mean()
        avg_ign = ign_mask.float().sum(dim=1).mean()
        return loss, avg_pos.detach(), avg_ign.detach()

    # ----- Hard-negative hinge -----
    def _hard_neg_hinge(self, z_local, t_local, gather=True):
        if self.hparams.hardneg_w <= 0 or self.hparams.hardneg_topk <= 0:
            return torch.zeros((), device=z_local.device), torch.zeros((), device=z_local.device)

        z = F.normalize(F.layer_norm(z_local, (z_local.shape[-1],)), dim=-1)
        with torch.no_grad():
            t = t_local
            t_all = self._gather(t, gather=gather)
            z_all = self._gather(z, gather=gather)
            B = z.shape[0]
            rank = getattr(self, "global_rank", 0)
            offset = rank * B if gather else 0
            sim_tt = t @ t_all.t()

            pos_mask = sim_tt >= self.hparams.pos_thresh
            ar = torch.arange(B, device=z.device)
            pos_mask[ar, offset + ar] = True

            hard_mask = (sim_tt >= self.hparams.ign_thresh) & (sim_tt < self.hparams.pos_thresh) & (~pos_mask)
            masked = sim_tt.masked_fill(~hard_mask, -1e9)
            k = min(self.hparams.hardneg_topk, masked.shape[1])
            topk_vals, topk_idx = torch.topk(masked, k=k, dim=1)  # (B,k)

        zz = z @ z_all.t()
        pos_idx = (offset + torch.arange(z.shape[0], device=z.device)).view(-1,1)
        pos_log = zz.gather(1, pos_idx)
        neg_log = zz.gather(1, topk_idx)

        hinge = F.relu(self.hparams.hardneg_margin + neg_log - pos_log)
        loss = hinge.mean()
        avg_hard = topk_vals[topk_vals > -1e8].mean() if (topk_vals > -1e8).any() else torch.tensor(0.0, device=z.device)
        return loss, avg_hard.detach()

    # ----- Spatial structure distillation -----
    def _spatial_losses(self, S_grid, T_tokens, pos_mask_local, token_subsample=128):
        B, Ds, H, W = S_grid.shape
        N = H * W

        S = S_grid.flatten(2).transpose(1, 2)  # (B,256,Ds) normalized tokens

        if token_subsample < N:
            idx = torch.randperm(N, device=S.device)[:token_subsample]
            S = S[:, idx, :]                 # (B,K,Ds)
            T = T_tokens[:, idx, :]         # (B,K,768)
        else:
            T = T_tokens                     # (B,256,768)

        A_T = row_softmax_affinity(T, tau=self.hparams.spatial_tau_self)  # (B,K,K)
        A_S = row_softmax_affinity(S, tau=self.hparams.spatial_tau_self)  # (B,K,K)
        loss_self = kl_rowwise(A_T, A_S)

        loss_cross = torch.zeros((), device=S.device)
        if self.hparams.spatial_cross_w > 0:
            count = 0
            for i in range(B):
                pos_js = torch.nonzero(pos_mask_local[i], as_tuple=False).flatten()
                pos_js = pos_js[pos_js != i]
                if pos_js.numel() == 0: continue
                j = pos_js[torch.randint(0, pos_js.numel(), (1,), device=S.device)].item()
                C_T = F.softmax((T[i] @ T[j].t()) / self.hparams.spatial_tau_cross, dim=-1)
                C_S = F.softmax((S[i] @ S[j].t()) / self.hparams.spatial_tau_cross, dim=-1)
                loss_cross = loss_cross + kl_rowwise(C_T.unsqueeze(0), C_S.unsqueeze(0))
                count += 1
            if count > 0:
                loss_cross = loss_cross / count

        return loss_self, loss_cross

    # ----- Pairwise cosine alignment (row-wise global) -----
    def _pair_align_losses(self, z_local, t_local, midrange=False):
        with torch.no_grad():
            t_all = self._gather(t_local, gather=True)  # (N_all, D_t) normalized

        z_local = F.normalize(z_local, dim=-1)                
        z_all   = F.normalize(self._gather(z_local, gather=True).detach(), dim=-1)

        # Row-wise similarities (local rows vs global pool)
        Sz = z_local @ z_all.t()    # (B, N_all)
        St = t_local @ t_all.t()    # (B, N_all)

        # --- 1) Ranking term (z-scored row-wise) ---
        Sz_z = (Sz - Sz.mean(dim=1, keepdim=True)) / (Sz.std(dim=1, keepdim=True) + 1e-8)
        St_z = (St - St.mean(dim=1, keepdim=True)) / (St.std(dim=1, keepdim=True) + 1e-8)
        if midrange:
            s = ((St_z.clamp(-1, 1) + 1.0) * 0.5)
            w = 4.0 * s * (1.0 - s)
            pair_rank = (w * (Sz_z - St_z).pow(2)).sum(dim=1).div(w.sum(dim=1) + 1e-8).mean()
        else:
            pair_rank = F.mse_loss(Sz_z, St_z)

        # --- 2) Absolute calibration on strong pos/neg only (small weight) ---
        calib_mask = (St >= self.hparams.abs_pos_thresh) | (St <= self.hparams.abs_neg_thresh)
        if calib_mask.any():
            pair_abs = F.mse_loss(Sz[calib_mask], St[calib_mask])
        else:
            pair_abs = torch.zeros((), device=Sz.device)

        # --- 3) Per-row mean/std match (nudges scale without fighting rank) ---
        mu_z  = Sz.mean(dim=1)
        mu_t  = St.mean(dim=1)
        sd_z  = Sz.std(dim=1) + 1e-8
        sd_t  = St.std(dim=1) + 1e-8
        row_mu  = F.l1_loss(mu_z, mu_t)
        row_sd  = F.l1_loss(sd_z, sd_t)
        pair_row_mustd = row_mu + row_sd

        return pair_rank, pair_abs, pair_row_mustd


    # ----- VICReg-like dispersion on z -----
    def _dispersion(self, z, gamma=0.5):
        zc = z - z.mean(0, keepdim=True)
        std = zc.std(0) + 1e-8
        var_loss = F.relu(gamma - std).mean()
        cov = (zc.T @ zc) / max(1, (z.shape[0] - 1))
        cov_loss = (off_diagonal(cov).pow(2)).mean()
        return var_loss, cov_loss

    # ----- tiny triplane augmentation -----
    def _tri_aug(self, tri):
        if self.hparams.tri_noise_std > 0:
            tri = tri + torch.randn_like(tri) * self.hparams.tri_noise_std
        if self.hparams.tri_drop_channel_prob > 0 and self.training:
            B, C, H, W = tri.shape  # C=16
            mask = (torch.rand(B, C, 1, 1, device=tri.device) > self.hparams.tri_drop_channel_prob).float()
            tri = tri * mask
        return tri

    # =========================
    # Training
    # =========================
    def training_step(self, batch, batch_idx):
        tri, img, _ = batch
        tri = self._tri_aug(tri)
        S_grid, z = self.student(tri)
        with torch.no_grad():
            T_tokens, t = self.teacher(img)

        # Global relational contrast + hard negs (use all_gather=True for big pool)
        con_loss, avg_pos, avg_ign = self._info_nce_rel(z, t, gather=True)
        hard_loss, avg_hard = self._hard_neg_hinge(z, t, gather=True)

        # Spatial structure (local teacher mask is fine)
        with torch.no_grad():
            sim_tt_local = t @ t.t()
            pos_mask_local = sim_tt_local >= self.hparams.pos_thresh
            ar = torch.arange(z.shape[0], device=z.device)
            pos_mask_local[ar, ar] = True

        loss_self, loss_cross = self._spatial_losses(
            S_grid, T_tokens, pos_mask_local, token_subsample=self.hparams.spatial_tokens
        )

        # Row-wise global alignment components (ranking + absolute + row mean/std)
        pair_rank, pair_abs, pair_row = self._pair_align_losses(
            z, t, midrange=self.hparams.pair_midrange_weight
        )

        # Dispersion (train only)
        var_loss, cov_loss = self._dispersion(z, gamma=self.hparams.var_gamma)

        # Total loss
        loss = (
            self.hparams.contrastive_w     * con_loss
        + self.hparams.hardneg_w         * hard_loss
        + self.hparams.spatial_self_w    * loss_self
        + self.hparams.spatial_cross_w   * loss_cross
        + self.hparams.pair_align_w      * pair_rank
        + self.hparams.abs_align_w       * pair_abs
        + self.hparams.row_mustd_w       * pair_row
        + self.hparams.var_w             * var_loss
        + self.hparams.cov_w             * cov_loss
        )

        # Log true tau (tau = exp(-logit_scale))
        tau = float(torch.exp(-self.logit_scale).detach())
        self.log("train/temperature", tau, on_step=True, on_epoch=True, prog_bar=False)

        B = tri.shape[0]
        self.log_dict(
            {
                "train/loss": loss,
                "train/contrastive": con_loss,
                "train/hardneg": hard_loss, "train/avg_hard_teacher_sim": avg_hard,
                "train/sp_self_step": loss_self, "train/sp_cross_step": loss_cross,
                "train/pair_rank": pair_rank, "train/pair_abs": pair_abs, "train/pair_row": pair_row,
                "train/var_z": var_loss, "train/cov_offdiag": cov_loss,
                "train/avg_pos_step": avg_pos, "train/avg_ign_step": avg_ign,
            },
            on_step=True, on_epoch=False, prog_bar=True, batch_size=B
        )
        self.log_dict(
            {
                "train/sp_self_epoch": loss_self,
                "train/sp_cross_epoch": loss_cross,
                "train/contrastive_epoch": con_loss,
                "train/avg_pos_epoch": avg_pos,
                "train/avg_ign_epoch": avg_ign,
                "train/hardneg_epoch": hard_loss,
            },
            on_step=False, on_epoch=True, prog_bar=False, batch_size=B
        )
        return loss

    # =========================
    # Validation: batch logs + epoch-level KPI
    # =========================
    def on_validation_epoch_start(self):
        self._val_z = []
        self._val_t = []

    def validation_step(self, batch, batch_idx):
        tri, img, _ = batch

        # ---- Student & teacher (local) ----
        S_grid, z_local = self.student(tri)  # (B, D)
        with torch.no_grad():
            T_tokens, t_local = self.teacher(img)  # (B, 256, 768), (B, 768)

        # ---- Local-only aux losses (fast) ----
        con_loss, avg_pos, avg_ign = self._info_nce_rel(z_local, t_local, gather=False)
        hard_loss, avg_hard = self._hard_neg_hinge(z_local, t_local, gather=False)

        pos_mask_local = (t_local @ t_local.t()) >= self.hparams.pos_thresh
        loss_self, loss_cross = self._spatial_losses(
            S_grid, T_tokens,
            pos_mask_local=pos_mask_local,
            token_subsample=self.hparams.spatial_tokens
        )

        # ---- Row-wise global pairwise alignment (gathers inside) ----
        pair_rank, pair_abs, pair_row = self._pair_align_losses(
            z_local, t_local, midrange=self.hparams.pair_midrange_weight
        )

        # ---- Total validation loss (no dispersion terms in val) ----
        loss = (
            self.hparams.contrastive_w     * con_loss
        + self.hparams.hardneg_w         * hard_loss
        + self.hparams.spatial_self_w    * loss_self
        + self.hparams.spatial_cross_w   * loss_cross
        + self.hparams.pair_align_w      * pair_rank
        + self.hparams.abs_align_w       * pair_abs
        + self.hparams.row_mustd_w       * pair_row
        )

        # ---- Logging ----
        B = tri.shape[0]
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=B)
        self.log("val/contrastive", con_loss, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/hardneg", hard_loss, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/sp_self", loss_self, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/sp_cross", loss_cross, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/pair_rank", pair_rank, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/pair_abs", pair_abs, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/pair_row", pair_row, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/avg_pos", avg_pos, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/avg_ign", avg_ign, on_epoch=True, sync_dist=True, batch_size=B)
        self.log("val/avg_hard_teacher_sim", avg_hard, on_epoch=True, sync_dist=True, batch_size=B)

        # ---- Accumulate for epoch-level KPI ----
        self._val_z.append(z_local.detach())
        self._val_t.append(t_local.detach())


    def on_validation_epoch_end(self):
        if len(self._val_z) == 0:
            return
        z_local = torch.cat(self._val_z, dim=0)   # (N_local, D)
        t_local = torch.cat(self._val_t, dim=0)   # (N_local, Dt)

        # pad to same length across ranks, all_gather, then unpad
        n_local = torch.tensor([z_local.shape[0]], device=self.device, dtype=torch.long)
        n_all = self.all_gather(n_local).view(-1)  # (world,)
        max_n = int(n_all.max().item())

        def pad_to(x, tgt_n):
            if x.shape[0] == tgt_n: return x
            pad = torch.zeros((tgt_n - x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)

        z_pad = F.normalize(pad_to(z_local, max_n), dim=-1)
        t_pad = pad_to(t_local, max_n)  # t already normalized upstream

        z_g = self.all_gather(z_pad)  # (world, max_n, D)
        t_g = self.all_gather(t_pad)  # (world, max_n, Dt)

        # build masks to drop padding
        mask = torch.arange(max_n, device=self.device).unsqueeze(0) < n_all.unsqueeze(1)  # (world, max_n)
        z_all = z_g[mask].view(-1, z_g.shape[-1])
        t_all = t_g[mask].view(-1, t_g.shape[-1])

        # full global KPIs
        with torch.no_grad():
            Kz = z_all @ z_all.t()
            Kt = t_all @ t_all.t()
            zvec = upper_triangular_flat(Kz)
            tvec = upper_triangular_flat(Kt)
            pair_mse = F.mse_loss(zvec, tvec)
            zc, tc = zvec - zvec.mean(), tvec - tvec.mean()
            pearson_r = (zc * tc).sum() / ((zc.norm() * tc.norm()).clamp(min=1e-8))

        # log epoch-level KPIs
        self.log("val/pair_mse_global_epoch", pair_mse, prog_bar=True, sync_dist=True)
        self.log("val/pearson_global_epoch",  pearson_r, prog_bar=True, sync_dist=True)

    # =========================
    # Data hooks done
    # =========================


# =========================
# DataModule
# =========================
class LatentDataModule(pl.LightningDataModule):
    def __init__(self, root, batch_size=512, num_workers=64, pin_memory=True):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.steps_per_epoch = None

    def setup(self, stage=None):
        self.train_ds = LatentsNet(self.root, split='train')
        self.val_ds   = LatentsNet(self.root, split='val')

    def train_dataloader(self):
        loader = torch.utils.data.DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            drop_last=True, persistent_workers=self.num_workers > 0,
        )
        self.steps_per_epoch = len(loader)
        return loader

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=self.pin_memory,
            drop_last=False, persistent_workers=self.num_workers > 0,
        )


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/mnt/localssd/direct3d_latents")
    parser.add_argument("--project", type=str, default="pretrain_semantic_predictor")
    parser.add_argument("--run_name", type=str, default="StudentRel-AlignDispersion-GN-epochKPI")
    parser.add_argument("--batch_size", type=int, default=768)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d_token", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--global_pool", type=str, default="attn", choices=["mean","attn"])
    parser.add_argument("--global_dim", type=int, default=768)
    parser.add_argument("--spatial_dim", type=int, default=256)

    parser.add_argument("--teacher_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--teacher_img_size", type=int, default=224)

    # Relational InfoNCE + hard negatives
    parser.add_argument("--contrastive_w", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learn_temp", action="store_true")
    parser.add_argument("--pos_thresh", type=float, default=0.83)
    parser.add_argument("--ign_thresh", type=float, default=0.65)
    parser.add_argument("--hardneg_w", type=float, default=0.25)
    parser.add_argument("--hardneg_topk", type=int, default=10)
    parser.add_argument("--hardneg_margin", type=float, default=0.25)

    # Spatial relational distillation
    parser.add_argument("--spatial_self_w", type=float, default=0.5)
    parser.add_argument("--spatial_cross_w", type=float, default=0.25)
    parser.add_argument("--spatial_tokens", type=int, default=128)
    parser.add_argument("--spatial_tau_self", type=float, default=0.1)
    parser.add_argument("--spatial_tau_cross", type=float, default=0.1)

    # Pairwise alignment & dispersion
    parser.add_argument("--pair_align_w", type=float, default=1.0)
    parser.add_argument("--pair_midrange_weight", action="store_true")
    parser.add_argument("--var_w", type=float, default=1.0)
    parser.add_argument("--cov_w", type=float, default=0.01)
    parser.add_argument("--var_gamma", type=float, default=0.5)

    # Tiny triplane noise/drop (default off)
    parser.add_argument("--tri_noise_std", type=float, default=0.0)
    parser.add_argument("--tri_drop_channel_prob", type=float, default=0.0)

    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    dm  = LatentDataModule(root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    lit = LitMetricStudent(
        lr=args.lr, weight_decay=args.weight_decay,
        depth=args.depth, heads=args.heads, d_token=args.d_token, dropout=args.dropout,
        global_pool=args.global_pool, global_dim=args.global_dim, spatial_dim=args.spatial_dim,
        teacher_model=args.teacher_model, teacher_img_size=args.teacher_img_size,
        contrastive_w=args.contrastive_w, temperature=args.temperature, learn_temp=True, #args.learn_temp,
        pos_thresh=args.pos_thresh, ign_thresh=args.ign_thresh,
        hardneg_w=args.hardneg_w, hardneg_topk=args.hardneg_topk, hardneg_margin=args.hardneg_margin,
        spatial_self_w=args.spatial_self_w, spatial_cross_w=args.spatial_cross_w,
        spatial_tokens=args.spatial_tokens, spatial_tau_self=args.spatial_tau_self, spatial_tau_cross=args.spatial_tau_cross,
        pair_align_w=args.pair_align_w, pair_midrange_weight=args.pair_midrange_weight,
        var_w=args.var_w, cov_w=args.cov_w, var_gamma=args.var_gamma,
        tri_noise_std=args.tri_noise_std, tri_drop_channel_prob=args.tri_drop_channel_prob
    )

    if args.compile:
        try:
            lit.student = torch.compile(lit.student)
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    logger = WandbLogger(project=args.project, name=args.run_name, log_model=True)

    ckpt  = ModelCheckpoint(
        monitor="val/pearson_global_epoch", save_last=True, save_top_k=2, mode="max",
        filename="sem_rel-{epoch:03d}-{pearson_epoch:.4f}", auto_insert_metric_name=False
    )
    lrmon = LearningRateMonitor(logging_interval="step")
    es    = EarlyStopping(monitor="val/pearson_global_epoch", patience=30, mode="max")

    trainer = pl.Trainer(
        accelerator='auto', devices=args.devices, strategy=args.strategy,
        max_epochs=args.max_epochs, precision=args.precision, logger=logger,
        callbacks=[ckpt, lrmon, es], gradient_clip_val=args.grad_clip, log_every_n_steps=10,
    )

    trainer.fit(lit, datamodule=dm)

    save_dir = osp.join(os.getcwd(), "pretrain_weights")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(lit.student.state_dict(), osp.join(save_dir, "semantic_student.pth"))
    print(f"Saved student weights to {osp.join(save_dir, 'semantic_student.pth')}")

if __name__ == "__main__":
    main()
