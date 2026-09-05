# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from typing import Optional, List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
import open_clip
from src.stage1.vision_transformer import DropPath
from src.stage2.diffloss import DiffLoss

def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)



#################################################################################
#                      Embedding Layers for OpenCLIP Feats                      #
#################################################################################
class ConditionEmbedder(nn.Module):
    """
    Projects OpenCLIP features (image or text) to model dim and emits a fixed
    number of conditioning tokens to prepend. Includes dropout for CFG.
    Inputs:
      feats: [B, E] or [B, L, E] OpenCLIP embeddings (image/text).
    Outputs:
      tokens: [B, C, D] where C = max_tokens (cls_token_num), D = model dim.
    """
    def __init__(self, in_dim: int, model_dim: int, max_tokens: int = 1, dropout_prob: float = 0.1):
        super().__init__()
        self.in_dim = in_dim
        self.model_dim = model_dim
        self.max_tokens = max_tokens
        self.dropout_prob = dropout_prob

        self.proj = nn.Linear(in_dim, model_dim, bias=False)
        self.ln = nn.LayerNorm(model_dim, eps=1e-5)
        # learned null token used when dropping for CFG or padding
        self.null_tokens = nn.Parameter(torch.zeros(1, max_tokens, model_dim))
        nn.init.normal_(self.null_tokens, std=0.02)

    def token_drop(self, tokens, train: bool, force_drop_ids: Optional[torch.Tensor] = None):
        if not train and force_drop_ids is None:
            return tokens
        B = tokens.shape[0]
        if force_drop_ids is None:
            drop = torch.rand(B, device=tokens.device) < self.dropout_prob
        else:
            drop = (force_drop_ids == 1)
        null = self.null_tokens.expand(B, self.max_tokens, -1)
        tokens = torch.where(drop.view(B, 1, 1), null, tokens)
        return tokens

    def forward(self, feats: torch.Tensor, train: bool, force_drop_ids: Optional[torch.Tensor] = None):
        assert feats.dim() in (2, 3), "cond feats must be [B,E] or [B,L,E]"
        if feats.dim() == 2:
            feats = feats.unsqueeze(1)  # [B,1,E]
        B, L, E = feats.shape
        x = self.proj(feats)           # [B,L,D]
        x = self.ln(x)
        # choose/pad to fixed number of condition tokens
        if L >= self.max_tokens:
            tokens = x[:, :self.max_tokens, :]
        else:
            pad = self.null_tokens[:, : self.max_tokens - L, :].expand(B, -1, -1)
            tokens = torch.cat([x, pad], dim=1)
        tokens = self.token_drop(tokens, train=train, force_drop_ids=force_drop_ids)
        return tokens


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        multiple_of: int = 256,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = find_multiple(hidden_dim, multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.ffn_dropout = nn.Dropout(ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype, device=None):
        super().__init__()
        device = device if device is not None else torch.device("cpu")
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype, device=device))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val
        return k_out, v_out


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
    ):
        super().__init__()
        assert dim % n_head == 0
        self.dim = dim
        self.head_dim = dim // n_head
        self.n_head = n_head

        self.wqkv = nn.Linear(dim, dim * 3, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.kv_cache = None
        self.attn_dropout_p = attn_dropout_p
        self.resid_dropout = nn.Dropout(resid_dropout_p)

    def forward(self, x: torch.Tensor, input_pos: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,S,D)
        xk = xk.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)             
        else:
            keys, values = xk, xv                                              

        if mask is not None and mask.dim() == 4 and mask.size(1) == 1:
            mask = mask.expand(-1, self.n_head, -1, -1).contiguous()

        out = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=False if mask is not None else True,
            dropout_p=self.attn_dropout_p if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.resid_dropout(self.wo(out))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        attn_dropout_p: float = 0.0,
        ffn_dropout_p: float = 0.1,
        resid_dropout_p: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attention = Attention(
            dim=dim,
            n_head=n_head,
            attn_dropout_p=attn_dropout_p,
            resid_dropout_p=resid_dropout_p,
        )
        self.feed_forward = FeedForward(
            dim=dim,
            multiple_of=multiple_of,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor] = None):
        h = x + self.drop_path(self.attention(self.attention_norm(x), start_pos, mask))
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int = 4096,
        n_layer: int = 32,
        n_head: int = 32,
        attn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.1,
        ffn_dropout_p: float = 0.1,
        drop_path_rate: float = 0.0,

        num_classes: Union[int, List[int], None] = None,
        class_dropout_prob: float = 0.0,

        cond_in_dim: Optional[int] = 512,         
        cond_dropout_prob: float = 0.1,            

        cls_token_num: int = 1,                    
        num_slots: int = 16,
        slot_dim: int = 256,

        diffloss_d: int = 3,
        diffloss_w: int = 1024,
        num_sampling_steps: str = '100',
        diffusion_batch_mul: int = 4,
        predict_xstart: bool = False,
        use_si: bool = False,
        cond_method: str = "adaln",                
        **kwargs,
    ):
        super().__init__()

        # Store configuration
        self.dim = dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.cls_token_num = cls_token_num
        self.diffusion_batch_mul = diffusion_batch_mul


        self.cond_in_dim = cond_in_dim
        self.cond_dropout_prob = cond_dropout_prob
        self.cond_embedder = None
        if cond_in_dim is not None and cls_token_num > 0:
            self.cond_embedder = ConditionEmbedder(
                in_dim=cond_in_dim, model_dim=dim,
                max_tokens=cls_token_num, dropout_prob=cond_dropout_prob
            )

        self.z_proj = nn.Linear(slot_dim, dim, bias=True)
        self.z_proj_ln = RMSNorm(dim)
        self.pos_embed_learned = nn.Parameter(torch.zeros(1, num_slots + cls_token_num, dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(n_layer):
            self.layers.append(TransformerBlock(
                dim=dim,
                n_head=n_head,
                ffn_dropout_p=ffn_dropout_p,
                attn_dropout_p=attn_dropout_p,
                resid_dropout_p=resid_dropout_p,
                drop_path=dpr[layer_id],
            ))

        for blk in self.layers:
            blk.attention._cond_len = int(self.cls_token_num)

        self.norm = RMSNorm(dim)

        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, num_slots, dim))

        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k'
        )
        self.clip = clip_model
        self.clip_preprocess = preprocess
        self.clip.eval()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.diffloss = DiffLoss(
            target_channels=slot_dim,
            z_channels=self.dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            predict_xstart=predict_xstart,
            use_si=use_si,
            cond_method=cond_method,
        )

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed_learned, std=0.02)
        nn.init.normal_(self.diffusion_pos_embed_learned, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(std=0.02)

    def encode_clip_conditions(self, img: torch.Tensor):
        """
        Encode images using OpenCLIP to get conditioning features.
        Inputs:
          img: [B, 3, H, W] input images normalized to CLIP range.
        Outputs:
          img_feats: [B, 512] normalized CLIP image features.
        """
        with torch.no_grad():
            img_feats = self.clip.encode_image(img)           
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        return img_feats


    def encode_clip_text(self, tokens):
        with torch.no_grad():
            text_feats = self.clip.encode_text(tokens)           
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        return text_feats

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        head_dim = self.dim // self.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        device = self.pos_embed_learned.device

        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.n_head, head_dim, dtype, device=device)
            b.attention.parent = self

        block_future = torch.triu(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool, device=device),
            diagonal=1
        )
        self.register_buffer(
            "causal_mask",
            block_future.unsqueeze(0).repeat(self.max_batch_size, 1, 1),  
            persistent=False,
        )


    def reset_caches(self):
        self.max_seq_length = -1
        self.max_batch_size = -1
        for b in self.layers:
            b.attention.kv_cache = None

    def forward_loss(self, z, target):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target)
        return loss

    def forward_cfg(self, h, cfg):
        if cfg > 1.0:
            h_cond, h_uncond = h.chunk(2, dim=0)
            h = h_uncond + cfg * (h_cond - h_uncond)
        return h

    def forward(
        self,
        slots: torch.Tensor,                            
        images: Optional[torch.Tensor] = None,          
        text: List[Optional[str]] = None,          
        input_pos:  Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,            
        cfg: float = 1.0,
        temperature: float = 1.0,
        force_drop_ids: Optional[torch.Tensor] = None,  
    ):
        """
        Training / naive inference: provide both slots and images.
        KV-cache incremental inference:
          - prefill: slots=None, images=..., input_pos=range(C)
          - decode:  slots=...,  images=None, input_pos=[...]
        """
        cond_tokens = None
        if text:
            text_tokens = self.tokenizer(text).to(images.device)   
            cond_feats = self.encode_clip_text(text_tokens)
            cond_tokens = self.cond_embedder(cond_feats, train=self.training, force_drop_ids=force_drop_ids)  
        elif images is not None:
            cond_feats = self.encode_clip_conditions(images)  
            cond_tokens = self.cond_embedder(cond_feats, train=self.training, force_drop_ids=force_drop_ids)  

        if slots is not None:
            z_tokens = self.z_proj(slots)  
            if (cond_tokens is not None) and (self.cls_token_num > 0):
                token_embeddings = torch.cat([cond_tokens, z_tokens], dim=1)  
            else:
                token_embeddings = z_tokens                                     
        else:
            assert cond_tokens is not None, "prefill requires images to produce condition tokens"
            token_embeddings = cond_tokens                                       

        attn_bias = None
        if (not self.training) and (input_pos is not None):
            bs = token_embeddings.shape[0]
            if input_pos.numel() == 1:
                mask_bool = self.causal_mask[:bs, None, input_pos]               
            else:
                mask_bool = self.causal_mask[:bs][:, input_pos, :].unsqueeze(1)  

            mdtype = token_embeddings.dtype
            attn_bias = torch.zeros_like(mask_bool, dtype=mdtype)
            attn_bias.masked_fill_(mask_bool, torch.finfo(mdtype).min)           

        if self.training or input_pos is None:
            h = token_embeddings + self.pos_embed_learned[:, :token_embeddings.shape[1], :]
        else:
            h = token_embeddings + self.pos_embed_learned[:, input_pos].view(1, -1, self.dim)

        h = self.z_proj_ln(h)

        for layer in self.layers:
            h = layer(h, input_pos, attn_bias)

        h = self.norm(h)

        if self.training:
            C = self.cls_token_num if ('cond_tokens' in locals() and cond_tokens is not None) else 0
            start = max(C - 1, 0)
            h = h[:, start:-1, :].contiguous()                            
            h = h + self.diffusion_pos_embed_learned[:, :h.shape[1], :]
            loss = self.forward_loss(h, slots.detach())
            return loss
        else:
            last = h[:, -1, :]  # [B, D]

            if input_pos is None:
                L = h.shape[1]  # C + generated_so_far
                slot_idx = L - 1 - (self.cls_token_num - 1)
                slot_idx = torch.tensor([slot_idx], device=h.device)
            else:
                slot_idx = input_pos[-1] - (self.cls_token_num - 1)
            slot_idx = torch.clamp(slot_idx, min=0, max=self.num_slots - 1).to(torch.long)

            pos = self.diffusion_pos_embed_learned[:, slot_idx].view(self.diffusion_pos_embed_learned.shape[-1])
            last = last + pos
            next_tokens = self.diffloss.sample(last, temperature=temperature, cfg=cfg)  
            return next_tokens


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
def GPT_7B(**kwargs):
    return Transformer(n_layer=32, n_head=32, dim=4096, **kwargs) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(n_layer=24, n_head=32, dim=3200, **kwargs) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(n_layer=22, n_head=32, dim=2048, **kwargs) # 1.2B

def GPT_XXXL(**kwargs):
    return Transformer(n_layer=48, n_head=40, dim=2560, **kwargs) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(n_layer=48, n_head=24, dim=1536, **kwargs) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(n_layer=36, n_head=20, dim=1280, **kwargs) # 775M

def GPT_L(**kwargs):
    return Transformer(n_layer=24, n_head=16, dim=1024, **kwargs) # 343M

def GPT_B(**kwargs):
    return Transformer(n_layer=12, n_head=12, dim=768, **kwargs) # 111M


GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B,
}
