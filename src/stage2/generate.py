import torch
from typing import Optional, List
from tqdm import tqdm

def _dup_for_cfg(x, text, cfg_scale):
    """Return (x_or_2x, force_drop_ids_or_None, batch_mult)."""
    if cfg_scale > 1.0:
        B = x.shape[0]
        x2 = torch.cat([x, x], dim=0)
        text2 = text * 2 if text is not None else None
        force = torch.cat(
            [
                torch.zeros(B, dtype=torch.long, device=x.device),
                torch.ones(B,  dtype=torch.long, device=x.device),
            ],
            dim=0,
        )
        return x2, text2, force, 2
    else:
        return x, text, None, 1


@torch.no_grad()
def generate(
    model,
    cond,                 
    max_new_tokens,       
    emb_masks=None,       
    cfg_scale=1.0,
    cfg_schedule="constant",   
    temperature: float = 1.0,
    text: Optional[List[str]] = None,          
):
    device = cond.device
    dtype = model.z_proj.weight.dtype
    if torch.is_autocast_enabled():
        dtype = torch.get_autocast_dtype(device_type=device.type)

    B = cond.shape[0]
    S = int(max_new_tokens)
    T = int(model.cls_token_num)

    seq = torch.empty((B, S, model.slot_dim), dtype=dtype, device=device)

    cond0, text0, force0, mult0 = _dup_for_cfg(cond, text, cfg_scale)
    tok0 = model(
        slots=None,
        images=cond0,
        input_pos=None,             
        cfg=cfg_scale,
        temperature=temperature,
        force_drop_ids=force0,
        text=text0,
    )                                
    seq[:, 0:1] = tok0.unsqueeze(1)

    for t in tqdm(range(1, S)):
        slots_so_far = seq[:, :t, :]                       
        cond_t, text_t, force_t, mult_t = _dup_for_cfg(cond, text, cfg_scale)
        if mult_t == 2:
            slots_dup = torch.cat([slots_so_far, slots_so_far], dim=0)  
        else:
            slots_dup = slots_so_far

        tok_t = model(
            slots=slots_dup,          
            images=cond_t,            
            input_pos=None,           
            cfg=cfg_scale,
            temperature=temperature,
            force_drop_ids=force_t,
            text=text_t,
        )                              
        seq[:, t:t+1] = tok_t.unsqueeze(1)

    return seq
