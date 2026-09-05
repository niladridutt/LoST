# LoST: Level of Semantics Tokenization for 3D Shapes

[Project Webpage](https://lost3d.github.io/) | [Paper](https://arxiv.org/abs/2603.17995)

![LoST teaser](teaser.png)

LoST is a tokenizer for 3D shapes that orders tokens by semantic salience. Short token prefixes decode into complete, plausible shapes that capture the main semantics of an object, while later tokens refine instance-specific geometry and semantic detail. The tokenizer is trained over Direct3D VAE triplane latents and can be used as a compact representation for autoregressive 3D generation.[^semanticist][^direct3d]

The training pipeline has three model stages:

- **Stage 0: RIDA pretraining.** Pretrain the semantic extractor used as a semantic alignment loss for the tokenizer.
- **Stage 1: semantic tokenizer.** Train LoST to encode Direct3D VAE latents into prefix-decodable semantic tokens.
- **Stage 2: autoregressive model.** Train a GPT-style model on LoST tokens for image-conditioned 3D generation.

## Installation

Create an environment and install the main LoST dependencies:

```bash
pip install -r requirements.txt
```

Direct3D is used for image-to-3D latent generation and VAE latent decoding. It has additional requirements:

```bash
cd Direct3D
pip install -r requirements.txt
pip install -e .
cd ..
```

Some data preparation scripts also use external services/models:

- `scripts/generate_prompts.py` uses Gemini and expects `GEMINI_API_KEY`.
- `scripts/generate_images.py` and `scripts/generate_images_multi.py` use FLUX.1-dev through `diffusers`.
- Direct3D scripts use `DreamTechAI/Direct3D`.

## Data Preparation

LoST is trained on Direct3D VAE latents. The released scripts build these latents from generated prompts and images.

### 1. Generate Text Prompts

Generate diverse object prompts with Gemini:

```bash
python scripts/generate_prompts.py
```

This writes prompts to `prompts.txt`. The prompt template follows the one used in the paper supplement: prompts focus on single, visually distinctive 3D objects with broad category diversity.

### 2. Generate Images from Prompts

Generate FLUX images from `prompts.txt`:

```bash
python scripts/generate_images.py
```

For multi-GPU generation:

```bash
python scripts/generate_images_multi.py
```

Both scripts write local PNGs, a `manifest.jsonl`, and a `done_lines.txt` file for resumability. Edit the constants at the top of each script to change paths, batch size, image resolution, or model settings.

### 3. Generate Direct3D VAE Latents

Convert generated images into Direct3D latents:

```bash
python Direct3D/generate_3d_latents.py
```

For multi-GPU generation:

```bash
python Direct3D/generate_3d_latents_multi.py
```

These scripts read `manifest.jsonl`, save local VAE latents under `LOCAL_LATENTS_DIR`, optionally save meshes, and write `manifest_3d.jsonl` plus `done_3d.txt`.

Use these latents as the training data for Stage 1.

## Stage 0: RIDA Pretraining

RIDA, Relational Inter-Distance Alignment, pretrains a semantic extractor that maps 3D triplane latents into a semantically structured feature space. This provides the semantic guidance used during Stage 1 tokenizer training.

RIDA needs:

- Direct3D VAE latents
- the corresponding FLUX images

Run:

```bash
python pretrain_rida.py
```

Edit paths and training constants inside [pretrain_rida.py](pretrain_rida.py) for your dataset layout.

## Stage 1: Train the Semantic Tokenizer

The tokenizer learns a sequence of LoST tokens from Direct3D VAE latents. It uses causal masking and nested dropout so that earlier prefixes carry the principal semantic content, while later tokens add finer details.

Train with:

```bash
accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/tokenizer_l.yaml
```

Important config fields live in [configs/tokenizer_l.yaml](configs/tokenizer_l.yaml):

- `trainer.params.dataset.params.root`: root directory containing Direct3D latents
- `trainer.params.test_dataset.params.root`: validation latent root
- `trainer.params.result_folder`: output checkpoint directory
- `trainer.params.model.params.num_slots`: maximum number of LoST tokens
- `trainer.params.model.params.enable_nest_after`: epoch after which nested dropout is enabled

We train initially without nested dropout, then enables nested dropout so the model first learns full-capacity reconstruction before learning the prefix hierarchy.

## Tokenize Latents for Stage 2

After Stage 1 is trained, encode Direct3D latents into LoST slot tokens:

```bash
python ar_slot_encode.py
```

This creates the tokenized training data used by the Stage 2 autoregressive model. Edit the constants at the top of [ar_slot_encode.py](ar_slot_encode.py), especially `CKPT_PATH`, `LATENT_DIR`, and `OUTPUT_DIR`.

To decode LoST slot tokens back into Direct3D VAE latents:

```bash
python ar_slot_decode.py
```

This is useful for checking tokenizer outputs and for decoding AR outputs before converting them into 3D meshes.

## Stage 2: Train the Autoregressive Model

Train the AR model on encoded LoST slot tokens:

```bash
accelerate launch --config_file=configs/onenode_config.yaml train_net.py --cfg configs/autoregressive_l.yaml
```

Important config fields live in [configs/autoregressive_l.yaml](configs/autoregressive_l.yaml):

- `trainer.params.dataset.params.root`: root directory containing encoded LoST slots
- `trainer.params.test_dataset.params.root`: validation slot root
- `trainer.params.train_num_slots`: number of slots used for AR training
- `trainer.params.gpt_model.params.num_slots`: generated sequence length
- `trainer.params.result_folder`: output checkpoint directory

## Demos and Inference

Tokenizer reconstruction demo:

```bash
python tokenizer_demo.py
```

Autoregressive image-conditioned generation demo:

```bash
python ar_demo.py
```

Both scripts are local-only and use constants at the top for paths, checkpoints, batch size, and inference settings.

## Direct3D Utilities

Direct3D is used as the VAE representation and final mesh decoder.

Image to 3D VAE latents:

```bash
python Direct3D/generate_3d_latents.py
python Direct3D/generate_3d_latents_multi.py
```

Decode VAE latents to `.glb` meshes:

```bash
python Direct3D/decode_vae.py \
  --inputs "/path/to/predictions/*.pt" \
  --output-dir /path/to/output_meshes
```

Encode meshes with the Direct3D VAE and decode them back for reconstruction checks:

```bash
python Direct3D/recon_vae.py
```

Direct3D-specific dependencies are listed in [Direct3D/requirements.txt](Direct3D/requirements.txt).

## Repository Map

- `scripts/`: data preparation, prompt generation, image generation
- `Direct3D/`: Direct3D latent generation, VAE decoding, VAE reconstruction tools
- `pretrain_rida.py`: Stage 0 RIDA semantic extractor pretraining
- `train_net.py`: shared entry point for Stage 1 and Stage 2 training
- `configs/tokenizer_l.yaml`: Stage 1 tokenizer config
- `configs/autoregressive_l.yaml`: Stage 2 AR config
- `ar_slot_encode.py`: encode Direct3D latents into LoST tokens
- `ar_slot_decode.py`: decode LoST tokens back into Direct3D latents
- `tokenizer_demo.py`: tokenizer reconstruction demo
- `ar_demo.py`: AR generation demo
- `src/stage1/`: LoST tokenizer modules
- `src/stage2/`: autoregressive model and sampling

## Citation

```bibtex
@article{dutt2026lost,
  title   = {LoST: Level of Semantics Tokenization for 3D Shapes},
  author  = {Dutt, Niladri Shekhar and Shi, Zifan and Guerrero, Paul and Huang, Chun-Hao Paul and Ceylan, Duygu and Mitra, Niloy J. and Chen, Xuelin},
  journal = {arXiv preprint arXiv:2603.17995},
  year    = {2026}
}
```

## Acknowledgements

We thank [Semanticist](https://github.com/visual-gen/semanticist), which we used as the base for LoST, and [FlexTok](https://github.com/apple/ml-flextok), which inspired the level-of-semantics tokenization design. We also thank [Direct3D](https://github.com/DreamTechAI/Direct3D) for the 3D VAE representation and image-to-3D data generation pipeline.

[^semanticist]: LoST builds on [Semanticist](https://github.com/visual-gen/semanticist) as a codebase foundation and takes additional inspiration from [FlexTok](https://github.com/apple/ml-flextok) for flexible level-of-semantics tokenization.
[^direct3d]: LoST uses [Direct3D](https://github.com/DreamTechAI/Direct3D) for the 3D VAE latent representation, image-to-3D data generation, and final latent-to-mesh decoding.
