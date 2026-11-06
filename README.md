# Offline Knowledge Distillation for Video Summarization

A frozen, query-conditioned teacher ([UniVTG](https://github.com/showlab/UniVTG), CLIP ViT-B/32) is run once on SumMe and TVSum and its scores are saved to disk. A small query-free Transformer student is then trained on the standard GoogLeNet features, using the ground-truth scores plus the saved teacher scores.

## Setup

### 1. Conda environment

```bash
conda create -n kd python=3.10 -y
conda activate kd
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch 2.1.2 built for CUDA 11.8. For a different CUDA version, install the matching `torch`/`torchvision` from [pytorch.org](https://pytorch.org/get-started/previous-versions/) first.

Teacher feature extraction also needs the `ffmpeg` binary:

```bash
sudo apt install ffmpeg        # or: conda install -c conda-forge ffmpeg
```

### 2. UniVTG code and checkpoint

```bash
mkdir -p third_party checkpoints
git clone https://github.com/showlab/UniVTG.git third_party/UniVTG
```

Download the **CLIP-B/32, fine-tuned on QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo** checkpoint from the [UniVTG model zoo](https://github.com/showlab/UniVTG/blob/main/model.md) and place `model_best.ckpt` and `opt.json` in `checkpoints/univtg_clipb32_ft/`.

The CLIP ViT-B/32 weights used for feature extraction are downloaded automatically to `~/.cache/clip` on first run.

### 3. Data

Put these files into `dataset/`:

| File | Source |
|---|---|
| `eccv16_dataset_summe_google_pool5.h5`, `eccv16_dataset_tvsum_google_pool5.h5` | Preprocessed SumMe/TVSum features from [DSN (K. Zhou)](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce) |
| `summe_splits.json`, `tvsum_splits.json` | Standard 5 splits, included in this repo |
| `ydata-tvsum50-anno.tsv`, `ydata-tvsum50-info.tsv` | [TVSum release](https://github.com/yalesong/tvsum) |

Raw videos are needed only to build the teacher inputs (step 1 below):

- SumMe videos from the [SumMe page](https://gyglim.github.io/me/vsum/index.html) go into `Videos/SumMe/` (spaces in file names become `_`, e.g. `Air_Force_One.mp4`).
- TVSum videos go into `Videos/TVSum/` as `<youtube_id>.mp4`.

`Videos/{summe,tvsum}_mapped_video_names.json` map file names to the `video_N` keys of the h5 files. They are included in this repo.

### Folder layout

```
.
├── common.py                    # paths, teacher clip -> student step mapping
├── evaluation.py                # F1 / Kendall tau / Spearman rho
├── extract_teacher_feats.py     # step 1
├── teacher_infer.py             # step 2
├── train_student.py             # step 3
├── models/
│   └── student.py
├── requirements.txt
├── checkpoints/
│   └── univtg_clipb32_ft/
│       ├── model_best.ckpt
│       └── opt.json
├── third_party/
│   └── UniVTG/                  # git clone
├── dataset/
│   ├── eccv16_dataset_summe_google_pool5.h5
│   ├── eccv16_dataset_tvsum_google_pool5.h5
│   ├── summe_splits.json
│   ├── tvsum_splits.json
│   ├── ydata-tvsum50-anno.tsv
│   ├── ydata-tvsum50-info.tsv
│   ├── {summe,tvsum}_video_clip.h5     # created by step 1
│   ├── {summe,tvsum}_queries_name.h5   # created by step 1
│   └── {summe,tvsum}_teacher.h5        # created by step 2
├── Videos/
│   ├── SumMe/*.mp4
│   ├── TVSum/*.mp4
│   ├── summe_mapped_video_names.json
│   └── tvsum_mapped_video_names.json
└── results/                     # created by step 3
```

## Usage

Run all commands from the repository root. A single GPU is enough (tested on a 12 GB TITAN Xp).

### Step 1. Teacher inputs

```bash
python extract_teacher_feats.py
```

Encodes every video with CLIP ViT-B/32 at one frame per 2 s into `dataset/{ds}_video_clip.h5` (`[Tc, 512]`). It also encodes one text query per video into `dataset/{ds}_queries_name.h5`: the video name for SumMe and the video title for TVSum.

### Step 2. Teacher scores

```bash
python teacher_infer.py
```

Runs the frozen teacher once and writes `dataset/{ds}_teacher.h5` with `{key}/saliency` and `{key}/foreground`, both `[Tc]`. It also prints the teacher's own F1 / τ / ρ on the 5 test splits.

### Step 3. Student training

Baseline without KD:

```bash
python train_student.py --dataset summe --exp summe_base --layers 4 --epochs 200
```

With KD:

```bash
python train_student.py --dataset tvsum --exp tvsum_kd_corr \
    --layers 4 --epochs 200 --kd --kd_loss corr --kd_signal f+s --kd_alpha 0.5
```

Main options:

| Option | Meaning |
|---|---|
| `--dataset` | `summe` or `tvsum` |
| `--exp` | output folder name under `results/` |
| `--seed` | one or more seeds (default `1 2 3 4 5`) |
| `--splits` | splits to run (default `0 1 2 3 4`) |
| `--select` | metric for picking the best epoch: `tau` (default) or `rho` |
| `--d`, `--heads`, `--layers`, `--ffn`, `--dropout`, `--attn_dropout`, `--no_pos` | student architecture |
| `--epochs`, `--lr`, `--wd` | training |
| `--kd` | enable distillation |
| `--kd_signal` | teacher signal: `s` saliency, `f` foreground, `f+s` sum of both (z-scored per video) |
| `--kd_loss` | `kl`: MSE + α·T²·KL(softmax) over time; `corr`: MSE + α·(1 − Pearson); `soft`: MSE against (1−α)·gt + α·teacher |
| `--kd_alpha` | KD weight (mixing ratio for `soft`) |
| `--kd_temp` | softmax temperature for `kl` |
| `--kd_interp` | map teacher clips to student steps by linear interpolation instead of nearest clip |

Run `python train_student.py -h` for the full list.

Quick smoke test:

```bash
python train_student.py --dataset summe --exp /tmp/smoke --splits 0 --seed 1 --epochs 2 --kd --kd_loss corr
```

### Output

```
results/<exp>/
├── config.json
├── results.txt                  # final F1 / tau / rho per seed and the mean over seeds
└── seed<s>/
    ├── results.txt              # mean ± std per split and the seed's average
    └── split<i>/best.pt         # weights of the best epoch
```

## Evaluation protocol

`evaluation.py` follows the PGL-SUM protocol. Step scores are expanded to frames and averaged per KTS segment, and a 0/1 knapsack selects 15% of the video. F1 is the max over users on SumMe and the mean over users on TVSum. Kendall τ and Spearman ρ are computed against the user annotations. The best epoch is chosen on the test split (`--select`), as in PGL-SUM.

## Acknowledgements

The teacher model and CLIP feature extractor come from [UniVTG](https://github.com/showlab/UniVTG). Preprocessed features are from [DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce).
