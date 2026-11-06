import os
import sys
import json
import argparse
import h5py
import numpy as np
import torch

from common import UNIVTG_DIR, ROOT, DATASET_DIR, video_feat_path, query_feat_path, teacher_path, to_steps
from evaluation import evaluate_video

sys.path.insert(0, UNIVTG_DIR)
from model.univtg import build_model  # noqa: E402

CKPT_DIR = os.path.join(ROOT, 'checkpoints', 'univtg_clipb32_ft')
EVAL_FIELDS = ['change_points', 'n_frames', 'n_frame_per_seg', 'picks', 'user_summary']
METRICS = ['f1', 'tau', 'rho']


def load_teacher(device):
    opt = argparse.Namespace(**json.load(open(os.path.join(CKPT_DIR, 'opt.json'))))
    opt.v_feat_dim += 2  # TEF (ctx_mode = video_tef), UniVTG config.py ile aynı
    opt.device = device
    model, _ = build_model(opt)
    ckpt = torch.load(os.path.join(CKPT_DIR, 'model_best.ckpt'), map_location='cpu')
    model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['model'].items()})
    return model.to(device).eval()


def l2norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@torch.no_grad()
def run_teacher(model, vid, txt, device):
    vid = torch.from_numpy(l2norm(vid))
    txt = torch.from_numpy(l2norm(txt))
    Tc = vid.shape[0]
    tef_st = torch.arange(Tc, dtype=torch.float32) / Tc
    vid = torch.cat([vid, torch.stack([tef_st, tef_st + 1.0 / Tc], dim=1)], dim=1)  # (Tc, 514)

    out = model(src_vid=vid[None].to(device), src_vid_mask=torch.ones(1, Tc, device=device),
                src_txt=txt[None].to(device), src_txt_mask=torch.ones(1, len(txt), device=device))
    return out['saliency_scores'][0].cpu().numpy(), out['pred_logits'][0, :, 0].cpu().numpy()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_teacher(device)

    for dataset in ['summe', 'tvsum']:
        eval_h5 = h5py.File(os.path.join(DATASET_DIR, f'eccv16_dataset_{dataset}_google_pool5.h5'), 'r')
        vid_h5 = h5py.File(video_feat_path(dataset), 'r')
        txt_h5 = h5py.File(query_feat_path(dataset, 'name'), 'r')

        data = {}
        with h5py.File(teacher_path(dataset), 'w') as f:
            for key in vid_h5.keys():
                s, fg = run_teacher(model, vid_h5[key][()], txt_h5[key][()], device)
                f[f'{key}/saliency'] = s
                f[f'{key}/foreground'] = fg

                v = {name: eval_h5[key][name][()] for name in EVAL_FIELDS}
                picks, n_frames = v['picks'], int(v['n_frames'])
                v['s'], v['f'] = to_steps(s, picks, n_frames), to_steps(fg, picks, n_frames)
                v['f+s'] = v['f'] + v['s']
                data[key] = v

        # sadece değerlendirme: student ile aynı 5 test kümesi, split ortalaması
        splits = json.load(open(os.path.join(DATASET_DIR, f'{dataset}_splits.json')))
        for signal in ['s', 'f', 'f+s']:
            per_split = []
            for split in splits:
                res = [evaluate_video(data[k][signal], data[k], k, dataset) for k in split['test_keys']]
                per_split.append({m: np.mean([r[m] for r in res]) for m in METRICS})
            mean = {m: np.mean([r[m] for r in per_split]) for m in METRICS}
            print(f'[{dataset}] teacher {signal:4s}  F1 {mean["f1"]:.2f}  tau {mean["tau"]:.3f}  rho {mean["rho"]:.3f}')


if __name__ == '__main__':
    main()
