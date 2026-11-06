import os
import json
import random
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import ROOT, DATASET_DIR, teacher_path, to_steps, to_steps_interp
from models.student import Student
from evaluation import evaluate_video

METRICS = ['f1', 'tau', 'rho']
EVAL_FIELDS = ['change_points', 'n_frames', 'n_frame_per_seg', 'picks', 'user_summary']


def zscore(x):
    return ((x - x.mean()) / (x.std() + 1e-8)).astype(np.float32)


def teacher_score(f, key, signal):
    s = f[key]['saliency'][()]
    p = np.clip(f[key]['foreground'][()], 1e-6, 1 - 1e-6)
    fg = np.log(p / (1 - p))
    if signal == 's':
        return zscore(s)
    if signal == 'f':
        return zscore(fg)
    return zscore(zscore(fg) + zscore(s))


def load_data(dataset, kd_signal=None, interp=False):
    align = to_steps_interp if interp else to_steps
    path = os.path.join(DATASET_DIR, f'eccv16_dataset_{dataset}_google_pool5.h5')
    data = {}
    with h5py.File(path, 'r') as f:
        for key in f.keys():
            v = {name: f[key][name][()] for name in EVAL_FIELDS}
            v['features'] = f[key]['features'][()].astype(np.float32)
            gt = f[key]['gtscore'][()].astype(np.float32)
            v['gtscore'] = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
            data[key] = v

    if kd_signal:
        with h5py.File(teacher_path(dataset), 'r') as f:
            for key, v in data.items():
                v['teacher'] = align(teacher_score(f, key, kd_signal), v['picks'], int(v['n_frames']))
    return data


def kd_loss(logit, teacher, T):
    return F.kl_div(F.log_softmax(logit / T, dim=1), F.softmax(teacher / T, dim=1),
                    reduction='batchmean') * T * T


def corr_loss(logit, teacher):
    x = logit - logit.mean(dim=1, keepdim=True)
    y = teacher - teacher.mean(dim=1, keepdim=True)
    r = (x * y).sum(dim=1) / (x.norm(dim=1) * y.norm(dim=1) + 1e-8)
    return (1 - r).mean()


def minmax(x):
    return ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)


def to_tensor(x, device):
    return torch.from_numpy(x).unsqueeze(0).to(device)  # (1, T, ...)


@torch.no_grad()
def predict(model, data, keys, device):
    model.eval()
    return {k: torch.sigmoid(model(to_tensor(data[k]['features'], device)))[0].cpu().numpy() for k in keys}


def evaluate(preds, data, dataset):
    per_video = [evaluate_video(p, data[k], k, dataset) for k, p in preds.items()]
    res = {}
    for m in METRICS:
        vals = [r[m] for r in per_video]
        res[m], res[m + '_std'] = float(np.mean(vals)), float(np.std(vals))
    return res


def train_split(args, data, split, out_dir, device):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.makedirs(out_dir, exist_ok=True)

    model = Student(d=args.d, heads=args.heads, layers=args.layers, ffn=args.ffn,
                    dropout=args.dropout, attn_dropout=args.attn_dropout,
                    use_pos=not args.no_pos).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = nn.MSELoss()

    train_keys = list(split['train_keys'])
    best = {m: -1.0 for m in METRICS}
    for epoch in range(1, args.epochs + 1):
        model.train()
        random.shuffle(train_keys)
        total, total_kd = 0.0, 0.0
        for key in train_keys:
            logit = model(to_tensor(data[key]['features'], device))  # (1, T)
            gt = data[key]['gtscore']
            if args.kd and args.kd_loss == 'soft':  # target = (1-λ)·gt + λ·minmax(teacher), λ = kd_alpha
                gt = (1 - args.kd_alpha) * gt + args.kd_alpha * minmax(data[key]['teacher'])
            loss = criterion(torch.sigmoid(logit), to_tensor(gt, device))
            total += loss.item()
            if args.kd and args.kd_loss != 'soft':
                teacher = to_tensor(data[key]['teacher'], device)
                kd = kd_loss(logit, teacher, args.kd_temp) if args.kd_loss == 'kl' else corr_loss(logit, teacher)
                loss = loss + args.kd_alpha * kd
                total_kd += kd.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        preds = predict(model, data, split['test_keys'], device)
        res = evaluate(preds, data, args.dataset)
        if res[args.select] > best[args.select]:
            best = {**res, 'epoch': epoch}
            torch.save(model.state_dict(), os.path.join(out_dir, 'best.pt'))

        if epoch % 10 == 0 or epoch == 1:
            print(f'  epoch {epoch:3d}  mse {total / len(train_keys):.4f}  kd {total_kd / len(train_keys):.4f}  '
                  f'f1 {res["f1"]:.2f}  tau {res["tau"]:.3f}  rho {res["rho"]:.3f}')
    return best


def summary_table(results, label='split', note=None):

    final = {}
    for m in METRICS:
        vals = [r[m] for r in results.values()]
        final[m], final[m + '_std'] = float(np.mean(vals)), float(np.std(vals))

    def row(name, r):
        return (f'{name:10s} {r.get("epoch", ""):>5}   {r["f1"]:6.2f} ± {r["f1_std"]:5.2f}   '
                f'{r["tau"]:.3f} ± {r["tau_std"]:.3f}   {r["rho"]:.3f} ± {r["rho_std"]:.3f}')

    lines = [f'{label:10s} {"epoch":>5s}   {"F1 (%)":15s}   {"tau":13s}   {"rho":13s}']
    lines += [row(name, r) for name, r in results.items()]
    lines.append(row('final', final))
    if note:
        lines.append('\n' + note)
    return final, '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['summe', 'tvsum'], required=True)
    p.add_argument('--exp', required=True, help='experiment name (folder under results/)')
    p.add_argument('--splits', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--select', choices=['tau', 'rho'], default='tau', help='metric used to pick the best epoch')
    # model (ablation)
    p.add_argument('--d', type=int, default=256)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--ffn', type=int, default=512)
    p.add_argument('--dropout', type=float, default=0.5)
    p.add_argument('--attn_dropout', type=float, default=0.3)
    p.add_argument('--no_pos', action='store_true', help='disable positional encoding')
    # train
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--seed', type=int, nargs='+', default=[1, 2, 3, 4, 5], help='one or more seeds')
    # knowledge distillation
    p.add_argument('--kd', action='store_true', help='enable KD (loss type set by --kd_loss)')
    p.add_argument('--kd_signal', choices=['s', 'f', 'f+s'], default='f+s',
                   help='teacher signal: saliency, foreground, or their sum')
    p.add_argument('--kd_loss', choices=['kl', 'corr', 'soft'], default='kl',
                   help='kl: MSE + α·KL(softmax) | corr: MSE + α·(1-Pearson) | soft: MSE((1-α)·gt + α·teacher)')
    p.add_argument('--kd_alpha', type=float, default=1.0, help='KD loss weight (for soft: mixing ratio λ)')
    p.add_argument('--kd_temp', type=float, default=1.0, help='KL temperature (applied after z-score)')
    p.add_argument('--kd_interp', action='store_true',
                   help='map teacher clips to student steps by linear interpolation instead of nearest clip')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_data(args.dataset, args.kd_signal if args.kd else None, args.kd_interp)
    splits = json.load(open(os.path.join(DATASET_DIR, f'{args.dataset}_splits.json')))

    exp_dir = os.path.join(ROOT, 'results', args.exp)
    os.makedirs(exp_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(exp_dir, 'config.json'), 'w'), indent=2)

    seed_results, report = {}, []
    for seed in args.seed:
        args_s = argparse.Namespace(**{**vars(args), 'seed': seed})
        seed_dir = os.path.join(exp_dir, f'seed{seed}')
        results = {}
        for i in args.splits:
            print(f'[{args.dataset}] seed {seed} split {i}')
            results[f'split{i}'] = train_split(args_s, data, splits[i], os.path.join(seed_dir, f'split{i}'), device)
        final, table = summary_table(results)
        seed_results[f'seed{seed}'] = final
        report.append(f'===== seed {seed} =====\n{table}')
        with open(os.path.join(seed_dir, 'results.txt'), 'w') as f:
            f.write(table)

    _, table = summary_table(seed_results, label='seed')
    report.append(f'===== seed summary =====\n{table}')
    report.append("split row std: across test videos | seed row std: across splits\n"
                  "seed summary final: mean over seeds, std across seeds\n")
    with open(os.path.join(exp_dir, 'results.txt'), 'w') as f:
        f.write('\n'.join(report))


if __name__ == '__main__':
    main()
