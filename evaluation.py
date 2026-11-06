import os
import csv
import functools
import numpy as np
from scipy.stats import kendalltau, spearmanr

TVSUM_ANNO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'dataset', 'ydata-tvsum50-anno.tsv')



def knapsack(values, weights, capacity):
    n = len(values)
    table = np.zeros((n + 1, capacity + 1))
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        table[i] = table[i - 1]
        if w <= capacity:
            table[i, w:] = np.maximum(table[i - 1, w:], table[i - 1, :capacity + 1 - w] + v)
    chosen, c = [], capacity
    for i in range(n, 0, -1):
        if table[i, c] != table[i - 1, c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    return chosen


def generate_summary(score, change_points, n_frames, n_frame_per_seg, picks, ratio=0.15):
    frame_score = np.zeros(n_frames)
    bounds = np.append(picks, n_frames)
    for i in range(len(picks)):
        frame_score[bounds[i]:bounds[i + 1]] = score[i]

    seg_score = [frame_score[s:e + 1].mean() for s, e in change_points]
    chosen = knapsack(seg_score, [int(n) for n in n_frame_per_seg], int(n_frames * ratio))

    summary = np.zeros(n_frames)
    for i in chosen:
        s, e = change_points[i]
        summary[s:e + 1] = 1
    return summary


# ---------------- Metrikler ----------------

def f1_score(summary, user_summary, dataset):
    f1s = []
    for user in user_summary:
        overlap = (summary * user).sum()
        p = overlap / (summary.sum() + 1e-8)
        r = overlap / (user.sum() + 1e-8)
        f1s.append(0.0 if overlap == 0 else 2 * p * r / (p + r))
    f1 = max(f1s) if dataset == 'summe' else np.mean(f1s)
    return float(f1 * 100)


@functools.lru_cache(maxsize=None)
def tvsum_user_scores():
    """{video_idx (1..50): [n_users, T]}"""
    rows = list(csv.reader(open(TVSUM_ANNO), delimiter='\t'))
    names = list(dict.fromkeys(r[0] for r in rows))  # dosyadaki sıra = video_1..50
    out = {}
    for idx, name in enumerate(names, start=1):
        users = [np.array(r[2].split(','), dtype=float) for r in rows if r[0] == name]
        out[idx] = [(u / u.max())[::15] for u in users]
    return out


def rank_correlation(score, summary, user_summary, video_key, dataset):
    """(tau, rho)"""
    if dataset == 'summe':
        gt = user_summary.mean(axis=0)
        return kendalltau(summary, gt)[0], spearmanr(summary, gt)[0]

    users = tvsum_user_scores()[int(video_key.split('_')[-1])]
    taus = [kendalltau(score, u)[0] for u in users]
    rhos = [spearmanr(score, u)[0] for u in users]
    return float(np.mean(taus)), float(np.mean(rhos))


def evaluate_video(score, video, video_key, dataset):
    summary = generate_summary(score, video['change_points'], int(video['n_frames']),
                               video['n_frame_per_seg'], video['picks'])
    tau, rho = rank_correlation(score, summary, video['user_summary'], video_key, dataset)
    return {'f1': f1_score(summary, video['user_summary'], dataset), 'tau': tau, 'rho': rho}
