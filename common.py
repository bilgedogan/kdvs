import os
import json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
UNIVTG_DIR = os.path.join(ROOT, 'third_party', 'UniVTG')
DATASET_DIR = os.path.join(ROOT, 'dataset')  
VIDEOS_DIR = os.path.join(ROOT, 'Videos')

VIDEO_DIR = {
    'summe': os.path.join(VIDEOS_DIR, 'SumMe'),
    'tvsum': os.path.join(VIDEOS_DIR, 'TVSum'),
}
NAME_MAP = {
    'summe': os.path.join(VIDEOS_DIR, 'summe_mapped_video_names.json'),
    'tvsum': os.path.join(VIDEOS_DIR, 'tvsum_mapped_video_names.json'),
}


def video_paths(dataset):
    """{'video_1': '/.../Air_Force_One.mp4', ...}"""
    name_map = json.load(open(NAME_MAP[dataset]))
    out = {}
    for name, key in name_map.items():
        path = os.path.join(VIDEO_DIR[dataset], name.replace(' ', '_') + '.mp4')
        assert os.path.exists(path), path
        out[key] = path
    return out


def video_feat_path(dataset):
    return os.path.join(DATASET_DIR, f'{dataset}_video_clip.h5')


def query_feat_path(dataset, query_source):
    return os.path.join(DATASET_DIR, f'{dataset}_queries_{query_source}.h5')


def teacher_path(dataset):
    return os.path.join(DATASET_DIR, f'{dataset}_teacher.h5')


def to_steps(clip_score, picks, n_frames):
    """Teacher klip skoru [Tc] (2 sn'lik) -> student adım skoru [T]. Kare p -> klip floor(p*Tc/n_frames)."""
    Tc = len(clip_score)
    idx = np.minimum(picks * Tc // n_frames, Tc - 1).astype(int)
    return clip_score[idx]


def to_steps_interp(clip_score, picks, n_frames):
    """to_steps'in yumuşak hali: klip skoru klip merkezine konur, kareler arası lineer interpolasyon.
    Basamaklı değil, klip içinde de değişen sinyal. Kenarlarda ilk/son klip değeri sabit."""
    Tc = len(clip_score)
    centers = (np.arange(Tc) + 0.5) * n_frames / Tc
    return np.interp(picks, centers, clip_score).astype(np.float32)
