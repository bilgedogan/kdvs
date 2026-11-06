import os
import sys
import csv
import json
import types
import h5py
from tqdm import tqdm

from common import UNIVTG_DIR, DATASET_DIR, NAME_MAP, video_paths, video_feat_path, query_feat_path

_pkg = types.ModuleType('run_on_video')
_pkg.__path__ = [os.path.join(UNIVTG_DIR, 'run_on_video')]
sys.modules['run_on_video'] = _pkg
from run_on_video.data_utils import ClipFeatureExtractor  # noqa: E402

def query_texts(dataset):
    """{video_key: sorgu metni}"""
    name_map = json.load(open(NAME_MAP[dataset]))
    if dataset == 'summe':
        return {key: name.replace('_', ' ') for name, key in name_map.items()}
    rows = csv.DictReader(open(os.path.join(DATASET_DIR, 'ydata-tvsum50-info.tsv')), delimiter='\t')
    title = {r['video_id']: r['title'] for r in rows}
    return {key: title[name] for name, key in name_map.items()}


def main():
    extractor = ClipFeatureExtractor(framerate=1 / 2, size=224, centercrop=True,
                                     model_name_or_path='ViT-B/32', device='cuda')

    for dataset in ['summe', 'tvsum']:
        with h5py.File(video_feat_path(dataset), 'w') as f:
            for key, path in tqdm(video_paths(dataset).items(), desc=f'{dataset} video'):
                f[key] = extractor.encode_video(path).float().cpu().numpy()

        texts = query_texts(dataset)
        keys = list(texts)
        feats = extractor.encode_text([texts[k] for k in keys])
        with h5py.File(query_feat_path(dataset, 'name'), 'w') as f:
            for key, feat in zip(keys, feats):
                f[key] = feat.float().cpu().numpy()
                f[key].attrs['text'] = texts[key]


if __name__ == '__main__':
    main()
