"""Audit submitted results against logs, source hashes and reconstructed images."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
import tensorflow as tf
import tifffile


def audit(root, output):
    result = {}
    for model in ['baseline', 'attention']:
        folder = root / 'runs' / f'3090-{model}-s42'
        config = json.loads((folder/'config.json').read_text())
        runtime = json.loads((folder/'runtime.json').read_text())
        with (folder/'metrics.csv').open() as stream:
            rows = [{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
        assert [r['epoch'] for r in rows] == list(range(1,51))
        assert all(np.isfinite(list(r.values())).all() for r in rows)
        logged=[json.loads(x) for x in (root/f'{model}-s42.log').read_text().splitlines() if x.startswith('{"epoch":')]
        assert logged==rows
        for name, digest in runtime['code_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
        truth=tifffile.imread(folder/'ground_truth.tiff')
        saved={}
        for epoch in [10,20,30,40,50]:
            raw=tifffile.imread(folder/f'epoch_{epoch:04d}_raw.tiff')
            display=tifffile.imread(folder/f'epoch_{epoch:04d}.tiff')
            assert raw.shape==display.shape==truth.shape and np.isfinite(raw).all()
            positive=np.maximum(raw,0.)
            scaled=(positive/positive.max()*255).astype(np.uint8) if positive.max()>0 else np.zeros_like(display)
            assert np.array_equal(scaled,display)
            mse=float(np.mean((display.astype(float)/255-truth.astype(float)/255)**2))
            assert np.isclose(mse,rows[epoch-1]['display_mse'],rtol=1e-10)
            ssim=float(np.mean([tf.image.ssim(p[...,None],g[...,None],max_val=255).numpy() for p,g in zip(display,truth)]))
            assert np.isclose(ssim,rows[epoch-1]['ssim'],atol=2e-5)
            saved[epoch]={'display_mse':mse,'recomputed_ssim':ssim,'raw_min':float(raw.min()),'raw_max':float(raw.max()),'raw_std':float(raw.std()),'display_std':float(display.std())}
        result[model]={'config':config,'runtime':runtime,'first':rows[0],'final':rows[-1],
          'best_ssim':max(rows,key=lambda r:r['ssim']),'lowest_train_mse':min(rows,key=lambda r:r['train_mse']),
          'training_seconds':sum(r['training_seconds'] for r in rows),'evaluation_seconds':sum(r['evaluation_seconds'] for r in rows),
          'snapshots':saved,'milestones':[rows[e-1] for e in [10,20,30,40,50]],
          'first_epoch_ssim_ge_0.6':next((r['epoch'] for r in rows if r['ssim']>=.6),None)}
    a,b=[result[x] for x in ['baseline','attention']]
    for key in ['dataset_sha256','epochs','seed','n_points','n_rays','depth','embedding_dim','learning_rate','max_projections','query_batch','save_every']:
        assert a['config'][key]==b['config'][key],key
    assert np.array_equal(tifffile.imread(root/'runs/3090-baseline-s42/ground_truth.tiff'),tifffile.imread(root/'runs/3090-attention-s42/ground_truth.tiff'))
    result['status']='PASS: both runs complete; matched experiment settings; CSV/log/source/image checks passed, including recomputed snapshot SSIM.'
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
    for model in ['baseline','attention']:
        r=result[model]
        print(model,json.dumps({k:r[k] for k in ['final','best_ssim','lowest_train_mse','training_seconds','evaluation_seconds','first_epoch_ssim_ge_0.6']}))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    audit(a.root,a.output)
