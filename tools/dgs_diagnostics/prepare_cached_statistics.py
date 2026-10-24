"""Build 10-task Gaussian statistics from existing 5-task per-image caches.

CPU only. Missing/misaligned images cause a report and failure, never silent
sampling or GPU extraction. Outputs are isolated from baseline statistics.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='4'
os.environ['MKL_NUM_THREADS']='4'
import argparse
import hashlib
import json
from pathlib import Path
import torch

SPLITS = dict(DIOR=['Task4-1','Task4-2','Task4-3','Task4-4'],
              PascalVOC=['Task4-1','Task4-2','Task4-3','Task4-4'],
              RUOD=['Task2-1','Task2-2'])
CACHES = dict(DIOR=['DIOR_Task2-1','DIOR_Task2-2'],
              PascalVOC=['PascalVOC_Task2-1','PascalVOC_Task2-2'],RUOD=['RUOD'])


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def regularize(cov,ratio):
    # Same operations as _svd_regularize_cov; no change to covariance estimator.
    cov=(cov+cov.T)/2
    eigenvalues,eigenvectors=torch.linalg.eigh(cov)
    floor=eigenvalues.max()*ratio
    eigenvalues=torch.clamp(eigenvalues,min=floor.item())
    return eigenvectors@torch.diag(eigenvalues)@eigenvectors.T


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    repo=args.repo.resolve();out=args.output_dir.resolve()
    if out==repo/'visualize/CDIOD_mlvlfeat' or 'work_dirs/CDIOD/repro_' in str(out):
        raise ValueError('Use an isolated diagnostic output directory')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    jobs=[];missing=[];duplicates=[];annotations={}
    for domain,splits in SPLITS.items():
        index={}
        for task in CACHES[domain]:
            suffix=task[len(domain)+1:] if task!=domain else ''
            name='annotations_without_background'+('_'+suffix if suffix else '')+'.json'
            ann=repo/'data/CDIOD'/domain/'train'/name
            payload=json.loads(ann.read_text());annotations[str(ann.relative_to(repo))]=sha(ann)
            for image in payload['images']:
                path=repo/'visualize/CDIOD_mlvlfeat'/task/'train'/f"img_{image['id']}.pt"
                if path.is_file():index.setdefault(image['file_name'],[]).append(path)
        for suffix in splits:
            task=f'{domain}_{suffix}'
            ann=repo/'data/CDIOD'/domain/'train'/f'annotations_without_background_{suffix}.json'
            payload=json.loads(ann.read_text());annotations[str(ann.relative_to(repo))]=sha(ann)
            images=sorted(payload['images'],key=lambda im:f"img_{im['id']}.pt")
            paths=[]
            for image in images:
                candidates=index.get(image['file_name'],[])
                if not candidates:
                    missing.append(dict(task=task,file_name=image['file_name'],image_id=image['id']))
                else:
                    paths.append(candidates[0])
                    if len(candidates)>1:duplicates.append(candidates)
            jobs.append((task,images,paths))
    report=dict(cpu_only=True,annotations_sha256=annotations,missing=missing,
        requested_tasks=[j[0] for j in jobs],source='existing train per-image feature caches',
        reuse_assumption='Cache extraction used the unchanged frozen backbone/neck, checkpoint and deterministic test preprocessing; exact per-file extraction manifests were not originally recorded.')
    (out/'cache_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    if missing:raise RuntimeError(f'{len(missing)} missing embeddings; see cache_coverage.json')
    # Check repeat images across task caches: input identity alone is not enough.
    duplicate_max=0.0
    for candidates in duplicates[:64]:
        a=torch.load(candidates[0],map_location='cpu')
        b=torch.load(candidates[1],map_location='cpu')
        delta=float((a-b).abs().max());duplicate_max=max(duplicate_max,delta)
        if not torch.allclose(a,b,rtol=1e-4,atol=1e-5):
            raise RuntimeError('Duplicate-image caches disagree; recheck extraction provenance')
    report.update(duplicate_checks=min(len(duplicates),64),duplicate_max_abs_difference=duplicate_max)
    distributions=[];tasks=[];provenance=[]
    for task,images,paths in jobs:
        features=[];hashes=hashlib.sha256()
        for image,path in zip(images,paths):
            blob=path.read_bytes();hashes.update(str(image['file_name']).encode());hashes.update(hashlib.sha256(blob).digest())
            feature=torch.load(path,map_location='cpu')
            if feature.ndim!=2 or feature.shape[-1]!=256 or not torch.isfinite(feature).all():
                raise ValueError(f'Invalid per-image feature: {path}')
            features.append(feature[-1])
        features=torch.stack(features)
        mean=features.mean(dim=0)
        raw_cov=torch.cov(features.T)+1e-6*torch.eye(features.shape[-1])
        cov=regularize(raw_cov,1e-3)
        distributions.append(torch.distributions.MultivariateNormal(mean,cov))
        tasks.append(task)
        # Store [1,D] and [1,D,D] to retain get_task_distribution(...)[-1]
        # compatibility. These files are diagnostic-only, last-level statistics.
        stats=dict(mean=mean[None],cov=cov[None],num_samples=len(images))
        torch.save(stats,out/f'{task}.pt')
        provenance.append(dict(task=task,n_images=len(images),feature_chain_sha256=hashes.hexdigest()))
        print(json.dumps(provenance[-1]),flush=True)
    matrix=[]
    for a in distributions:
        matrix.append([float((torch.distributions.kl.kl_divergence(a,b)+torch.distributions.kl.kl_divergence(b,a))/2) for b in distributions])
    result=dict(tasks=tasks,distance_matrix=matrix,provenance=provenance,
        min_eig_ratio=1e-3,eps=1e-6,selected_feature_level=-1,
        features='train only; same image may occur in multiple class-incremental tasks',
        no_training=True,no_gpu=True)
    (out/'distance_matrix.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'cache_coverage.json').write_text(json.dumps(report,indent=2)+'\n')
    print('STATISTICS_COMPLETE',out/'distance_matrix.json',flush=True)


if __name__=='__main__':main()
