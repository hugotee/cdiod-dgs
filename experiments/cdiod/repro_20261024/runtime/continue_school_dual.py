"""Continue phases 2--5 of the existing single-card upstream DGS run.

Runs one GPU job at a time, preserves phase-zero artifacts and the upstream
method, writes fail-fast status, and records unfiltered metrics after each phase.
"""
import os,sys,json,math,shutil,subprocess,fcntl,signal,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

REPO=Path(os.environ.get('DGS_REPO','/home/data/dq/cdiod-dgs')).resolve()
ROOT=REPO/'work_dirs/CDIOD/repro_20261022_seed0_5phases'
PYTHON=sys.executable
TASKS=['DIOR_Task2-1','DIOR_Task2-2','PascalVOC_Task2-1','PascalVOC_Task2-2','RUOD']
SEED=1498185232
os.chdir(REPO);sys.path.insert(0,str(REPO))
os.environ.update(PATH=str(Path(PYTHON).parent)+':'+os.environ['PATH'],PYTHONPATH=str(REPO),
                  CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES','2,3'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
                  TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                  PYTHONUNBUFFERED='1',PORT=os.environ.get('PORT','29527'),
                  MASTER_PORT=os.environ.get('PORT','29527'),TMPDIR=str(REPO.parent/'tmp'),
                  XDG_CACHE_HOME=str(REPO.parent/'.cache'),TORCH_HOME=str(REPO.parent/'.cache/torch'),
                  HF_HOME=str(REPO.parent/'.cache/huggingface'))
from mmengine.config import Config
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules
import yaml
register_all_modules(init_default_scope=True)
import projects.DGS.dgs

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

CURRENT={}
def status(stage,**kwargs):
    CURRENT.update(kwargs)
    value=dict(stage=stage,updated_at=now(),driver_pid=os.getpid(),**CURRENT)
    write_json(ROOT/'status.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True)

CHILD=None
def terminate(signum,frame):
    if CHILD is not None and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)
    status('interrupted',signal=signum)
    raise SystemExit(128+signum)

signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)

def run(stage,cmd,log):
    global CHILD
    status(stage,command=cmd,log=str(log),child_pid=None)
    with log.open('a') as stream:
        CHILD=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        status(stage,child_pid=CHILD.pid)
        code=CHILD.wait()
    if code:raise RuntimeError(f'{stage} exited with {code}; log={log}')
    CHILD=None

def common(cfg,i,mapping_path=None):
    task=TASKS[i];seen=TASKS[:i+1]
    if '_Task' in task:
        domain,suffix=task.split('_',1);ann=f'train/annotations_without_background_{suffix}.json'
    else:domain=task;ann='train/annotations_without_background.json'
    cfg.train_dataloader.dataset.update(data_root=f'data/CDIOD/{domain}/',ann_file=ann,seen_tasks=','.join(seen))
    cfg.model.update(num_tasks=5,task_id=i,seen_tasks=seen)
    cfg.model.domain_predictor_cfg.stats_path=str(ROOT/'feature_stats/svd')
    cfg.model.domain_predictor_cfg.task_id_mapping_path=str(mapping_path or ROOT/'task_id_mapping.yaml')
    if cfg.train_dataloader.dataset.get('distn_cfg') is not None:
        cfg.train_dataloader.dataset.distn_cfg.task_id_mapping_path=str(mapping_path or ROOT/'task_id_mapping.yaml')
    cfg.val_evaluator.evaluation_tasks=','.join(TASKS)
    cfg.test_evaluator.evaluation_tasks=','.join(TASKS)
    cfg.load_from=str(ROOT/TASKS[i-1]/'epoch_12.pth')
    cfg.model.language_model.name=str(REPO/'weights/bert-base-uncased')
    cfg.resume=False;cfg.randomness=dict(seed=SEED)
    cfg.work_dir=str(ROOT/task)
    cfg.default_hooks.checkpoint.update(save_optimizer=True,save_param_scheduler=True,max_keep_ckpts=2)
    return cfg

def collect(i):
    task=TASKS[i]
    fs=sorted((ROOT/task).glob('*/vis_data/scalars.json'),key=lambda f:f.stat().st_mtime)
    if not fs:raise RuntimeError(f'No scalar file for {task}')
    recs=[json.loads(l) for l in fs[-1].read_text().splitlines()]
    vals=[r for r in recs if 'DIOR/coco/bbox_mAP' in r]
    if not vals:raise RuntimeError(f'No completed evaluation for {task}')
    metrics=vals[-1]
    core=[metrics[f'{domain}/coco/bbox_mAP'] for domain in ['DIOR','PascalVOC','RUOD']]
    assert all(math.isfinite(v) and v>=0 for v in core)
    result=dict(phase_index=i,phase_number=i+1,task=task,completed_at=now(),
                checkpoint=str(ROOT/task/'epoch_12.pth'),scalar_file=str(fs[-1]),
                raw_metrics=metrics,domain_mean_including_zero=sum(core)/3)
    write_json(ROOT/f'phase{i}_results.json',result)
    return result

def relocate(value):
    if isinstance(value,str):
        return value.replace('/root/autodl-tmp/cdiod-dgs',str(REPO))
    if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
    if isinstance(value,list):return [relocate(v) for v in value]
    if isinstance(value,tuple):return tuple(relocate(v) for v in value)
    return value

def prepare_phase(i):
    task=TASKS[i]
    # Reuse the verified task mapping and exact source config for phase two.
    if i==1:
        cfg=Config(relocate(Config.fromfile(str(ROOT/'phase1.py')).to_dict()))
        cfg.model.language_model.name=str(REPO/'weights/bert-base-uncased')
        return cfg
    mapping_cfg=common(Config.fromfile('projects/DGS/configs/CDIOD/dgs_stage1.py'),i)
    mapping_cfg.train_dataloader.dataset.metainfo=task
    ann=REPO/mapping_cfg.train_dataloader.dataset.data_root/mapping_cfg.train_dataloader.dataset.ann_file
    ids=[v['id'] for v in json.loads(ann.read_text())['images']]
    cache=REPO/'visualize/CDIOD_mlvlfeat'/task/'train'
    if not all((cache/f'img_{v}.pt').is_file() and (cache/f'img_{v}.pt').stat().st_size>0 for v in ids):
        run('features',[PYTHON,'projects/DGS/configs/CDIOD/extract_cdiod_features.py','--steps','5','--tasks',task],ROOT/f'phase{i}_features.school.log')
    mapping_file=ROOT/f'phase{i}_mapping.school.py';mapping_cfg.dump(str(mapping_file))
    run('mapping',[PYTHON,'projects/DGS/configs/adaptive_task_mapping.py','--config',str(mapping_file),
        '--task_id',str(i),'--seen_tasks',','.join(TASKS[:i+1]),'--num_tasks','5','--work_dirs',str(ROOT)],ROOT/f'phase{i}_mapping.school.log')
    mapping=yaml.safe_load((ROOT/'task_id_mapping.yaml').read_text())
    assert all(k in mapping for k in range(i+1))
    same_group=sum(v==mapping[i] for v in mapping.values())>1
    snapshot=ROOT/f'phase_snapshots/phase{i}';snapshot.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'task_id_mapping.yaml',snapshot/'task_id_mapping.yaml')
    cfg=common(Config.fromfile('projects/DGS/configs/CDIOD/dgs_stage2.py' if same_group else 'projects/DGS/configs/CDIOD/dgs_stage1.py'),i,snapshot/'task_id_mapping.yaml')
    cfg.train_dataloader.dataset.metainfo=','.join(TASKS[:i+1]) if same_group else task
    return cfg

def apply_resume(cfg,checkpoint):
    cfg.resume=checkpoint is not None
    if checkpoint is None:return cfg
    cfg.load_from=str(checkpoint)
    # The upstream group-initialization hook copies the old expert over the
    # current expert on every checkpoint load. On a same-phase resume that
    # would erase the trained student while retaining its optimizer state.
    # Keep normal phase-transition initialization; skip it only for resume.
    for hook in cfg.get('custom_hooks',[]):
        if hook['type']=='WeightsTransformHook':
            transforms=hook.get('cfg',[])
            if isinstance(transforms,dict):transforms=[transforms]
            hook['cfg']=[v for v in transforms if v.get('type')!='moe_group_init']
    return cfg

def main():
    lock=(ROOT/'continuation.school.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=json.loads((ROOT/'migration_cutover.json').read_text())
    assert marker['source_training_stopped'] is True
    assert marker['task_order']==TASKS and marker['seed']==SEED
    (ROOT/'continuation.school.dual.pid').write_text(str(os.getpid())+'\n')
    archive=ROOT/'migration_source_snapshot';archive.mkdir(exist_ok=True)
    if not (archive/'status.json').exists():shutil.copy2(ROOT/'status.json',archive/'status.json')
    results=[collect(0)]
    for i in range(1,5):
        task=TASKS[i]
        CURRENT.clear();CURRENT.update(phase_index=i,phase_number=i+1,total_phases=5,task=task,host='school',gpu_ids=[2,3],num_gpus=2,global_batch_size=2)
        if (ROOT/task/'epoch_12.pth').exists():
            results.append(collect(i));continue
        if shutil.disk_usage(ROOT).free<6*1024**3:raise RuntimeError('Less than 6 GiB free')
        cfg=prepare_phase(i)
        checkpoints=list((ROOT/task).glob('epoch_*.pth'))
        latest=max(checkpoints,key=lambda p:int(p.stem.split('_')[-1])) if checkpoints else None
        cfg=apply_resume(cfg,latest)
        cfg.train_dataloader.batch_size=1
        cfg.default_hooks.checkpoint.update(save_optimizer=True,save_param_scheduler=True,max_keep_ckpts=2)
        cfg.work_dir=str(ROOT/task)
        assert Path(cfg.load_from).is_file()
        ds=DATASETS.build(cfg.train_dataloader.dataset)
        plan=dict(phase_index=i,phase_number=i+1,task=task,train_images=len(ds),prompt_classes=len(ds.metainfo['classes']),
            batch_size=cfg.train_dataloader.batch_size,epochs=cfg.train_cfg.max_epochs,lr=cfg.optim_wrapper.optimizer.lr,
            resume_checkpoint=str(latest) if latest else None,load_from=cfg.load_from,
            global_batch_size=2,num_gpus=2,gpu_ids=[2,3],
            iterations_per_epoch=math.ceil(len(ds)/2),
            resume_group_initialization_skipped=latest is not None)
        assert plan['batch_size']*plan['num_gpus']==2 and plan['epochs']==12
        del ds
        config=ROOT/f'phase{i}.school.dual.py';cfg.dump(str(config))
        write_json(ROOT/f'phase{i}.school_dual_manifest.json',plan)
        status('prepared',**plan)
        cmd=['bash','tools/dist_train.sh',str(config),'2','--amp','--work-dir',str(ROOT/task)]
        if latest:cmd+=['--resume',str(latest)]
        run('train',cmd,ROOT/f'phase{i}_train.school.dual.log')
        assert (ROOT/task/'epoch_12.pth').is_file()
        results.append(collect(i));write_json(ROOT/'phase_results_unfiltered.json',results)
        status('phase_complete',checkpoint=str(ROOT/task/'epoch_12.pth'))
    CURRENT.clear()
    status('complete',host='school',completed_phases=5,checkpoint=str(ROOT/TASKS[-1]/'epoch_12.pth'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        traceback.print_exc();status('failed',error=str(exc));raise
