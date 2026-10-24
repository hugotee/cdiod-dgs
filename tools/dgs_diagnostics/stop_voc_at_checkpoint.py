"""Stop only this user's VOC driver after a specified full epoch is saved."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(2)
    root = args.root.resolve()
    original = json.loads((root/'status.json').read_text())
    pid = original['driver_pid']
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    if os.stat(f'/proc/{pid}').st_uid != os.getuid() or str(root) not in command or 'voc_ab.py run' not in command:
        raise RuntimeError('Driver identity or ownership differs; refusing to signal')
    directory = Path(original['phase_dir'])
    checkpoint = directory/f'epoch_{args.epoch}.pth'
    deadline = time.monotonic()+args.timeout
    while time.monotonic() < deadline:
        current = json.loads((root/'status.json').read_text())
        if current['driver_pid'] != pid or current.get('phase_dir') != str(directory):
            raise RuntimeError('Active stage changed; refusing to stop another stage')
        last = directory/'last_checkpoint'
        if checkpoint.is_file() and last.is_file() and last.read_text().strip() == str(checkpoint):
            saved = torch.load(checkpoint, map_location='cpu')
            if saved['meta']['epoch'] != args.epoch or not all(
                    key in saved for key in ['optimizer', 'param_schedulers', 'state_dict']):
                raise RuntimeError('Incomplete epoch checkpoint')
            if 'loss_scaler' not in saved['optimizer']:
                raise RuntimeError('AMP scaler was not saved')
            meta = dict(epoch=saved['meta']['epoch'], iteration=saved['meta']['iter'])
            del saved
            os.kill(pid, signal.SIGTERM)
            lock = (root/'driver.lock').open('a+')
            for _ in range(30):
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(1)
            else:
                raise RuntimeError('Old driver did not release its lock')
            archive = root/'gpu13_migration'
            archive.mkdir(exist_ok=True)
            copy = archive/checkpoint.name
            if copy.exists():
                raise FileExistsError(copy)
            shutil.copy2(checkpoint, copy)
            record = dict(original_status=original, requested_epoch=args.epoch,
                          checkpoint=str(checkpoint), retained_checkpoint=str(copy), **meta,
                          old_driver_stopped=True, checkpoint_has_optimizer_scheduler_amp=True,
                          caveat='A few microbatches after checkpoint saving may be discarded; resume uses the full epoch boundary.')
            (archive/'stop_record.json').write_text(json.dumps(record, indent=2)+'\n')
            print(json.dumps(record, indent=2))
            return
        time.sleep(1)
    raise TimeoutError('No complete target checkpoint; training was left running')


if __name__ == '__main__':
    main()
