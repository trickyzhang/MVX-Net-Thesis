"""python -m thesis_adapter {run,export,query,doctor}."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .geometry import Calibration
from .records import build_observation
from .storage import connect, query_targets, save_observation


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_manifest(path):
    path = Path(path).resolve()
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError('Manifest must be a non-empty JSON array')
    seen = set()
    for sample in entries:
        frame = sample['frame_id']
        if not isinstance(frame, str) or not frame or frame in seen:
            raise ValueError('frame_id must be a unique non-empty string')
        seen.add(frame)
        for key in ('image', 'points', 'calib'):
            p = Path(sample[key]).expanduser()
            if not p.is_absolute():
                p = path.parent / p
            if not p.is_file():
                raise FileNotFoundError(p)
            sample[key] = str(p.resolve())
        if Path(sample['points']).suffix.lower() != '.bin':
            raise ValueError('This KITTI runner expects float32 XYZI .bin point clouds')
        if sample.get('timestamp') is not None and not np.isfinite(float(sample['timestamp'])):
            raise ValueError('timestamp must be finite seconds or null')
    return entries


def run(args):
    from PIL import Image
    import torch
    from .backends import LCFBackend, MVXBackend, unpack_predictions
    if args.warmup < 0:
        raise ValueError('warmup must be non-negative')
    cuda = torch.device(args.device).type == 'cuda'
    if cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; check the GPU environment')
    if args.precision == 'amp-fp16' and not cuda:
        raise ValueError('Experimental AMP FP16 requires CUDA')
    samples = read_manifest(args.manifest)
    backend = {'mvxnet': MVXBackend, 'lcf3d': LCFBackend}[args.backend](args.config, args.checkpoint, args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'observations.jsonl').exists():
        raise FileExistsError('Use a new output directory for each run')
    db = connect(output / 'targets.sqlite')
    provenance = dict(backend=args.backend, device=args.device, precision=args.precision,
                      config=str(Path(args.config).resolve()),
                      config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                      checkpoint=args.checkpoint, classes=backend.classes,
                      torch_version=torch.__version__, python=sys.version,
                      manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                      warmup=args.warmup)
    write_json(output / 'run.json', dict(provenance, resolved_config=backend.resolved_config))

    def predict(sample, calibration):
        context = torch.autocast('cuda', dtype=torch.float16) if args.precision == 'amp-fp16' else nullcontext()
        with torch.inference_mode(), context:
            return backend.predict(sample, calibration)

    def sync():
        if cuda:
            torch.cuda.synchronize(args.device)

    try:
        first_calib = Calibration.from_kitti(samples[0]['calib'])
        for _ in range(args.warmup):
            predict(samples[0], first_calib)
        sync()
        if cuda:
            torch.cuda.reset_peak_memory_stats(args.device)
        inference_times, export_times, frame_times = [], [], []
        start = time.perf_counter()
        with (output / 'observations.jsonl').open('w') as stream:
            for sample in samples:
                frame_start = time.perf_counter()
                calib = Calibration.from_kitti(sample['calib'])
                with Image.open(sample['image']) as im:
                    size = im.size
                sync()
                t = time.perf_counter()
                pred = predict(sample, calib)
                sync()
                inference_ms = 1000 * (time.perf_counter() - t)
                t = time.perf_counter()
                boxes, scores, labels = unpack_predictions(pred)
                points = None
                if args.surface_range:
                    points = np.fromfile(sample['points'], dtype=np.float32).reshape(-1, 4)
                observation = build_observation(
                    boxes, scores, labels, backend.classes, calib, size,
                    run_id=args.run_id, frame_id=sample['frame_id'], backend=args.backend,
                    image_path=sample['image'], pointcloud_path=sample['points'],
                    timestamp=sample.get('timestamp'), points=points,
                    score_threshold=args.score_threshold, min_points=args.min_points)
                observation['runtime'] = provenance
                observation['timing'] = dict(inference_ms=inference_ms)
                save_observation(db, observation)
                stream.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + '\n')
                export_times.append(1000 * (time.perf_counter() - t))
                inference_times.append(inference_ms)
                frame_times.append(1000 * (time.perf_counter() - frame_start))
                print(f"{sample['frame_id']}: {len(observation['detections'])} targets, {inference_ms:.1f} ms inference")
        elapsed = time.perf_counter() - start
        def stats(values):
            return dict(mean_ms=float(np.mean(values)), median_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)))
        write_json(output / 'performance.json', dict(
            runtime=provenance, frames=len(samples), elapsed_s=elapsed,
            pipeline_fps=len(samples) / elapsed, inference=stats(inference_times),
            export_and_storage=stats(export_times), frame_total=stats(frame_times),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device) if cuda else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device) if cuda else None,
            timing_scope='After model initialization and warmup; inference includes upstream preprocessing and postprocessing. Frame total also includes calibration, geometry and storage.'))
    finally:
        db.close()


def export(args):
    payload = json.loads(Path(args.predictions).read_text())
    calib = Calibration.from_kitti(args.calib)
    observation = build_observation(
        payload['boxes_lidar'], payload['scores'], payload['labels'], payload['classes'],
        calib, payload['image_size'], run_id=args.run_id, frame_id=payload['frame_id'],
        backend=payload['backend'], image_path=payload['image_path'],
        pointcloud_path=payload['pointcloud_path'], timestamp=payload.get('timestamp'),
        score_threshold=args.score_threshold)
    write_json(args.output, observation)
    if args.database:
        db = connect(args.database)
        try:
            save_observation(db, observation)
        finally:
            db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='Single-camera KITTI batch inference; GPU dependencies required')
    p.add_argument('--backend', required=True, choices=['mvxnet', 'lcf3d'])
    p.add_argument('--manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--precision', choices=['fp32', 'amp-fp16'], default='fp32')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--surface-range', action='store_true')
    p.add_argument('--min-points', type=int, default=3)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--score-threshold', type=float, default=0.1)
    p.set_defaults(action=run)
    p = sub.add_parser('export', help='CPU-only export of canonical MMDetection3D box arrays')
    p.add_argument('--predictions', required=True)
    p.add_argument('--calib', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--database')
    p.add_argument('--score-threshold', type=float, default=0.1)
    p.set_defaults(action=export)
    p = sub.add_parser('query', help='Structured query; does not infer unseen categories')
    p.add_argument('--database', required=True)
    p.add_argument('--category', action='append', default=[])
    p.add_argument('--max-range', type=float)
    p.add_argument('--side', choices=['left', 'center', 'right'])
    p.add_argument('--since', type=float)
    p.add_argument('--until', type=float)
    p.add_argument('--run-id')
    p.add_argument('--limit', type=int, default=100)
    p.set_defaults(action=lambda a: query_command(a))
    p = sub.add_parser('doctor', help='Inspect environment without importing CUDA libraries')
    p.set_defaults(action=lambda a: doctor())
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, ImportError, KeyError) as exc:
        parser.exit(2, f'Error: {exc}\n')


def query_command(args):
    if not Path(args.database).is_file():
        raise FileNotFoundError(args.database)
    db = connect(args.database)
    try:
        result = query_targets(db, categories=args.category, max_range=args.max_range,
                               side=args.side, since=args.since, until=args.until,
                               run_id=args.run_id, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    finally:
        db.close()


def doctor():
    versions = {}
    for name in ('numpy', 'Pillow', 'torch', 'mmcv', 'mmengine', 'mmdet', 'mmdet3d', 'spconv-cu117', 'spconv-cu118'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    print(json.dumps(dict(python=sys.version, platform=platform.platform(), packages=versions), indent=2))


if __name__ == '__main__':
    main()
