"""Canonical per-observation records. Instance IDs do not imply tracking IDs."""
import hashlib

import numpy as np

from .geometry import box_corners, boxes_array, points_in_box, project_box


def build_observation(boxes, scores, labels, classes, calibration, image_size,
                      *, run_id, frame_id, backend, image_path, pointcloud_path,
                      timestamp=None, points=None, score_threshold=0.1, min_points=3):
    boxes = boxes_array(boxes)
    scores, raw_labels = np.asarray(scores, dtype=float), np.asarray(labels)
    if scores.shape != (len(boxes),) or raw_labels.shape != (len(boxes),):
        raise ValueError('Box, score and label counts must agree')
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Scores must be finite probabilities in [0,1]')
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('Unique, non-empty class vocabulary required')
    labels = raw_labels.astype(int)
    if not np.equal(raw_labels, labels).all() or ((labels < 0) | (labels >= len(classes))).any():
        raise ValueError('Labels must index the supplied model class vocabulary')
    if not run_id or not frame_id or not 0 <= score_threshold <= 1 or min_points < 1:
        raise ValueError('Invalid observation identifier or export settings')
    if (len(image_size) != 2 or not np.isfinite(image_size).all()
            or min(image_size) <= 0 or any(int(v) != v for v in image_size)):
        raise ValueError('image_size must be [width,height]')
    if timestamp is not None and not np.isfinite(float(timestamp)):
        raise ValueError('timestamp must be finite seconds or null')
    if points is not None:
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError('Points must have shape (N,>=3)')
        points = points[np.isfinite(points[:, :3]).all(axis=1)]
    corners = box_corners(boxes)
    centers = boxes[:, :3].copy()
    centers[:, 2] += boxes[:, 5] / 2  # MMDetection3D stores bottom centres.
    camera_centers = calibration.camera_points(centers)
    records = []
    for idx, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < score_threshold:
            continue
        bbox, status = project_box(corners[idx], calibration, image_size)
        surface_distance, point_count = None, None
        if points is not None:
            selected = points[points_in_box(points, box), :3]
            point_count = len(selected)
            if point_count >= min_points:
                surface_distance = float(np.median(np.linalg.norm(selected, axis=1)))
        key = hashlib.sha256(f'{run_id}\0{frame_id}\0{idx}'.encode()).hexdigest()[:24]
        records.append(dict(
            instance_id=key, detection_index=idx, category=str(classes[label]),
            label_id=int(label), confidence=float(score), bbox_2d=bbox,
            projection_valid=bbox is not None, projection_status=status,
            box_3d_lidar=box.tolist(), center_lidar_m=centers[idx].tolist(),
            range_center_lidar_m=float(np.linalg.norm(centers[idx])),
            depth_camera_m=float(camera_centers[idx, 2]),
            depth_valid=bool(camera_centers[idx, 2] > 0),
            surface_median_range_m=surface_distance, lidar_points_in_box=point_count,
            surface_range_valid=surface_distance is not None))
    return dict(schema_version='1.0', run_id=run_id, frame_id=frame_id, backend=backend,
                timestamp=timestamp, image_path=str(image_path),
                pointcloud_path=str(pointcloud_path), image_size=list(image_size),
                coordinate_system='mmdet3d_lidar_x_forward_y_left_z_up',
                distance_origin='lidar_sensor', box_origin='bottom_center',
                detections=records)
