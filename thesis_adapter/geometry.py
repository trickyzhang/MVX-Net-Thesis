"""KITTI calibration and MMDetection3D LiDAR box geometry (metres).

Boxes: x,y,z_bottom,dx,dy,dz,yaw; yaw rotates +x towards +y.
The projection uses rectified reference-camera coordinates and P2, including
its baseline translation. depth_camera_m is z in that rectified frame.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Calibration:
    lidar_to_camera: np.ndarray
    camera_to_image: np.ndarray

    def __post_init__(self):
        t = np.asarray(self.lidar_to_camera, dtype=np.float64)
        p = np.asarray(self.camera_to_image, dtype=np.float64)
        if t.shape != (4, 4) or p.shape != (3, 4):
            raise ValueError('Expected lidar_to_camera 4x4 and camera_to_image 3x4')
        if not np.isfinite(t).all() or not np.isfinite(p).all():
            raise ValueError('Non-finite calibration')
        if not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError('Invalid homogeneous transform')
        if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-3):
            raise ValueError('Extrinsic rotation is not orthonormal')
        if not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-3):
            raise ValueError('Extrinsic rotation must preserve handedness')
        if not np.allclose(p[2], [0, 0, 1, 0], atol=1e-6):
            raise ValueError('Expected rectified KITTI projection matrix')
        if p[0, 0] <= 0 or p[1, 1] <= 0:
            raise ValueError('Camera focal lengths must be positive')
        object.__setattr__(self, 'lidar_to_camera', t)
        object.__setattr__(self, 'camera_to_image', p)

    @classmethod
    def from_kitti(cls, path):
        fields = {}
        for line in Path(path).read_text().splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                if key in ('P2', 'R0_rect', 'Tr_velo_to_cam'):
                    fields[key] = np.array([float(v) for v in value.split()])
        missing = {'P2', 'R0_rect', 'Tr_velo_to_cam'} - fields.keys()
        if missing:
            raise ValueError('Missing calibration fields: ' + ', '.join(sorted(missing)))
        r, t = np.eye(4), np.eye(4)
        r[:3, :3] = fields['R0_rect'].reshape(3, 3)
        t[:3] = fields['Tr_velo_to_cam'].reshape(3, 4)
        return cls(r @ t, fields['P2'].reshape(3, 4))

    def camera_points(self, points):
        xyz = np.asarray(points, dtype=np.float64)
        return xyz @ self.lidar_to_camera[:3, :3].T + self.lidar_to_camera[:3, 3]


def boxes_array(boxes):
    b = np.asarray(boxes, dtype=np.float64)
    if b.size == 0:
        return np.empty((0, 7))
    if b.ndim != 2 or b.shape[1] != 7:
        raise ValueError('boxes_lidar must have shape (N,7)')
    if not np.isfinite(b).all() or (b[:, 3:6] <= 0).any():
        raise ValueError('Boxes must be finite with strictly positive dimensions')
    return b


def box_corners(boxes):
    b = boxes_array(boxes)
    pattern = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
                        [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]])
    local = (pattern - [0.5, 0.5, 0]) * b[:, None, 3:6]
    c, s = np.cos(b[:, 6])[:, None], np.sin(b[:, 6])[:, None]
    x, y = local[:, :, 0].copy(), local[:, :, 1].copy()
    local[:, :, 0], local[:, :, 1] = c * x - s * y, s * x + c * y
    return local + b[:, None, :3]


EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
         (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def project_box(corners, calibration, image_size, near=0.1):
    """Clip box edges at the near plane before projecting, then clip to image.

    A box enclosing the whole image is retained even when no corner lies
    inside it. Invalid/behind-camera boxes return (None, reason).
    """
    width, height = image_size
    if width <= 0 or height <= 0 or near <= 0:
        raise ValueError('Positive image dimensions and near plane required')
    cam = calibration.camera_points(corners)
    front = cam[:, 2] >= near
    vertices = list(cam[front])
    for a, b in EDGES:
        if front[a] != front[b]:
            q = cam[a] + (near - cam[a, 2]) / (cam[b, 2] - cam[a, 2]) * (cam[b] - cam[a])
            vertices.append(q)
    if not vertices:
        return None, 'behind_camera'
    vertices = np.asarray(vertices)
    uvw = vertices @ calibration.camera_to_image[:, :3].T + calibration.camera_to_image[:, 3]
    uv = uvw[:, :2] / uvw[:, 2:3]
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    xyxy = np.array([max(0., lo[0]), max(0., lo[1]), min(float(width), hi[0]), min(float(height), hi[1])])
    if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
        return None, 'outside_image'
    return xyxy.tolist(), 'near_plane_clipped' if not front.all() else 'valid'


def points_in_box(points, box):
    """Boolean membership in a yaw-oriented box; one vectorized pass per box."""
    delta = np.asarray(points)[:, :3] - box[:3]
    c, s = np.cos(box[6]), np.sin(box[6])
    x, y = c * delta[:, 0] + s * delta[:, 1], -s * delta[:, 0] + c * delta[:, 1]
    return ((abs(x) <= box[3] / 2) & (abs(y) <= box[4] / 2)
            & (delta[:, 2] >= 0) & (delta[:, 2] <= box[5]))
