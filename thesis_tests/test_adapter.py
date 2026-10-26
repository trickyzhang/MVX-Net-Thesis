"""CPU regression tests. Synthetic geometry is not a detector benchmark."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from thesis_adapter.geometry import Calibration, box_corners, points_in_box, project_box
from thesis_adapter.records import build_observation
from thesis_adapter.storage import connect, query_targets, save_observation
from thesis_adapter.__main__ import read_manifest


def calibration():
    # Standard LiDAR x-forward, y-left, z-up -> camera x-right,y-down,z-forward.
    t = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1.]])
    return Calibration(t, np.array([[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0.]]))


def observation(boxes=None, **kwargs):
    defaults = dict(run_id='mvx-test', frame_id='000001', backend='synthetic',
                    image_path='image.png', pointcloud_path='points.bin', timestamp=12.)
    defaults.update(kwargs)
    if boxes is None:
        boxes = [[10, 0, -1, 2, 2, 2, 0]]
    return build_observation(boxes, [0.9] * len(boxes), [0] * len(boxes),
                             ['Car'], calibration(), [100, 100], **defaults)


class GeometryTests(unittest.TestCase):
    def test_known_projection_and_center(self):
        r = observation()['detections'][0]
        np.testing.assert_allclose(r['bbox_2d'], [50 - 100/9, 50 - 100/9, 50 + 100/9, 50 + 100/9])
        self.assertEqual(r['center_lidar_m'], [10, 0, 0])
        self.assertEqual(r['range_center_lidar_m'], 10)
        self.assertEqual(r['depth_camera_m'], 10)

    def test_range_is_not_axial_depth(self):
        r = observation([[10, 4, -1, 2, 2, 2, 0]])['detections'][0]
        self.assertAlmostEqual(r['range_center_lidar_m'], np.sqrt(116))
        self.assertEqual(r['depth_camera_m'], 10)

    def test_behind_camera(self):
        r = observation([[-10, 0, -1, 2, 2, 2, 0]])['detections'][0]
        self.assertIsNone(r['bbox_2d'])
        self.assertFalse(r['depth_valid'])
        self.assertEqual(r['projection_status'], 'behind_camera')

    def test_near_plane_clipping(self):
        r = observation([[0.5, 0, -1, 2, 2, 2, 0]])['detections'][0]
        self.assertEqual(r['projection_status'], 'near_plane_clipped')
        self.assertEqual(r['bbox_2d'], [0, 0, 100, 100])
        json.dumps(r, allow_nan=False)

    def test_box_encloses_image_without_inside_corner(self):
        r = observation([[10, 0, -20, 2, 40, 40, 0]])['detections'][0]
        self.assertEqual(r['bbox_2d'], [0, 0, 100, 100])

    def test_outside_image(self):
        r = observation([[10, 100, -1, 2, 2, 2, 0]])['detections'][0]
        self.assertEqual(r['projection_status'], 'outside_image')

    def test_yaw_and_point_membership(self):
        box = np.array([10, 0, 0, 4, 1, 2, np.pi / 2])
        mask = points_in_box(np.array([[10, 1.5, 1], [11.5, 0, 1], [10, 0, 3]]), box)
        np.testing.assert_array_equal(mask, [True, False, False])
        corners = box_corners([box])[0]
        np.testing.assert_allclose(np.ptp(corners, axis=0), [1, 4, 2])

    def test_sparse_surface_validity(self):
        r = observation(points=np.array([[10, 0, 0], [10, .1, 0]]))['detections'][0]
        self.assertEqual(r['lidar_points_in_box'], 2)
        self.assertFalse(r['surface_range_valid'])
        self.assertIsNone(r['surface_median_range_m'])

    def test_surface_measurement(self):
        r = observation(points=np.array([[9.5, 0, 0], [10, 0, 0], [10.5, 0, 0], [100, 0, 0]]))['detections'][0]
        self.assertEqual(r['lidar_points_in_box'], 3)
        self.assertEqual(r['surface_median_range_m'], 10)

    def test_invalid_labels_dimensions_and_probabilities(self):
        for boxes in ([[10, 0, 0, -1, 2, 2, 0]], [[float('nan'), 0, 0, 1, 2, 2, 0]]):
            with self.assertRaises(ValueError):
                observation(boxes)
        for scores, labels in (([1.1], [0]), ([.5], [1]), ([.5], [.5])):
            with self.assertRaises(ValueError):
                build_observation([[10, 0, 0, 1, 2, 2, 0]], scores, labels, ['Car'],
                                  calibration(), [100, 100], run_id='r', frame_id='f',
                                  backend='test', image_path='i', pointcloud_path='p')

    def test_calibration_parser_composes_rectification(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'calib.txt'
            c = calibration()
            p.write_text('P2: ' + ' '.join(map(str, c.camera_to_image.ravel()))
                         + '\nR0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: '
                         + ' '.join(map(str, c.lidar_to_camera[:3].ravel())))
            loaded = Calibration.from_kitti(p)
            np.testing.assert_allclose(loaded.lidar_to_camera, c.lidar_to_camera)
            p.write_text('P2: 1 2 3\n')
            with self.assertRaises(ValueError):
                Calibration.from_kitti(p)

    def test_nonidentity_rectification_order(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'calib.txt'
            c = calibration()
            r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]])
            p.write_text('P2: ' + ' '.join(map(str, c.camera_to_image.ravel()))
                         + '\nR0_rect: ' + ' '.join(map(str, r.ravel()))
                         + '\nTr_velo_to_cam: ' + ' '.join(map(str, c.lidar_to_camera[:3].ravel())))
            loaded = Calibration.from_kitti(p)
            np.testing.assert_allclose(loaded.lidar_to_camera[:3, :3], r @ c.lidar_to_camera[:3, :3])


class ManifestTests(unittest.TestCase):
    def test_relative_paths_and_frame_uniqueness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('image.png', 'points.bin', 'calib.txt'):
                (root / name).touch()
            sample = dict(frame_id='1', image='image.png', points='points.bin', calib='calib.txt')
            p = root / 'manifest.json'
            p.write_text(json.dumps([sample]))
            self.assertEqual(read_manifest(p)[0]['image'], str(root / 'image.png'))
            p.write_text(json.dumps([sample, sample]))
            with self.assertRaises(ValueError):
                read_manifest(p)

    def test_empty_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'manifest.json'
            p.write_text('[]')
            with self.assertRaises(ValueError):
                read_manifest(p)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.db = connect(':memory:')

    def tearDown(self):
        self.db.close()

    def test_reimport_and_empty_frame_remove_stale_targets(self):
        save_observation(self.db, observation())
        save_observation(self.db, observation())
        self.assertEqual(len(query_targets(self.db)), 1)
        save_observation(self.db, observation([]))
        self.assertEqual(query_targets(self.db), [])
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 1)

    def test_runs_do_not_overwrite_each_other(self):
        save_observation(self.db, observation(run_id='mvx'))
        save_observation(self.db, observation(run_id='lcf'))
        self.assertEqual(len(query_targets(self.db)), 2)
        self.assertEqual(len(query_targets(self.db, run_id='lcf')), 1)

    def test_numeric_side_time_and_evidence(self):
        save_observation(self.db, observation([[10, 4, -1, 2, 2, 2, 0]]))
        self.assertEqual(len(query_targets(self.db, categories=['Car'], side='left', max_range=11, since=11, until=13)), 1)
        self.assertEqual(query_targets(self.db, max_range=10), [])
        self.assertEqual(query_targets(self.db, side='right'), [])
        self.assertEqual(query_targets(self.db, since=13), [])
        self.assertEqual(query_targets(self.db)[0]['evidence']['image_path'], 'image.png')

    def test_query_is_parameterized(self):
        save_observation(self.db, observation())
        self.assertEqual(query_targets(self.db, categories=["Car' OR 1=1 --"]), [])
        self.assertEqual(len(query_targets(self.db)), 1)

    def test_transaction_rolls_back_failed_replacement(self):
        original = observation()
        save_observation(self.db, original)
        broken = copy.deepcopy(original)
        del broken['detections'][0]['category']
        with self.assertRaises(KeyError):
            save_observation(self.db, broken)
        self.assertEqual(len(query_targets(self.db)), 1)

    def test_invisible_targets_are_not_returned(self):
        save_observation(self.db, observation([[-10, 0, -1, 2, 2, 2, 0]]))
        self.assertEqual(query_targets(self.db), [])


if __name__ == '__main__':
    unittest.main()
