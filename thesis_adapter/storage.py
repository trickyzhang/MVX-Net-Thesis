"""JSONL evidence storage and parameterized SQLite attribute queries."""
import json
import sqlite3


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS observations (
        run_id TEXT NOT NULL, frame_id TEXT NOT NULL, timestamp REAL,
        payload TEXT NOT NULL, PRIMARY KEY(run_id, frame_id));
      CREATE TABLE IF NOT EXISTS targets (
        instance_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, frame_id TEXT NOT NULL,
        category TEXT NOT NULL, confidence REAL NOT NULL, range_m REAL NOT NULL,
        projection_valid INTEGER NOT NULL, center_x_norm REAL, payload TEXT NOT NULL,
        FOREIGN KEY(run_id,frame_id) REFERENCES observations(run_id,frame_id));
      CREATE INDEX IF NOT EXISTS idx_target_class_range ON targets(category,range_m);
      CREATE INDEX IF NOT EXISTS idx_observation_time ON observations(timestamp);
    ''')
    return db


def save_observation(db, observation):
    """Atomic idempotent replacement of ONE run/frame, including empty frames."""
    o = observation
    with db:
        db.execute('DELETE FROM targets WHERE run_id=? AND frame_id=?', (o['run_id'], o['frame_id']))
        db.execute('INSERT OR REPLACE INTO observations VALUES(?,?,?,?)',
                   (o['run_id'], o['frame_id'], o['timestamp'], json.dumps(o, allow_nan=False)))
        for r in o['detections']:
            bbox = r['bbox_2d']
            center = None if bbox is None else (bbox[0] + bbox[2]) / (2 * o['image_size'][0])
            db.execute('INSERT INTO targets VALUES(?,?,?,?,?,?,?,?,?)',
                       (r['instance_id'], o['run_id'], o['frame_id'], r['category'],
                        r['confidence'], r['range_center_lidar_m'], int(r['projection_valid']),
                        center, json.dumps(r, allow_nan=False)))


def query_targets(db, *, categories=(), max_range=None, side=None,
                  since=None, until=None, run_id=None, limit=100):
    """Distance is predicted 3D centre range; side means IMAGE left/right thirds."""
    clauses, params = ['t.projection_valid=1'], []
    if not isinstance(limit, int) or not 1 <= limit <= 10000:
        raise ValueError('limit must be an integer in [1,10000]')
    if categories:
        clauses.append('t.category IN (' + ','.join('?' for _ in categories) + ')')
        params.extend(categories)
    if max_range is not None:
        if max_range < 0:
            raise ValueError('max_range must be non-negative')
        clauses.append('t.range_m<=?')
        params.append(max_range)
    if side:
        if side not in ('left', 'center', 'right'):
            raise ValueError('side must be left, center or right')
        clauses.append({'left': 't.center_x_norm < 1.0/3',
                        'center': 't.center_x_norm >= 1.0/3 AND t.center_x_norm <= 2.0/3',
                        'right': 't.center_x_norm > 2.0/3'}[side])
    for value, clause in ((since, 'o.timestamp>=?'), (until, 'o.timestamp<=?'), (run_id, 't.run_id=?')):
        if value is not None:
            clauses.append(clause)
            params.append(value)
    rows = db.execute('SELECT t.payload,o.payload AS observation FROM targets t '
                      'JOIN observations o USING(run_id,frame_id) WHERE '
                      + ' AND '.join(clauses)
                      + ' ORDER BY t.range_m,t.instance_id LIMIT ?', params + [limit])
    results = []
    for row in rows:
        r, o = json.loads(row['payload']), json.loads(row['observation'])
        r['evidence'] = {k: o[k] for k in ('run_id', 'frame_id', 'timestamp', 'image_path', 'pointcloud_path', 'backend')}
        results.append(r)
    return results
