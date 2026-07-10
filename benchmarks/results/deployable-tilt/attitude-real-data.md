# Real-data attitude evaluation

- commit: `19f0dbf` · 13th Gen Intel(R) Core(TM) i9-13980HX · NumPy 1.26.4 · Python 3.10.12
- estimator: Eskf (gate 0.999, gravity always, mag when referenced)
- initialization: **accel_tilt** (deployable)
- magnetic reference: **calibration**
- noise config: universal {'gyro_nd': 0.005, 'gyro_bw': 1e-05, 'acc_sigma': 0.05, 'mag_sigma': 0.1}
- trials: 52 succeeded, 0 failed (rate 0.000)

| collection | trials | median RMSE [°] | median tilt [°] | median heading [°] | heading status | mag-aided | median RTF |
|---|---|---|---|---|---|---|---|
| Caruso-Sassari | 8 | 2.91 | 1.97 | 2.26 | aligned-absolute (8/8 trials) | 8/8 | 25x |
| EuRoC-MAV | 6 | 80.50 | 3.48 | n/a | unobservable (drift-only) | 0/6 | 20x |
| Myon | 8 | 47.35 | 0.60 | n/a | unobservable (drift-only) | 0/8 | 13x |
| OxIOD | 8 | 4.28 | 2.49 | 3.33 | aligned-absolute (8/8 trials) | 8/8 | 24x |
| RepoIMU | 8 | 0.96 | 0.52 | 0.68 | aligned-absolute (8/8 trials) | 8/8 | 24x |
| TUM-VI | 6 | 94.12 | 5.62 | n/a | unobservable (drift-only) | 0/6 | 20x |
| diodem | 8 | 5.12 | 0.82 | 4.52 | aligned-absolute (8/8 trials) | 8/8 | 61x |

_Aggregation: median over trials per collection; metrics exclude calibration segment and ground-truth gaps._
