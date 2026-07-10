# Real-data attitude evaluation

- commit: `46886db-dirty` · 13th Gen Intel(R) Core(TM) i9-13980HX · NumPy 1.26.4 · Python 3.10.12
- estimator: Eskf (gate 0.999, gravity always, mag when referenced)
- initialization: **ground_truth** *(oracle information)*
- magnetic reference: **oracle** *(oracle information)*
- noise config: universal {'gyro_nd': 0.005, 'gyro_bw': 1e-05, 'acc_sigma': 0.05, 'mag_sigma': 0.1}
- trials: 52 succeeded, 0 failed (rate 0.000)

| collection | trials | median RMSE [°] | median tilt [°] | median heading [°] | heading status | mag-aided | median RTF |
|---|---|---|---|---|---|---|---|
| Caruso-Sassari | 8 | 2.85 | 1.91 | 2.24 | aligned-absolute (8/8 trials) | 8/8 | 7x |
| EuRoC-MAV | 6 | 81.22 | 3.49 | n/a | unobservable (drift-only) | 0/6 | 6x |
| Myon | 8 | 39.63 | 0.59 | n/a | unobservable (drift-only) | 0/8 | 4x |
| OxIOD | 8 | 4.29 | 2.46 | 3.32 | aligned-absolute (8/8 trials) | 8/8 | 10x |
| RepoIMU | 8 | 0.97 | 0.55 | 0.68 | aligned-absolute (8/8 trials) | 8/8 | 8x |
| TUM-VI | 6 | 92.86 | 5.60 | n/a | unobservable (drift-only) | 0/6 | 6x |
| diodem | 8 | 2.32 | 0.65 | 2.18 | aligned-absolute (8/8 trials) | 8/8 | 15x |

_Aggregation: median over trials per collection; metrics exclude calibration segment and ground-truth gaps._
