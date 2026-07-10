# Real-data attitude evaluation

- commit: `49f3b90` · 13th Gen Intel(R) Core(TM) i9-13980HX · NumPy 1.26.4 · Python 3.10.12
- estimator: Eskf (gate 0.999, gravity always, mag when referenced)
- initialization: **ground_truth** *(oracle information)*
- magnetic reference: **oracle** *(oracle information)*
- noise config: universal {'gyro_nd': 0.005, 'gyro_bw': 1e-05, 'acc_sigma': 0.05, 'mag_sigma': 0.1}
- trials: 52 succeeded, 0 failed (rate 0.000)

| collection | trials | median RMSE [°] | median tilt [°] | median heading [°] | mag-aided | median RTF |
|---|---|---|---|---|---|---|
| Caruso-Sassari | 8 | 2.85 | 1.91 | 2.24 | 8/8 | 24x |
| EuRoC-MAV | 6 | 81.22 | 3.49 | 81.06 | 0/6 | 18x |
| Myon | 8 | 39.63 | 0.59 | 39.62 | 0/8 | 13x |
| OxIOD | 8 | 4.29 | 2.46 | 3.32 | 8/8 | 23x |
| RepoIMU | 8 | 0.97 | 0.55 | 0.68 | 8/8 | 23x |
| TUM-VI | 6 | 92.86 | 5.60 | 92.77 | 0/6 | 18x |
| diodem | 8 | 2.32 | 0.65 | 2.18 | 8/8 | 57x |

_Aggregation: median over trials per collection; metrics exclude calibration segment and ground-truth gaps._
