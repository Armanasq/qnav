# Test fixtures

`repoimu_tstick_04_1_40s.npz` — first 40 s (100 Hz, float32) of the
RepoIMU `TStick_04_1` trial (Szczęsna et al., "Reference data set for
accuracy evaluation of orientation estimation algorithms for inertial
motion capture systems", public research dataset), converted to the qnav
NPZ layout (`dt, gyro [rad/s], accel [m/s²], mag, q_ref (w,x,y,z,
reference z-up), movement`). Used by `tests/test_real_data.py` so real-data
replay tests run without the full multi-hundred-MB benchmark collection.
