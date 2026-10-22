from pathlib import Path
import numpy as np

trace_file = Path('traces/mode2_external_validation/synthetic_highway_rho_35_20s.txt')
data = np.loadtxt(trace_file)
# Columns: time, vehicleID, x, y, speed
times = np.unique(data[:, 0])
vids = np.unique(data[:, 1])

print(f"Trace duration: {times.min():.1f}s to {times.max():.1f}s ({len(times)} time steps)")
print(f"Vehicle count: {len(vids)} (IDs {int(vids.min())} to {int(vids.max())})")

# Vehicle positions at t = 0
t0_data = data[data[:, 0] == 0.0]
x_pos = t0_data[:, 2]
y_pos = t0_data[:, 3]
speeds = t0_data[:, 4]

print(f"t=0 X positions: min={x_pos.min():.1f}m, max={x_pos.max():.1f}m, span={x_pos.max()-x_pos.min():.1f}m")
print(f"Mean inter-vehicle spacing: {np.mean(np.diff(np.sort(x_pos))):.2f}m")
print(f"Speeds: min={speeds.min():.2f} m/s, max={speeds.max():.2f} m/s ({np.mean(speeds)*3.6:.1f} km/h)")
