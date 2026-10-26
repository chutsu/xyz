#!/usr/bin/env python3
"""
Plot the output of test_inertial_odometry_batch() (src/test_xyz.c).

That test runs a batch of IMU preintegration factors over a perturbed
initial guess and solves for the trajectory, writing out three CSVs via
inertial_odometry_save():

  /tmp/imu_odom-gnd.csv   ground truth
  /tmp/imu_odom-init.csv  perturbed initial guess the solver starts from
  /tmp/imu_odom-est.csv   solved estimate

Each has columns: ts,rx,ry,rz,qw,qx,qy,qz,vx,vy,vz,ba_x,ba_y,ba_z,bg_x,
bg_y,bg_z. Run the test first (e.g. `run_test test_xyz
test_inertial_odometry_batch` in scripts/run.sh) to generate them, then:

  python3 scripts/plot_inertial_odometry.py
"""
import argparse

import numpy as np
import pandas
import matplotlib.pyplot as plt


def load_odom_csv(csv_path):
  """Load an inertial_odometry_save() CSV into a dict of numpy arrays"""
  df = pandas.read_csv(csv_path)
  df = df.rename(columns=lambda x: x.strip())

  return {
      "ts": df["#ts"].to_numpy(),
      "r_WB": df[["rx", "ry", "rz"]].to_numpy(),
      "q_WB": df[["qw", "qx", "qy", "qz"]].to_numpy(),
      "v_WB": df[["vx", "vy", "vz"]].to_numpy(),
      "ba": df[["ba_x", "ba_y", "ba_z"]].to_numpy(),
      "bg": df[["bg_x", "bg_y", "bg_z"]].to_numpy(),
  }


TRAJ_STYLES = {
    # key: (line style, label, alpha)
    "gnd": ("k--", "Ground-Truth", 1.0),
    "init": ("r:", "Initial Guess", 0.3),
    "est": ("b-", "Estimate", 1.0),
}


def plot_trajectory_topdown(ax, odom_data):
  """Plot ground-truth / initial-guess / estimate trajectories from
  directly above (x-y plane)"""
  for key, (style, label, alpha) in TRAJ_STYLES.items():
    r_WB = odom_data[key]["r_WB"]
    ax.plot(r_WB[:, 0], r_WB[:, 1], style, marker="o", markersize=6,
             linewidth=2, alpha=alpha, label=label)

  ax.set_xlabel("x [m]")
  ax.set_ylabel("y [m]")
  ax.set_title("Top-Down (x-y)")
  ax.axis("equal")
  ax.legend(loc=0)


def plot_height(ax, odom_data):
  """Plot ground-truth / initial-guess / estimate height (z) per state"""
  for key, (style, label, alpha) in TRAJ_STYLES.items():
    r_WB = odom_data[key]["r_WB"]
    state_idx = np.arange(r_WB.shape[0])
    ax.plot(state_idx, r_WB[:, 2], style, marker="o", markersize=6,
             linewidth=2, alpha=alpha, label=label)

  ax.set_xlabel("State index")
  ax.set_ylabel("z [m]")
  ax.set_title("Height (z)")
  ax.legend(loc=0)


def plot_position_error(ax, odom_data):
  """Plot per-state position error of the initial guess and the
  estimate against ground truth, showing how much the solve recovered"""
  gnd_r = odom_data["gnd"]["r_WB"]
  init_r = odom_data["init"]["r_WB"]
  est_r = odom_data["est"]["r_WB"]

  state_idx = np.arange(gnd_r.shape[0])
  init_err = np.linalg.norm(init_r - gnd_r, axis=1)
  est_err = np.linalg.norm(est_r - gnd_r, axis=1)

  init_style, init_label, init_alpha = TRAJ_STYLES["init"]
  est_style, est_label, est_alpha = TRAJ_STYLES["est"]
  ax.plot(state_idx, init_err, init_style, marker="o", alpha=init_alpha,
           label=init_label)
  ax.plot(state_idx, est_err, est_style, marker="o", alpha=est_alpha,
           label=est_label)
  ax.set_xlabel("State index")
  ax.set_ylabel("Position error [m]")
  ax.set_title("Position Error vs. Ground-Truth")
  ax.legend(loc=0)


def main():
  parser = argparse.ArgumentParser(
      description="Plot test_inertial_odometry_batch()'s output")
  parser.add_argument(
      "--prefix",
      default="/tmp/imu_odom",
      help="Path prefix test_inertial_odometry_batch() saved its "
      "{prefix}-gnd.csv / -init.csv / -est.csv files under "
      "(default: /tmp/imu_odom)")
  args = parser.parse_args()

  odom_data = {
      "gnd": load_odom_csv(f"{args.prefix}-gnd.csv"),
      "init": load_odom_csv(f"{args.prefix}-init.csv"),
      "est": load_odom_csv(f"{args.prefix}-est.csv"),
  }

  fig = plt.figure(figsize=(16, 5))
  ax0 = fig.add_subplot(1, 3, 1)
  plot_trajectory_topdown(ax0, odom_data)

  ax1 = fig.add_subplot(1, 3, 2)
  plot_height(ax1, odom_data)

  ax2 = fig.add_subplot(1, 3, 3)
  plot_position_error(ax2, odom_data)

  fig.tight_layout()
  plt.show()


if __name__ == "__main__":
  main()
