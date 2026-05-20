#!/usr/bin/env python3
"""
Convert Kalibr camchain.yaml to YAC format
"""
import os
import sys
import yaml

import numpy as np


def print_usage():
  """ Print usage """
  print("Usage: kalibr2yac.py camchain.yaml")


if __name__ == "__main__":
  # Check CLI args
  if len(sys.argv) != 2:
    print_usage()
    sys.exit(-1)

  # Setup
  calib_path = sys.argv[1]
  calib = yaml.safe_load(open(calib_path, 'r').read())

  # Extract camera parameters
  data = {}
  for cam_str in ["cam0", "cam1"]:
    data[cam_str] = {}
    data[cam_str]["proj_params"] = calib[cam_str]["intrinsics"]
    data[cam_str]["dist_params"] = calib[cam_str]["distortion_coeffs"]

  # Extract camera-camera extrinsics
  T_C1C0 = np.reshape(np.array(calib['cam1']['T_cn_cnm1']), (4, 4))
  T_C0C1 = np.linalg.inv(T_C1C0)

  # Extract camera-imu extrinsics
  T_C0S = np.reshape(np.array(calib['cam0']['T_cam_imu']), (4, 4))
  T_SC0 = np.linalg.inv(T_C0S)

  output_path = f"""{calib_path.replace('.yaml', '-yac.yaml')}"""
  output_yaml = open(output_path, "w")
  output_yaml.write(f"""\
calib_target:
  target_type: "aprilgrid"
  tag_rows: 6
  tag_cols: 6
  tag_size: 0.037500
  tag_spacing: 0.300000

cam0:
  resolution: [640, 480]
  proj_model: "pinhole"
  dist_model: "radtan4"
  proj_params: {data['cam0']['proj_params']}
  dist_params: {data['cam0']['dist_params']}

cam1:
  resolution: [640, 480]
  proj_model: "pinhole"
  dist_model: "radtan4"
  proj_params: {data['cam1']['proj_params']}
  dist_params: {data['cam1']['dist_params']}

imu0:
  rate: 250.000000          # [Hz]
  a_max: 160.000000         # [m/s^2]
  g_max: 10.000000          # [rad/s]
  sigma_g_c: 2.780000e-03   # [rad/s/sqrt(Hz)]
  sigma_a_c: 2.520000e-02   # [m/s^2/sqrt(Hz)]
  sigma_gw_c: 1.650000e-05  # [rad/s^s/sqrt(Hz)]
  sigma_aw_c: 4.410000e-04  # [m/s^2/sqrt(Hz)]
  sigma_bg: 3.000000e-02    # [rad/s]
  sigma_ba: 1.000000e-01    # [m/s^2]
  g: 9.810070               # [m/s^2]

T_cam0_cam1:
  rows: 4
  cols: 4
  data: [
    {T_C0C1[0, 0]}, {T_C0C1[0, 1]}, {T_C0C1[0, 2]}, {T_C0C1[0, 3]},
    {T_C0C1[1, 0]}, {T_C0C1[1, 1]}, {T_C0C1[1, 2]}, {T_C0C1[1, 3]},
    {T_C0C1[2, 0]}, {T_C0C1[2, 1]}, {T_C0C1[2, 2]}, {T_C0C1[2, 3]},
    {T_C0C1[3, 0]}, {T_C0C1[3, 1]}, {T_C0C1[3, 2]}, {T_C0C1[3, 3]}
  ]

T_imu0_cam0:
  rows: 4
  cols: 4
  data: [
    {T_SC0[0, 0]}, {T_SC0[0, 1]}, {T_SC0[0, 2]}, {T_SC0[0, 3]},
    {T_SC0[1, 0]}, {T_SC0[1, 1]}, {T_SC0[1, 2]}, {T_SC0[1, 3]},
    {T_SC0[2, 0]}, {T_SC0[2, 1]}, {T_SC0[2, 2]}, {T_SC0[2, 3]},
    {T_SC0[3, 0]}, {T_SC0[3, 1]}, {T_SC0[3, 2]}, {T_SC0[3, 3]}
  ]
\n""")
