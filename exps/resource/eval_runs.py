""" Evaluation Script """
#!/usr/bin/env python3
import os
import sys
import glob
import json
import datetime
import time
import shutil
import yaml

import pandas
import rosbag
import seaborn
import numpy as np
import matplotlib.pylab as plt
from matplotlib.patches import PathPatch
from PIL import Image, ImageChops

from proto import tf_trans
from proto import tf_quat
from proto import rad2deg
from proto import quat2euler

# # update latex preamble
# plt.rcParams.update({
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
#     "pgf.texsystem": 'pdflatex',  # default is xetex
#     "pgf.preamble": [r"\usepackage[T1]{fontenc}", r"\usepackage{mathpazo}"]
# })

##############################################################################
# UTILS
##############################################################################


def list_flatten(xss):
  """ Flatten a list """
  return [x for xs in xss for x in xs]


def trim_image(image_path):
  """ Trim whitespace in image """
  im = Image.open(image_path)
  bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
  diff = ImageChops.difference(im, bg)
  diff = ImageChops.add(diff, diff, 2.0, -100)
  bbox = diff.getbbox()

  padding = 10
  start_x = max(bbox[0] - padding, 0)
  start_y = max(bbox[1] - padding, 0)
  end_x = min(bbox[2] + padding, im.width)
  end_y = min(bbox[3] + padding, im.height)
  bbox = (start_x, start_y, end_x, end_y)

  cropped_im = im.crop(bbox)
  cropped_im.save(image_path)


def load_json(json_path):
  """ Load JSON """
  with open(json_path, "r") as json_file:
    data = json.load(json_file)
  return data


def adjust_box_widths(g, fac):
  """ Adjust the withs of a seaborn-generated boxplot """
  for ax in g.axes:
    for c in ax.get_children():
      if isinstance(c, PathPatch):
        # getting current width of box:
        p = c.get_path()
        verts = p.vertices
        verts_sub = verts[:-1]
        xmin = np.min(verts_sub[:, 0])
        xmax = np.max(verts_sub[:, 0])
        xmid = 0.5 * (xmin + xmax)
        xhalf = 0.5 * (xmax - xmin)

        # setting new width of box
        xmin_new = xmid - fac * xhalf
        xmax_new = xmid + fac * xhalf
        verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
        verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

        # setting new width of median line
        for l in ax.lines:
          if np.all(l.get_xdata() == [xmin, xmax]):
            l.set_xdata([xmin_new, xmax_new])


class YamlUnknown(yaml.SafeLoader):
  def ignore_unknown(self, node):
    return None


def extract_time_delay(calib_file):
  """ Get the cam-imu time delay from the calibration file """
  imu_cam_td = None

  with open(calib_file, "r") as s:
    _ = s.readline()  # Skip first line

    YamlUnknown.add_constructor(None, YamlUnknown.ignore_unknown)
    calib = yaml.load(s.read(), Loader=YamlUnknown)
    imu_cam_td = calib["Camera.timeshift_cam_imu"]

  return imu_cam_td


def check_rosbag_topic(rosbag_path, topic):
  """ Check ROS bag topic exists """
  bag = rosbag.Bag(rosbag_path, 'r')
  info = bag.get_type_and_topic_info()

  if topic in info.topics:
    return True
  return False


##############################################################################
# EUROC EXPERIMENTS
##############################################################################


def plot_euroc(kalibr_json, yac_json, **kwargs):
  """ Plot EuRoC Results """
  save_fig = kwargs.get("save_fig", True)
  save_fig_path = kwargs.get("save_fig_path", "euroc_results.png")
  show_plot = kwargs.get("show_plot", False)
  metric = kwargs.get("metric", "rmse")

  # Load euroc results data
  kalibr_result = load_json(kalibr_json)
  yac_result = load_json(yac_json)
  result_data = [("Kalibr", kalibr_result), ("Our Method", yac_result)]

  # Setup
  seaborn.set_style("darkgrid")
  dpi = 96
  plot_width = 600
  plot_height = 600
  plot_rows = 3
  plot_cols = 1
  fig_size = (plot_width / dpi, plot_height / dpi)
  _, axs = plt.subplots(plot_rows, plot_cols, figsize=fig_size, dpi=dpi)

  # -- Machine Hall
  MH = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05"]
  df = {"seq": [], "type": [], "ate-rmse": []}
  for result_type, result in result_data:
    for seq in MH:
      for metric in result[seq]:
        df["seq"].append(seq)
        df["type"].append(result_type)
        df["ate-rmse"].append(metric["ate"]["rmse"])

  seaborn.boxplot(ax=axs[0],
                  data=df,
                  hue="type",
                  x="seq",
                  y="ate-rmse",
                  showfliers=False)
  axs[0].set_xticklabels(MH)
  axs[0].set_ylabel("RMSE ATE [m]")
  axs[0].set_title("Machine Hall")

  # -- Vicon Room 01
  V1 = ["V1_01", "V1_02", "V1_03"]
  df = {"seq": [], "type": [], "ate-rmse": []}
  for result_type, result in result_data:
    for seq in V1:
      for metric in result[seq]:
        df["seq"].append(seq)
        df["type"].append(result_type)
        df["ate-rmse"].append(metric["ate"]["rmse"])

  seaborn.boxplot(ax=axs[1],
                  data=df,
                  hue="type",
                  x="seq",
                  y="ate-rmse",
                  showfliers=False)
  axs[1].set_xticklabels(V1)
  axs[1].set_ylabel("RMSE ATE [m]")
  axs[1].set_title("Vicon-Room 01")

  # -- Vicon Room 02
  V2 = ["V2_01", "V2_02", "V2_03"]
  df = {"seq": [], "type": [], "ate-rmse": []}
  for result_type, result in result_data:
    for seq in V2:
      for metric in result[seq]:
        df["seq"].append(seq)
        df["type"].append(result_type)
        df["ate-rmse"].append(metric["ate"]["rmse"])

  seaborn.boxplot(ax=axs[2],
                  data=df,
                  hue="type",
                  x="seq",
                  y="ate-rmse",
                  showfliers=False)
  axs[2].set_xticklabels(V2)
  axs[2].set_ylabel(f"RMSE ATE [m]")
  axs[2].set_title("Vicon-Room 02")

  # -- Subplot adjust
  plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05)

  # -- Save
  if save_fig:
    plt.savefig(save_fig_path)
    trim_image(save_fig_path)

  # -- Show
  if show_plot:
    plt.show()


def plot_convergence(kalibr_csv, yac_yaml, **kwargs):
  """ Plot convergence """
  save_fig = kwargs.get("save_fig", True)
  save_path = kwargs.get("save_path", "convergence.png")
  show_plot = kwargs.get("show_plot", False)
  plot_width = kwargs.get("plot_width", 550)
  plot_height = kwargs.get("plot_height", 550)

  data = pandas.read_csv(kalibr_csv)
  kalibr = {}
  kalibr["view_idx"] = data["view_idx"].to_numpy()
  kalibr["view_ts"] = data["view_ts"].to_numpy()
  kalibr["view_accepted"] = data["view_accepted"].to_numpy()
  kalibr["info"] = -1.0 * data["info"].to_numpy()
  kalibr["entropy"] = data["entropy"].to_numpy()

  data = yaml.safe_load(open(yac_yaml, 'r').read())
  data = np.array(data["convergence"]["data"])
  nb_rows = int(data.shape[0] / 9)
  nb_cols = int(9)
  data = np.resize(data, (nb_rows, nb_cols))

  yac = {}
  yac["view_idx"] = []
  yac["view_ts"] = []
  yac["view_accepted"] = []
  yac["info"] = []
  yac["entropy"] = []
  for row in data:
    yac["view_idx"].append(row[0])
    yac["view_ts"].append(row[1])
    yac["view_accepted"].append(row[2])
    yac["info"].append(row[3])
    yac["entropy"].append(row[4])

  dpi = 96
  plt.figure(figsize=(plot_width / dpi, plot_height / dpi), dpi=dpi)
  plt.plot(kalibr["view_idx"][10:-2], kalibr["entropy"][10:-2], label="kalibr")
  plt.plot(yac["view_idx"][10:-2], yac["entropy"][10:-2], label="yac")
  plt.xlabel("View Index")
  plt.ylabel("Shannon Entropy [nats]")
  plt.legend(loc=0)

  if save_fig:
    plt.savefig(save_path)
    trim_image(save_path)

  if show_plot:
    plt.show()


def plot_euroc_results():
  """ Plot EuRoC Results """
  # Stereo Calibration Result
  kalibr_res = "./euroc_results/plots/orbslam3-stereo-kalibr.json"
  yac_res = "./euroc_results/plots/orbslam3-stereo-yac.json"
  save_path = "./plots/euroc-stereo.png"
  plot_euroc(kalibr_res, yac_res, save_fig_path=save_path)

  # Stereo-IMU Calibration Result
  kalibr_res = "./euroc_results/plots/orbslam3-stereo_imu-kalibr.json"
  yac_res = "./euroc_results/plots/orbslam3-stereo_imu-yac.json"
  save_path = "./plots/euroc-stereo_imu.png"
  plot_euroc(kalibr_res, yac_res, save_fig_path=save_path)


##############################################################################
# YAC EXPERIMENTS
##############################################################################


def load_pose_data(csv_file):
  """ Load CSV data """
  names = ["ts", "x", "y", "z", "qx", "qy", "qz", "qw"]
  csv_data = pandas.read_csv(csv_file, names=names, header=0, sep=" ")

  # Extract position
  x = csv_data['x'].to_numpy()
  y = csv_data['y'].to_numpy()
  z = csv_data['z'].to_numpy()
  positions = np.array([x, y, z])

  # Form timestamps
  timestamps = []
  times = []
  ts0 = None
  for ts in csv_data["ts"]:
    timestamps.append(ts)

    if ts0 is None:
      times.append(0.0)
      ts0 = ts
    else:
      times.append((ts - ts0) * 1e-9)

  return (timestamps, times, positions)


def bag2csv(bag_path, topic, output_path):
  """ Convert ROSBAG to CSV file """
  cmd = f"python3 scripts/bag2csv.py {bag_path} {topic} {output_path}"
  os.system(cmd)


def convert_to_msckf_calib(calib_format, calib_file, output_path):
  """ Convert calibration file to MSCKF format """
  # Load calibration file
  calib_yaml = open(calib_file, 'r')
  calib = yaml.safe_load(calib_yaml)

  cam0_K = None
  cam0_D = None
  cam1_K = None
  cam1_D = None
  T_C0S = None
  T_C1S = None
  T_C1C0 = None

  if calib_format == "yac":
    cam0_K = calib['cam0']['proj_params']
    cam0_D = calib['cam0']['dist_params']
    cam1_K = calib['cam1']['proj_params']
    cam1_D = calib['cam1']['dist_params']

    T_SC0 = np.reshape(np.array(calib['T_imu0_cam0']['data']), (4, 4))
    T_C0C1 = np.reshape(np.array(calib['T_cam0_cam1']['data']), (4, 4))
    T_SC1 = T_SC0 @ T_C0C1

    T_C0S = np.linalg.inv(T_SC0).flatten().tolist()
    T_C1S = np.linalg.inv(T_SC1).flatten().tolist()
    T_C1C0 = np.linalg.inv(T_C0C1).flatten().tolist()

  elif calib_format == "kalibr":
    cam0_K = calib['cam0']['intrinsics']
    cam0_D = calib['cam0']['distortion_coeffs']
    cam1_K = calib['cam1']['intrinsics']
    cam1_D = calib['cam1']['distortion_coeffs']

    T_C0S = list_flatten(calib['cam0']['T_cam_imu'])
    T_C1S = list_flatten(calib['cam1']['T_cam_imu'])
    T_C1C0 = list_flatten(calib['cam1']['T_cn_cnm1'])

  else:
    print("Unsupported format [%s]!" % calib_format)
    sys.exit(-1)

  CALIB_FILE = f'''\
cam0:
  T_cam_imu:
    [{T_C0S[0]}, {T_C0S[1]}, {T_C0S[2]}, {T_C0S[3]},
     {T_C0S[4]}, {T_C0S[5]}, {T_C0S[6]}, {T_C0S[7]},
     {T_C0S[8]}, {T_C0S[9]}, {T_C0S[10]}, {T_C0S[11]},
     {T_C0S[12]}, {T_C0S[13]}, {T_C0S[14]}, {T_C0S[15]}]
  camera_model: pinhole
  distortion_coeffs: {cam0_D}
  distortion_model: radtan
  intrinsics: {cam0_K}
  resolution: [640, 480]
  rostopic: /rs/ir0/image
  timeshift_cam_imu: 0.0

cam1:
  T_cam_imu:
    [{T_C1S[0]}, {T_C1S[1]}, {T_C1S[2]}, {T_C1S[3]},
     {T_C1S[4]}, {T_C1S[5]}, {T_C1S[6]}, {T_C1S[7]},
     {T_C1S[8]}, {T_C1S[9]}, {T_C1S[10]}, {T_C1S[11]},
     {T_C1S[12]}, {T_C1S[13]}, {T_C1S[14]}, {T_C1S[15]}]
  T_cn_cnm1:
    [{T_C1C0[0]}, {T_C1C0[1]}, {T_C1C0[2]}, {T_C1C0[3]},
     {T_C1C0[4]}, {T_C1C0[5]}, {T_C1C0[6]}, {T_C1C0[7]},
     {T_C1C0[8]}, {T_C1C0[9]}, {T_C1C0[10]}, {T_C1C0[11]},
     {T_C1C0[12]}, {T_C1C0[13]}, {T_C1C0[14]}, {T_C1C0[15]}]
  camera_model: pinhole
  distortion_coeffs: {cam1_D}
  distortion_model: radtan
  intrinsics: {cam1_K}
  resolution: [640, 480]
  rostopic: /rs/ir1/image
  timeshift_cam_imu: 0.0

T_imu_body:
  [1.0000, 0.0000, 0.0000, 0.0000,
   0.0000, 1.0000, 0.0000, 0.0000,
   0.0000, 0.0000, 1.0000, 0.0000,
   0.0000, 0.0000, 0.0000, 1.0000]
'''
  outfile = open(output_path, 'w')
  outfile.write(CALIB_FILE)
  outfile.close()


def convert_to_orbslam_calib(mode, calib_format, calib_file, output_path):
  """ Convert calibration file to ORBSLAM format """
  # Load calibration file
  calib_yaml = open(calib_file, 'r')
  calib = yaml.safe_load(calib_yaml)

  cam0_K = None
  cam0_D = None
  cam1_K = None
  cam1_D = None
  T_C0C1 = None
  T_SC0 = None
  cam_td = None

  if calib_format == "yac":
    cam0_K = calib['cam0']['proj_params']
    cam0_D = np.array(calib['cam0']['dist_params'] + [0.0])
    cam1_K = calib['cam1']['proj_params']
    cam1_D = np.array(calib['cam1']['dist_params'] + [0.0])

    if mode == "VO":
      T_C0C1 = np.reshape(np.array(calib['T_cam0_cam1']['data']), (4, 4))
    elif mode == "VIO":
      T_C0C1 = np.reshape(np.array(calib['T_cam0_cam1']['data']), (4, 4))
      T_SC0 = np.reshape(np.array(calib['T_imu0_cam0']['data']), (4, 4))
      cam_td = calib['time_delay']
    else:
      print("Unsupported mode [%s]!" % mode)
      sys.exit(-1)

  elif calib_format == "kalibr":
    cam0_K = calib['cam0']['intrinsics']
    cam0_D = np.array(calib['cam0']['distortion_coeffs'] + [0.0])
    cam1_K = calib['cam1']['intrinsics']
    cam1_D = np.array(calib['cam1']['distortion_coeffs'] + [0.0])

    if mode == "VO":
      T_C1C0 = np.reshape(np.array(calib['cam1']['T_cn_cnm1']), (4, 4))
      T_C0C1 = np.linalg.inv(T_C1C0)
    elif mode == "VIO":
      T_C1C0 = np.reshape(np.array(calib['cam1']['T_cn_cnm1']), (4, 4))
      T_C0S = np.reshape(np.array(calib['cam0']['T_cam_imu']), (4, 4))
      T_C0C1 = np.linalg.inv(T_C1C0)
      T_SC0 = np.linalg.inv(T_C0S)
      cam0_td = calib['cam0']['timeshift_cam_imu']
      cam1_td = calib['cam1']['timeshift_cam_imu']
      cam_td = (cam0_td + cam1_td) / 2.0
    else:
      print("Unsupported mode [%s]!" % mode)
      sys.exit(-1)

  else:
    print("Unsupported format [%s]!" % calib_format)
    sys.exit(-1)

  # Camera Extrinsics to string
  cam_exts = []
  for i in range(4):
    for j in range(4):
      cam_exts.append(str(T_C0C1[i, j]))

  # IMU Extrinsics to string
  imu_exts = []
  if T_SC0 is not None:
    for i in range(4):
      for j in range(4):
        imu_exts.append(str(T_SC0[i, j]))

  CALIB_FILE = ""
  if mode == "VO":
    CALIB_FILE = f'''\
%YAML:1.0

#--------------------------------------------------------------------------------------------
# System config
#--------------------------------------------------------------------------------------------

# When the variables are commented, the system doesn't load a previous session or not store the current one

# If the LoadFile doesn't exist, the system give a message and create a new Atlas from scratch
#System.LoadAtlasFromFile: "Session_MH01_MH02_MH03_Stereo60_Pseudo"

# The store file is created from the current session, if a file with the same name exists it is deleted
#System.SaveAtlasToFile: "Session_MH01_MH02_MH03_Stereo60_Pseudo"

# Disable Loop Closure
System.LoopClosing: 0

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------
File.version: "1.0"

Camera.type: "PinHole"

# Camera calibration and distortion parameters (OpenCV)
Camera1.fx: {cam0_K[0]}
Camera1.fy: {cam0_K[1]}
Camera1.cx: {cam0_K[2]}
Camera1.cy: {cam0_K[3]}

Camera1.k1: {cam0_D[0]}
Camera1.k2: {cam0_D[1]}
Camera1.p1: {cam0_D[2]}
Camera1.p2: {cam0_D[3]}

Camera2.fx: {cam1_K[0]}
Camera2.fy: {cam1_K[1]}
Camera2.cx: {cam1_K[2]}
Camera2.cy: {cam1_K[3]}

Camera2.k1: {cam1_D[0]}
Camera2.k2: {cam1_D[1]}
Camera2.p1: {cam1_D[2]}
Camera2.p2: {cam1_D[3]}

Camera.width: 640
Camera.height: 480

# Camera frames per second
Camera.fps: 20

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

Stereo.ThDepth: 60.0
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{cam_exts[0]}, {cam_exts[1]}, {cam_exts[2]}, {cam_exts[3]},
         {cam_exts[4]}, {cam_exts[5]}, {cam_exts[6]}, {cam_exts[7]},
         {cam_exts[8]}, {cam_exts[9]}, {cam_exts[10]}, {cam_exts[11]},
         {cam_exts[12]}, {cam_exts[13]}, {cam_exts[14]}, {cam_exts[15]}]

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1200

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
'''

  elif mode == "VIO":
    CALIB_FILE = f'''\
%YAML:1.0

#--------------------------------------------------------------------------------------------
# System config
#--------------------------------------------------------------------------------------------

# When the variables are commented, the system doesn't load a previous session or not store the current one

# If the LoadFile doesn't exist, the system give a message and create a new Atlas from scratch
#System.LoadAtlasFromFile: "Session_MH01_MH02_MH03_Stereo60_Pseudo"

# The store file is created from the current session, if a file with the same name exists it is deleted
#System.SaveAtlasToFile: "Session_MH01_MH02_MH03_Stereo60_Pseudo"

#--------------------------------------------------------------------------------------------
# Camera Parameters. Adjust them!
#--------------------------------------------------------------------------------------------
File.version: "1.0"

Camera.type: "PinHole"

# Camera calibration and distortion parameters (OpenCV)
Camera1.fx: {cam0_K[0]}
Camera1.fy: {cam0_K[1]}
Camera1.cx: {cam0_K[2]}
Camera1.cy: {cam0_K[3]}

Camera1.k1: {cam0_D[0]}
Camera1.k2: {cam0_D[1]}
Camera1.p1: {cam0_D[2]}
Camera1.p2: {cam0_D[3]}

Camera2.fx: {cam1_K[0]}
Camera2.fy: {cam1_K[1]}
Camera2.cx: {cam1_K[2]}
Camera2.cy: {cam1_K[3]}

Camera2.k1: {cam1_D[0]}
Camera2.k2: {cam1_D[1]}
Camera2.p1: {cam1_D[2]}
Camera2.p2: {cam1_D[3]}

Camera.width: 640
Camera.height: 480
Camera.timeshift_cam_imu: {cam_td}

# Camera frames per second
Camera.fps: 20

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1

# Close/Far threshold. Baseline times.
Stereo.ThDepth: 60.0
Stereo.T_c1_c2: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{cam_exts[0]}, {cam_exts[1]}, {cam_exts[2]}, {cam_exts[3]},
         {cam_exts[4]}, {cam_exts[5]}, {cam_exts[6]}, {cam_exts[7]},
         {cam_exts[8]}, {cam_exts[9]}, {cam_exts[10]}, {cam_exts[11]},
         {cam_exts[12]}, {cam_exts[13]}, {cam_exts[14]}, {cam_exts[15]}]

# Transformation from camera 0 to body-frame (imu)
IMU.T_b_c1: !!opencv-matrix
  rows: 4
  cols: 4
  dt: f
  data: [{imu_exts[0]}, {imu_exts[1]}, {imu_exts[2]}, {imu_exts[3]},
         {imu_exts[4]}, {imu_exts[5]}, {imu_exts[6]}, {imu_exts[7]},
         {imu_exts[8]}, {imu_exts[9]}, {imu_exts[10]}, {imu_exts[11]},
         {imu_exts[12]}, {imu_exts[13]}, {imu_exts[14]}, {imu_exts[15]}]

# IMU noise
IMU.NoiseGyro: 1.6968e-04
IMU.NoiseAcc: 2.0e-03
IMU.GyroWalk: 1.9393e-05
IMU.AccWalk: 3.0e-03
IMU.Frequency: 200.0

#--------------------------------------------------------------------------------------------
# ORB Parameters
#--------------------------------------------------------------------------------------------

# ORB Extractor: Number of features per image
ORBextractor.nFeatures: 1200

# ORB Extractor: Scale factor between levels in the scale pyramid
ORBextractor.scaleFactor: 1.2

# ORB Extractor: Number of levels in the scale pyramid
ORBextractor.nLevels: 8

# ORB Extractor: Fast threshold
# Image is divided in a grid. At each cell FAST are extracted imposing a minimum response.
# Firstly we impose iniThFAST. If no corners are detected we impose a lower value minThFAST
# You can lower these values if your images have low contrast
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

#--------------------------------------------------------------------------------------------
# Viewer Parameters
#--------------------------------------------------------------------------------------------
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
Viewer.imageViewScale: 1.0
  '''

  # Output calib file
  outfile = open(output_path, 'w')
  outfile.write(CALIB_FILE)
  outfile.close()


def extract_ground_truth(rosbag_path, save_path):
  """ Extract ground truth from ROS bag """
  run_name = os.path.basename(rosbag_path).split(".")[0]

  mocap_topic = ""
  if check_rosbag_topic(rosbag_path, "/vicon/chris_d435i/chris_d435i"):
    mocap_topic = "/vicon/chris_d435i/chris_d435i"
  else:
    mocap_topic = "/vicon/realsense_d435i/realsense_d435i"

  # Create ground truth file
  gt_csv_path = save_path
  if not os.path.exists(gt_csv_path):
    # -- Convert odom rosbag to csv file
    bag2csv(rosbag_path, mocap_topic, gt_csv_path)
    # -- Reformat CSV file
    csv = pandas.read_csv(gt_csv_path)
    timestamps = (csv["secs"] * 1e9).astype(int) + csv["nsecs"]
    positions = csv[["x", "y", "z"]]
    quaternions = csv[["qw", "qx", "qy", "qz"]]
    data = {
        "ts": timestamps,
        "x": positions["x"],
        "y": positions["y"],
        "z": positions["z"],
        "qx": quaternions["qx"],
        "qy": quaternions["qy"],
        "qz": quaternions["qz"],
        "qw": quaternions["qw"],
    }
    df = pandas.DataFrame(data)
    df.to_csv(gt_csv_path, index=False, sep=" ", header=False)


def run_msckf(calib_file, rosbag_path, record_path, dryrun=False):
  """ Run MSCKF """
  run_name = os.path.basename(rosbag_path).split(".")[0]
  # mocap_topic = "/vicon/realsense_d435i/realsense_d435i"
  mocap_topic = "/vicon/chris_d435i/chris_d435i"
  odom_topic = "/yac_experiments/vio/odom"
  odom_csv_path = record_path.replace(".bag", ".csv")

  # Check if calib is already evaluated
  if os.path.exists(odom_csv_path):
    return

  # Run MSCKF
  run_cmd = f"""\
  make run_msckf \
  calib_file={calib_file} \
  rosbag_file={rosbag_path} \
  output_file={record_path} \
  """
  if dryrun:
    print(run_cmd)
    return
  os.system(run_cmd)

  # Process odom data
  # -- Convert odom rosbag to csv file
  bag2csv(record_path, odom_topic, odom_csv_path)
  # -- Reformat CSV file
  csv = pandas.read_csv(odom_csv_path)
  timestamps = (csv["secs"] * 1e9).astype(int) + csv["nsecs"]
  positions = csv[["x", "y", "z"]]
  quaternions = csv[["qw", "qx", "qy", "qz"]]
  data = {
      "ts": timestamps,
      "x": positions["x"],
      "y": positions["y"],
      "z": positions["z"],
      "qx": quaternions["qx"],
      "qy": quaternions["qy"],
      "qz": quaternions["qz"],
      "qw": quaternions["qw"],
  }
  df = pandas.DataFrame(data)
  df.to_csv(odom_csv_path, index=False, sep=" ", header=False)
  # -- Remove record rosbag
  if os.path.exists(record_path):
    os.remove(record_path)


def run_orbslam3_stereo_imu(calib_file, rosbag_path, record_path, dryrun=False):
  """ Run ORBSLAM3 """

  # Check if calib is already evaluated
  imu_cam_td_s = extract_time_delay(calib_file)
  bag2euroc_cmd = f"python3 scripts/bag2euroc.py {rosbag_path} {imu_cam_td_s}"
  orbslam3_cmd = f"""\
  make run_orbslam3_stereo_imu \
    calib_file={calib_file} \
    data_path={rosbag_path.replace('.bag', '')} \
    camera_timestamps={rosbag_path.replace('.bag', '')}/camera_timestamps.txt \
    record_path={record_path}
	"""
  if dryrun:
    print(bag2euroc_cmd)
    print(orbslam3_cmd)
    return

  # Convert ROS bag to EuRoC format
  print(f"Converting ROSBAG [{rosbag_path}] to EuRoC format")
  print(bag2euroc_cmd)
  data_path = rosbag_path.replace(".bag", "")
  if os.path.exists(data_path) and os.path.isdir(data_path):
    shutil.rmtree(data_path)
  os.system(bag2euroc_cmd)

  # Run ORBSLAM3
  if os.path.exists(record_path) is False:
    print("Evaluating with ORBSLAM3")
    print(orbslam3_cmd)
    os.system(orbslam3_cmd)

    # # Reformat results
    # names = ["ts", "rx", "ry", "rz", "qx", "qy", "qz", "qw"]
    # dtype = {
    #     "ts": int,
    #     "rx": float,
    #     "ry": float,
    #     "rz": float,
    #     "qx": float,
    #     "qy": float,
    #     "qz": float,
    #     "qw": float
    # }
    # df = pandas.read_csv(record_path,
    #                      header=None,
    #                      names=names,
    #                      delimiter=" ",
    #                      dtype=dtype)
    # df.to_csv(record_path, index=False, sep=" ", header=False)

  # Clean up data directory after
  data_path = rosbag_path.replace(".bag", "")
  if os.path.exists(data_path) and os.path.isdir(data_path):
    shutil.rmtree(data_path)


def run_orbslam3_stereo(calib_file, rosbag_path, record_path, dryrun=False):
  """ Run ORBSLAM3 """
  # Check if calib is already evaluated
  bag2euroc_cmd = f"python3 scripts/bag2euroc.py {rosbag_path}"
  orbslam3_cmd = f"""\
  make run_orbslam3_stereo \
    calib_file={calib_file} \
    data_path={rosbag_path.replace('.bag', '')} \
    camera_timestamps={rosbag_path.replace('.bag', '')}/camera_timestamps.txt \
    record_path={record_path}
	"""
  if dryrun:
    print(bag2euroc_cmd)
    print(orbslam3_cmd)
    return

  # Convert ROS bag to EuRoC format
  if os.path.exists(rosbag_path.replace(".bag", "")) is False:
    print("Converting ROSBAG [{rosbag_path}] to EuRoC format")
    print(bag2euroc_cmd)
    os.system(bag2euroc_cmd)

  # Run ORBSLAM3
  if os.path.exists(record_path) is False:
    print("Evaluating with ORBSLAM3")
    print(orbslam3_cmd)
    os.system(orbslam3_cmd)

    # Reformat results
    names = ["ts", "rx", "ry", "rz", "qx", "qy", "qz", "qw"]
    dtype = {
        "ts": int,
        "rx": float,
        "ry": float,
        "rz": float,
        "qx": float,
        "qy": float,
        "qz": float,
        "qw": float
    }
    df = pandas.read_csv(record_path,
                         header=None,
                         names=names,
                         delimiter=" ",
                         dtype=dtype)
    df.to_csv(record_path, index=False, sep=" ", header=False)


def eval_kalibr_calibs(rosbags_dir, kalibr_dir, estimates_dir, dryrun=False):
  """ Evaluate Kalibr Calibrations """
  calib_format = "kalibr"
  kalibr_paths = f"{kalibr_dir}/*/calib_imu-camchain.yaml"

  # Create estimates directory
  if not os.path.exists(estimates_dir):
    os.makedirs(estimates_dir)

  # # Evaluate - MSCKF
  # for calib_file in sorted(glob.glob(kalibr_paths)):
  #   # Convert Kalibr camchain file to MSCKF format
  #   calib_seq = os.path.dirname(calib_file).split("/")[-1]
  #   calib_outfile = f"kalibr_calib.yaml"
  #   convert_to_msckf_calib(calib_format, calib_file, calib_outfile)

  #   # Evaluate Kalibr calibration by running MSCKF on all evaluation rosbags
  #   for rosbag_path in sorted(glob.glob(f"{rosbags_dir}/*.bag")):
  #     seq = os.path.basename(rosbag_path).replace(".bag", "")
  #     record_path = f"{estimates_dir}/{seq}-msckf-{calib_format}-{calib_seq}.bag"
  #     run_msckf(calib_outfile, rosbag_path, record_path, dryrun)

  # Evaluate - ORBSLAM3
  for calib_file in sorted(glob.glob(kalibr_paths)):
    # Convert Kalibr camchain file to ORBSLAM format
    calib_seq = os.path.dirname(calib_file).split("/")[-1]
    calib_outfile = f"/data/yac_experiments/kalibr_calib.yaml"
    convert_to_orbslam_calib("VIO", calib_format, calib_file, calib_outfile)

    # Evaluate Kalibr calibration by running MSCKF on all evaluation rosbags
    for rosbag_path in sorted(glob.glob(f"{rosbags_dir}/*.bag")):
      seq = os.path.basename(rosbag_path).replace(".bag", "")
      record_path = f"{estimates_dir}/{seq}-orbslam3-{calib_format}-{calib_seq}.csv"
      run_orbslam3_stereo_imu(calib_outfile, rosbag_path, record_path, dryrun)


def eval_yac_calibs(rosbags_dir, yac_dir, estimates_dir, dryrun=False):
  """ Evaluate Kalibr Calibrations """
  calib_format = "yac"
  yac_paths = f"{yac_dir}/*/calib_imu/calib-results.yaml"
  # yac_paths = f"{yac_dir}/*/calib-imu-rerun.yaml"

  # Create estimates directory
  if not os.path.exists(estimates_dir):
    os.makedirs(estimates_dir)

  # # Evaluate - MSCKF
  # for calib_file in sorted(glob.glob(yac_paths)):
  #   # Convert Kalibr camchain file to MSCKF format
  #   calib_seq = os.path.dirname(calib_file).split("/")[-2]
  #   calib_outfile = f"yac_calib.yaml"
  #   convert_to_msckf_calib(calib_format, calib_file, calib_outfile)

  #   # Evaluate Kalibr calibration by running MSCKF on all evaluation rosbags
  #   for rosbag_path in sorted(glob.glob(f"{rosbags_dir}/*.bag")):
  #     rosbag = os.path.basename(rosbag_path).replace(".bag", "")
  #     record_path = f"{estimates_dir}/{rosbag}-msckf-{calib_format}-{calib_seq}.bag"
  #     run_msckf(calib_outfile, rosbag_path, record_path, dryrun)

  # Evaluate - ORBSLAM3
  for calib_file in sorted(glob.glob(yac_paths)):
    # Convert YAC calib file to ORBSLAM format
    calib_seq = os.path.dirname(calib_file).split("/")[-2]
    calib_outfile = f"/data/yac_experiments/yac_calib.yaml"
    convert_to_orbslam_calib("VIO", calib_format, calib_file, calib_outfile)

    # Evaluate Kalibr calibration by running MSCKF on all evaluation rosbags
    for rosbag_path in sorted(glob.glob(f"{rosbags_dir}/*.bag")):
      seq = os.path.basename(rosbag_path).replace(".bag", "")
      record_path = f"{estimates_dir}/{seq}-orbslam3-{calib_format}-{calib_seq}.csv"
      run_orbslam3_stereo_imu(calib_outfile, rosbag_path, record_path, dryrun)


def plot_odom(data_files, seq, algo, **kwargs):
  """ Plot Odometry """
  save_fig = kwargs.get("save_fig", True)
  show_plots = kwargs.get("show_plots", False)

  kalibr_data = {}
  for _, calib_seq in enumerate(data_files[seq][algo]["kalibr"]):
    data = load_pose_data(data_files[seq][algo]["kalibr"][calib_seq])
    kalibr_data[calib_seq] = {
        "timestamps": data[0],
        "times": data[1],
        "pos": data[2]
    }

  yac_data = {}
  for _, calib_seq in enumerate(data_files[seq][algo]["yac"]):
    data = load_pose_data(data_files[seq][algo]["yac"][calib_seq])
    yac_data[calib_seq] = {
        "timestamps": data[0],
        "times": data[1],
        "pos": data[2]
    }

  # Plot odometry
  dpi = 96
  plot_width = 1000
  plot_height = 600
  plt.figure(figsize=(plot_width / dpi, plot_height / dpi), dpi=dpi)

  plt.subplot(121)
  for _, calib_seq in enumerate(kalibr_data):
    plt.plot(kalibr_data[calib_seq]["pos"][0, :],
             kalibr_data[calib_seq]["pos"][2, :])
  plt.xlabel("x [m]")
  plt.ylabel("y [m]")
  plt.title("Kalibr")
  plt.axis("equal")

  plt.subplot(122)
  for _, calib_seq in enumerate(yac_data):
    plt.plot(yac_data[calib_seq]["pos"][0, :], yac_data[calib_seq]["pos"][2, :])
  plt.xlabel("x [m]")
  plt.ylabel("y [m]")
  plt.title("YAC")
  plt.axis("equal")

  plt.subplots_adjust(left=0.10,
                      right=0.85,
                      top=0.95,
                      bottom=0.07,
                      wspace=0.4,
                      hspace=0.8)

  if save_fig:
    fig_path = f"./plots/odom-{seq}.png"
    plt.savefig(fig_path)
    trim_image(fig_path)

  if show_plots:
    plt.show()


def plot_odoms(estimates_dir, algo):
  """ Plot Odometries """
  # Aggregate data
  data_files = {}
  for odom_path in glob.glob(f"{estimates_dir}/*-calib*.csv"):
    fname = os.path.basename(odom_path)
    seq, algo, calib_type, calib_seq = fname.replace(".csv", "").split("-")

    if seq not in data_files:
      data_files[seq] = {}

    if algo not in data_files[seq]:
      data_files[seq][algo] = {}

    if calib_type not in data_files[seq][algo]:
      data_files[seq][algo][calib_type] = {}

    if calib_seq not in data_files[seq][algo][calib_type]:
      data_files[seq][algo][calib_type][calib_seq] = {}

    data_files[seq][algo][calib_type][calib_seq] = odom_path

  # Plot run
  for seq in data_files:
    plot_odom(data_files, seq, algo)


def load_kalibr_data(data_dir):
  """ Load Kalibr data """
  kalibr_data = {
      "cam0": {
          "fx": [],
          "fy": [],
          "cx": [],
          "cy": [],
          "k1": [],
          "k2": [],
          "p1": [],
          "p2": [],
      },
      "cam1": {
          "fx": [],
          "fy": [],
          "cx": [],
          "cy": [],
          "k1": [],
          "k2": [],
          "p1": [],
          "p2": [],
      },
      "T_cam0_cam1": {
          "tf": [],
          "x": [],
          "y": [],
          "z": [],
          "qw": [],
          "qx": [],
          "qy": [],
          "qz": [],
          "roll": [],
          "pitch": [],
          "yaw": [],
      },
      "T_imu0_cam0": {
          "tf": [],
          "x": [],
          "y": [],
          "z": [],
          "qw": [],
          "qx": [],
          "qy": [],
          "qz": [],
          "roll": [],
          "pitch": [],
          "yaw": [],
      }
  }

  kalibr_dirs = []
  for path in sorted(os.listdir(data_dir)):
    path = os.path.join(data_dir, path)
    if os.path.isdir(path):
      kalibr_dirs.append(path)

  for kalibr_dir in kalibr_dirs:
    calib_results = os.path.join(kalibr_dir, "calib_imu-camchain.yaml")
    calib = yaml.safe_load(open(calib_results, 'r').read())

    # Extract camera parameters
    for cam_str in ["cam0", "cam1"]:
      fx, fy, cx, cy = calib[cam_str]["intrinsics"]
      k1, k2, p1, p2 = calib[cam_str]["distortion_coeffs"]

      kalibr_data[cam_str]["fx"].append(fx)
      kalibr_data[cam_str]["fy"].append(fy)
      kalibr_data[cam_str]["cx"].append(cx)
      kalibr_data[cam_str]["cy"].append(cy)

      kalibr_data[cam_str]["k1"].append(k1)
      kalibr_data[cam_str]["k2"].append(k2)
      kalibr_data[cam_str]["p1"].append(p1)
      kalibr_data[cam_str]["p2"].append(p2)

    # Extract camera-camera extrinsics
    T_C1C0 = np.reshape(np.array(calib['cam1']['T_cn_cnm1']), (4, 4))
    T_C0C1 = np.linalg.inv(T_C1C0)
    x, y, z = tf_trans(T_C0C1)
    qw, qx, qy, qz = tf_quat(T_C0C1)
    yaw, pitch, roll = quat2euler(np.array([qw, qx, qy, qz]))
    kalibr_data["T_cam0_cam1"]["tf"].append(T_C0C1)
    kalibr_data["T_cam0_cam1"]["x"].append(x)
    kalibr_data["T_cam0_cam1"]["y"].append(y)
    kalibr_data["T_cam0_cam1"]["z"].append(z)
    kalibr_data["T_cam0_cam1"]["qw"].append(qw)
    kalibr_data["T_cam0_cam1"]["qx"].append(qx)
    kalibr_data["T_cam0_cam1"]["qy"].append(qy)
    kalibr_data["T_cam0_cam1"]["qz"].append(qz)
    kalibr_data["T_cam0_cam1"]["yaw"].append(yaw)
    kalibr_data["T_cam0_cam1"]["pitch"].append(pitch)
    kalibr_data["T_cam0_cam1"]["roll"].append(roll)

    # Extract camera-imu extrinsics
    T_C0S = np.reshape(np.array(calib['cam0']['T_cam_imu']), (4, 4))
    T_SC0 = np.linalg.inv(T_C0S)
    x, y, z = tf_trans(T_SC0)
    qw, qx, qy, qz = tf_quat(T_SC0)
    yaw, pitch, roll = quat2euler(np.array([qw, qx, qy, qz]))
    kalibr_data["T_imu0_cam0"]["tf"].append(T_SC0)
    kalibr_data["T_imu0_cam0"]["x"].append(x)
    kalibr_data["T_imu0_cam0"]["y"].append(y)
    kalibr_data["T_imu0_cam0"]["z"].append(z)
    kalibr_data["T_imu0_cam0"]["qw"].append(qw)
    kalibr_data["T_imu0_cam0"]["qx"].append(qx)
    kalibr_data["T_imu0_cam0"]["qy"].append(qy)
    kalibr_data["T_imu0_cam0"]["qz"].append(qz)
    kalibr_data["T_imu0_cam0"]["yaw"].append(yaw)
    kalibr_data["T_imu0_cam0"]["pitch"].append(pitch)
    kalibr_data["T_imu0_cam0"]["roll"].append(roll)

  return kalibr_data


def load_yac_data(data_dir):
  """ Load YAC data """
  yac_data = {
      "cam0": {
          "fx": [],
          "fy": [],
          "cx": [],
          "cy": [],
          "k1": [],
          "k2": [],
          "p1": [],
          "p2": [],
      },
      "cam1": {
          "fx": [],
          "fy": [],
          "cx": [],
          "cy": [],
          "k1": [],
          "k2": [],
          "p1": [],
          "p2": [],
      },
      "T_cam0_cam1": {
          "tf": [],
          "x": [],
          "y": [],
          "z": [],
          "qw": [],
          "qx": [],
          "qy": [],
          "qz": [],
          "roll": [],
          "pitch": [],
          "yaw": [],
      },
      "T_imu0_cam0": {
          "tf": [],
          "x": [],
          "y": [],
          "z": [],
          "qw": [],
          "qx": [],
          "qy": [],
          "qz": [],
          "roll": [],
          "pitch": [],
          "yaw": [],
      }
  }
  for yac_dir in sorted(os.listdir(data_dir)):
    calib_results = os.path.join(data_dir, yac_dir, "calib_imu",
                                 "calib-results.yaml")
    calib = yaml.safe_load(open(calib_results, 'r').read())

    # Extract camera parameters
    for cam_str in ["cam0", "cam1"]:
      fx, fy, cx, cy = calib[cam_str]["proj_params"]
      k1, k2, p1, p2 = calib[cam_str]["dist_params"]

      yac_data[cam_str]["fx"].append(fx)
      yac_data[cam_str]["fy"].append(fy)
      yac_data[cam_str]["cx"].append(cx)
      yac_data[cam_str]["cy"].append(cy)

      yac_data[cam_str]["k1"].append(k1)
      yac_data[cam_str]["k2"].append(k2)
      yac_data[cam_str]["p1"].append(p1)
      yac_data[cam_str]["p2"].append(p2)

    # Extract camera-camera extrinsics
    T_C0C1 = np.reshape(np.array(calib['T_cam0_cam1']["data"]), (4, 4))
    x, y, z = tf_trans(T_C0C1)
    qw, qx, qy, qz = tf_quat(T_C0C1)
    yaw, pitch, roll = quat2euler(np.array([qw, qx, qy, qz]))
    yac_data["T_cam0_cam1"]["tf"].append(T_C0C1)
    yac_data["T_cam0_cam1"]["x"].append(x)
    yac_data["T_cam0_cam1"]["y"].append(y)
    yac_data["T_cam0_cam1"]["z"].append(z)
    yac_data["T_cam0_cam1"]["qw"].append(qw)
    yac_data["T_cam0_cam1"]["qx"].append(qx)
    yac_data["T_cam0_cam1"]["qy"].append(qy)
    yac_data["T_cam0_cam1"]["qz"].append(qz)
    yac_data["T_cam0_cam1"]["yaw"].append(yaw)
    yac_data["T_cam0_cam1"]["pitch"].append(pitch)
    yac_data["T_cam0_cam1"]["roll"].append(roll)

    # Extract camera-imu extrinsics
    T_SC0 = np.reshape(np.array(calib['T_imu0_cam0']["data"]), (4, 4))
    x, y, z = tf_trans(T_SC0)
    qw, qx, qy, qz = tf_quat(T_SC0)
    yaw, pitch, roll = quat2euler(np.array([qw, qx, qy, qz]))
    yac_data["T_imu0_cam0"]["tf"].append(T_SC0)
    yac_data["T_imu0_cam0"]["x"].append(x)
    yac_data["T_imu0_cam0"]["y"].append(y)
    yac_data["T_imu0_cam0"]["z"].append(z)
    yac_data["T_imu0_cam0"]["qw"].append(qw)
    yac_data["T_imu0_cam0"]["qx"].append(qx)
    yac_data["T_imu0_cam0"]["qy"].append(qy)
    yac_data["T_imu0_cam0"]["qz"].append(qz)
    yac_data["T_imu0_cam0"]["yaw"].append(yaw)
    yac_data["T_imu0_cam0"]["pitch"].append(pitch)
    yac_data["T_imu0_cam0"]["roll"].append(roll)

  return yac_data


def read_file_list(filename, remove_bounds):
  """
  Reads a trajectory from a text file.
  File format:
  The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp
  (to be matched) and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D
  orientation) associated to this timestamp.
  Input:
  filename -- File name
  Output:
  dict -- dictionary of (stamp,data) tuples
  """
  file = open(filename)
  data = file.read()
  lines = data.replace(",", " ").replace("\t", " ").split("\n")
  if remove_bounds:
    lines = lines[100:-100]
  data = [[v.strip()
           for v in line.split(" ")
           if v.strip() != ""]
          for line in lines
          if len(line) > 0 and line[0] != "#"]
  data = [(float(l[0]), l[1:]) for l in data if len(l) > 1]
  return dict(data)


def associate(est_data, gnd_data, max_difference=(0.02 * 1e9)):
  """
    Associate two dictionaries of (stamp,data). As the time stamps never match
    exactly, we aim to find the closest match for every input tuple.

    Input:
      est_data -- first dictionary of (stamp,data) tuples
      gnd_data -- second dictionary of (stamp,data) tuples
      max_difference -- search radius for candidate generation

    Output:
      matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
  """
  est_keys = list(est_data)
  gnd_keys = list(gnd_data)

  matches = []
  anchor = 0

  for ts_a in est_keys:
    closest_ts = None
    closest_diff = None

    for idx, ts_b in enumerate(gnd_keys[anchor:]):
      ts_diff = abs(ts_a - ts_b)

      if closest_ts is None:
        closest_ts = ts_b
        closest_diff = ts_diff
        anchor = idx
      elif ts_diff < closest_diff:
        closest_ts = ts_b
        closest_diff = ts_diff
        anchor = idx

    if closest_diff < max_difference:
      matches.append((ts_a, closest_ts))

  return matches


def align(model, data):
  """Align two trajectories using the method of Horn (closed-form).
    Input:
        model -- first trajectory (3xn)
        data -- second trajectory (3xn)
    Output:
        rot -- rotation matrix (3x3)
        trans -- translation vector (3x1)
        trans_error -- translational error per point (1xn)
    """
  np.set_printoptions(precision=3, suppress=True)
  model_zerocentered = model - model.mean(1)
  data_zerocentered = data - data.mean(1)

  W = np.zeros((3, 3))
  for column in range(model.shape[1]):
    W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
  U, _, Vh = np.linalg.linalg.svd(W.transpose())

  S = np.matrix(np.identity(3))
  if np.linalg.det(U) * np.linalg.det(Vh) < 0:
    S[2, 2] = -1
  rot = U * S * Vh

  rotmodel = rot * model_zerocentered
  dots = 0.0
  norms = 0.0

  for column in range(data_zerocentered.shape[1]):
    dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:,
                                                                      column])
    normi = np.linalg.norm(model_zerocentered[:, column])
    norms += normi * normi

  s = float(dots / norms)

  transGT = data.mean(1) - s * rot * model.mean(1)
  trans = data.mean(1) - rot * model.mean(1)

  model_alignedGT = s * rot * model + transGT
  model_aligned = rot * model + trans

  alignment_errorGT = model_alignedGT - data
  alignment_error = model_aligned - data

  trans_errorGT = np.sqrt(
      np.sum(np.multiply(alignment_errorGT, alignment_errorGT), 0)).A[0]
  trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error),
                               0)).A[0]

  return rot, transGT, trans_errorGT, trans, trans_error, s


def eval_traj(gnd_file, est_file, **kwargs):
  """ Evaluate Trajectory """
  verbose = kwargs.get("verbose", False)

  # Read files and associate them
  gnd_data = read_file_list(gnd_file, False)
  est_data = read_file_list(est_file, False)
  matches = associate(est_data, gnd_data)
  if len(matches) < 2:
    sys.exit("Matches < 2! Did you choose the correct sequence?")

  # Extract matches between ground-truth and estimates and form matrices
  gnd_mat = []
  est_mat = []
  for ts_a, ts_b in matches:
    est_data_row = [float(value) for value in est_data[ts_a][0:3]]
    gnd_data_row = [float(value) for value in gnd_data[ts_b][0:3]]
    gnd_mat.append(gnd_data_row)
    est_mat.append(est_data_row)
  gnd_mat = np.matrix(gnd_mat).transpose()
  est_mat = np.matrix(est_mat).transpose()

  # Align both estimates and ground-truth
  rot, transGT, trans_errorGT, trans, trans_error, scale = align(
      est_mat, gnd_mat)

  # Calculate errors
  metrics = {
      "rmse": np.sqrt(np.dot(trans_error, trans_error) / len(trans_error)),
      "mean": np.mean(trans_error),
      "median": np.median(trans_error),
      "std": np.std(trans_error),
      "min": np.min(trans_error),
      "max": np.max(trans_error)
  }

  if verbose:
    print("compared_pose_pairs %d pairs" % (len(trans_error)))
    print("ATE.rmse %f m" % metrics["rmse"])
    print("ATE.mean %f m" % metrics["mean"])
    print("ATE.median %f m" % metrics["median"])
    print("ATE.std %f m" % metrics["std"])
    print("ATE.min %f m" % metrics["min"])
    print("ATE.max %f m" % metrics["max"])

  return metrics


def eval_trajs(estimates_dir):
  """ Evaluate Trajectories """
  # Load results json if it exists
  results_path = f"{estimates_dir}/results.json"
  if os.path.exists(results_path):
    results_file = open(results_path, "r")
    results = results_file.read()
    return json.loads(results)

  # Extract ground truths
  for rosbag_path in sorted(glob.glob(f"{rosbags_dir}/*.bag")):
    seq = os.path.basename(rosbag_path).split(".")[0]
    save_path = f"{estimates_dir}/{seq}-ground_truth.csv"
    if not os.path.exists(save_path):
      extract_ground_truth(rosbag_path, save_path)

  # Evaluate trajectories
  metrics = {}
  for odom_file in sorted(glob.glob(f"{estimates_dir}/*-calib*.csv")):
    print(f"Evaluating [{odom_file}]", flush=True)
    fname = os.path.basename(odom_file)
    seq, algo, calib_type, calib_seq = fname.replace(".csv", "").split("-")

    if seq not in metrics:
      metrics[seq] = {}

    if algo not in metrics[seq]:
      metrics[seq][algo] = {}

    if calib_type not in metrics[seq][algo]:
      metrics[seq][algo][calib_type] = {
          "rmse": [],
          "mean": [],
          "median": [],
          "std": [],
          "min": [],
          "max": [],
      }

    gnd_file = f"{estimates_dir}/{seq}-ground_truth.csv"
    est_file = f"{estimates_dir}/{seq}-{algo}-{calib_type}-{calib_seq}.csv"
    ate = eval_traj(gnd_file, est_file)

    metrics[seq][algo][calib_type]["rmse"].append(ate["rmse"])
    metrics[seq][algo][calib_type]["mean"].append(ate["mean"])
    metrics[seq][algo][calib_type]["median"].append(ate["median"])
    metrics[seq][algo][calib_type]["std"].append(ate["std"])
    metrics[seq][algo][calib_type]["min"].append(ate["min"])
    metrics[seq][algo][calib_type]["max"].append(ate["max"])

  # Output results to file
  results_file = open(results_path, "w")
  results_file.write(json.dumps(metrics, indent=2))
  results_file.close()

  return metrics


def plot_estimates(estimates_dir, **kwargs):
  """ Plot Estimates """
  save_fig = kwargs.get("save_fig", True)
  show_plots = kwargs.get("show_plots", False)

  results = eval_trajs(estimates_dir)
  df = {"seq": [], "algo": [], "calib_type": [], "rmse": []}
  for seq in results.keys():
    for algo in ["orbslam3"]:
      for calib_type in ["kalibr", "yac"]:
        for rmse in results[seq][algo][calib_type]["rmse"]:
          df["seq"].append(seq)
          df["algo"].append(algo)
          if calib_type == "kalibr":
            df["calib_type"].append("Kalibr")
          else:
            df["calib_type"].append("Our Method")
          df["rmse"].append(rmse)

  seaborn.set_style("darkgrid")
  dpi = 96
  plot_width = 700
  plot_height = 400
  plot_rows = 1
  plot_cols = 1
  fig_size = (plot_width / dpi, plot_height / dpi)
  _, axs = plt.subplots(plot_rows, plot_cols, figsize=fig_size, dpi=dpi)
  seaborn.boxplot(ax=axs,
                  hue="calib_type",
                  x="seq",
                  y="rmse",
                  data=df,
                  showfliers=False)
  axs.set_xlabel("Run Sequence")
  axs.set_ylabel("RMSE ATE [m]")

  if save_fig:
    fig_path = f"./plots/odom-boxplot.png"
    plt.savefig(fig_path)
    trim_image(fig_path)

  if show_plots:
    plt.show()


def plot_camera_params(kalibr_data, yac_data, **kwargs):
  """ Plot boxplot camera params """
  save_fig = kwargs.get("save_fig", True)
  show_plot = kwargs.get("show_plot", False)
  plot_width = kwargs.get("plot_width", 800)
  plot_height = kwargs.get("plot_height", 700)

  df_data = {
      "fx": [],
      "fy": [],
      "cx": [],
      "cy": [],
      "k1": [],
      "k2": [],
      "p1": [],
      "p2": [],
      "cam_idx": [],
      "source": [],
  }

  # Kalibr data
  for key in kalibr_data["cam0"]:
    value = kalibr_data["cam0"][key]
    df_data[key].extend(value)
  df_data["cam_idx"].extend(0.0 * np.ones(len(value)))

  for key in kalibr_data["cam1"]:
    value = kalibr_data["cam1"][key]
    df_data[key].extend(value)
  df_data["cam_idx"].extend(1.0 * np.ones(len(value)))

  for _ in range(len(kalibr_data["cam0"]["fx"]) * 2):
    df_data["source"].append("kalibr")

  # YAC data
  for key in yac_data["cam0"]:
    value = yac_data["cam0"][key]
    df_data[key].extend(value)
  df_data["cam_idx"].extend(0.0 * np.ones(len(value)))

  for key in yac_data["cam1"]:
    value = yac_data["cam1"][key]
    df_data[key].extend(value)
  df_data["cam_idx"].extend(1.0 * np.ones(len(value)))

  for _ in range(len(yac_data["cam0"]["fx"]) * 2):
    df_data["source"].append("yac")

  # Form data frame
  df = pandas.DataFrame(df_data)

  # Plot boxplot
  dpi = 96
  seaborn.set_style("darkgrid")
  plot_rows = 4
  plot_cols = 2
  fig_size = (plot_width / dpi, plot_height / dpi)
  _, axs = plt.subplots(plot_rows, plot_cols, figsize=fig_size, dpi=dpi)

  y_keys = np.array([
      ["fx", "fy"],
      ["cx", "cy"],
      ["k1", "k2"],
      ["p1", "p2"],
  ])
  titles = np.array([
      [
          "Focal Length - fx",
          "Focal Length - fy",
      ],
      [
          "Principal Center - cx",
          "Principal Center - cy",
      ],
      [
          "Radial Distortion - k1",
          "Radial Distortion - k2",
      ],
      [
          "Tangential Distortion - p1",
          "Tangential Distortion - p2",
      ],
  ])

  for i in range(plot_rows):
    for j in range(plot_cols):
      seaborn.boxplot(ax=axs[i, j],
                      x="cam_idx",
                      y=y_keys[i, j],
                      hue="source",
                      data=df,
                      showfliers=False)
      axs[i, j].set_xlabel("")
      if i < 2:
        axs[i, j].set_ylabel("Pixels [px]")
      else:
        axs[i, j].set_ylabel("")
      axs[i, j].set_xticklabels(["Camera 0", "Camera 1"])
      axs[i, j].set_title(titles[i, j])

      if (i == 0 and j == 1):
        axs[i, j].legend(bbox_to_anchor=(1.1, 1.0), loc=2, borderaxespad=0.)
      else:
        axs[i, j].legend([], [], frameon=False)

  # Adjust subplots
  plt.subplots_adjust(left=0.10,
                      right=0.85,
                      top=0.95,
                      bottom=0.07,
                      wspace=0.4,
                      hspace=0.8)

  # -- Save
  if save_fig:
    fig_path = "./plots/camera_params-boxplot.png"
    plt.savefig(fig_path)
    trim_image(fig_path)

  # -- Show
  if show_plot:
    plt.show()


def plot_extrinsics(kalibr_data, yac_data, ext_key, **kwargs):
  """ Plot boxplot camera params """
  save_fig = kwargs.get("save_fig", True)
  show_plot = kwargs.get("show_plot", False)
  plot_width = kwargs.get("plot_width", 1000)
  plot_height = kwargs.get("plot_height", 600)

  df_data = {
      "x": [],
      "y": [],
      "z": [],
      "roll": [],
      "pitch": [],
      "yaw": [],
      "source": [],
  }

  df_data["x"].extend(kalibr_data[ext_key]["x"])
  df_data["y"].extend(kalibr_data[ext_key]["y"])
  df_data["z"].extend(kalibr_data[ext_key]["z"])
  df_data["roll"].extend(rad2deg(kalibr_data[ext_key]["roll"]))
  df_data["pitch"].extend(rad2deg(kalibr_data[ext_key]["pitch"]))
  df_data["yaw"].extend(rad2deg(kalibr_data[ext_key]["yaw"]))
  for _ in range(len(kalibr_data[ext_key]["x"])):
    df_data["source"].append("kalibr")

  df_data["x"].extend(yac_data[ext_key]["x"])
  df_data["y"].extend(yac_data[ext_key]["y"])
  df_data["z"].extend(yac_data[ext_key]["z"])
  df_data["roll"].extend(rad2deg(yac_data[ext_key]["roll"]))
  df_data["pitch"].extend(rad2deg(yac_data[ext_key]["pitch"]))
  df_data["yaw"].extend(rad2deg(yac_data[ext_key]["yaw"]))
  for _ in range(len(yac_data[ext_key]["x"])):
    df_data["source"].append("yac")

  # Form data frame
  df = pandas.DataFrame(df_data)

  # Plot boxplot
  dpi = 96
  seaborn.set_style("darkgrid")
  plot_rows = 2
  plot_cols = 3
  fig_size = (plot_width / dpi, plot_height / dpi)
  _, axs = plt.subplots(plot_rows, plot_cols, figsize=fig_size, dpi=dpi)

  y_keys = np.array([
      ["x", "y", "z"],
      ["roll", "pitch", "yaw"],
  ])

  titles = np.array([
      ["Position - x", "Position - y", "Position - z"],
      ["Roll", "Pitch", "Yaw"],
  ])

  for i in range(plot_rows):
    for j in range(plot_cols):
      seaborn.boxplot(ax=axs[i, j],
                      x="source",
                      y=y_keys[i, j],
                      hue="source",
                      data=df,
                      showfliers=False)

      axs[i, j].set_title(titles[i, j])

      if i == 0:
        axs[i, j].set_ylabel("Displacement [m]")
      else:
        axs[i, j].set_ylabel("Attitude [deg]")

      if (i == 0 and j == 2):
        axs[i, j].legend(bbox_to_anchor=(1.1, 1.0), loc=2, borderaxespad=0.)
      else:
        axs[i, j].legend([], [], frameon=False)

  # Adjust subplots
  plt.subplots_adjust(left=0.1,
                      bottom=0.07,
                      right=0.88,
                      top=0.95,
                      wspace=0.7,
                      hspace=0.5)

  # -- Save
  if save_fig:
    fig_path = kwargs["fig_path"]
    plt.savefig(fig_path)
    trim_image(fig_path)

  # -- Show
  if show_plot:
    plt.show()


def print_calib_stats(kalibr_data, yac_data):
  """ Print Calibration Stats """
  k = {
      "cam0": {
          "fx_mean": np.mean(kalibr_data['cam0']['fx']),
          "fx_std": np.std(kalibr_data['cam0']['fx']),
          "fy_mean": np.mean(kalibr_data['cam0']['fy']),
          "fy_std": np.std(kalibr_data['cam0']['fy']),
          "cx_mean": np.mean(kalibr_data['cam0']['cx']),
          "cx_std": np.std(kalibr_data['cam0']['cx']),
          "cy_mean": np.mean(kalibr_data['cam0']['cy']),
          "cy_std": np.std(kalibr_data['cam0']['cy']),
          "k1_mean": np.mean(kalibr_data['cam0']['k1']),
          "k1_std": np.std(kalibr_data['cam0']['k1']),
          "k2_mean": np.mean(kalibr_data['cam0']['k2']),
          "k2_std": np.std(kalibr_data['cam0']['k2']),
          "p1_mean": np.mean(kalibr_data['cam0']['p1']),
          "p1_std": np.std(kalibr_data['cam0']['p1']),
          "p2_mean": np.mean(kalibr_data['cam0']['p2']),
          "p2_std": np.std(kalibr_data['cam0']['p2']),
      },
      "cam1": {
          "fx_mean": np.mean(kalibr_data['cam1']['fx']),
          "fx_std": np.std(kalibr_data['cam1']['fx']),
          "fy_mean": np.mean(kalibr_data['cam1']['fy']),
          "fy_std": np.std(kalibr_data['cam1']['fy']),
          "cx_mean": np.mean(kalibr_data['cam1']['cx']),
          "cx_std": np.std(kalibr_data['cam1']['cx']),
          "cy_mean": np.mean(kalibr_data['cam1']['cy']),
          "cy_std": np.std(kalibr_data['cam1']['cy']),
          "k1_mean": np.mean(kalibr_data['cam1']['k1']),
          "k1_std": np.std(kalibr_data['cam1']['k1']),
          "k2_mean": np.mean(kalibr_data['cam1']['k2']),
          "k2_std": np.std(kalibr_data['cam1']['k2']),
          "p1_mean": np.mean(kalibr_data['cam1']['p1']),
          "p1_std": np.std(kalibr_data['cam1']['p1']),
          "p2_mean": np.mean(kalibr_data['cam1']['p2']),
          "p2_std": np.std(kalibr_data['cam1']['p2']),
      },
      "cam_exts": {
          "x_mean": np.mean(kalibr_data['T_cam0_cam1']['x']),
          "x_std": np.std(kalibr_data['T_cam0_cam1']['x']),
          "y_mean": np.mean(kalibr_data['T_cam0_cam1']['y']),
          "y_std": np.std(kalibr_data['T_cam0_cam1']['y']),
          "z_mean": np.mean(kalibr_data['T_cam0_cam1']['z']),
          "z_std": np.std(kalibr_data['T_cam0_cam1']['z']),
          "roll_mean": rad2deg(np.mean(kalibr_data['T_cam0_cam1']['roll'])),
          "roll_std": rad2deg(np.std(kalibr_data['T_cam0_cam1']['roll'])),
          "pitch_mean": rad2deg(np.mean(kalibr_data['T_cam0_cam1']['pitch'])),
          "pitch_std": rad2deg(np.std(kalibr_data['T_cam0_cam1']['pitch'])),
          "yaw_mean": rad2deg(np.mean(kalibr_data['T_cam0_cam1']['yaw'])),
          "yaw_std": rad2deg(np.std(kalibr_data['T_cam0_cam1']['yaw'])),
      },
      "imu_exts": {
          "x_mean": np.mean(kalibr_data['T_imu0_cam0']['x']),
          "x_std": np.std(kalibr_data['T_imu0_cam0']['x']),
          "y_mean": np.mean(kalibr_data['T_imu0_cam0']['y']),
          "y_std": np.std(kalibr_data['T_imu0_cam0']['y']),
          "z_mean": np.mean(kalibr_data['T_imu0_cam0']['z']),
          "z_std": np.std(kalibr_data['T_imu0_cam0']['z']),
          "roll_mean": rad2deg(np.mean(kalibr_data['T_imu0_cam0']['roll'])),
          "roll_std": rad2deg(np.std(kalibr_data['T_imu0_cam0']['roll'])),
          "pitch_mean": rad2deg(np.mean(kalibr_data['T_imu0_cam0']['pitch'])),
          "pitch_std": rad2deg(np.std(kalibr_data['T_imu0_cam0']['pitch'])),
          "yaw_mean": rad2deg(np.mean(kalibr_data['T_imu0_cam0']['yaw'])),
          "yaw_std": rad2deg(np.std(kalibr_data['T_imu0_cam0']['yaw'])),
      }
  }

  y = {
      "cam0": {
          "fx_mean": np.mean(yac_data['cam0']['fx']),
          "fx_std": np.std(yac_data['cam0']['fx']),
          "fy_mean": np.mean(yac_data['cam0']['fy']),
          "fy_std": np.std(yac_data['cam0']['fy']),
          "cx_mean": np.mean(yac_data['cam0']['cx']),
          "cx_std": np.std(yac_data['cam0']['cx']),
          "cy_mean": np.mean(yac_data['cam0']['cy']),
          "cy_std": np.std(yac_data['cam0']['cy']),
          "k1_mean": np.mean(yac_data['cam0']['k1']),
          "k1_std": np.std(yac_data['cam0']['k1']),
          "k2_mean": np.mean(yac_data['cam0']['k2']),
          "k2_std": np.std(yac_data['cam0']['k2']),
          "p1_mean": np.mean(yac_data['cam0']['p1']),
          "p1_std": np.std(yac_data['cam0']['p1']),
          "p2_mean": np.mean(yac_data['cam0']['p2']),
          "p2_std": np.std(yac_data['cam0']['p2']),
      },
      "cam1": {
          "fx_mean": np.mean(yac_data['cam1']['fx']),
          "fx_std": np.std(yac_data['cam1']['fx']),
          "fy_mean": np.mean(yac_data['cam1']['fy']),
          "fy_std": np.std(yac_data['cam1']['fy']),
          "cx_mean": np.mean(yac_data['cam1']['cx']),
          "cx_std": np.std(yac_data['cam1']['cx']),
          "cy_mean": np.mean(yac_data['cam1']['cy']),
          "cy_std": np.std(yac_data['cam1']['cy']),
          "k1_mean": np.mean(yac_data['cam1']['k1']),
          "k1_std": np.std(yac_data['cam1']['k1']),
          "k2_mean": np.mean(yac_data['cam1']['k2']),
          "k2_std": np.std(yac_data['cam1']['k2']),
          "p1_mean": np.mean(yac_data['cam1']['p1']),
          "p1_std": np.std(yac_data['cam1']['p1']),
          "p2_mean": np.mean(yac_data['cam1']['p2']),
          "p2_std": np.std(yac_data['cam1']['p2']),
      },
      "cam_exts": {
          "x_mean": np.mean(yac_data['T_cam0_cam1']['x']),
          "x_std": np.std(yac_data['T_cam0_cam1']['x']),
          "y_mean": np.mean(yac_data['T_cam0_cam1']['y']),
          "y_std": np.std(yac_data['T_cam0_cam1']['y']),
          "z_mean": np.mean(yac_data['T_cam0_cam1']['z']),
          "z_std": np.std(yac_data['T_cam0_cam1']['z']),
          "roll_mean": rad2deg(np.mean(yac_data['T_cam0_cam1']['roll'])),
          "roll_std": rad2deg(np.std(yac_data['T_cam0_cam1']['roll'])),
          "pitch_mean": rad2deg(np.mean(yac_data['T_cam0_cam1']['pitch'])),
          "pitch_std": rad2deg(np.std(yac_data['T_cam0_cam1']['pitch'])),
          "yaw_mean": rad2deg(np.mean(yac_data['T_cam0_cam1']['yaw'])),
          "yaw_std": rad2deg(np.std(yac_data['T_cam0_cam1']['yaw'])),
      },
      "imu_exts": {
          "x_mean": np.mean(yac_data['T_imu0_cam0']['x']),
          "x_std": np.std(yac_data['T_imu0_cam0']['x']),
          "y_mean": np.mean(yac_data['T_imu0_cam0']['y']),
          "y_std": np.std(yac_data['T_imu0_cam0']['y']),
          "z_mean": np.mean(yac_data['T_imu0_cam0']['z']),
          "z_std": np.std(yac_data['T_imu0_cam0']['z']),
          "roll_mean": rad2deg(np.mean(yac_data['T_imu0_cam0']['roll'])),
          "roll_std": rad2deg(np.std(yac_data['T_imu0_cam0']['roll'])),
          "pitch_mean": rad2deg(np.mean(yac_data['T_imu0_cam0']['pitch'])),
          "pitch_std": rad2deg(np.std(yac_data['T_imu0_cam0']['pitch'])),
          "yaw_mean": rad2deg(np.mean(yac_data['T_imu0_cam0']['yaw'])),
          "yaw_std": rad2deg(np.std(yac_data['T_imu0_cam0']['yaw'])),
      }
  }

  # print(f"""\
  # Kalibr:
  # ------------------------------
  # cam0
  # fx: {k['cam0']['fx_mean']:.4f} +- {k['cam0']['fx_std']:.4f}
  # fy: {k['cam0']['fy_mean']:.4f} +- {k['cam0']['fy_std']:.4f}
  # cx: {k['cam0']['cx_mean']:.4f} +- {k['cam0']['cx_std']:.4f}
  # cy: {k['cam0']['cy_mean']:.4f} +- {k['cam0']['cy_std']:.4f}

  # k1: {k['cam0']['k1_mean']:.4f} +- {k['cam0']['k1_std']:.4f}
  # k2: {k['cam0']['k2_mean']:.4f} +- {k['cam0']['k2_std']:.4f}
  # p1: {k['cam0']['p1_mean']:.4f} +- {k['cam0']['p1_std']:.4f}
  # p2: {k['cam0']['p2_mean']:.4f} +- {k['cam0']['p2_std']:.4f}

  # cam1
  # fx: {k['cam1']['fx_mean']:.4f} +- {k['cam1']['fx_std']:.4f}
  # fy: {k['cam1']['fy_mean']:.4f} +- {k['cam1']['fy_std']:.4f}
  # cx: {k['cam1']['cx_mean']:.4f} +- {k['cam1']['cx_std']:.4f}
  # cy: {k['cam1']['cy_mean']:.4f} +- {k['cam1']['cy_std']:.4f}

  # k1: {k['cam1']['k1_mean']:.4f} +- {k['cam1']['k1_std']:.4f}
  # k2: {k['cam1']['k2_mean']:.4f} +- {k['cam1']['k2_std']:.4f}
  # p1: {k['cam1']['p1_mean']:.4f} +- {k['cam1']['p1_std']:.4f}
  # p2: {k['cam1']['p2_mean']:.4f} +- {k['cam1']['p2_std']:.4f}

  # T_cam0_cam1:
  # x:     {k['cam_exts']['x_mean']:.4f} +- {k['cam_exts']['x_std']:.4f}
  # y:     {k['cam_exts']['y_mean']:.4f} +- {k['cam_exts']['y_std']:.4f}
  # z:     {k['cam_exts']['z_mean']:.4f} +- {k['cam_exts']['z_std']:.4f}
  # roll:  {k['cam_exts']['roll_mean']:.4f} +- {k['cam_exts']['roll_std']:.4f}
  # pitch: {k['cam_exts']['pitch_mean']:.4f} +- {k['cam_exts']['pitch_std']:.4f}
  # yaw:   {k['cam_exts']['yaw_mean']:.4f} +- {k['cam_exts']['yaw_std']:.4f}

  # T_imu0_cam0:
  # x:     {k['imu_exts']['x_mean']:.4f} +- {k['imu_exts']['x_std']:.4f}
  # y:     {k['imu_exts']['y_mean']:.4f} +- {k['imu_exts']['y_std']:.4f}
  # z:     {k['imu_exts']['z_mean']:.4f} +- {k['imu_exts']['z_std']:.4f}
  # roll:  {k['imu_exts']['roll_mean']:.4f} +- {k['imu_exts']['roll_std']:.4f}
  # pitch: {k['imu_exts']['pitch_mean']:.4f} +- {k['imu_exts']['pitch_std']:.4f}
  # yaw:   {k['imu_exts']['yaw_mean']:.4f} +- {k['imu_exts']['yaw_std']:.4f}

  # YAC:
  # ------------------------------
  # cam0
  # fx: {y['cam0']['fx_mean']:.4f} +- {y['cam0']['fx_std']:.4f}
  # fy: {y['cam0']['fy_mean']:.4f} +- {y['cam0']['fy_std']:.4f}
  # cx: {y['cam0']['cx_mean']:.4f} +- {y['cam0']['cx_std']:.4f}
  # cy: {y['cam0']['cy_mean']:.4f} +- {y['cam0']['cy_std']:.4f}

  # k1: {y['cam0']['k1_mean']:.4f} +- {y['cam0']['k1_std']:.4f}
  # k2: {y['cam0']['k2_mean']:.4f} +- {y['cam0']['k2_std']:.4f}
  # p1: {y['cam0']['p1_mean']:.4f} +- {y['cam0']['p1_std']:.4f}
  # p2: {y['cam0']['p2_mean']:.4f} +- {y['cam0']['p2_std']:.4f}

  # cam1
  # fx: {y['cam1']['fx_mean']:.4f} +- {y['cam1']['fx_std']:.4f}
  # fy: {y['cam1']['fy_mean']:.4f} +- {y['cam1']['fy_std']:.4f}
  # cx: {y['cam1']['cx_mean']:.4f} +- {y['cam1']['cx_std']:.4f}
  # cy: {y['cam1']['cy_mean']:.4f} +- {y['cam1']['cy_std']:.4f}

  # k1: {y['cam1']['k1_mean']:.4f} +- {y['cam1']['k1_std']:.4f}
  # k2: {y['cam1']['k2_mean']:.4f} +- {y['cam1']['k2_std']:.4f}
  # p1: {y['cam1']['p1_mean']:.4f} +- {y['cam1']['p1_std']:.4f}
  # p2: {y['cam1']['p2_mean']:.4f} +- {y['cam1']['p2_std']:.4f}

  # T_cam0_cam1:
  # x:     {y['cam_exts']['x_mean']:.4f} +- {y['cam_exts']['x_std']:.4f}
  # y:     {y['cam_exts']['y_mean']:.4f} +- {y['cam_exts']['y_std']:.4f}
  # z:     {y['cam_exts']['z_mean']:.4f} +- {y['cam_exts']['z_std']:.4f}
  # roll:  {y['cam_exts']['roll_mean']:.4f} +- {y['cam_exts']['roll_std']:.4f}
  # pitch: {y['cam_exts']['pitch_mean']:.4f} +- {y['cam_exts']['pitch_std']:.4f}
  # yaw:   {y['cam_exts']['yaw_mean']:.4f} +- {y['cam_exts']['yaw_std']:.4f}

  # T_imu0_cam0:
  # x:     {y['imu_exts']['x_mean']:.4f} +- {y['imu_exts']['x_std']:.4f}
  # y:     {y['imu_exts']['y_mean']:.4f} +- {y['imu_exts']['y_std']:.4f}
  # z:     {y['imu_exts']['z_mean']:.4f} +- {y['imu_exts']['z_std']:.4f}
  # roll:  {y['imu_exts']['roll_mean']:.4f} +- {y['imu_exts']['roll_std']:.4f}
  # pitch: {y['imu_exts']['pitch_mean']:.4f} +- {y['imu_exts']['pitch_std']:.4f}
  # yaw:   {y['imu_exts']['yaw_mean']:.4f} +- {y['imu_exts']['yaw_std']:.4f}
  # """)

  print(f"""\
\\begin{{table}}[!h]
\\centering
\\label{{table:calib_compare}}
\\caption{{Calibration Comparison}}
\\begin{{tabular}}{{ccc}}
\\hline
\\textbf{{Parameter}}
& \\textbf{{Kalibr}}
& \\textbf{{YAC}} \\\\
\\hline
  $C_0$ \\\\
  fx [px]
  & ${k['cam0']['fx_mean']:.4f} \pm {k['cam0']['fx_std']:.4f}$
  & ${y['cam0']['fx_mean']:.4f} \pm {y['cam0']['fx_std']:.4f}$ \\\\
  fy [px]
  & ${k['cam0']['fy_mean']:.4f} \pm {k['cam0']['fy_std']:.4f}$
  & ${y['cam0']['fy_mean']:.4f} \pm {y['cam0']['fy_std']:.4f}$ \\\\
  cx [px]
  & ${k['cam0']['cx_mean']:.4f} \pm {k['cam0']['cx_std']:.4f}$
  & ${y['cam0']['cx_mean']:.4f} \pm {y['cam0']['cx_std']:.4f}$ \\\\
  cy [px]
  & ${k['cam0']['cy_mean']:.4f} \pm {k['cam0']['cy_std']:.4f}$
  & ${y['cam0']['cy_mean']:.4f} \pm {y['cam0']['cy_std']:.4f}$ \\\\
\\\\
  $D_0$ \\\\
  k1
  & ${k['cam0']['k1_mean']:.4f} \pm {k['cam0']['k1_std']:.4f}$
  & ${y['cam0']['k1_mean']:.4f} \pm {y['cam0']['k1_std']:.4f}$ \\\\
  k2
  & ${k['cam0']['k2_mean']:.4f} \pm {k['cam0']['k2_std']:.4f}$
  & ${y['cam0']['k2_mean']:.4f} \pm {y['cam0']['k2_std']:.4f}$ \\\\
  p1
  & ${k['cam0']['p1_mean']:.4f} \pm {k['cam0']['p1_std']:.4f}$
  & ${y['cam0']['p1_mean']:.4f} \pm {y['cam0']['p1_std']:.4f}$ \\\\
  p2
  & ${k['cam0']['p2_mean']:.4f} \pm {k['cam0']['p2_std']:.4f}$
  & ${y['cam0']['p2_mean']:.4f} \pm {y['cam0']['p2_std']:.4f}$ \\\\
\\\\
  $C_1$ \\\\
  fx [px]
  & ${k['cam1']['fx_mean']:.4f} \pm {k['cam1']['fx_std']:.4f}$
  & ${y['cam1']['fx_mean']:.4f} \pm {y['cam1']['fx_std']:.4f}$ \\\\
  fy [px]
  & ${k['cam1']['fy_mean']:.4f} \pm {k['cam1']['fy_std']:.4f}$
  & ${y['cam1']['fy_mean']:.4f} \pm {y['cam1']['fy_std']:.4f}$ \\\\
  cx [px]
  & ${k['cam1']['cx_mean']:.4f} \pm {k['cam1']['cx_std']:.4f}$
  & ${y['cam1']['cx_mean']:.4f} \pm {y['cam1']['cx_std']:.4f}$ \\\\
  cy [px]
  & ${k['cam1']['cy_mean']:.4f} \pm {k['cam1']['cy_std']:.4f}$
  & ${y['cam1']['cy_mean']:.4f} \pm {y['cam1']['cy_std']:.4f}$ \\\\
\\\\
  $D_1$ \\\\
  k1
  & ${k['cam1']['k1_mean']:.4f} \pm {k['cam1']['k1_std']:.4f}$
  & ${y['cam1']['k1_mean']:.4f} \pm {y['cam1']['k1_std']:.4f}$ \\\\
  k2
  & ${k['cam1']['k2_mean']:.4f} \pm {k['cam1']['k2_std']:.4f}$
  & ${y['cam1']['k2_mean']:.4f} \pm {y['cam1']['k2_std']:.4f}$ \\\\
  p1
  & ${k['cam1']['p1_mean']:.4f} \pm {k['cam1']['p1_std']:.4f}$
  & ${y['cam1']['p1_mean']:.4f} \pm {y['cam1']['p1_std']:.4f}$ \\\\
  p2
  & ${k['cam1']['p2_mean']:.4f} \pm {k['cam1']['p2_std']:.4f}$
  & ${y['cam1']['p2_mean']:.4f} \pm {y['cam1']['p2_std']:.4f}$ \\\\
\\\\
  $\mathbf{{T}}_{{C_0C_1}}$ \\\\
  x [m]
  & ${k['cam_exts']['x_mean']:.4f} \pm {k['cam_exts']['x_std']:.2e}$
  & ${y['cam_exts']['x_mean']:.4f} \pm {y['cam_exts']['x_std']:.2e}$ \\\\
  y [m]
  & ${k['cam_exts']['y_mean']:.4f} \pm {k['cam_exts']['y_std']:.2e}$
  & ${y['cam_exts']['y_mean']:.4f} \pm {y['cam_exts']['y_std']:.2e}$ \\\\
  z [m]
  & ${k['cam_exts']['z_mean']:.4f} \pm {k['cam_exts']['z_std']:.2e}$
  & ${y['cam_exts']['z_mean']:.4f} \pm {y['cam_exts']['z_std']:.2e}$ \\\\
  roll [deg]
  & ${k['cam_exts']['roll_mean']:.4f} \pm {k['cam_exts']['roll_std']:.4f}$
  & ${y['cam_exts']['roll_mean']:.4f} \pm {y['cam_exts']['roll_std']:.4f}$ \\\\
  pitch [deg]
  & ${k['cam_exts']['pitch_mean']:.4f} \pm {k['cam_exts']['pitch_std']:.4f}$
  & ${y['cam_exts']['pitch_mean']:.4f} \pm {y['cam_exts']['pitch_std']:.4f}$ \\\\
  yaw [deg]
  & ${k['cam_exts']['yaw_mean']:.4f} \pm {k['cam_exts']['yaw_std']:.4f}$
  & ${y['cam_exts']['yaw_mean']:.4f} \pm {y['cam_exts']['yaw_std']:.4f}$ \\\\
\\\\
  $\mathbf{{T}}_{{SC_0}}$ \\\\
  x [m]
  & ${k['imu_exts']['x_mean']:.4f} \pm {k['imu_exts']['x_std']:.2e}$
  & ${y['imu_exts']['x_mean']:.4f} \pm {y['imu_exts']['x_std']:.2e}$ \\\\
  y [m]
  & ${k['imu_exts']['y_mean']:.4f} \pm {k['imu_exts']['y_std']:.2e}$
  & ${y['imu_exts']['y_mean']:.4f} \pm {y['imu_exts']['y_std']:.2e}$ \\\\
  z [m]
  & ${k['imu_exts']['z_mean']:.4f} \pm {k['imu_exts']['z_std']:.2e}$
  & ${y['imu_exts']['z_mean']:.4f} \pm {y['imu_exts']['z_std']:.2e}$ \\\\
  roll [deg]
  & ${k['imu_exts']['roll_mean']:.4f} \pm {k['imu_exts']['roll_std']:.4f}$
  & ${y['imu_exts']['roll_mean']:.4f} \pm {y['imu_exts']['roll_std']:.4f}$ \\\\
  pitch [deg]
  & ${k['imu_exts']['pitch_mean']:.4f} \pm {k['imu_exts']['pitch_std']:.4f}$
  & ${y['imu_exts']['pitch_mean']:.4f} \pm {y['imu_exts']['pitch_std']:.4f}$ \\\\
  yaw [deg]
  & ${k['imu_exts']['yaw_mean']:.4f} \pm {k['imu_exts']['yaw_std']:.4f}$
  & ${y['imu_exts']['yaw_mean']:.4f} \pm {y['imu_exts']['yaw_std']:.4f}$ \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
""")


def plot_num_frames(kalibr_dir, yac_dir, **kwargs):
  """ Compare calibrations """
  save_fig = kwargs.get("save_fig", True)
  save_path = kwargs.get("save_path", "./plots/calib_camera-num_frames.png")
  show_plots = kwargs.get("show_plots", False)
  plot_width = kwargs.get("plot_width", 700)
  plot_height = kwargs.get("plot_height", 700)

  # Kalibr
  df = {"type": [], "nb_views": [], "total_views": []}
  for kalibr_path in sorted(os.listdir(kalibr_dir)):
    kalibr_path = os.path.join(kalibr_dir, kalibr_path)
    if os.path.isdir(kalibr_path) is False:
      continue

    info_csv = os.path.join(kalibr_path, "calib_camera-info.csv")
    views_accepted = pandas.read_csv(info_csv)["view_accepted"].to_numpy()
    nb_views = np.sum(views_accepted)
    total = views_accepted.shape[0]

    df["type"].append("Kalibr")
    df["nb_views"].append(nb_views)
    df["total_views"].append(total)

  # YAC
  for yac_path in sorted(os.listdir(yac_dir)):
    yac_path = os.path.join(yac_dir, yac_path)
    if os.path.isdir(yac_path) is False:
      continue

    calib_camera = os.path.join(yac_path, "calib_camera", "calib-results.yaml")
    calib_camera = yaml.safe_load(open(calib_camera, 'r').read())

    df["type"].append("Our Method")
    df["nb_views"].append(calib_camera["total_reproj_error"]["nb_views"])
    df["total_views"].append(calib_camera["total_reproj_error"]["nb_views"])

  # Plot
  dpi = 96
  plot_rows = 1
  plot_cols = 1
  figsize = (plot_width / dpi, plot_height / dpi)
  _, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, dpi=dpi)
  seaborn.boxplot(ax=axs, data=df, x="type", y="nb_views", showfliers=False)
  axs.set_ylabel("Number of Views")
  axs.set_title("Number of Views Used for Camera Calibration")

  # df = pandas.DataFrame(df)
  # print(df.groupby("type")["nb_views"].mean())

  # Save figure
  if save_fig:
    plt.savefig(save_path)
    trim_image(save_path)

  # Show plots
  if show_plots:
    plt.show()


def plot_calibs(kalibr_dir, yac_dir, **kwargs):
  """ Compare calibrations """
  show_plots = kwargs.get("show_plots", False)
  kalibr_data = load_kalibr_data(kalibr_dir)
  yac_data = load_yac_data(yac_dir)
  plot_camera_params(kalibr_data, yac_data)
  plot_extrinsics(kalibr_data,
                  yac_data,
                  "T_cam0_cam1",
                  fig_path="./plots/camera_extrinsics-boxplot.png")
  plot_extrinsics(kalibr_data,
                  yac_data,
                  "T_imu0_cam0",
                  fig_path="./plots/imu_extrinsics-boxplot.png")

  print_calib_stats(kalibr_data, yac_data)

  if show_plots:
    plt.show()


def plot_calib_camera_infos(kalibr_dir, yac_dir, **kwargs):
  """ Plot Camera Calibration Information """
  save_fig = kwargs.get("save_fig", True)
  save_path = kwargs.get("save_path", "./plots/calib_camera-info.png")
  show_plots = kwargs.get("show_plots", False)
  plot_width = kwargs.get("plot_width", 320)
  plot_height = kwargs.get("plot_height", 220)

  # Setup figure
  seaborn.set_style("darkgrid")
  df = {"type": [], "info": [], "entropy": []}
  dpi = 96
  plot_rows = 1
  plot_cols = 1
  figsize = (plot_width / dpi, plot_height / dpi)
  fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, dpi=dpi)

  # Kalibr
  for kalibr_path in sorted(os.listdir(kalibr_dir)):
    kalibr_path = os.path.join(kalibr_dir, kalibr_path)
    if os.path.isdir(kalibr_path) is False:
      continue

    run = kalibr_path.split("/")[-1]
    info_csv = os.path.join(kalibr_path, "calib_camera-info.csv")
    info_data = pandas.read_csv(info_csv)

    view_info = info_data["info"].to_numpy()[-1]

    det_covar = np.exp(view_info * np.log(2))
    n = 8 + 8 + 6  # 2 camera parameters + 1 cam-cam extrinsics
    k = pow(2.0 * np.pi * np.exp(1), n)
    entropy = 0.5 * np.log(k * det_covar)

    df["type"].append("Kalibr")
    df["info"].append(view_info)
    df["entropy"].append(entropy)

  # YAC
  for yac_path in sorted(os.listdir(yac_dir)):
    yac_path = os.path.join(yac_dir, yac_path)
    if os.path.isdir(yac_path) is False:
      continue

    run = yac_path.split("/")[-1]
    info_csv = os.path.join(yac_path, "calib_camera", "calib_info.csv")
    info_data = pandas.read_csv(info_csv)

    view_idxs = np.array(range(info_data.shape[0]))
    view_info = info_data["info"].to_numpy()[-1]

    det_covar = np.exp(view_info * np.log(2))
    n = 8 + 8 + 6  # 2 camera parameters + 1 cam-cam extrinsics
    k = pow(2.0 * np.pi * np.exp(1), n)
    entropy = 0.5 * np.log(k * det_covar)

    df["type"].append("Our Method")
    df["info"].append(view_info)
    df["entropy"].append(entropy)

  # Plot
  seaborn.boxplot(ax=axs,
                  data=df,
                  x="type",
                  y="entropy",
                  width=0.3,
                  showfliers=False)
  axs.set_ylabel("Shannon Entropy [nats]")
  axs.set_aspect(0.08)
  fig.tight_layout()

  # Save fig
  if save_fig:
    plt.savefig(save_path)
    trim_image(save_path)

  # Show plots
  if show_plots:
    plt.show()


def plot_calib_imu_infos(kalibr_dir, yac_dir, **kwargs):
  """ Plot Camera-IMU Calibration Information """
  save_fig = kwargs.get("save_fig", True)
  save_path = kwargs.get("save_path", "./plots/calib_imu-info.png")
  show_plots = kwargs.get("show_plots", False)
  plot_width = kwargs.get("plot_width", 320)
  plot_height = kwargs.get("plot_height", 220)

  # Setup figure
  df = {"type": [], "info": [], "entropy": []}
  dpi = 96
  plot_rows = 1
  plot_cols = 1
  figsize = (plot_width / dpi, plot_height / dpi)
  fig, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, dpi=dpi)

  # Kalibr
  for kalibr_path in sorted(os.listdir(kalibr_dir)):
    kalibr_path = os.path.join(kalibr_dir, kalibr_path)
    if os.path.isdir(kalibr_path) is False:
      continue

    run = kalibr_path.split("/")[-1]
    info_csv = os.path.join(kalibr_path, "calib_imu-info.csv")
    info_data = pandas.read_csv(info_csv)

    view_info = info_data["info"].to_numpy()[-1]

    det_covar = np.exp(view_info * np.log(2))
    n = 6  # 1 cam-imu extrinsics
    k = pow(2.0 * np.pi * np.exp(1), n)
    entropy = 0.5 * np.log(k * det_covar)

    df["type"].append("Kalibr")
    df["info"].append(view_info)
    df["entropy"].append(entropy)

  # YAC
  for yac_path in sorted(os.listdir(yac_dir)):
    yac_path = os.path.join(yac_dir, yac_path)
    if os.path.isdir(yac_path) is False:
      continue

    run = yac_path.split("/")[-1]
    info_csv = os.path.join(yac_path, "calib_imu", "calib_info.csv")
    info_data = pandas.read_csv(info_csv)

    nbt_idx = np.array(range(info_data.shape[0]))
    view_info = info_data["info_current"].to_numpy()[-1]

    det_covar = np.exp(view_info * np.log(2))
    n = 6  # 1 cam-imu extrinsics
    k = pow(2.0 * np.pi * np.exp(1), n)
    entropy = 0.5 * np.log(k * det_covar)

    df["type"].append("Our Method")
    df["info"].append(view_info)
    df["entropy"].append(entropy)

  # Plot
  seaborn.set_style("darkgrid")
  seaborn.boxplot(ax=axs,
                  data=df,
                  x="type",
                  y="entropy",
                  width=0.3,
                  showfliers=False)
  axs.set_ylabel("Shannon Entropy [nats]")
  axs.set_aspect(0.08)
  axs.set_ylim([-40.0, -20.0])
  fig.tight_layout()

  # Save fig
  if save_fig:
    plt.savefig(save_path)
    trim_image(save_path)

  # Show plots
  if show_plots:
    plt.show()


def plot_timings(kalibr_dir, yac_dir, **kwargs):
  """ Plot timings """
  save_fig = kwargs.get("save_fig", True)
  save_path = kwargs.get("save_path", "./plots")
  show_plots = kwargs.get("show_plots", False)
  plot_width = kwargs.get("plot_width", 700)
  plot_height = kwargs.get("plot_height", 320)

  # Kalibr
  kalibr_camera_capture_time = 0.0
  kalibr_camera_solve_time = 0.0
  kalibr_camera_imu_capture_time = 0.0
  kalibr_camera_imu_solve_time = 0.0
  kalibr_data = {
      "calib_camera_capture": [],
      "calib_camera_solve": [],
      "calib_imu_capture": [],
      "calib_imu_solve": [],
      "calib_total_time": [],
  }

  for kalibr_path in sorted(os.listdir(kalibr_dir)):
    kalibr_path = os.path.join(kalibr_dir, kalibr_path)
    if os.path.isdir(kalibr_path) is False:
      continue

    calib_camera_rosbag = os.path.join(kalibr_path, "calib_camera.bag")
    calib_camera_rosbag = rosbag.Bag(calib_camera_rosbag)
    start_time = calib_camera_rosbag.get_start_time()
    end_time = calib_camera_rosbag.get_end_time()
    calib_camera_capture_time = (end_time - start_time)

    calib_camera_solve_time = os.path.join(kalibr_path, "calib_camera-time.txt")
    calib_camera_solve_time = open(calib_camera_solve_time, "r").read()
    calib_camera_solve_time = calib_camera_solve_time.split(" ")
    calib_camera_solve_time = calib_camera_solve_time[2].replace("elapsed", "")
    calib_camera_solve_time = calib_camera_solve_time.split(".")[0]
    x = time.strptime(calib_camera_solve_time, '%M:%S')
    calib_camera_solve_time = datetime.timedelta(
        hours=x.tm_hour, minutes=x.tm_min, seconds=x.tm_sec).total_seconds()

    calib_imu_rosbag = os.path.join(kalibr_path, "calib_imu.bag")
    calib_imu_rosbag = rosbag.Bag(calib_imu_rosbag)
    start_time = calib_imu_rosbag.get_start_time()
    end_time = calib_imu_rosbag.get_end_time()
    calib_imu_capture_time = (end_time - start_time)

    calib_imu_solve_time = os.path.join(kalibr_path, "calib_imu-time.txt")
    calib_imu_solve_time = open(calib_imu_solve_time, "r").read()
    calib_imu_solve_time = calib_imu_solve_time.split(" ")
    calib_imu_solve_time = calib_imu_solve_time[2].replace("elapsed", "")
    calib_imu_solve_time = calib_imu_solve_time.split(".")[0]
    x = time.strptime(calib_imu_solve_time, '%M:%S')
    calib_imu_solve_time = datetime.timedelta(hours=x.tm_hour,
                                              minutes=x.tm_min,
                                              seconds=x.tm_sec).total_seconds()

    calib_total_time = calib_camera_capture_time + calib_camera_solve_time
    calib_total_time += calib_imu_capture_time + calib_imu_solve_time

    kalibr_data["calib_camera_capture"].append(calib_camera_capture_time)
    kalibr_data["calib_camera_solve"].append(calib_camera_solve_time)
    kalibr_data["calib_imu_capture"].append(calib_imu_capture_time)
    kalibr_data["calib_imu_solve"].append(calib_imu_solve_time)
    kalibr_data["calib_total_time"].append(calib_total_time)

    kalibr_camera_capture_time += calib_camera_capture_time
    kalibr_camera_solve_time += calib_camera_solve_time
    kalibr_camera_imu_capture_time += calib_imu_capture_time
    kalibr_camera_imu_solve_time += calib_imu_solve_time

  # YAC
  yac_camera_capture_time = 0.0
  yac_camera_solve_time = 0.0
  yac_camera_imu_capture_time = 0.0
  yac_camera_imu_solve_time = 0.0
  yac_data = {
      "calib_camera_capture": [],
      "calib_camera_solve": [],
      "calib_imu_capture": [],
      "calib_imu_solve": [],
      "calib_total_time": [],
  }

  for yac_path in sorted(os.listdir(yac_dir)):
    yac_path = os.path.join(yac_dir, yac_path)
    if os.path.isdir(yac_path) is False:
      continue

    calib_camera = os.path.join(yac_path, "calib_camera", "calib-results.yaml")
    calib_camera = yaml.safe_load(open(calib_camera, 'r').read())
    calib_imu = os.path.join(yac_path, "calib_imu", "calib-results.yaml")
    calib_imu = yaml.safe_load(open(calib_imu, 'r').read())

    calib_total_time = calib_camera["profiling"]["total_time"]
    calib_total_time += calib_imu["profiling"]["total_time"]

    calib_camera_capture_time = calib_camera["profiling"]["data_collection"]
    calib_camera_solve_time = calib_camera["profiling"]["final_solve"]
    calib_imu_capture_time = calib_imu["profiling"]["data_collection"]
    calib_imu_solve_time = calib_imu["profiling"]["final_solve"]

    yac_data["calib_camera_capture"].append(calib_camera_capture_time)
    yac_data["calib_camera_solve"].append(calib_camera_solve_time)
    yac_data["calib_imu_capture"].append(calib_imu_capture_time)
    yac_data["calib_imu_solve"].append(calib_imu_solve_time)
    yac_data["calib_total_time"].append(calib_total_time)

    yac_camera_capture_time += calib_camera_capture_time
    yac_camera_solve_time += calib_camera_solve_time
    yac_camera_imu_capture_time += calib_imu_capture_time
    yac_camera_imu_solve_time += calib_imu_solve_time

  # Plot
  dpi = 96
  figsize = (plot_width / dpi, plot_height / dpi)

  stack_data = [[], []]
  stack_data[0].append("Kalibr")
  stack_data[0].append(np.mean(kalibr_data["calib_camera_capture"]))
  stack_data[0].append(np.mean(kalibr_data["calib_camera_solve"]))
  stack_data[0].append(np.mean(kalibr_data["calib_imu_capture"]))
  stack_data[0].append(np.mean(kalibr_data["calib_imu_solve"]))
  stack_data[1].append("Our Method")
  stack_data[1].append(np.mean(yac_data["calib_camera_capture"]))
  stack_data[1].append(np.mean(yac_data["calib_camera_solve"]))
  stack_data[1].append(np.mean(yac_data["calib_imu_capture"]))
  stack_data[1].append(np.mean(yac_data["calib_imu_solve"]))

  columns = [
      "type", "Camera Capture Time", "Camera Solve Time",
      "Camera-IMU Capture Time", "Camera-IMU Solve Time"
  ]
  colors = ["#ff3333", "#e78b8b", "#00ff13", "#a2ffa9"]

  pandas_df = pandas.DataFrame(stack_data, columns=columns)
  ax = pandas_df.plot(kind='bar',
                      x='type',
                      stacked=True,
                      figsize=figsize,
                      color=colors)
  ax.set_xlabel("")
  ax.set_ylabel("Time [s]")
  plt.setp(ax.xaxis.get_ticklabels(), rotation=0)

  # Save fig
  if save_fig:
    figpath = os.path.join(save_path, "timings-stacked.png")
    plt.savefig(figpath)
    trim_image(figpath)

  # # Plot Break Downs
  # dpi = 96
  # plot_rows = 2
  # plot_cols = 2
  # figsize = (plot_width / dpi, plot_height / dpi)
  # _, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, dpi=dpi)

  # seaborn.boxplot(ax=axs[0, 0],
  #                 data=df,
  #                 x="type",
  #                 y="calib_camera_capture",
  #                 showfliers=False)
  # axs[0, 0].set_ylabel("Time [s]")
  # axs[0, 0].set_title("Camera Capture Time")

  # seaborn.boxplot(ax=axs[0, 1],
  #                 data=df,
  #                 x="type",
  #                 y="calib_camera_solve",
  #                 showfliers=False)
  # axs[0, 1].set_ylabel("Time [s]")
  # axs[0, 1].set_title("Camera Solve Time")

  # seaborn.boxplot(ax=axs[1, 0],
  #                 data=df,
  #                 x="type",
  #                 y="calib_imu_capture",
  #                 showfliers=False)
  # axs[1, 0].set_ylabel("Time [s]")
  # axs[1, 0].set_title("Camera-IMU Capture Time")

  # seaborn.boxplot(ax=axs[1, 1],
  #                 data=df,
  #                 x="type",
  #                 y="calib_imu_solve",
  #                 showfliers=False)
  # axs[1, 1].set_ylabel("Time [s]")
  # axs[1, 1].set_title("Camera-IMU Solve Time")

  # # Save fig
  # if save_fig:
  #   figpath = os.path.join(save_path, "timings-breakdown.png")
  #   plt.savefig(figpath)
  #   trim_image(figpath)

  # # Plot Total Time
  # dpi = 96
  # plot_rows = 1
  # plot_cols = 1
  # figsize = (plot_width / dpi, plot_height / dpi)
  # _, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, dpi=dpi)

  # seaborn.boxplot(ax=axs,
  #                 data=df,
  #                 x="type",
  #                 y="calib_total_time",
  #                 showfliers=False)
  # axs.set_ylabel("Time [s]")
  # axs.set_title("Total Time")

  # # Save fig
  # if save_fig:
  #   figpath = os.path.join(save_path, "timings-total.png")
  #   plt.savefig(figpath)
  #   trim_image(figpath)

  # # Show plot
  # if show_plots:
  #   plt.show()


def plot_time_delays(kalibr_dir, yac_dir, **kwargs):
  """ Plot time delays """
  # Settings
  time_delays_file = kwargs.get("time_delays_file", "/tmp/time_delays.csv")
  force_reeval = kwargs.get("force_reeval", False)
  save_fig = kwargs.get("save_fig", True)
  save_path = kwargs.get("save_path", "./plots")
  show_plots = kwargs.get("show_plots", False)
  plot_width = kwargs.get("plot_width", 700)
  plot_height = kwargs.get("plot_height", 320)

  # Extract time delays and write to file
  if not os.path.exists(time_delays_file) or force_reeval:
    df = {"type": [], "seq": [], "time_delay": []}

    # Extract Kalibr Time Delays
    print("Extracting Kalibr Time Delays ...", flush=True)
    for kalibr_path in sorted(os.listdir(kalibr_dir)):
      kalibr_path = os.path.join(kalibr_dir, kalibr_path)
      calib_seq = os.path.basename(kalibr_path)
      if os.path.isdir(kalibr_path) is False:
        continue

      calib_file = f"{kalibr_path}/calib_imu-camchain.yaml"
      time_delay = None
      with open(calib_file, "r") as s:
        calib = yaml.load(s.read())
        cam0_td = calib["cam0"]["timeshift_cam_imu"]
        cam1_td = calib["cam1"]["timeshift_cam_imu"]
        time_delay = (cam0_td + cam1_td) / 2.0

      df["type"].append("Kalibr")
      df["seq"].append(calib_seq)
      df["time_delay"].append(time_delay)

    # Extract YAC Time Delays
    print("Extracting YAC Time Delays ...", flush=True)
    yac_time_delays = []
    for yac_path in sorted(os.listdir(yac_dir)):
      yac_path = os.path.join(yac_dir, yac_path)
      calib_seq = os.path.basename(yac_path)
      if os.path.isdir(yac_path) is False:
        continue

      calib_file = f"{yac_path}/calib-imu-rerun.yaml"
      time_delay = None
      with open(calib_file, "r") as s:
        calib = yaml.load(s.read())
        time_delay = calib["time_delay"]
      yac_time_delays.append(time_delay)

      df["type"].append("YAC")
      df["seq"].append(calib_seq)
      df["time_delay"].append(time_delay)

    pandas_df = pandas.DataFrame(df)
    pandas_df.to_csv(time_delays_file, index=False)

  # Load time delays summary file
  df = pandas.read_csv(time_delays_file)

  print(df)

  # Plot
  dpi = 96
  plot_rows = 1
  plot_cols = 1
  figsize = (plot_width / dpi, plot_height / dpi)
  _, axs = plt.subplots(plot_rows, plot_cols, figsize=figsize, dpi=dpi)

  seaborn.scatterplot(ax=axs, data=df, x="seq", y="time_delay", hue="Type")
  axs.set_ylabel("Time Delay [s]")
  axs.set_title("Calibration Sequence")

  # Save fig
  if save_fig:
    figpath = os.path.join(save_path, "time_delays.png")
    plt.savefig(figpath)
    trim_image(figpath)

  # Show plot
  if show_plots:
    plt.show()


if __name__ == "__main__":
  rosbags_dir = "/data/yac_experiments/human_trials/rosbags"
  kalibr_dir = "/data/yac_experiments/human_trials/kalibr"
  yac_dir = "/data/yac_experiments/human_trials/yac"
  estimates_dir = "/data/yac_experiments/human_trials/estimates"

  # mode = "VIO"
  # calib_format = "kalibr"
  # calib_file = os.path.join(kalibr_dir, "calib0/calib_imu-camchain.yaml")
  # calib_path = "/data/yac_experiments/orbslam3_calib.yaml"
  # convert_to_orbslam_calib(mode, calib_format, calib_file, calib_path)

  # rosbag_path = os.path.join(rosbags_dir, "run0.bag")
  # record_path = "/data/yac_experiments/orbslam3_results.csv"
  # run_orbslam3_stereo_imu(calib_path, rosbag_path, record_path)

  # eval_kalibr_calibs(rosbags_dir, kalibr_dir, estimates_dir)
  # eval_yac_calibs(rosbags_dir, yac_dir, estimates_dir)
  # eval_trajs(estimates_dir)

  # kalibr_csv = "./euroc_results/configs/kalibr/cam_april-info.csv"
  # yac_yaml = "./euroc_results/configs/yac/calib_camera-results.yaml"
  # save_path = "./euroc-cam_april-convergence.png"
  # plot_convergence(kalibr_csv, yac_yaml, save_path=save_path)

  # plot_euroc_results()
  # plot_estimates(estimates_dir)
  # plot_odoms(estimates_dir, "orbslam3")
  # plot_num_frames(kalibr_dir, yac_dir)
  # plot_calibs(kalibr_dir, yac_dir)
  # plot_calib_camera_infos(kalibr_dir, yac_dir)
  # plot_calib_imu_infos(kalibr_dir, yac_dir)
  # plot_infos(kalibr_dir, yac_dir)
  # plot_timings(kalibr_dir, yac_dir)
  # plot_time_delays(kalibr_dir, yac_dir)
