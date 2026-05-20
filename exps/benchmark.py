#!/usr/bin/env python3
"""
Benchmark
"""
import os
import sys
import time
import shutil
import uuid
import logging
import yaml
import glob
import argparse
import subprocess
from pathlib import Path

import pandas
# import bag2csv

# Setup Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

# Global settings
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/data"
RESOURCE_DIR = f"{SCRIPT_DIR}/resource"

################################################################################
# UTILS
################################################################################


def docker_command(docker_image, command):
  """ Form docker command """
  return f"""\
docker run -e DISPLAY \
  --privileged \
  -v /tmp:/tmp \
  -v {DATA_DIR}:{DATA_DIR} \
  -v {RESOURCE_DIR}:/home/docker/resource \
  -it \
  --rm \
  {docker_image} \
  {command}"""


class YamlUnknown(yaml.SafeLoader):
  """ YAML Unknown Loader """
  def ignore_unknown(self, _):
    """ Ignore unknown """
    return None


def extract_time_delay(calib_file):
  """ Get the cam-imu time delay from the calibration file """
  imu_cam_td = None

  with open(calib_file, "r") as stream:
    _ = stream.readline()  # Skip first line

    YamlUnknown.add_constructor(None, YamlUnknown.ignore_unknown)
    calib = yaml.load(stream.read(), Loader=YamlUnknown)
    imu_cam_td = calib["Camera.timeshift_cam_imu"]

  return imu_cam_td


def dataset_adjust_timestamps(src_dir, dst_dir, dataset_name, calib_file):
  """
  Using the estimated IMU-camera time delay we adjust the EuRoC dataset
  timestamps, we do this  because ORBSLAM3 assumes the IMU-camera are
  perfectly synchronized with 0 time delay and there is no other way to input
  this.
  """
  # Extract the time delay
  imu_cam_td_s = extract_time_delay(calib_file)
  imu_cam_td_ns = int(imu_cam_td_s * 1e9)  # convert s to ns

  # Copy original to tmp directory
  src_path = f"{src_dir}/{dataset_name}"
  dst_path = f"{dst_dir}/{dataset_name}"
  if os.path.exists(dst_path) and os.path.isdir(dst_path):
    shutil.rmtree(dst_path)
  shutil.copytree(src_path, dst_path)

  # Adjust IMU timestamps
  logger.info("Adjusting [%s] IMU timestamps", dataset_name)
  imu_path = f"{dst_dir}/{dataset_name}/mav0/imu0/data.csv"
  imu_df = pandas.read_csv(imu_path)
  timestamps_original = imu_df["#timestamp [ns]"]
  timestamps_adjusted = imu_df["#timestamp [ns]"] + imu_cam_td_ns
  diff = timestamps_adjusted[0] - timestamps_original[0]
  imu_df["#timestamp [ns]"] = timestamps_adjusted
  assert diff == imu_cam_td_ns

  # Overwrite new IMU data
  imu_df.to_csv(imu_path, index=False)


################################################################################
# ORBSLAM3
################################################################################


def run_orbslam3(mode, ds_path, run_name, config_path, res_dir, **kwargs):
  """ Run ORBSLAM """
  # Setup
  retries = kwargs.get("retries", 3)
  cmd = ""
  f_file = ""
  kf_file = ""
  uuid_str = str(uuid.uuid4())

  # ORBSLAM3 settings
  timestamps_path = f"/tmp/orbslam3-{uuid_str}-timestamps.txt"
  orbslam3_path = "/home/docker/ORB_SLAM3"
  docker_image = "benchmark/orbslam3"

  # Check dataset path
  if os.path.exists(ds_path) is False:
    raise RuntimeError(f"Dataset [{ds_path}] does not exist")

  # Check calib file
  if os.path.exists(config_path) is False:
    raise RuntimeError(f"Calibration file [{config_path}] does not exist")

  # Create a timestamp file for ORBSLAM3
  # -- Get timestamps from camera images
  timestamps = {}
  num_cams = len(glob.glob(f"{ds_path}/*/cam*"))
  for cam_dir in glob.glob(f"{ds_path}/*/cam*"):
    for image_file in glob.glob(f"{cam_dir}/data/*"):
      # Extract timestamp from filename
      timestamp = int(os.path.basename(image_file).split(".")[0])
      if timestamp not in timestamps:
        timestamps[timestamp] = 1
      else:
        timestamps[timestamp] += 1

  # -- Check if number of cameras found
  if num_cams == 0:
    raise RuntimeError(f"Found 0 cameras in [{ds_path}]?")

  # -- Check number of timestamps
  if len(timestamps) == 0:
    raise RuntimeError(f"Found 0 timestamps in [{ds_path}]?")

  # -- Write out timestamps to file
  timestamps_file = open(timestamps_path, "w")
  for timestamp, count in timestamps.items():
    if count == num_cams:
      timestamps_file.write("%ld\n" % timestamp)
  timestamps_file.close()

  # Run ORBSLAM3
  if mode == "mono":
    run_cmd = f"""\
{orbslam3_path}/Examples/Monocular/mono_euroc \
{orbslam3_path}/Vocabulary/ORBvoc.txt \
{config_path} \
{ds_path} \
{timestamps_path} \
dataset-{uuid_str}-{run_name}_mono"""
    cmd = docker_command(docker_image, run_cmd)
    f_file = f"f_dataset-{uuid_str}-{run_name}_mono.txt"
    kf_file = f"kf_dataset-{uuid_str}-{run_name}_mono.txt"

  elif mode == "stereo":
    run_cmd = f"""\
{orbslam3_path}/Examples/Stereo/stereo_euroc \
{orbslam3_path}/Vocabulary/ORBvoc.txt \
{config_path} \
{ds_path} \
{timestamps_path} \
dataset-{uuid_str}-{run_name}_stereo"""
    cmd = docker_command(docker_image, run_cmd)
    f_file = f"f_dataset-{uuid_str}-{run_name}_stereo.txt"
    kf_file = f"kf_dataset-{uuid_str}-{run_name}_stereo.txt"

  elif mode == "stereo_imu":
    run_cmd = f"""\
{orbslam3_path}/Examples/Stereo-Inertial/stereo_inertial_euroc \
{orbslam3_path}/Vocabulary/ORBvoc.txt \
{config_path} \
{ds_path} \
{timestamps_path} \
dataset-{uuid_str}-{run_name}_stereo_imu"""
    cmd = docker_command(docker_image, run_cmd)
    f_file = f"f_dataset-{uuid_str}-{run_name}_stereo_imu.txt"
    kf_file = f"kf_dataset-{uuid_str}-{run_name}_stereo_imu.txt"
  else:
    logger.error("ERROR! ORBSLAM3 [%s] mode not supported!", mode)
    sys.exit(-1)

  # Check if results already exists
  f_dst = Path(res_dir, f_file.replace(f"-{uuid_str}", ""))
  kf_dst = Path(res_dir, kf_file.replace(f"-{uuid_str}", ""))
  if os.path.exists(f_dst) or os.path.exists(kf_dst):
    return True

  # Run
  for _ in range(retries):
    # Run ORBSLAM3
    time.sleep(2)
    result = subprocess.run(cmd.split())

    # Check if result files exists
    if os.path.exists(f_file) is False:
      logger.error("ERROR! [%s] DOES NOT EXIST! RETRYING!", f_file)
      continue
    if os.path.exists(kf_file) is False:
      logger.error("ERROR! [%s] DOES NOT EXIST! RETRYING!", kf_file)
      continue

    # Move results
    os.system(f"mv {f_file} {f_dst}")
    os.system(f"rm {kf_file}")  # Remove (not useful for evaluation)
    return True

  # Failed to run orbslam
  logger.error("FAILED TO RUN ORBSLAM")
  logger.error("COMMAND:")
  logger.error("%s\n", cmd)
  return False


def run_orbslam3_euroc(run_name, mode, config_path, res_dir, **kwargs):
  """Run OBSLAM3 on EuRoC dataset"""
  data_dir = kwargs.get("data_dir", "/data/euroc")
  sequences = kwargs.get("sequences", [
      "MH_01",
      "MH_02",
      "MH_03",
      "MH_04",
      "MH_05",
      "V1_01",
      "V1_02",
      "V1_03",
      "V2_01",
      "V2_02",
      "V2_03",
  ])
  for seq in sequences:
    run_orbslam3(mode, f"{data_dir}/{seq}", run_name, config_path, res_dir)


################################################################################
# VINS-FUSION
################################################################################


def run_vins_fusion(mode, ds_path, run_name, config_path, output, **kwargs):
  """ Run VINS-Fusion """
  # Setup
  retries = kwargs.get("retries", 3)
  docker_image = kwargs.get("docker_image", "benchmark/vins-fusion")
  uuid_str = str(uuid.uuid4())

  # Check dataset path
  if os.path.exists(ds_path) is False:
    raise RuntimeError(f"Dataset [{ds_path}] does not exist")

  # Run VINS-Fusion
  ds_dir = os.path.dirname(ds_path)
  est_path = f"{ds_dir}/vins-fusion-estimation-{uuid_str}.bag"
  odom_topic = "/vins_fusion/odometry"
  cmd = ""
  if mode in ["mono_imu", "stereo", "stereo_imu"]:
    run_cmd = f"""\
roslaunch configs/vins-fusion/vins-fusion.launch \
  rosbag_input_path:={ds_path} \
  rosbag_output_path:={est_path} \
  config_file:={config_path}"""
    cmd = docker_command(docker_image, run_cmd)

  else:
    logger.error("ERROR! VINS-Fusion [%s] mode not supported!", mode)
    sys.exit(-1)

  for _ in range(retries):
    time.sleep(2)
    print(f"{cmd}")

    # Run VINS-Fusion
    os.system(cmd)

    # # Convert recorded odometry ROS bag to csv
    # bag2csv.bag2csv(est_path, odom_topic, output)

    # # Remove ROS bag
    # if os.path.exists(est_path):
    #   os.remove(est_path)

    return True

  # Failed to run orbslam
  logger.error("FAILED TO RUN VINS-Fusion")
  logger.error("COMMAND:")
  logger.error("%s\n", cmd)
  return False


################################################################################
# OKVIS
################################################################################


def run_okvis(mode, ds_path, run_name, config_path, output, **kwargs):
  """ Run OKVIS """
  # Setup
  retries = kwargs.get("retries", 3)
  docker_image = kwargs.get("docker_image", "benchmark/okvis")
  uuid_str = str(uuid.uuid4())

  # Run VINS-Fusion
  ds_dir = os.path.dirname(ds_path)
  cmd = ""
  if mode in ["stereo_imu"]:
    run_cmd = f"okvis_app_synchronous {config_path} {ds_path}"
    cmd = docker_command(docker_image, run_cmd)

  else:
    logger.error("ERROR! OKVIS [%s] mode not supported!", mode)
    sys.exit(-1)

  for _ in range(retries):
    time.sleep(2)
    print(f"{cmd}")

    # Run VINS-Fusion
    os.system(cmd)

    return True

  # Failed to run orbslam
  logger.error("FAILED TO RUN OKVIS")
  logger.error("COMMAND:")
  logger.error("%s\n", cmd)
  return False


################################################################################
# MAIN
################################################################################

if __name__ == "__main__":
  # # Parse command line arguments
  # prog_name = "benchmark"
  # prog_desc = "Benchmark SLAM algorithms"
  # parser = argparse.ArgumentParser(prog=prog_name, description=prog_desc)
  # parser.add_argument("-a", "--algo", required=True)
  # parser.add_argument("-m", "--mode", required=True)
  # parser.add_argument("-d", "--dataset", required=True)
  # parser.add_argument("-n", "--run_name", required=True)
  # parser.add_argument("-c", "--config", required=True)
  # parser.add_argument("-o", "--output", required=True)
  #
  # # Extract command line arguments
  # args = parser.parse_args()
  # mode = args.mode
  # dataset = args.dataset
  # run_name = args.run_name
  # config = args.config
  # output = args.output

  # if args.algo == "orbslam3":
  #   run_orbslam3(mode, dataset, run_name, config, output)
  # elif args.algo == "vins-fusion":
  #   run_vins_fusion(mode, dataset, run_name, config, output)
  # elif args.algo == "okvis":
  #   run_okvis(mode, dataset, run_name, config, output)
  # else:
  #   logger.error("ERROR! algorithm [%s] not supported!", args.algo)

  mode = "mono"
  dataset = "/data/euroc/MH_01"
  run_name = "euroc-mh01-mono"
  config = "./resource/configs/orbslam3/euroc/euroc-mono.yaml"
  output = "/data/orbslam3-exp"
  run_orbslam3(mode, dataset, run_name, config, output)
