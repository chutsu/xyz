#!/usr/bin/env python3
import os
import glob
import urllib.request
import zipfile


def download(src, dst):
    """ Download """
    if not os.path.exists(dst):
        print(f"Downloading [{src}] -> [{dst}]", flush=True)
        urllib.request.urlretrieve(src, dst)
        print("")


def download_sequences(dst_dir):
    """ Download sequences """
    base_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
    seqs = [
        "calibration_datasets/cam_april/cam_april.zip",
        "calibration_datasets/imu_april/imu_april.zip",
        "machine_hall/MH_01_easy/MH_01_easy.zip",
        "machine_hall/MH_02_easy/MH_02_easy.zip",
        "machine_hall/MH_03_medium/MH_03_medium.zip",
        "machine_hall/MH_04_difficult/MH_04_difficult.zip",
        "machine_hall/MH_05_difficult/MH_05_difficult.zip",
        "machine_hall/MH_01_easy/MH_01_easy.zip",
        "machine_hall/MH_02_easy/MH_02_easy.zip",
        "machine_hall/MH_03_medium/MH_03_medium.zip",
        "machine_hall/MH_04_difficult/MH_04_difficult.zip",
        "machine_hall/MH_05_difficult/MH_05_difficult.zip",
        "vicon_room1/V1_01_easy/V1_01_easy.zip",
        "vicon_room1/V1_02_medium/V1_02_medium.zip",
        "vicon_room1/V1_03_difficult/V1_03_difficult.zip",
        "vicon_room2/V2_01_easy/V2_01_easy.zip",
        "vicon_room2/V2_02_medium/V2_02_medium.zip",
        "vicon_room2/V2_03_difficult/V2_03_difficult.zip",
    ]

    # Make destination folder if it doesn't exist already
    os.makedirs(dst_dir, exist_ok=True)

    # Download sequence
    for seq in seqs:
        src = os.path.join(base_url, seq)
        dst = os.path.join(dst_dir, os.path.basename(seq))
        download(src, dst)


def download_rosbags(dst_dir):
    """ Download ROS bags """
    base_url = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"
    bags = [
        "machine_hall/MH_01_easy/MH_01_easy.bag",
        "machine_hall/MH_02_easy/MH_02_easy.bag",
        "machine_hall/MH_03_medium/MH_03_medium.bag",
        "machine_hall/MH_04_difficult/MH_04_difficult.bag",
        "machine_hall/MH_05_difficult/MH_05_difficult.bag",
        "machine_hall/MH_01_easy/MH_01_easy.bag",
        "machine_hall/MH_02_easy/MH_02_easy.bag",
        "machine_hall/MH_03_medium/MH_03_medium.bag",
        "machine_hall/MH_04_difficult/MH_04_difficult.bag",
        "machine_hall/MH_05_difficult/MH_05_difficult.bag",
        "vicon_room1/V1_01_easy/V1_01_easy.bag",
        "vicon_room1/V1_02_medium/V1_02_medium.bag",
        "vicon_room1/V1_03_difficult/V1_03_difficult.bag",
        "vicon_room2/V2_01_easy/V2_01_easy.bag",
        "vicon_room2/V2_02_medium/V2_02_medium.bag",
        "vicon_room2/V2_03_difficult/V2_03_difficult.bag",
    ]

    # Make destination folder if it doesn't exist already
    os.makedirs(dst_dir, exist_ok=True)

    # Download sequence
    for bag in bags:
        bag_name = os.path.basename(bag)
        bag_name = bag_name.replace("_easy", "")
        bag_name = bag_name.replace("_medium", "")
        bag_name = bag_name.replace("_difficult", "")

        src = os.path.join(base_url, bag)
        dst = os.path.join(dst_dir, bag_name)
        download(src, dst)


def extract_zip(src, dst):
    """ Extract zip file """
    dir_name = os.path.basename(src).replace(".zip", "")
    dir_name = dir_name.replace("_easy", "")
    dir_name = dir_name.replace("_medium", "")
    dir_name = dir_name.replace("_difficult", "")
    dst_path = os.path.join(dst, dir_name)

    print(f"Extracting [{src}] -> [{dst_path}]", flush=True)
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(dst_path)


def extract_zips(src_dir, dst):
    """ Extract zip files """
    for zipfile in sorted(glob.glob(os.path.join(src_dir, "*.zip"))):
        extract_zip(zipfile, dst)


if __name__ == "__main__":
    dst_dir = "./archive"
    bag_dir = "./rosbags"
    # download_sequences(dst_dir)
    # extract_zips(dst_dir, ".")

    download_rosbags(bag_dir)
