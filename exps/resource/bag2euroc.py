""" Script to convert ROS bag to EuRoC dataset format """
#!/usr/bin/env python
import sys
import os
from os.path import join

import cv2
import rosbag
import rospy
from cv_bridge import CvBridge

import numpy as np
# import matplotlib.pylab as plt


def print_usage():
    """ Print usage """
    print("Usage: bag2euroc.py <ros bag> <time_delay>")
    print("Example: bag2euroc.py record.bag 1e-5")


def extract_camera_data(bag_data, target_topic, outdir):
    """ Extract camera data from rosbag """
    br = CvBridge()

    # Create data directory for images
    data_dir = join(outdir, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Csv file for timestamps + filename
    data_csv = open(join(outdir, "data.csv"), "w")
    data_csv.write("#timestamp [ns],filename\n")

    # Extract data
    for _, msg, _ in bag_data.read_messages(topics=[target_topic]):
        # Convert image message to np.array
        image = br.imgmsg_to_cv2(msg)

        # Convert image timestamp to string
        secs = msg.header.stamp.secs
        nsecs = msg.header.stamp.nsecs
        timestamp = rospy.Time(secs, nsecs)
        ts = str(timestamp.to_nsec())

        # Write image to file
        image_fname = join(data_dir, ts + ".png")
        cv2.imwrite(image_fname, image)

        # Write image file to csv
        data_csv.write(f"{ts},{image_fname}\n")


def extract_imu_data(bag_data, target_topic, outdir, time_delay):
    """ Extract imu data from rosbag """

    imu_csv = open(os.path.join(outdir, "data.csv"), "w")
    imu_csv.write("#timestamp [ns],")
    imu_csv.write("w_RS_S_x [rad s^-1],")
    imu_csv.write("w_RS_S_y [rad s^-1],")
    imu_csv.write("w_RS_S_z [rad s^-1],")
    imu_csv.write("a_RS_S_x [m s^-2],")
    imu_csv.write("a_RS_S_y [m s^-2],")
    imu_csv.write("a_RS_S_z [m s^-2]\n")

    time_delay_ns = int(time_delay * 1e9)
    ts0 = None
    time = []
    gyr = []
    acc = []

    for _, msg, _ in bag_data.read_messages(topics=[target_topic]):
        # Convert image timestamp to string
        secs = msg.header.stamp.secs
        nsecs = msg.header.stamp.nsecs
        timestamp = rospy.Time(secs, nsecs)
        ts = timestamp.to_nsec() + time_delay_ns

        # Extract angular velocity
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z

        # Extract acceleration
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z

        # Write to csv
        imu_csv.write(f"{str(ts)},")
        imu_csv.write(f"{wx},{wy},{wz},")
        imu_csv.write(f"{ax},{ay},{az}\n")

        # Keep track
        if ts0 is None:
            ts0 = ts
        time.append((ts - ts0) * 1e-9)
        gyr.append(np.array([wx, wy, wz]))
        acc.append(np.array([ax, ay, az]))

    time = np.array(time)
    gyr = np.array(gyr)
    acc = np.array(acc)

    # plt.figure()
    # plt.subplot(211)
    # plt.plot(time, gyr[:, 0], 'r-', label="wx")
    # plt.plot(time, gyr[:, 1], 'g-', label="wy")
    # plt.plot(time, gyr[:, 2], 'b-', label="wz")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Angular Velocity [rad/s]")
    # plt.subplot(212)
    # plt.plot(time, acc[:, 0], 'r-', label="ax")
    # plt.plot(time, acc[:, 1], 'g-', label="ay")
    # plt.plot(time, acc[:, 2], 'b-', label="az")
    # plt.xlabel("Time [s]")
    # plt.ylabel("Acceleration [m/s^2]")
    # plt.show()


def create_timestamps_file(cam0_path, cam1_path, outdir):
    """ Create timestamps file """
    cam0_files = os.listdir(join(cam0_path, "data"))
    cam1_files = os.listdir(join(cam1_path, "data"))
    cam0_files.sort()
    cam1_files.sort()

    if len(cam0_files) != len(cam1_files):
        raise RuntimeError("len(cam0_files) != len(cam1_files)")

    timestamps_file = open(os.path.join(outdir, "camera_timestamps.txt"), "w")
    for fname in cam0_files:
        ts_str = fname.split(".png")[0]
        timestamps_file.write(f"{ts_str}\n")


if __name__ == "__main__":
    # Check CLI args
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(-1)

    # Parse CLI args
    bag_path = sys.argv[1]
    time_delay = float(sys.argv[2])  # IMU ts = Camera ts + time_delay
    bag = rosbag.Bag(bag_path, 'r')
    imu0_topic = '/rs/imu0/data'
    cam0_topic = '/rs/ir0/image'
    cam1_topic = '/rs/ir1/image'

    # Create output directory
    output_dir = bag_path.split(".bag")[0]
    imu0_path = os.path.join(output_dir, "mav0", "imu0")
    cam0_path = os.path.join(output_dir, "mav0", "cam0")
    cam1_path = os.path.join(output_dir, "mav0", "cam1")
    for d in [imu0_path, cam0_path, cam1_path]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Check if topic is in bag
    info = bag.get_type_and_topic_info()
    for topic in [imu0_topic, cam0_topic, cam1_topic]:
        if topic not in info.topics:
            raise RuntimeError("Opps! topic not in bag!")

    # Check image message type
    msg_type = info.topics[topic].msg_type
    supported_msgs = ["sensor_msgs/Image", "sensor_msgs/Imu"]
    if msg_type not in supported_msgs:
        err_msg = "Script only supports %s!" % " or ".join(supported_msgs)
        raise RuntimeError(err_msg)

    # Extract data
    extract_imu_data(bag, imu0_topic, imu0_path, time_delay)
    extract_camera_data(bag, cam0_topic, cam0_path)
    extract_camera_data(bag, cam1_topic, cam1_path)
    create_timestamps_file(cam0_path, cam1_path, output_dir)
