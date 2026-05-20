#!/usr/bin/env python3
import sys
from math import atan2
from math import asin
from math import degrees
# from cStringIO import StringIO # Python 2
from io import StringIO

import rosbag
import rospy

import numpy as np


def print_usage():
  """ Print Usage """
  print("Usage: bag2csv.py <ros bag> <ros topic> <output path>")
  print("Example: bag2csv.py record.bag /robot/pose robot_images.csv")


def quat2euler(q):
  """ Quaternion to Euler """
  qw = q[0]
  qx = q[1]
  qy = q[2]
  qz = q[3]

  qw2 = qw * qw
  qx2 = qx * qx
  qy2 = qy * qy
  qz2 = qz * qz

  x = atan2(2 * (qx * qw + qz * qy), (qw2 - qx2 - qy2 + qz2))
  y = asin(2 * (qy * qw - qx * qz))
  z = atan2(2 * (qx * qy + qz * qw), (qw2 + qx2 - qy2 - qz2))

  return [x, y, z]


class std_msgs:
  supported_msgs = ["std_msgs/Header" "std_msgs/String"]

  @staticmethod
  def header_to_str(msg):
    header = "seq,frame_id,secs,nsecs"

    seq = msg.seq
    frame_id = msg.frame_id
    secs = msg.stamp.secs
    nsecs = msg.stamp.nsecs
    timestamp = rospy.Time(secs, nsecs)

    ts = str(timestamp.to_nsec())
    secs = str(ts[0:10])
    nsecs = str(ts[10:19])

    data = ",".join([str(seq), str(frame_id), secs, nsecs])
    return (header, data)

  @staticmethod
  def string_to_str(field_name, data):
    header = field_name
    return (header, data)


class geometry_msgs:
  supported_msgs = [
      "geometry_msgs/Point", "geometry_msgs/PointStamped",
      "geometry_msgs/Vector3", "geometry_msgs/Quaternion", "geometry_msgs/Pose",
      "geometry_msgs/PoseStamped", "geometry_msgs/PoseWithCovarianceStamped",
      "geometry_msgs/Twist", "geometry_msgs/TwistStamped",
      "geometry_msgs/TwistWithCovarianceStamped",
      "geometry_msgs/TransformStamped"
  ]

  @staticmethod
  def point_to_str(msg, prefix=""):
    axis = ["x", "y", "z"]
    header = ",".join([prefix + ax for ax in axis])
    data = ",".join([str(msg.x), str(msg.y), str(msg.z)])
    return (header, data)

  @staticmethod
  def point_stamped_to_str(msg, prefix=""):
    msg_header, header_data = std_msgs.header_to_str(msg.header)
    point_header, point_data = geometry_msgs.point_to_str(msg.point)

    header = msg_header + "," + point_header
    data = header_data + "," + point_data
    return (header, data)

  @staticmethod
  def vector3_to_str(msg, prefix=""):
    axis = ["x", "y", "z"]
    header = ",".join([prefix + ax for ax in axis])
    data = ",".join([str(msg.x), str(msg.y), str(msg.z)])
    return (header, data)

  @staticmethod
  def quaternion_to_str(msg):
    header = "qw,qx,qy,qz,roll[deg],pitch[deg],yaw[deg]"
    rpy = quat2euler([msg.w, msg.x, msg.y, msg.z])
    data = ",".join([
        str(msg.w),
        str(msg.x),
        str(msg.y),
        str(msg.z),
        str(degrees(rpy[0])),
        str(degrees(rpy[1])),
        str(degrees(rpy[2]))
    ])
    return (header, data)

  @staticmethod
  def covariance_to_str(covar_data, prefix=""):
    header = ",".join([prefix + "covar_" + str(i) for i in range(36)])
    data = ",".join([str(x) for x in covar_data])
    return (header, data)

  @staticmethod
  def pose_to_str(msg):
    pos_header, pos_data = geometry_msgs.point_to_str(msg.position)
    rot_header, rot_data = geometry_msgs.quaternion_to_str(msg.orientation)

    header = pos_header + "," + rot_header
    data = pos_data + "," + rot_data
    return (header, data)

  @staticmethod
  def tf_to_str(msg):
    pos_header, pos_data = geometry_msgs.vector3_to_str(msg.translation)
    rot_header, rot_data = geometry_msgs.quaternion_to_str(msg.rotation)

    header = pos_header + "," + rot_header
    data = pos_data + "," + rot_data
    return (header, data)

  @staticmethod
  def pose_with_covariance_to_str(msg):
    pose_header, pose_data = geometry_msgs.pose_to_str(msg.pose)
    covar_header, covar_data = geometry_msgs.covariance_to_str(
        msg.covariance, "pose_")

    header = pose_header + "," + covar_header
    data = pose_data + "," + covar_data
    return (header, data)

  @staticmethod
  def pose_stamped_to_str(msg):
    msg_header, header_data = std_msgs.header_to_str(msg.header)
    pose_header, pose_data = geometry_msgs.pose_to_str(msg.pose)

    header = msg_header + "," + pose_header
    data = header_data + "," + pose_data
    return (header, data)

  @staticmethod
  def pose_with_covariance_stamped_to_str(msg):
    msg_header, header_data = std_msgs.header_to_str(msg.header)
    pose_header, pose_data = geometry_msgs.pose_with_covariance_to_str(
        msg.pose, "pose_")

    header = msg_header + "," + pose_header
    data = header_data + "," + pose_data
    return (header, data)

  @staticmethod
  def twist_to_str(msg):
    linear_header, linear_data = geometry_msgs.vector3_to_str(msg.linear, "a")
    angular_header, angular_data = geometry_msgs.vector3_to_str(
        msg.angular, "w")

    header = linear_header + "," + angular_header
    data = linear_data + "," + angular_data
    return (header, data)

  @staticmethod
  def twist_with_covariance_to_str(msg):
    twist_header, twist_data = geometry_msgs.twist_to_str(msg.twist)
    covar_header, covar_data = geometry_msgs.covariance_to_str(
        msg.covariance, "twist_")

    header = twist_header + "," + covar_header
    data = twist_data + "," + covar_data
    return (header, data)

  @staticmethod
  def twist_stamped_to_str(msg):
    msg_header, header_data = std_msgs.header_to_str(msg.header)
    twist_header, twist_data = geometry_msgs.twist_to_str(msg.twist)

    header = msg_header + "," + twist_header
    data = msg_data + "," + twist_data
    return (header, data)

  @staticmethod
  def twist_with_covariance_stamped_to_str(msg):
    msg_header, header_data = std_msgs.header_to_str(msg.header)
    twist_header, twist_data = geometry_msgs.twist_to_str(msg.twist)
    covar_header, covar_data = geometry_msgs.covariance_to_str(
        msg.covariance, "twist_")

    header = msg_header + "," + twist_header + "," + covar_header
    data = msg_data + "," + twist_data + "," + covar_data
    return (header, data)

  @staticmethod
  def transform_stamped_to_str(msg):
    msg_header, header_data = std_msgs.header_to_str(msg.header)
    tf_header, tf_data = geometry_msgs.tf_to_str(msg.transform)

    header = msg_header + "," + tf_header
    data = header_data + "," + tf_data
    return (header, data)


class nav_msgs:
  supported_msgs = [
      "nav_msgs/Odometry",
  ]

  @staticmethod
  def odometry_to_str(msg):
    msg_header, msg_data = std_msgs.header_to_str(msg.header)
    _, str_data = std_msgs.string_to_str("child_frame_id", msg.child_frame_id)
    pose_header, pose_data = geometry_msgs.pose_with_covariance_to_str(msg.pose)
    twist_header, twist_data = geometry_msgs.twist_with_covariance_to_str(
        msg.twist)

    header = msg_header + "," + "child_frame_id" + "," + pose_header + "," + twist_header
    data = msg_data + "," + str_data + "," + pose_data + "," + twist_data
    return (header, data)


class sensor_msgs:
  supported_msgs = ["sensor_msgs/Imu", "sensor_msgs/BatteryState"]

  @staticmethod
  def imu_to_str(msg):
    msg_header, msg_data = std_msgs.header_to_str(msg.header)
    rot_header, quat = geometry_msgs.quaternion_to_str(msg.orientation)
    _, gyr = geometry_msgs.vector3_to_str(msg.angular_velocity)
    _, acc = geometry_msgs.vector3_to_str(msg.linear_acceleration)
    header = msg_header + "," + rot_header + "," + "wx,wy,wz,ax,ay,az"
    data = msg_data + "," + quat + "," + gyr + "," + acc
    return (header, data)

  @staticmethod
  def battery_state_to_str(msg):
    msg_header, msg_data = std_msgs.header_to_str(msg.header)

    header = msg_header + ","
    header += "voltage,"
    header += "current,"
    header += "charge,"
    header += "capacity,"
    header += "design_capacity,"
    header += "percentage,"
    header += "power_supply_status,"
    header += "power_supply_health,"
    header += "power_supply_technology,"
    header += "present,"
    header += "location,"
    header += "serial_number"

    data = msg_data + ","
    data += str(msg.voltage) + ","
    data += str(msg.current) + ","
    data += str(msg.charge) + ","
    data += str(msg.capacity) + ","
    data += str(msg.design_capacity) + ","
    data += str(msg.percentage) + ","
    data += str(msg.power_supply_status) + ","
    data += str(msg.power_supply_health) + ","
    data += str(msg.power_supply_technology) + ","
    data += str(msg.present) + ","
    data += str(msg.location) + ","
    data += str(msg.serial_number)

    return (header, data)


class mavros_msgs:
  supported_msgs = ["mavros_msgs/AttitudeTarget"]

  @staticmethod
  def attitude_target_to_str(msg):
    msg_header, msg_data = std_msgs.header_to_str(msg.header)
    header = msg_header + ",wx,wy,wz,ax,ay,az"

    rot_header, quat = geometry_msgs.quaternion_to_str(msg.orientation)
    _, vec = geometry_msgs.vector3_to_str(msg.body_rate)
    body_rate_header = "body_rate_x,body_rate_y,body_rate_z"

    header = msg_header + "," + rot_header + "," + body_rate_header + "," + "thrust"
    data = msg_data + "," + quat + "," + vec + "," + str(msg.thrust)
    return (header, data)


class aabm_comms:
  supported_msgs = ["aabm_comms/Trajectory"]

  @staticmethod
  def traj_to_str(msg):
    waypoints = msg.trajectory

    time = []
    pos = []
    vel = []
    acc = []
    yaw = []

    ts = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
    for wp in msg.trajectory:
      pos.append([wp.position.x, wp.position.y, wp.position.z])
      vel.append(
          [wp.linearVelocity.x, wp.linearVelocity.y, wp.linearVelocity.z])
      acc.append([
          wp.linearAcceleration.x, wp.linearAcceleration.y,
          wp.linearAcceleration.z
      ])
      yaw.append(wp.yaw)
      time.append(ts + wp.timeMilliseconds * 1e-3)

    header = "timestamp[s]"
    header += ",pos_x[m],pos_y[m],pos_z[m]"
    header += ",vel_x[m],vel_y[m],vel_z[m]"
    header += ",acc_x[m],acc_y[m],acc_z[m]"
    header += ",yaw[rad]"

    time = np.array(time).reshape(len(time), 1)
    pos = np.array(pos)
    vel = np.array(vel)
    acc = np.array(acc)
    yaw = np.array(yaw).reshape(len(yaw), 1)
    df = np.block([time, pos, vel, acc, yaw])

    str_io = StringIO()
    np.savetxt(str_io, df, fmt='%.9f', delimiter=',')
    data = str_io.getvalue()

    return (header, data)


def check_topic_exists(bag, topic):
  info = bag.get_type_and_topic_info()
  if topic not in info.topics:
    raise RuntimeError("Opps! topic not in bag!")


def check_topic_type(bag, topic):
  info = bag.get_type_and_topic_info()
  msg_type = info.topics[topic].msg_type
  supported_msgs = std_msgs.supported_msgs
  supported_msgs += geometry_msgs.supported_msgs
  supported_msgs += nav_msgs.supported_msgs
  supported_msgs += sensor_msgs.supported_msgs
  supported_msgs += mavros_msgs.supported_msgs
  supported_msgs += aabm_comms.supported_msgs

  if msg_type not in supported_msgs:
    supported_list = ""
    for x in supported_msgs:
      supported_list += "  - " + str(x) + "\n"

    err_msg = "bag2csv does not support msg type: [%s]\n" % msg_type
    err_msg += "bag2csv currently only supports:\n%s" % supported_list
    raise RuntimeError(err_msg)

  return msg_type


def get_msg_converter(bag, topic):
  info = bag.get_type_and_topic_info()
  msg_type = info.topics[topic].msg_type

  # STD MSGS
  if msg_type == "std_msgs/Header":
    return std_msgs.header_to_str

  # GEOMETRY MSGS
  if msg_type == "geometry_msgs/Point":
    return geometry_msgs.point_to_str
  if msg_type == "geometry_msgs/PointStamped":
    return geometry_msgs.point_stamped_to_str
  if msg_type == "geometry_msgs/Vector3":
    return geometry_msgs.vector3_to_str
  if msg_type == "geometry_msgs/Quaternion":
    return geometry_msgs.quaternion_to_str
  if msg_type == "geometry_msgs/Pose":
    return geometry_msgs.pose_to_str
  if msg_type == "geometry_msgs/PoseStamped":
    return geometry_msgs.pose_stamped_to_str
  if msg_type == "geometry_msgs/PoseWithCovarianceStamped":
    return geometry_msgs.pose_with_covariance_stamped_to_str
  if msg_type == "geometry_msgs/TransformStamped":
    return geometry_msgs.transform_stamped_to_str

  # ODOMETRY MSGS
  if msg_type == "nav_msgs/Odometry":
    return nav_msgs.odometry_to_str

  # SENSOR MSGS
  if msg_type == "sensor_msgs/Imu":
    return sensor_msgs.imu_to_str
  if msg_type == "sensor_msgs/BatteryState":
    return sensor_msgs.battery_state_to_str

  # MAVROS MSGS
  if msg_type == "mavros_msgs/AttitudeTarget":
    return mavros_msgs.attitude_target_to_str

  # AABM MSGS
  if msg_type == "aabm_comms/Trajectory":
    return aabm_comms.traj_to_str


def bag2csv(bag_path, topic, output_path):
  # Checks
  bag = rosbag.Bag(bag_path, 'r')
  check_topic_exists(bag, topic)
  msg_type = check_topic_type(bag, topic)
  msg_converter = get_msg_converter(bag, topic)

  # Output csv file
  print("Processing rosbag: [%s]" % (bag_path))
  print("Extracting rostopic: [%s]" % (topic))
  print("Saving to: [%s]" % (output_path))
  # -- Output header
  csv_file = open(output_path, "w")
  topic, msg, t = next(bag.read_messages(topics=[topic]))
  header, data = msg_converter(msg)
  csv_file.write("#" + header + "\n")
  csv_file.write(data + "\n")
  # -- Special messages
  if msg_type == "aabm_comms/Trajectory":
    print("Done!")
    exit(0)
  # -- Output data
  for topic, msg, t in bag.read_messages(topics=[topic]):
    _, data = msg_converter(msg)
    csv_file.write(data + "\n")
    csv_file.flush()
  csv_file.close()


if __name__ == "__main__":
  # Check CLI args
  if len(sys.argv) != 4:
    print_usage()
    exit(-1)

  # Parse CLI args
  bag_path = sys.argv[1]
  topic = sys.argv[2]
  output_path = sys.argv[3]
  bag2csv(bag_path, topic, output_path)
  print("Done!")
