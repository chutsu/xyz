#!/usr/bin/env python3
import unittest

import numpy as np
from numpy import eye
from numpy import tan
import matplotlib.pyplot as plt

from xyz import euler321
from xyz import tf
from xyz import normalize
from xyz import plot_tf
from xyz import plot_bbox
from xyz import plot_set_axes_equal



def float_to_uint10(x, min_val = 0.0, max_val = 1.0):
  """Convert float in [min_val, max_val] to 10-bit integer (0..1023)."""
  x_clipped = np.clip((x - min_val) / (max_val - min_val), 0.0, 1.0)
  return int(x_clipped * 1023.0)


def part1by2(n):
  """
  Interleave 10-bit integer with two zeros between each bit.

  The function take the bits of 1 number and insert 2 zero bits between each of
  them. This is sometimes called **bit partitioning** or **bit dilation**. It
  is essentially "spreading out" the bits of a number so that you can
  interleave them with other numbers.

  """
  n &= 0x3FF
  n = (n | (n << 16)) & 0x30000FF
  n = (n | (n << 8)) & 0x300F00F
  n = (n | (n << 4)) & 0x30C30C3
  n = (n | (n << 2)) & 0x9249249
  return n


def compact1by2(n):
  """
  The opposite of part1by2(), it reverses the operation to de-interlave the
  bits to obtain the original integer.
  """
  n &= 0x09249249
  n = (n ^ (n >> 2)) & 0x030C30C3
  n = (n ^ (n >> 4)) & 0x0300F00F
  n = (n ^ (n >> 8)) & 0x030000FF
  n = (n ^ (n >> 16)) & 0x000003FF
  return n


def morton_encode(x, y, z):
  """
  Create 3D Morton code from three integer numbers in x, y, and z axis.

  The function uses 10 bits per axis, so the final Morton code fits in a 30-bit
  integer.

  Notes:

  B = 30      # Number of bits for Morton code in 3D
  v = 0.01    # Voxel size in meters
  b = B // 3  # Bits per axis = 10
  N = 2 ** b  # Number of voxels per axis
  S = v * N   # 0.01 * 1024 = 10.24 meters

  """
  return (part1by2(z) << 2) | (part1by2(y) << 1) | part1by2(x)


def morton_decode(code):
  """
  Decode morton code back to its x, y, z components.
  """
  x = compact1by2(code)
  y = compact1by2(code >> 1)
  z = compact1by2(code >> 2)
  return (x, y, z)


# def get_parent(x, y, z):
#   return (x >> 1, y >> 1, z >> 1)
#
# def get_children(x, y, z):
#   children = []
#   base_x = x << 1
#   base_y = y << 1
#   base_z = z << 1
#   for dx in [0, 1]:
#     for dy in [0, 1]:
#       for dz in [0, 1]:
#         children.append((base_x + dx, base_y + dy, base_z + dz))
#   return children

# def morton_parent(code, level):
#   """Morton code Parent"""
#   # x, y, z = morton_decode(code)
#   # px, py, pz = x >> 1, y >> 1, z >> 1
#   return morton_xyz_f32(px, py, pz)

# def morton_children(code):
#   x, y, z = morton_decode(code)
#   return [
#     morton3D((x << 1) + dx, (y << 1) + dy, (z << 1) + dz)
#     for dx in [0, 1]
#     for dy in [0, 1]
#     for dz in [0, 1]
#   ]

# def morton_neighbors(code):
#   x, y, z = morton_decode(code)
#   return [
#     morton3D(x + dx, y + dy, z + dz)
#     for dx in [-1, 0, 1]
#     for dy in [-1, 0, 1]
#     for dz in [-1, 0, 1]
#     if not (dx == dy == dz == 0)
#   ]


class Plane:
  """Plane"""
  def __init__(
    self,
    normal,
    point = None,
    dist = None,
  ):
    self.normal = normal
    if point is not None:
      self.dist = float(point @ self.normal)
      self.point = point
    elif dist is not None:
      self.dist = dist
      n = self.normal / np.linalg.norm(self.normal)
      self.point = -self.dist * n

  def vector(self):
    """Plane coefficients as a vector (nx, ny, nz, d)"""
    return np.array([self.normal[0], self.normal[1], self.normal[2], self.dist])

  def transform(self, T):
    """Transform plane"""
    x, y, z, d = np.transpose(np.linalg.inv(T)) @ self.vector()
    self.normal = np.array([x, y, z])

    length = np.linalg.norm(self.normal)
    self.normal = self.normal / length
    self.dist = d / length

  def distance(self, p):
    """Point to plane distance"""
    a, b, c = self.normal
    d = self.dist
    x, y, z = p
    return a * x + b * y + c * z - d

  def get_transform(self):
    """Plane homogeneous transform"""
    world_up = np.array([0, 1, 0])
    z_axis = self.normal / np.linalg.norm(self.normal)
    x_axis = np.cross(z_axis, world_up)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    p = self.point

    T = eye(4, 4)
    T[0:3, 0] = x_axis
    T[0:3, 1] = y_axis
    T[0:3, 2] = z_axis
    T[0:3, 3] = p

    return T

  def plot(
      self,
      ax,
      color="r",
      xrange=np.linspace(-1.0, 1.0, 10),
      yrange=np.linspace(-1.0, 1.0, 10),
  ):
    """Plot the plane"""
    xx, yy = np.meshgrid(xrange, yrange)
    a, b, c, d = self.vector()
    zz = (d - a * xx - b * yy) / c
    ax.plot_surface(
      xx,
      yy,
      zz,
      alpha=0.5,
      rstride=100,
      cstride=100,
      color=color,
    )


class Frustum:
  """Frustum"""
  def __init__(
    self,
    hfov,
    aspect,
    znear,
    zfar,
    frustum_pose = None,
  ):
    self.hfov = hfov
    self.aspect = aspect
    self.znear = znear
    self.zfar = zfar

    wnear = 2.0 * tan(np.deg2rad(hfov) / 2.0) * znear
    hnear = wnear * (1.0 / aspect)
    wfar = 2.0 * tan(np.deg2rad(hfov) / 2.0) * zfar
    hfar = wfar * (1.0 / aspect)

    front = np.array([0, 0, -1])
    right = np.array([1, 0, 0])
    up = np.array([0, 1, 0])
    cam_pos = np.array([0, 0, 0])

    nc = cam_pos + front * znear
    self.ntl = nc + (up * hnear / 2.0) - (right * wnear / 2.0)
    self.ntr = nc + (up * hnear / 2.0) + (right * wnear / 2.0)
    self.nbl = nc - (up * hnear / 2.0) - (right * wnear / 2.0)
    self.nbr = nc - (up * hnear / 2.0) + (right * wnear / 2.0)

    fc = cam_pos + front * zfar
    self.ftl = fc + (up * hfar / 2.0) - (right * wfar / 2.0)
    self.ftr = fc + (up * hfar / 2.0) + (right * wfar / 2.0)
    self.fbl = fc - (up * hfar / 2.0) - (right * wfar / 2.0)
    self.fbr = fc - (up * hfar / 2.0) + (right * wfar / 2.0)

    # Points on left, right, top and bottom
    p_left = (nc - right * wnear / 2.0) - cam_pos
    p_right = (nc + right * wnear / 2.0) - cam_pos
    p_top = (nc + up * hnear / 2.0) - cam_pos
    p_bottom = (nc - up * hnear / 2.0) - cam_pos

    # Form left, right, top and bottom normals using the cross product
    normal_left = np.cross(normalize(p_left), up)
    normal_right = np.cross(up, normalize(p_right))
    normal_top = np.cross(-right, normalize(p_top))
    normal_bottom = np.cross(right, normalize(p_bottom))

    # OpenGL Frustum
    self.near = Plane(normal=front, point=nc)
    self.far = Plane(normal=-front, point=fc)
    self.left = Plane(normal=normal_left, point=p_left)
    self.right = Plane(normal=normal_right, point=p_right)
    self.top = Plane(normal=normal_top, point=p_top)
    self.bottom = Plane(normal=normal_bottom, point=p_bottom)

    if frustum_pose is not None:
      self.near.transform(frustum_pose)
      self.far.transform(frustum_pose)
      self.left.transform(frustum_pose)
      self.right.transform(frustum_pose)
      self.top.transform(frustum_pose)
      self.bottom.transform(frustum_pose)

  def plot(
    self,
    ax,
    points=None,
    plot_planes = False,
    plot_plane_frames = False,
  ):
    """Plot Frustum"""
    # Plot planes
    if plot_planes:
      self.near.plot(ax, color="r")
      self.far.plot(ax, color="g")
      self.left.plot(ax, color="r")
      self.right.plot(ax, color="g")
      self.top.plot(ax, color="r")
      self.bottom.plot(ax, color="g")

    # Plot plane frames
    if plot_plane_frames:
      T_near = self.near.get_transform()
      T_far = self.far.get_transform()
      T_left = self.left.get_transform()
      T_right = self.right.get_transform()
      T_top = self.top.get_transform()
      T_bottom = self.bottom.get_transform()

      plot_tf(ax, T_near)
      plot_tf(ax, T_far)
      plot_tf(ax, T_left)
      plot_tf(ax, T_right)
      plot_tf(ax, T_top)
      plot_tf(ax, T_bottom)

    # Plot points
    if points is not None:
      inside = []
      outside = []
      for p in points:
        if (self.near.distance(p) >= 0 and self.far.distance(p) >= 0 and
            self.left.distance(p) >= 0 and self.right.distance(p) >= 0 and
            self.top.distance(p) >= 0 and self.bottom.distance(p) >= 0):
          inside.append(p)
        else:
          outside.append(p)
      inside = np.array(inside)
      outside = np.array(outside)

      if inside.shape[0]:
        ax.scatter(
          inside[:, 0],
          inside[:, 1],
          inside[:, 2],
          c="g",
          alpha=0.2,
          label="inside",
        )
      if outside.shape[0]:
        ax.scatter(
          outside[:, 0],
          outside[:, 1],
          outside[:, 2],
          c="r",
          alpha=0.2,
          label="outside",
        )

    # Plot near plane
    near_points = [self.ntl, self.nbl, self.nbr, self.ntr]
    for i in range(4):
      p1 = near_points[i - 1]
      p2 = near_points[i]
      ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k-")

    # Plot far plane
    far_points = [self.ftl, self.fbl, self.fbr, self.ftr]
    for i in range(4):
      p1 = far_points[i - 1]
      p2 = far_points[i]
      ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k-")

    # Plot corner lines
    corner_pairs = [
      (self.ntl, self.ftl),
      (self.ntr, self.ftr),
      (self.nbl, self.fbl),
      (self.nbr, self.fbr),
    ]
    for p1, p2 in corner_pairs:
      ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        [p1[2], p2[2]],
        "k-",
      )
    ax.legend(loc=0)


# class Ray:
#   """3D Ray"""
#   def __init__(self, origin, dir):
#     self.origin = origin
#     self.dir = dir
#     self.invdir = 1.0 / dir
#     self.sign = [
#       (self.invdir[0] < 0),
#       (self.invdir[1] < 0),
#       (self.invdir[2] < 0),
#     ]


class OctreeNode:
  """Octree Node"""
  def __init__(self, center, size, depth, max_depth):
    self.center = center
    self.size = size
    self.depth = depth
    self.max_depth = max_depth
    self.children = [None for _ in range(8)]
    self.data = []

    vmin = center - size / 2.0
    vmax = center + size / 2.0
    self.bounds = [vmin, vmax]

  def insert(self, point):
    if self.depth == self.max_depth:
      self.data.append(point)
      return

    index = 0
    for i in range(3):
      if point[i] < self.center[i]:
        index |= 1 << i

    half_size = self.size / 2.0
    quarter_size = self.size / 4.0
    offset_x = (-1)**((index >> 0) & 1) * quarter_size
    offset_y = (-1)**((index >> 1) & 1) * quarter_size
    offset_z = (-1)**((index >> 2) & 1) * quarter_size

    child = self.children[index]
    if child is None:
      new_center = self.center + np.array([offset_x, offset_y, offset_z])
      child = OctreeNode(
        center=new_center,
        size=half_size,
        depth=self.depth + 1,
        max_depth=self.max_depth,
      )
      self.children[index] = child  # pyright: ignore
    self.children[index].insert(point)  # pyright: ignore

  def intersect(self, r):
    # Check intersect in x-y
    tx_min = (self.bounds[0 - r.sign[0]][0] - r.origin[0]) * r.invdir[0]
    tx_max = (self.bounds[1 - r.sign[0]][0] - r.origin[0]) * r.invdir[0]
    ty_min = (self.bounds[0 - r.sign[1]][1] - r.origin[1]) * r.invdir[1]
    ty_max = (self.bounds[1 - r.sign[1]][1] - r.origin[1]) * r.invdir[1]
    if (tx_min > ty_max) or (ty_min > tx_max):
      return (False, -1)

    if ty_min > tx_min:
      tx_min = ty_min
    if ty_max < tx_max:
      tx_max = ty_max

    # Check intersect in z
    tz_min = (self.bounds[0 - r.sign[2]][2] - r.origin[2]) * r.invdir[2]
    tz_max = (self.bounds[1 - r.sign[2]][2] - r.origin[2]) * r.invdir[2]
    if (tx_min > tz_max) or (tz_min > tx_max):
      return (False, -1)

    if tz_min > tx_min:
      tx_min = tz_min
    if tz_max < tx_max:
      tx_max = tz_max

    # Form results
    if tx_min < 0 and tx_max < 0:
      return (False, -1)
    elif tx_min < 0:
      t = tx_min
    else:
      t = tx_max

    return (True, t)


class Octree:
  """Octree"""
  def __init__(self, points, max_depth=3):
    self.center = np.array([0.0, 0.0, 0.0])
    self.size = 2.0
    self.root = OctreeNode(self.center, self.size, 0, max_depth)
    for point in points:
      self.root.insert(point)

  def get_points_and_bboxes(self, node, points_list, bboxes_list):
    # Get points
    if node.data:
      points_list.extend(node.data)

    # Get bounding boxes
    bboxes_list.append((node.center, node.size))

    # DFS get points and bboxes
    for child in node.children:
      if child:
        self.get_points_and_bboxes(child, points_list, bboxes_list)


class TestPlane(unittest.TestCase):
  """Test Plane"""
  def test_plane(self):
    # Define the coefficients of the plane
    # ax + by + cz = d
    d = 1.0
    normal = np.array([0, 0, 1])  # Example normal vector (a, b, c)
    a, b, c = normal

    # Create a grid of x, y values
    x = np.linspace(-10, 10, 10)
    y = np.linspace(-10, 10, 10)
    x, y = np.meshgrid(x, y)

    # Calculate corresponding z values
    z = (d - a * x - b * y) / c

    debug = False
    if debug:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection="3d")

      # Plot the surface
      ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100)

      # Set labels
      ax.set_xlabel("X axis")
      ax.set_ylabel("Y axis")
      ax.set_zlabel("Z axis")

      # Show the plot
      plt.show()


class TestFrustum(unittest.TestCase):
  """Test Frustum"""
  def test_frustum(self):
    # C_WC = euler321(-pi / 2.0, 0.0, -pi / 2.0)
    C_WC = euler321(0.0, 0.0, 0.0)
    r_WC = np.array([0.0, 0.0, 0.0])
    T_WC = tf(C_WC, r_WC)

    hfov = 60.0
    aspect = 1.0
    frustum = Frustum(
      hfov=hfov,
      aspect=aspect,
      znear=0.1,
      zfar=5.0,
      frustum_pose=T_WC,
    )
    points = np.random.uniform(-6.0, 6.0, (500, 3))

    # Visualize
    debug = False
    if debug:
      figsize = (10, 10)
      fig = plt.figure(figsize=figsize)
      ax = fig.add_subplot(111, projection="3d")
      plot_tf(ax, T_WC, size=1.0)
      frustum.plot(ax, points=points)

      plot_set_axes_equal(ax)
      ax.set_xlabel("X axis")
      ax.set_ylabel("Y axis")
      ax.set_zlabel("Z axis")
      plt.show()


class TestOctree(unittest.TestCase):
  """Test Octree"""
  def test_octree(self):
    points = [np.random.rand(3) for _ in range(100)]
    center = [0.0, 0.0, 0.0]
    size = 100.0
    octree = Octree(points)

    octree_points = []
    octree_bboxes = []
    octree.get_points_and_bboxes(octree.root, octree_points, octree_bboxes)

    # Visualize
    debug = False
    if debug:
      fig = plt.figure()
      ax = fig.add_subplot(111, projection="3d")

      # -- Plot bounding boxes
      for center, size in octree_bboxes:
        plot_bbox(ax, center, [size, size, size])

      # -- Plot points
      for p in octree_points:
        ax.plot(p[0], p[1], p[2], "r.")

      plt.show()
