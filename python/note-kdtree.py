import unittest
import numpy as np
import matplotlib.pyplot as plt


class KdNode:
  def __init__(self, point, k, left=None, right=None):
    self.point = point
    self.k = k
    self.left = left
    self.right = right


def kdtree_build(points, depth=0):
  if points is None or len(points) == 0:
    return None

  kdim = len(points[0])
  axis = depth % kdim
  sorted_points = sorted(points, key=lambda p: p[axis])
  median_index = len(sorted_points) // 2
  median_point = sorted_points[median_index]

  node = KdNode(median_point, axis)
  node.left = kdtree_build(sorted_points[:median_index], depth + 1)
  node.right = kdtree_build(sorted_points[median_index + 1:], depth + 1)
  return node


def kdtree_nn(root, target):
  best = [None, float("inf")]  # [best_point, best_dist]

  def search(node, depth):
    if node is None:
      return

    # Calculate distance and keep track of best
    dist = np.linalg.norm(target - node.point)
    if dist < best[1]:
      best[0] = node.point
      best[1] = dist

    # Determine which side to search first
    axis = node.k
    diff = target[axis] - node.point[axis]

    # Search the closer subtree first
    if diff <= 0:
      closer, farther = (node.left, node.right)
    else:
      closer, farther = (node.right, node.left)
    search(closer, depth + 1)

    # Search the farther subtree
    if abs(diff) < best[1]:
      search(farther, depth + 1)

  # Search
  search(root, 0)

  return (best[0], best[1])


class TestKDTree(unittest.TestCase):
  """Test KDTree"""
  def test_kdtree(self):
    points = np.array([
      [1.0, 2.0],
      [3.0, 5.0],
      [4.0, 2.0],
      [7.0, 8.0],
      [8.0, 1.0],
      [9.0, 6.0],
    ])

    target_point = [5.0, 3.0]
    kdtree = kdtree_build(points)
    best_point, _ = kdtree_nn(kdtree, target_point)

    debug = False
    if debug:
      plt.plot(points[:, 0], points[:, 1], "b.")
      plt.plot(target_point[0], target_point[1], "ko")
      plt.plot(best_point[0], best_point[1], "rx")
      plt.show()
