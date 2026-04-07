import scipy
import numpy as np


def convolve2d(image, kernel):
  """Convolve 2D image with kernel"""
  # f is an image and is indexed by (v, w)
  # kernel is a filter kernel and is indexed by (s, t),
  #   it needs odd dimensions
  # h is the output image and is indexed by (x, y),
  #   it is not cropped
  if kernel.shape[0] % 2 != 1 or kernel.shape[1] % 2 != 1:
    raise ValueError("Only odd dimensions on filter supported")

  # smid and tmid are number of pixels between the center pixel
  # and the edge, ie for a 5x5 filter they will be 2.
  #
  # The output size is calculated by adding smid, tmid to each
  # side of the dimensions of the input image.
  vmax = image.shape[0]
  wmax = image.shape[1]
  smax = kernel.shape[0]
  tmax = kernel.shape[1]
  smid = smax // 2
  tmid = tmax // 2
  xmax = vmax + 2 * smid
  ymax = wmax + 2 * tmid

  # Allocate result image.
  out = np.zeros((xmax, ymax), dtype=image.dtype)

  # Do convolution
  for x in range(xmax):
    for y in range(ymax):
      # Calculate pixel value for out at (x,y). Sum one component
      # for each pixel (s, t) of the kernel filter.
      s_from = max(smid - x, -smid)
      s_to = min((xmax - x) - smid, smid + 1)
      t_from = max(tmid - y, -tmid)
      t_to = min((ymax - y) - tmid, tmid + 1)
      value = 0
      for s in range(s_from, s_to):
        for t in range(t_from, t_to):
          v = x - smid + s
          w = y - tmid + t
          value += kernel[smid - s, tmid - t] * image[v, w]
      out[x, y] = value

  return out


def harris_corner(image_gray, **kwargs):
  """Harris Corner Detector

  For educational purposes only, this implementation is slower than OpenCV's.

  """
  assert len(image_gray.shape) == 2  # Ensure image is 1 channel (grayscale)
  assert image_gray.dtype == "uint8"
  k = kwargs.get("k", 0.05)
  radius = kwargs.get("radius", 5)
  min_dist = kwargs.get("min_dist", 10)

  # Apply Sobel filter find image gradients in x and y directions
  img = image_gray / 255.0
  sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  Ix = scipy.signal.convolve2d(img, sobel_x, mode="same")
  Iy = scipy.signal.convolve2d(img, sobel_y, mode="same")

  # Compute element-wise product of gradients and apply Gaussian filter
  gauss_kern = 1.0 / 16.0 * np.array([[0, 2, 0], [2, 4, 2], [0, 2, 0]])
  Ixx = scipy.signal.convolve2d(Ix * Ix, gauss_kern, mode="same")
  Ixy = scipy.signal.convolve2d(Ix * Iy, gauss_kern, mode="same")
  Iyy = scipy.signal.convolve2d(Iy * Iy, gauss_kern, mode="same")

  # Calculate Harris corner response
  detM = Ixx * Iyy - Ixy**2
  traceM = Ixx + Iyy
  R = detM - k * traceM**2

  # Extract corners
  corners = []
  image_h, image_w = image_gray.shape

  for i, R_row in enumerate(R):
    for j, r in enumerate(R_row):
      # Check pixel is not too close to image boundary
      x_ok = i > radius and i < (image_h - radius)
      y_ok = j > radius and j < (image_w - radius)
      if not x_ok or not y_ok:
        continue

      # Region is a corner
      if r > 0:
        corners.append([i, j, r])

  # Sort corners by responses
  corners = sorted(corners, key=lambda x: x[2], reverse=True)

  # Make sure corners are N pixels apart
  mask = np.zeros((image_h, image_w))
  filtered_corners = []
  offset = int(min_dist / 2)
  for corner in corners:
    cx, cy, _ = corner

    row_start = max(0, cy - offset)
    col_start = max(0, cx - offset)
    row_end = min(image_h, cy + offset)
    col_end = min(image_w, cx + offset)

    occuppied = False
    for i in range(row_start, row_end):
      for j in range(col_start, col_end):
        if mask[i, j] == 1:
          occuppied = True
          break

    if occuppied is False:
      mask[row_start:row_end, col_start:col_end] = 1
      filtered_corners.append((cx, cy))

  return filtered_corners


def shi_tomasi_corner(image_gray, **kwargs):
  """Shi-Tomasi Corner Detector

  For educational purposes only, this implementation is slower than OpenCV's.

  """
  assert len(image_gray.shape) == 2  # Ensure image is 1 channel (grayscale)
  assert image_gray.dtype == "uint8"
  from scipy.signal import convolve2d

  radius = kwargs.get("radius", 5)
  min_dist = kwargs.get("min_dist", 10)
  thresh = 0.1
  offset = 0

  # Apply Sobel filter find image gradients in x and y directions
  img = image_gray / 255.0
  img_h, img_w = image_gray.shape
  sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
  Ix = convolve2d(img, sobel_x, mode="same")
  Iy = convolve2d(img, sobel_y, mode="same")

  # Compute element-wise product of gradients and apply Gaussian filter
  gauss_kern = 1.0 / 16.0 * np.array([[0, 2, 0], [2, 4, 2], [0, 2, 0]])
  Ixx = convolve2d(Ix * Ix, gauss_kern, mode="same")
  Iyy = convolve2d(Iy * Iy, gauss_kern, mode="same")

  # Extract corners
  corners = []
  for i in range(offset, img_h - offset):
    for j in range(offset, img_w - offset):
      # Check pixel is not too close to image boundary
      x_ok = i > radius and i < (img_h - radius)
      y_ok = j > radius and j < (img_w - radius)
      if not x_ok or not y_ok:
        continue

      # Calculate sum of squares
      Sxx = Ixx[i - offset:i + offset + 1, j - offset:j + offset + 1].sum()
      Syy = Iyy[i - offset:i + offset + 1, j - offset:j + offset + 1].sum()
      r = min(Sxx, Syy)

      # Threshold for corner
      if r > thresh:
        corners.append((i, j, r))

  # Sort corners by responses
  corners = sorted(corners, key=lambda x: x[2], reverse=True)

  # Make sure corners are N pixels apart
  mask = np.zeros((img_h, img_w))
  filtered_corners = []
  offset = int(min_dist / 2)
  for corner in corners:
    cx, cy, _ = corner

    row_start = max(0, cy - offset)
    col_start = max(0, cx - offset)
    row_end = min(img_h, cy + offset)
    col_end = min(img_w, cx + offset)

    occuppied = False
    for i in range(row_start, row_end):
      for j in range(col_start, col_end):
        if mask[i, j] == 1:
          occuppied = True
          break

    if occuppied is False:
      mask[row_start:row_end, col_start:col_end] = 1
      filtered_corners.append((cx, cy))

  return filtered_corners


  # @unittest.skip("Fix Me!")
  # def test_harris_corner(self):
  #   """Test harris_corner()"""
  #   img_file = "./test_data/images/checker_board-5x5.png"
  #   img_path = os.path.join(SCRIPT_DIR, img_file)
  #   img  = cv2.imread(img_path)
  #   img_gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #   corners = harris_corner(img_gray)
  #
  #   assert img is not None
  #   for corner in corners:
  #     x = int(corner[0])
  #     y = int(corner[1])
  #     img[x, y] = [0, 0, 255]
  #
  #   debug = False
  #   if debug:
  #     cv2.imshow("Image", img)
  #     cv2.waitKey(0)
  #
  #   self.assertTrue(len(corners))
  #
  # @unittest.skip("Fix Me!")
  # def test_shi_tomasi_corner(self):
  #   """Test shi_tomasi_corner()"""
  #   img_file = "./test_data/images/checker_board-5x5.png"
  #   img_path = os.path.join(SCRIPT_DIR, img_file)
  #   img = cv2.imread(img_path)
  #   assert img is not None
  #
  #   img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #   corners = shi_tomasi_corner(img_gray)
  #   for corner in corners:
  #     x, y = corner
  #     img[x, y] = [0, 0, 255]
  #
  #   # print(f"num corners: {len(corners)}")
  #   debug = False
  #   if debug:
  #     cv2.imshow("Image", img)
  #     cv2.waitKey(0)
  #
  #   self.assertTrue(len(corners))
