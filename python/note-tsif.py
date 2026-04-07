#!/usr/bin/env python3
import cv2

from xyz import EurocDataset
from xyz import CameraEvent
from xyz import good_grid


if __name__ == "__main__":
  data_path = "/data/euroc/V1_01"
  dataset = EurocDataset(data_path)

  timestamps = dataset.timeline.get_timestamps()
  for ts in timestamps:
    for event in dataset.timeline.get_events(ts):
      if isinstance(event, CameraEvent) and event.cam_idx == 0:
        image = dataset.get_camera_image(event.cam_idx, ts)

        # good_grid(
        #   image=image,
        #   max_keypoints= 2000,
        #   quality_level = 0.001,
        #   use_harris = True,
        #   min_dist = 20,
        #   grid_rows = 2,
        #   grid_cols = 3,
        #   prev_kps=[],
        #   debug = True,
        # )

        cv2.imshow("Image", image)
        cv2.waitKey(1)
