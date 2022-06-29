#!/usr/bin/env python3
## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.
## Modified by Sotiris Papatheodorou in 2019

#####################################################
## librealsense tutorial #1 - Accessing depth data ##
#####################################################

# First import the library
import pyrealsense2 as rs

print("Depth measurements up to 1 meter from the sensor")

try:
    # Create a context object. This object owns the handles to all connected
    # realsense devices.
    pipeline = rs.pipeline()
    pipeline.start()

    # Move the cursor down to make space for printing the frame.
    print("\033[25B\033[0D")

    while True:
        # This call waits until a new coherent set of frames is available on a
        # device. Calls to get_frame_data(...) and get_frame_timestamp(...) on
        # a device will return stable values until wait_for_frames(...) is
        # called.
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        if not depth: continue

        # The ASCII representation of a single camera depth frame.
        frame = ""

        # Print a simple text-based representation of the image, by breaking it
        # into 10x20 pixel regions and approximating the coverage of pixels
        # within one meter.
        coverage = [0] * 64
        for y in range(480):
            for x in range(640):
                dist = depth.get_distance(x, y)
                if 0 < dist and dist < 1:
                    coverage[x // 10] += 1

            if y % 20 is 19:
                line = ""
                for c in coverage:
                    line += " .:nhBXWW"[c // 25]
                coverage = [0] * 64
                frame += line + "\n"

        # Print the current frame and use ANSI escape sequences to move the
        # cursor back up.
        print("\033[26A\033[0D")
        print(frame)
    exit(0)
#except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
except Exception as e:
    print(e)
pass
