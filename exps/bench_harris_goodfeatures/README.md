# bench_harris_goodfeatures

Benchmarks `xyz`'s `image_harris()` and `image_good_features()` against
OpenCV's `goodFeaturesToTrack()` (Shi-Tomasi and Harris modes) on the same
synthetic grayscale image.

## Files

- `bench.cpp` — C++ main: generates a synthetic corner-rich image, times both
  libraries, compares detected corner counts and recall between the two.
- `bench_xyz.c` / `bench_xyz.h` — plain-C wrappers around the `xyz` API so the
  C++ main can call it (xyz.h has no `extern "C"` guards).
- `Makefile` — builds `xyz` in release mode, compiles the benchmark, runs it.

## Usage

```sh
make        # build and run (640x480)
make run    # run after building
make build  # build only
make clean  # remove build artifacts
```

Run at other resolutions:

```sh
make run   # or directly:
./../../build/bench_corners 1280 720
```

## Output

```
function      xyz    opencv  speedup  count   recall
harris        6.0 ms  2.5 ms   0.43x 3569/266  1.00/0.98
good_features 6.0 ms  4.7 ms   0.78x  266/726  1.00/1.00
```

- `count` = keypoints found (xyz / OpenCV).
- `recall` (xyz→OpenCV / OpenCV→xyz) = fraction of the source set whose corners
  fall within 4px of a corner in the other set.

Note: thresholds/blocksize are not directly mappable between the two libraries,
so corner counts and recall are indicative, not exact.
