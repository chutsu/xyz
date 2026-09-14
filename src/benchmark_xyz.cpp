#include "xyz.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

/*******************************************************************************
 * CONVOLUTION: image_convolve() vs image_convolution_fast() vs
 * image_convolution_fast2()
 ******************************************************************************/

static void benchmark_convolve(void) {
  // EuRoC MH_01-sized synthetic image
  const int width = 752;
  const int height = 480;
  image_t *img = image_malloc(width, height, 1);
  for (int i = 0; i < width * height; i++) {
    img->data[i] = (uint8_t) (i % 256);
  }

  const int sizes[] = {3, 5, 7, 9, 15, 21};
  const int num_sizes = (int) (sizeof(sizes) / sizeof(sizes[0]));

  printf("Benchmark: image_convolve() vs image_convolution_fast() vs "
         "image_convolution_fast2() (%dx%d)\n",
         width,
         height);
  for (int s = 0; s < num_sizes; s++) {
    const int size = sizes[s];
    const int radius = size / 2;
    const float sigma = (float) size / 6.0f;
    const float s2 = 2.0f * sigma * sigma;

    // 1D Gaussian kernel
    float kernel_1d[64];
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
      const float val = expf(-(float) (i * i) / s2);
      kernel_1d[i + radius] = val;
      sum += val;
    }
    for (int i = 0; i < size; i++) {
      kernel_1d[i] /= sum;
    }

    // Equivalent full 2D kernel (outer product)
    float *kernel_2d = (float *) malloc(sizeof(float) * size * size);
    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        kernel_2d[y * size + x] = kernel_1d[y] * kernel_1d[x];
      }
    }

    tic();
    image_t *slow = image_convolve(img, kernel_2d, size, size);
    const double slow_time = toc();

    tic();
    image_t *fast =
        image_convolution_fast(img, kernel_1d, size, kernel_1d, size);
    const double fast_time = toc();

    tic();
    image_t *fast2 =
        image_convolution_fast2(img, kernel_1d, size, kernel_1d, size);
    const double fast2_time = toc();

    printf("kernel %2dx%-2d | image_convolve: %6.4f s | "
           "image_convolution_fast: %6.4f s (%5.2fx) | "
           "image_convolution_fast2: %6.4f s (%5.2fx)\n",
           size,
           size,
           slow_time,
           fast_time,
           slow_time / fast_time,
           fast2_time,
           slow_time / fast2_time);

    image_free(slow);
    image_free(fast);
    image_free(fast2);
    free(kernel_2d);
  }

  image_free(img);
}

/*******************************************************************************
 * CORNER DETECTION: image_harris() / image_good_features() vs OpenCV
 ******************************************************************************/

// Synthetic 8-bit grayscale image with many corner-like features (a grid of
// bright squares over a darker background plus noise), so detectors find
// plenty of keypoints to work on.
static cv::Mat make_corner_image(int w, int h) {
  cv::Mat img(h, w, CV_8UC1);
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> noise(-12, 12);
  const int cell = 32;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      bool on = ((x / cell) + (y / cell)) % 2 == 0;
      int v = on ? 200 : 60;
      v += noise(rng);
      if (v < 0)
        v = 0;
      if (v > 255)
        v = 255;
      img.at<unsigned char>(y, x) = static_cast<unsigned char>(v);
    }
  }
  return img;
}

struct CornerRes {
  int count_xyz;
  int count_cv;
  double recall_xyz; // fraction of xyz corners within radius of an OpenCV one
  double recall_cv;  // fraction of OpenCV corners within radius of a xyz one
};

static CornerRes compute_recall(const keypoint_t *kps,
                                int count,
                                const std::vector<cv::Point2f> &corners_cv) {
  int hit_xyz = 0;
  for (int i = 0; i < count; i++) {
    for (const auto &c : corners_cv) {
      if (std::abs(c.x - kps[i].x) <= 4 && std::abs(c.y - kps[i].y) <= 4) {
        hit_xyz++;
        break;
      }
    }
  }
  int hit_cv = 0;
  for (const auto &c : corners_cv) {
    for (int i = 0; i < count; i++) {
      if (std::abs(c.x - kps[i].x) <= 4 && std::abs(c.y - kps[i].y) <= 4) {
        hit_cv++;
        break;
      }
    }
  }
  double r_xyz = count > 0 ? (double) hit_xyz / count : 1.0;
  double r_cv =
      !corners_cv.empty() ? (double) hit_cv / corners_cv.size() : 1.0;
  return {count, static_cast<int>(corners_cv.size()), r_xyz, r_cv};
}

static image_t make_xyz_view(const cv::Mat &img) {
  image_t view;
  view.width = img.cols;
  view.height = img.rows;
  view.channels = 1;
  view.data = (uint8_t *) img.data;
  return view;
}

static CornerRes run_harris(const cv::Mat &img,
                            int block_size,
                            float sigma,
                            float threshold,
                            double *t_xyz,
                            double *t_cv) {
  // OpenCV's Harris-with-keypoints is goodFeaturesToTrack(useHarrisDetector).
  std::vector<cv::Point2f> corners_cv;
  {
    auto t0 = Clock::now();
    cv::goodFeaturesToTrack(
        img, corners_cv, 1000, 0.01, 10, cv::noArray(), block_size, true, 0.04);
    auto t1 = Clock::now();
    *t_cv = ms(t0, t1);
  }

  image_t view = make_xyz_view(img);
  keypoint_t *kps = NULL;
  int count = 0;
  auto t0 = Clock::now();
  image_harris(&view, 0.04f, block_size, sigma, threshold, &kps, &count);
  auto t1 = Clock::now();
  *t_xyz = ms(t0, t1);

  CornerRes r = compute_recall(kps, count, corners_cv);
  free(kps);
  return r;
}

static CornerRes run_good_features(const cv::Mat &img,
                                   int block_size,
                                   float sigma,
                                   float threshold,
                                   double *t_xyz,
                                   double *t_cv) {
  std::vector<cv::Point2f> corners_cv;
  {
    auto t0 = Clock::now();
    cv::goodFeaturesToTrack(img,
                            corners_cv,
                            1000,
                            0.01,
                            10,
                            cv::noArray(),
                            block_size,
                            false,
                            0.04);
    auto t1 = Clock::now();
    *t_cv = ms(t0, t1);
  }

  image_t view = make_xyz_view(img);
  keypoint_t *kps = NULL;
  int count = 0;
  auto t0 = Clock::now();
  image_good_features(&view, block_size, sigma, threshold, &kps, &count);
  auto t1 = Clock::now();
  *t_xyz = ms(t0, t1);

  CornerRes r = compute_recall(kps, count, corners_cv);
  free(kps);
  return r;
}

static void benchmark_corners(int w, int h) {
  cv::Mat img = make_corner_image(w, h);

  printf("\nBenchmark: xyz vs OpenCV corner detection (%dx%d)\n", w, h);
  printf("%-14s %9s %9s %8s  %6s  recall(recall)\n",
         "function",
         "xyz",
         "opencv",
         "speedup",
         "count");
  printf("%-14s %9s %9s %8s  %6s  ----------\n",
         "--------",
         "---",
         "------",
         "-------",
         "-----");

  struct Job {
    const char *name;
    int block;
    float sigma;
    float thresh;
    CornerRes (*fn)(const cv::Mat &, int, float, float, double *, double *);
  };
  Job jobs[] = {
      {"harris", 3, 1.0f, 1e6f, run_harris},
      {"good_features", 3, 1.0f, 1e4f, run_good_features},
  };

  // Warm up (also forces OpenCV's internal threads/tables to initialize so
  // the first timed call is not inflated by lazy init).
  for (auto &j : jobs) {
    double a = 0, b = 0;
    j.fn(img, j.block, j.sigma, j.thresh, &a, &b);
  }

  for (auto &j : jobs) {
    // Run several times and report the minimum timing for each
    // implementation (least noisy), using the last result for the
    // recall/count check.
    double tx = 1e18, tc = 1e18;
    CornerRes r{0, 0, 0.0, 0.0};
    for (int i = 0; i < 5; i++) {
      double a = 0, b = 0;
      CornerRes ri = j.fn(img, j.block, j.sigma, j.thresh, &a, &b);
      if (a < tx)
        tx = a;
      if (b < tc)
        tc = b;
      r = ri;
    }
    // Format: count(xyz/cv) and recall(recall_xyz/recall_cv).
    printf("%-14s %8.3f ms %8.3f ms %7.2fx  %d/%d  %.2f/%.2f\n",
           j.name,
           tx,
           tc,
           tc / tx,
           r.count_xyz,
           r.count_cv,
           r.recall_xyz,
           r.recall_cv);
  }
}

/*******************************************************************************
 * OPTICAL FLOW: lk_track() / lk_track2() vs OpenCV calcOpticalFlowPyrLK()
 ******************************************************************************/

typedef void (*lk_track_fn_t)(const image_t *,
                              const image_t *,
                              const keypoint_t *,
                              const int,
                              const int,
                              const float,
                              keypoint_t *,
                              int *);

struct FlowRes {
  int tracked_xyz;
  int both_tracked; // tracked by both xyz and OpenCV
  double mean_err;  // mean pixel distance vs OpenCV, for points both tracked
  double max_err;
};

// Runs one xyz lk_track-family function and scores it against OpenCV's
// already-computed track (cv_next/cv_status) for the same keypoints.
static FlowRes run_lk_track(lk_track_fn_t fn,
                            const image_t *img0,
                            const image_t *img1,
                            const keypoint_t *kps,
                            int kps_count,
                            const std::vector<cv::Point2f> &cv_next,
                            const std::vector<uchar> &cv_status,
                            double *t_xyz) {
  keypoint_t *kp_out = (keypoint_t *) malloc(sizeof(keypoint_t) * kps_count);
  int *status = (int *) malloc(sizeof(int) * kps_count);

  auto t0 = Clock::now();
  fn(img0, img1, kps, kps_count, 3, 1.0f, kp_out, status);
  auto t1 = Clock::now();
  *t_xyz = ms(t0, t1);

  FlowRes r{};
  double err_sum = 0.0;
  for (int i = 0; i < kps_count; i++) {
    if (status[i]) {
      r.tracked_xyz++;
    }
    if (status[i] && cv_status[i]) {
      double ex = kp_out[i].x - cv_next[i].x;
      double ey = kp_out[i].y - cv_next[i].y;
      double e = std::sqrt(ex * ex + ey * ey);
      err_sum += e;
      if (e > r.max_err) {
        r.max_err = e;
      }
      r.both_tracked++;
    }
  }
  r.mean_err = r.both_tracked > 0 ? err_sum / r.both_tracked : 0.0;

  free(kp_out);
  free(status);
  return r;
}

static void benchmark_lk_track(void) {
  const char *data_path = "/data/euroc/MH_01";
  euroc_data_t *test_data = euroc_data_load(data_path);
  euroc_camera_t *cam0_data = test_data->cam0_data;

  image_t *img0 = image_load(cam0_data->image_paths[0]);
  image_t *img1 = image_load(cam0_data->image_paths[1]);

  keypoint_t *kps;
  int kps_count;
  image_good_features(img0, 3, 1.0f, 1e4f, &kps, &kps_count);

  cv::Mat cv_img0(img0->height, img0->width, CV_8UC1, img0->data);
  cv::Mat cv_img1(img1->height, img1->width, CV_8UC1, img1->data);

  std::vector<cv::Point2f> cv_prev(kps_count);
  for (int i = 0; i < kps_count; i++) {
    cv_prev[i] = cv::Point2f((float) kps[i].x, (float) kps[i].y);
  }

  // xyz uses window_size=21, 3 pyramid levels (maxLevel=2), 20 iters @ 0.01px
  std::vector<cv::Point2f> cv_next;
  std::vector<uchar> cv_status;
  std::vector<float> cv_err;
  cv::TermCriteria criteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                            20,
                            0.01);

  // Warm up, then time each implementation over several runs and keep the
  // minimum (least noisy); OpenCV's tracked positions are the reference the
  // xyz implementations are scored against.
  cv::calcOpticalFlowPyrLK(cv_img0,
                           cv_img1,
                           cv_prev,
                           cv_next,
                           cv_status,
                           cv_err,
                           cv::Size(21, 21),
                           2,
                           criteria);

  double t_cv = 1e18;
  for (int i = 0; i < 5; i++) {
    auto t0 = Clock::now();
    cv::calcOpticalFlowPyrLK(cv_img0,
                             cv_img1,
                             cv_prev,
                             cv_next,
                             cv_status,
                             cv_err,
                             cv::Size(21, 21),
                             2,
                             criteria);
    auto t1 = Clock::now();
    const double t = ms(t0, t1);
    if (t < t_cv) {
      t_cv = t;
    }
  }
  const int tracked_cv = (int) cv::countNonZero(cv_status);

  double t_xyz = 1e18;
  FlowRes r1{};
  for (int i = 0; i < 5; i++) {
    double t;
    FlowRes r =
        run_lk_track(lk_track, img0, img1, kps, kps_count, cv_next, cv_status, &t);
    if (t < t_xyz) {
      t_xyz = t;
      r1 = r;
    }
  }

  double t_xyz2 = 1e18;
  FlowRes r2{};
  for (int i = 0; i < 5; i++) {
    double t;
    FlowRes r = run_lk_track(
        lk_track2, img0, img1, kps, kps_count, cv_next, cv_status, &t);
    if (t < t_xyz2) {
      t_xyz2 = t;
      r2 = r;
    }
  }

  printf("\nBenchmark: lk_track() / lk_track2() vs OpenCV "
         "calcOpticalFlowPyrLK() (%dx%d, %d keypoints)\n",
         img0->width,
         img0->height,
         kps_count);
  printf("%-10s %11s %8s  %9s  %11s  %9s\n",
         "function",
         "time",
         "speedup",
         "tracked",
         "agree(cv)",
         "mean err");
  printf("%-10s %11s %8s  %9s  %11s  %9s\n",
         "--------",
         "----",
         "-------",
         "-------",
         "---------",
         "--------");
  printf("%-10s %8.3f ms %8s  %4d/%-4d  %11s  %9s\n",
         "opencv",
         t_cv,
         "-",
         tracked_cv,
         kps_count,
         "-",
         "-");
  printf("%-10s %8.3f ms %7.2fx  %4d/%-4d  %5d/%-5d  %7.3f px\n",
         "lk_track",
         t_xyz,
         t_cv / t_xyz,
         r1.tracked_xyz,
         kps_count,
         r1.both_tracked,
         r1.tracked_xyz,
         r1.mean_err);
  printf("%-10s %8.3f ms %7.2fx  %4d/%-4d  %5d/%-5d  %7.3f px\n",
         "lk_track2",
         t_xyz2,
         t_cv / t_xyz2,
         r2.tracked_xyz,
         kps_count,
         r2.both_tracked,
         r2.tracked_xyz,
         r2.mean_err);

  free(kps);
  image_free(img0);
  image_free(img1);
  euroc_data_free(test_data);
}

int main(int argc, char **argv) {
  benchmark_convolve();

  int w = 640, h = 480;
  if (argc >= 3) {
    w = std::atoi(argv[1]);
    h = std::atoi(argv[2]);
  }
  benchmark_corners(w, h);

  benchmark_lk_track();

  return 0;
}
