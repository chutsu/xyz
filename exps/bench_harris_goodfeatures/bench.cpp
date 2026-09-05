// Benchmark xyz's image_harris() and image_good_features() against OpenCV's
// cornerHarris() and goodFeaturesToTrack() on the same synthetic image.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "bench_xyz.h"

using Clock = std::chrono::steady_clock;

static double ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

// Synthetic 8-bit grayscale image with many corner-like features (a grid of
// bright squares over a darker background plus noise), so detectors find plenty
// of keypoints to work on.
static cv::Mat make_image(int w, int h) {
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

struct Res {
  int count_xyz;
  int count_cv;
  double recall_xyz;  // fraction of xyz corners within radius of an OpenCV one
  double recall_cv;   // fraction of OpenCV corners within radius of a xyz one
};

static Res compute_res(const bkp_t *kps, int count,
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
  double r_cv = !corners_cv.empty() ? (double) hit_cv / corners_cv.size() : 1.0;
  return {count, static_cast<int>(corners_cv.size()), r_xyz, r_cv};
}

static Res run_harris(const cv::Mat &img, int block_size, float sigma,
                      float threshold, double *t_xyz, double *t_cv) {
  const int w = img.cols, h = img.rows;

  // OpenCV's Harris-with-keypoints is goodFeaturesToTrack(useHarrisDetector).
  std::vector<cv::Point2f> corners_cv;
  {
    auto t0 = Clock::now();
    cv::goodFeaturesToTrack(img, corners_cv, 1000, 0.01, 10, cv::noArray(),
                            block_size, true, 0.04);
    auto t1 = Clock::now();
    *t_cv = ms(t0, t1);
  }

  bkp_t *kps = nullptr;
  int count = 0;
  auto t0 = Clock::now();
  bench_xyz_harris(img.data, w, h, 0.04f, block_size, sigma, threshold, &kps,
                   &count);
  auto t1 = Clock::now();
  *t_xyz = ms(t0, t1);

  Res r = compute_res(kps, count, corners_cv);
  std::free(kps);
  return r;
}

static Res run_good_features(const cv::Mat &img, int block_size, float sigma,
                             float threshold, double *t_xyz, double *t_cv) {
  const int w = img.cols, h = img.rows;

  std::vector<cv::Point2f> corners_cv;
  {
    auto t0 = Clock::now();
    cv::goodFeaturesToTrack(img, corners_cv, 1000, 0.01, 10, cv::noArray(),
                            block_size, false, 0.04);
    auto t1 = Clock::now();
    *t_cv = ms(t0, t1);
  }

  bkp_t *kps = nullptr;
  int count = 0;
  auto t0 = Clock::now();
  bench_xyz_good_features(img.data, w, h, block_size, sigma, threshold, &kps,
                          &count);
  auto t1 = Clock::now();
  *t_xyz = ms(t0, t1);

  Res r = compute_res(kps, count, corners_cv);
  std::free(kps);
  return r;
}

int main(int argc, char **argv) {
  int w = 640, h = 480;
  if (argc >= 3) {
    w = std::atoi(argv[1]);
    h = std::atoi(argv[2]);
  }

  cv::Mat img = make_image(w, h);

  printf("Benchmark: xyz vs OpenCV corner detection (%dx%d)\n", w, h);
  printf("%-14s %9s %9s %8s  %6s  recall(recall)\n", "function", "xyz",
         "opencv", "speedup", "count");
  printf("%-14s %9s %9s %8s  %6s  ----------\n", "--------", "---", "------",
         "-------", "-----");

  struct Job {
    const char *name;
    int block;
    float sigma;
    float thresh;
    Res (*fn)(const cv::Mat &, int, float, float, double *, double *);
  };
  Job jobs[] = {
      {"harris", 3, 1.0f, 1e6f, run_harris},
      {"good_features", 3, 1.0f, 1e4f, run_good_features},
  };

  // Warm up (also forces OpenCV's internal threads/tables to initialize so the
  // first timed call is not inflated by lazy init).
  for (auto &j : jobs) {
    double a = 0, b = 0;
    j.fn(img, j.block, j.sigma, j.thresh, &a, &b);
  }

  for (auto &j : jobs) {
    // Run several times and report the minimum timing for each implementation
    // (least noisy), using the last result for the recall/count check.
    double tx = 1e18, tc = 1e18;
    Res r{0, 0, 0.0, 0.0};
    for (int i = 0; i < 5; i++) {
      double a = 0, b = 0;
      Res ri = j.fn(img, j.block, j.sigma, j.thresh, &a, &b);
      if (a < tx)
        tx = a;
      if (b < tc)
        tc = b;
      r = ri;
    }
    // Format: count(xyz/cv) and recall(recall_xyz/recall_cv).
    printf("%-14s %8.3f ms %8.3f ms %7.2fx  %d/%d  %.2f/%.2f\n", j.name, tx, tc,
           tc / tx, r.count_xyz, r.count_cv, r.recall_xyz, r.recall_cv);
  }

  return 0;
}
