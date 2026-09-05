#include "bench_xyz.h"

#include <stdlib.h>
#include <string.h>

#include "../../src/xyz.h"

/* Wraps xyz's image_harris so it can be called from the C++ benchmark main.
 * xyz.h has no extern "C" guards, so these functions wrap it and re-expose the
 * keypoint results through a plain-C struct. */
void bench_xyz_harris(const unsigned char *gray,
                      int w,
                      int h,
                      float k,
                      int block_size,
                      float sigma,
                      float threshold,
                      bkp_t **out,
                      int *out_count) {
  image_t img;
  img.width = w;
  img.height = h;
  img.channels = 1;
  img.data = (uint8_t *) gray;

  keypoint_t *kps = NULL;
  int count = 0;
  image_harris(&img, k, block_size, sigma, threshold, &kps, &count);

  bkp_t *buf = malloc(sizeof(bkp_t) * (size_t) (count > 0 ? count : 1));
  for (int i = 0; i < count; i++) {
    buf[i].x = kps[i].x;
    buf[i].y = kps[i].y;
    buf[i].score = kps[i].score;
  }
  free(kps);
  *out = buf;
  *out_count = count;
}

void bench_xyz_good_features(const unsigned char *gray,
                             int w,
                             int h,
                             int block_size,
                             float sigma,
                             float threshold,
                             bkp_t **out,
                             int *out_count) {
  image_t img;
  img.width = w;
  img.height = h;
  img.channels = 1;
  img.data = (uint8_t *) gray;

  keypoint_t *kps = NULL;
  int count = 0;
  image_good_features(&img, block_size, sigma, threshold, &kps, &count);

  bkp_t *buf = malloc(sizeof(bkp_t) * (size_t) (count > 0 ? count : 1));
  for (int i = 0; i < count; i++) {
    buf[i].x = kps[i].x;
    buf[i].y = kps[i].y;
    buf[i].score = kps[i].score;
  }
  free(kps);
  *out = buf;
  *out_count = count;
}

void bench_xyz_gaussian_blur(const unsigned char *gray,
                             int w,
                             int h,
                             unsigned char *out) {
  image_t img;
  img.width = w;
  img.height = h;
  img.channels = 1;
  img.data = (uint8_t *) gray;

  image_t *res = image_gaussian_blur(&img, 7, 1.0f);
  memcpy(out, res->data, (size_t) w * h);
  image_free(res);
}
