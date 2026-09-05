#ifndef BENCH_XYZ_H
#define BENCH_XYZ_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bkp_t {
  int x;
  int y;
  float score;
} bkp_t;

void bench_xyz_harris(const unsigned char *gray,
                      int w,
                      int h,
                      float k,
                      int block_size,
                      float sigma,
                      float threshold,
                      bkp_t **out,
                      int *out_count);

void bench_xyz_good_features(const unsigned char *gray,
                             int w,
                             int h,
                             int block_size,
                             float sigma,
                             float threshold,
                             bkp_t **out,
                             int *out_count);

void bench_xyz_gaussian_blur(const unsigned char *gray,
                             int w,
                             int h,
                             unsigned char *out);

#ifdef __cplusplus
}
#endif

#endif /* BENCH_XYZ_H */
