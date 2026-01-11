#include "munit.h"
#include "xyz.h"
#include "xyz_gui.h"

void vis_scan(gl_points3d_t *gl_points,
              const gl_float_t *pcd,
              const size_t num_points) {
  gl_float_t point_size = 2.0;
  gl_float_t *points_data = malloc(sizeof(gl_float_t) * num_points * 6);

  for (size_t i = 0; i < num_points; ++i) {
    // Point positions
    // FLU -> OPENGL
    //   x -> -z
    //   y -> x
    //   z -> y
    points_data[i * 6 + 0] = pcd[i * 3 + 1];
    points_data[i * 6 + 1] = pcd[i * 3 + 2] + 1.6;
    points_data[i * 6 + 2] = -pcd[i * 3 + 0];

    // Point color - based on height
    gl_float_t r = 0.0f;
    gl_float_t g = 0.0f;
    gl_float_t b = 0.0f;
    gl_jet_colormap((pcd[i * 3 + 2] + 1.6) / 3.0, &r, &g, &b);
    points_data[i * 6 + 3] = r;
    points_data[i * 6 + 4] = g;
    points_data[i * 6 + 5] = b;
  }
  gl_points3d_update(gl_points, points_data, num_points, point_size);
  free(points_data);
}

void icp_jacobian(const float pose_est[4 * 4], float J[3 * 6]) {
  const float neye[3 * 3] = {-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0};
  TF_ROT(pose_est, R_est);

  // J = zeros((3, 6))
  // J[0:3, 0:3] = -1.0 * eye(3)
  // J[0:3, 3:6] = R_est @ hat(p_dst_est[:, i])
  // mat_block_set(J, 6, 0, 2, 0, 2, neye);
}

// void estimate_pose(const pcd_t *pcd_km1,
//                    const pcd_t *pcd_k,
//                    const float T_WB_km1[4 * 4],
//                    float T_WB_k[4 * 4]) {
//   assert(pcd_km1);
//   assert(pcd_k);
//   assert(T_WB_km1);
//   assert(T_WB_k);
//
//   // Setup
//   const float dist_threshold = 0.1;
//   const kdtree_t *kdtree = kdtree_malloc(pcd_km1->data, pcd_km1->num_points);
//   const size_t n = MIN(pcd_km1->num_points, pcd_k->num_points);
//   float *points_i = malloc(sizeof(float) * 3 * n);
//   float *points_j = malloc(sizeof(float) * 3 * n);
//   float *r = malloc(sizeof(float) * 3 * n);
//
//   // Find correspondences
//   size_t m = 0;
//   for (size_t j = 0; j < pcd_k->num_points; ++j) {
//     float best_point[3] = {0};
//     float best_dist = INFINITY;
//     kdtree_nn(kdtree, &pcd_k->data[j * 3], best_point, &best_dist);
//     if (best_dist < dist_threshold) {
//       points_i[m * 3 + 0] = best_point[0];
//       points_i[m * 3 + 1] = best_point[1];
//       points_i[m * 3 + 2] = best_point[2];
//
//       points_j[m * 3 + 0] = pcd_k->data[j * 3 + 0];
//       points_j[m * 3 + 1] = pcd_k->data[j * 3 + 1];
//       points_j[m * 3 + 2] = pcd_k->data[j * 3 + 2];
//       m++;
//     }
//   }
//
//   float *H = malloc(sizeof(float) * 6 * 6);
//   for (size_t idx = 0; idx < m; ++idx) {
//     r[idx * 3 + 0] = points_i[idx * 3 + 0] - points_j[idx * 3 + 0];
//     r[idx * 3 + 1] = points_i[idx * 3 + 1] - points_j[idx * 3 + 1];
//     r[idx * 3 + 2] = points_i[idx * 3 + 2] - points_j[idx * 3 + 2];
//
//     // Calculate jacobian
//
//     // Form hessian
//   }
//
//   // perform least squares
//
//   // update pose
//
//   return;
// }

pcd_t *kitti_lidar_pcd(const kitti_raw_t *kitti, const size_t index) {
  const char *pcd_path = kitti->velodyne->pcd_paths[index];
  const float voxel_size = 0.5;
  size_t num_points = 0;
  const float *points = kitti_lidar_xyz(pcd_path, voxel_size, &num_points);
  const timestamp_t ts_start = kitti->velodyne->timestamps_start[index];
  const timestamp_t ts_end = kitti->velodyne->timestamps_end[index];

  return pcd_malloc(ts_start, ts_end, points, NULL, num_points);
}

int test_icp(void) {
  // Kitti data
  const char *data_dir = "/data/kitti_raw/2011_09_26";
  const char *seq_name = "2011_09_26_drive_0001_sync";
  kitti_raw_t *kitti = kitti_raw_load(data_dir, seq_name);

  pcd_t *pcd0 = kitti_lidar_pcd(kitti, 0);
  pcd_t *pcd1 = kitti_lidar_pcd(kitti, 10);

  tic();

  real_t scale_est[1] = {0};
  real_t R_est[3 * 3] = {0};
  real_t t_est[3] = {0};
  kdtree_data_t *nn_data =
      kdtree_nns(pcd0->kdtree, pcd1->data, pcd1->num_points);
  umeyama(nn_data->points,
          pcd1->data,
          pcd1->num_points,
          scale_est,
          R_est,
          t_est);
  free(nn_data->points);
  free(nn_data);

  printf("\n");
  printf("scale: %f\n", scale_est[0]);
  print_matrix("R_est", R_est, 3, 3);
  print_vector("t_est", t_est, 3);
  printf("\n");

  {
    FILE *fp = fopen("/tmp/pcd0.csv", "w");
    for (int i = 0; i < pcd0->num_points; i++) {
      fprintf(fp, "%f ", pcd0->data[i * 3 + 0]);
      fprintf(fp, "%f ", pcd0->data[i * 3 + 1]);
      fprintf(fp, "%f\n", pcd0->data[i * 3 + 2]);
    }
    fclose(fp);
  }

  {
    real_t T[4 * 4] = {0};
    real_t T_inv[4 * 4] = {0};
    tf_cr(R_est, t_est, T);
    tf_inv(T, T_inv);

    FILE *fp = fopen("/tmp/pcd1.csv", "w");
    for (int i = 0; i < pcd1->num_points; i++) {
      // p_dst = R * p + t
      const real_t px = pcd1->data[i * 3 + 0];
      const real_t py = pcd1->data[i * 3 + 1];
      const real_t pz = pcd1->data[i * 3 + 2];

      real_t p[3] = {px, py, pz};
      real_t p_dst[3] = {0};
      tf_point(T_inv, p, p_dst);

      fprintf(fp, "%f ", p_dst[0]);
      fprintf(fp, "%f ", p_dst[1]);
      fprintf(fp, "%f\n", p_dst[2]);
    }
    fclose(fp);
  }

  // printf("\n");
  // printf("time taken: %f\n", toc());

  // Clean up
  pcd_free(pcd0);
  pcd_free(pcd1);
  kitti_raw_free(kitti);

  return 0;
}

int test_kitti(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Grid
  gl_int_t grid_rows = 10;
  gl_int_t grid_cols = 10;
  gl_float_t grid_size = 0.5f;
  gl_float_t grid_lw = 1.0f;
  gl_color_t grid_color = (gl_color_t){1.0, 1.0, 1.0};
  gl_grid3d_t *gl_grid =
      gl_grid3d_malloc(grid_rows, grid_cols, grid_size, grid_color, grid_lw);

  // KITTI
  const char *data_dir = "/data/kitti_raw/2011_09_26";
  const char *seq_name = "2011_09_26_drive_0001_sync";
  kitti_raw_t *kitti = kitti_raw_load(data_dir, seq_name);
  gl_points3d_t *gl_points = gl_points3d_malloc(NULL, 0, 1.0);

  int pcd_index = 0;
  double time_prev = gui_time();
  pcd_t *pcd_km1 = NULL;
  pcd_t *pcd_k = NULL;
  float T_WB_km1[4 * 4] = {0};
  float T_WB_k[4 * 4] = {0};

  while (gui_poll(gui)) {
    double time_dt = gui_time() - time_prev;
    if (pcd_index >= kitti->velodyne->num_timestamps) {
      break;
    }

    // if (*gui->key_n || time_dt > 0.1) {
    if (*gui->key_n) {
      printf("[pcd_index]: %d\n", pcd_index);

      // Load lidar points
      const timestamp_t ts_start = kitti->velodyne->timestamps_start[pcd_index];
      const timestamp_t ts_end = kitti->velodyne->timestamps_end[pcd_index];
      const char *pcd_path = kitti->velodyne->pcd_paths[pcd_index];
      const float voxel_size = 0.5;
      size_t num_points = 0;
      float *points = kitti_lidar_xyz(pcd_path, voxel_size, &num_points);
      float *time_diffs = malloc(sizeof(float) * num_points);
      for (int i = 0; i < num_points; ++i) {
        time_diffs[i] = 0.0;
      }
      pcd_k = pcd_malloc(ts_start, ts_end, points, time_diffs, num_points);

      // Estimate
      // estimate_pose(pcd_km1, pcd_k, T_WB_km1, T_WB_k);
      // -- Update point cloud
      pcd_free(pcd_km1);
      pcd_km1 = pcd_k;
      pcd_k = NULL;

      // Visualize
      vis_scan(gl_points, points, num_points);
      time_prev = gui_time();

      // Clean up
      free(points);
      free(time_diffs);

      // Update
      pcd_index++;
    }

    draw_grid3d(gl_grid);
    draw_points3d(gl_points);
    gui_update(gui);
  }

  // Clean up
  pcd_free(pcd_km1);
  pcd_free(pcd_k);
  kitti_raw_free(kitti);
  gl_points3d_free(gl_points);
  gl_grid3d_free(gl_grid);
  gui_free(gui);

  return 0;
}

void test_suite(void) {
  MU_ADD_TEST(test_icp);
  MU_ADD_TEST(test_kitti);
}

MU_RUN_TESTS(test_suite)
