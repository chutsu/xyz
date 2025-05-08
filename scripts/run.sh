#!/bin/bash
set -e

ctags src/*.c src/*.h
# ctags xyz/xyz.py

run_gdb() {
  gdb \
    -ex=run \
    -ex=bt \
    -ex="set confirm off" \
    -ex=quit \
    --args "$1" "$2" "$3"
}

run_memcheck() {
  valgrind --leak-check=full $1 $2 $3
}

###############################################################################
# PYTHON
###############################################################################

# python3 scripts/plot_frames.py

# python3 src/xyz.py
# python3 src/xyz.py TestNetwork.test_http_parse_request
# python3 src/xyz.py TestNetwork.test_websocket_hash
# python3 src/xyz.py TestNetwork.test_websocket_encode_frame
# python3 src/xyz.py TestNetwork.test_debug_server
# python3 src/xyz.py TestLinearAlgebra
# python3 src/xyz.py TestLie
# python3 src/xyz.py TestLie.test_sandbox
# python3 src/xyz.py TestTransform
# python3 src/xyz.py TestTransform.test_quat2rot
# python3 src/xyz.py TestTransform.test_rot2quat
# python3 src/xyz.py TestTransform.test_rot2euler
# python3 src/xyz.py TestTransform.test_quat_inv
# python3 src/xyz.py TestTransform.test_quat_conj
# python3 src/xyz.py TestTransform.test_quat_slerp
# python3 src/xyz.py TestMav
# python3 src/xyz.py TestMav.test_symdiff_velocity
# python3 src/xyz.py TestMav.test_plot
# python3 src/xyz.py TestMav.test_mav_attitude_control
# python3 src/xyz.py TestMav.test_mav_velocity_control
# python3 src/xyz.py TestMav.test_mav_position_control
# python3 src/xyz.py TestMav.test_mav_trajectory_control
# python3 src/xyz.py TestCV
# python3 src/xyz.py TestCV.test_linear_triangulation
# python3 src/xyz.py TestCV.test_parallax
# python3 src/xyz.py TestCV.test_homography_find
# python3 src/xyz.py TestCV.test_homography_pose
# python3 src/xyz.py TestCV.test_dlt_pose
# python3 src/xyz.py TestCV.test_solvepnp
# python3 src/xyz.py TestCV.test_harris_corner
# python3 src/xyz.py TestCV.test_shi_tomasi_corner
# python3 src/xyz.py TestPointCloud
# python3 src/xyz.py TestOctree
# python3 src/xyz.py TestOctree.test_octree
# python3 src/xyz.py TestOctree.test_point_plane
# python3 src/xyz.py TestKDTree
# python3 src/xyz.py TestFrustum
# python3 src/xyz.py TestFrustum.test_frustum
# python3 src/xyz.py TestFrustum.test_livox
# python3 src/xyz.py TestPoseFactor
# python3 src/xyz.py TestBAFactor
# python3 src/xyz.py TestVisionFactor
# python3 src/xyz.py TestCameraFactor
# python3 src/xyz.py TestCalibVisionFactor
# python3 src/xyz.py TestTwoStateVisionFactor
# python3 src/xyz.py TestCalibGimbalFactor
# python3 src/xyz.py TestIMUFactor
# python3 src/xyz.py TestIMUFactor.test_imu_buffer_with_interpolation
# python3 src/xyz.py TestIMUFactor.test_imu_factor_propagate
# python3 src/xyz.py TestIMUFactor.test_imu_factor
# python3 src/xyz.py TestIMUFactor.test_imu_propagation_jacobians
# python3 src/xyz.py TestIMUFactor.test_imu_factor2_propagate
# python3 src/xyz.py TestIMUFactor.test_imu_factor2
# python3 src/xyz.py TestMargFactor
# python3 src/xyz.py TestFactorGraph
# python3 src/xyz.py TestFactorGraph.test_factor_graph_add_param
# python3 src/xyz.py TestFactorGraph.test_factor_graph_add_factor
# python3 src/xyz.py TestFactorGraph.test_factor_graph_solve_vo
# python3 src/xyz.py TestFactorGraph.test_factor_graph_solve_io
# python3 src/xyz.py TestFactorGraph.test_factor_graph_solve_vio
# python3 src/xyz.py TestFeatureTracking
# python3 src/xyz.py TestFeatureTracking.test_feature_grid_cell_index
# python3 src/xyz.py TestFeatureTracking.test_feature_grid_count
# python3 src/xyz.py TestFeatureTracking.test_spread_keypoints
# python3 src/xyz.py TestFeatureTracking.test_grid_detect
# python3 src/xyz.py TestFeatureTracking.test_good_grid
# python3 src/xyz.py TestFeatureTracking.test_optflow_track
# python3 src/xyz.py TestFeatureTracking.test_feature_track
# python3 src/xyz.py TestFeatureTracking.test_estimate_pose
# python3 src/xyz.py TestFeatureTracking.test_euroc_mono
# python3 src/xyz.py TestFeatureTracking.test_euroc
# python3 src/xyz.py TestCalibration
# python3 src/xyz.py TestCalibration.test_aprilgrid
# python3 src/xyz.py TestCalibration.test_calibrator
# python3 src/xyz.py TestEuroc
# python3 src/xyz.py TestKitti
# python3 src/xyz.py TestKalmanFilter
# python3 src/xyz.py TestOctree
# python3 src/xyz.py TestOctree.test_octree
# python3 src/xyz.py TestSimulation
# python3 src/xyz.py TestSimulation.test_create_3d_features
# python3 src/xyz.py TestSimulation.test_create_3d_features_perimeter
# python3 src/xyz.py TestSimulation.test_sim_camera_frame
# python3 src/xyz.py TestSimulation.test_sim_data
# python3 src/xyz.py TestSimulation.test_sim_feature_tracker
# python3 src/xyz.py TestSimulation.test_sim_arm
# python3 src/xyz.py TestViz.test_multiplot
# python3 src/xyz.py TestViz.test_server
# python3 src/xyz.py TestSandbox.test_gimbal
# python3 src/xyz.py TestPoE.test_scene

# tmux send-keys -t dev -R C-l C-m
# tmux send-keys -t dev -R "python3 xyz.py TestOctree" C-m C-m
# exit

###############################################################################
# C
###############################################################################

run_all_tests() {
  tmux send-keys -t dev -R C-l C-m
  tmux send-keys -t dev -R "\
    cd ~/code/xyz/src \
      && clear \
      && make test_xyz -j \
  " C-m C-m
  exit
}

run_test() {
  TARGET="dev"
  tmux send-keys -t $TARGET -R C-l C-m
  tmux send-keys -t $TARGET -R "\
    cd ~/code/xyz \
      && clear \
      && make libxyz -j \
      && time make tests -j \
      && cd build \
      && ./$1 --target $2 \
      && cd ~/code/xyz
  " C-m C-m

#   tmux send-keys -t $TARGET -R "\
# python3 - <<EOF
# import numpy as np
# import matplotlib.pyplot as plt
#
# points = np.genfromtxt('/tmp/points.csv')
# points_out = np.genfromtxt('/tmp/points_downsampled.csv')
#
# fig = plt.figure(figsize=(12, 6))
# ax1 = fig.add_subplot(121, projection='3d')
# ax1.scatter(points[:, 0], points[:, 1], points[:, 2])
# ax2 = fig.add_subplot(122, projection='3d')
# ax2.scatter(points_out[:, 0], points_out[:, 1], points_out[:, 2])
#
# plt.tight_layout()
# plt.show()
# EOF
#   " C-m C-m

  exit
}

# CMD="cd ~/code/xyz && make libxyz -j"
# CMD="cd ~/code/xyz && make ci"
# CMD="cd ~/code/xyz && make tests -j"
# CMD="cd ~/code/xyz && make clean && make libxyz -j"
# tmux send-keys -t dev -R "q" C-m C-l
# tmux send-keys -t dev -R "${CMD}" C-m C-m
# exit


# XYZ
# run_all_tests
## XYZ-MACROS
# run_test test_xyz test_median_value
# run_test test_xyz test_mean_value
## XYZ-TIME
# run_test test_xyz test_tic_toc
# run_test test_xyz test_mtoc
# run_test test_xyz test_time_now
## XYZ-DARRAY
# run_test test_xyz test_darray_new_and_destroy
# run_test test_xyz test_darray_push_pop
# run_test test_xyz test_darray_contains
# run_test test_xyz test_darray_copy
# run_test test_xyz test_darray_new_element
# run_test test_xyz test_darray_set_and_get
# run_test test_xyz test_darray_update
# run_test test_xyz test_darray_remove
# run_test test_xyz test_darray_expand_and_contract
## XYZ-LIST
# run_test test_xyz test_list_malloc_and_free
# run_test test_xyz test_list_push_pop
# run_test test_xyz test_list_shift
# run_test test_xyz test_list_unshift
# run_test test_xyz test_list_remove
# run_test test_xyz test_list_remove_destroy
## XYZ-RED-BLACK-TREE
# run_test test_xyz test_rbt_node_malloc_and_free
# run_test test_xyz test_rbt_node_min_max
# run_test test_xyz test_rbt_node_height_size
# run_test test_xyz test_rbt_node_keys
# run_test test_xyz test_rbt_node_flip_colors
# run_test test_xyz test_rbt_node_rotate
# run_test test_xyz test_rbt_node_move_red_left
# run_test test_xyz test_rbt_node_move_red_right
# run_test test_xyz test_rbt_node_insert
# run_test test_xyz test_rbt_node_delete
# run_test test_xyz test_rbt_malloc_and_free
# run_test test_xyz test_rbt_insert
# run_test test_xyz test_rbt_delete
# run_test test_xyz test_rbt_search
# run_test test_xyz test_rbt_contains
# run_test test_xyz test_rbt_min_max
# run_test test_xyz test_rbt_keys
# run_test test_xyz test_rbt_rank
# run_test test_xyz test_rbt_select
# run_test test_xyz test_rbt_sandbox
## XYZ-HASHMAP
# run_test test_xyz test_hm_malloc_and_free
# run_test test_xyz test_hm_set_and_get
## XYZ-NETWORK
# run_test test_xyz test_tcp_server_setup
## XYZ-MATH
# run_test test_xyz test_min
# run_test test_xyz test_max
# run_test test_xyz test_randf
# run_test test_xyz test_deg2rad
# run_test test_xyz test_rad2deg
# run_test test_xyz test_wrap_180
# run_test test_xyz test_wrap_360
# run_test test_xyz test_wrap_pi
# run_test test_xyz test_wrap_2pi
# run_test test_xyz test_fltcmp
# run_test test_xyz test_fltcmp2
# run_test test_xyz test_cumsum
# run_test test_xyz test_logspace
# run_test test_xyz test_pythag
# run_test test_xyz test_lerp
# run_test test_xyz test_lerp3
# run_test test_xyz test_sinc
# run_test test_xyz test_mean
# run_test test_xyz test_median
# run_test test_xyz test_var
# run_test test_xyz test_stddev
## XYZ-LINEAR ALGEBRA
# run_test test_xyz test_eye
# run_test test_xyz test_ones
# run_test test_xyz test_zeros
# run_test test_xyz test_mat_set
# run_test test_xyz test_mat_val
# run_test test_xyz test_mat_copy
# run_test test_xyz test_mat_row_set
# run_test test_xyz test_mat_col_set
# run_test test_xyz test_mat_block_get
# run_test test_xyz test_mat_block_set
# run_test test_xyz test_mat_diag_get
# run_test test_xyz test_mat_diag_set
# run_test test_xyz test_mat_triu
# run_test test_xyz test_mat_tril
# run_test test_xyz test_mat_trace
# run_test test_xyz test_mat_transpose
# run_test test_xyz test_mat_add
# run_test test_xyz test_mat_sub
# run_test test_xyz test_mat_scale
# run_test test_xyz test_vec_add
# run_test test_xyz test_vec_sub
# run_test test_xyz test_dot
# run_test test_xyz test_bdiag_inv
# run_test test_xyz test_hat
# run_test test_xyz test_check_jacobian
# run_test test_xyz test_svd
# run_test test_xyz test_pinv
# run_test test_xyz test_svd_det
# run_test test_xyz test_chol
# run_test test_xyz test_chol_solve
# run_test test_xyz test_qr
# run_test test_xyz test_eig_sym
# run_test test_xyz test_eig_inv
# run_test test_xyz test_schur_complement
## XYZ-SUITE-SPARSE
# run_test test_xyz test_suitesparse_chol_solve
## XYZ-TRANSFORMS
# run_test test_xyz test_tf_rot_set
# run_test test_xyz test_tf_trans_set
# run_test test_xyz test_tf_trans_get
# run_test test_xyz test_tf_rot_get
# run_test test_xyz test_tf_quat_get
# run_test test_xyz test_tf_inv
# run_test test_xyz test_tf_point
# run_test test_xyz test_tf_hpoint
# run_test test_xyz test_tf_perturb_rot
# run_test test_xyz test_tf_perturb_trans
# run_test test_xyz test_tf_chain
# run_test test_xyz test_euler321
# run_test test_xyz test_rot2quat
# run_test test_xyz test_quat2euler
# run_test test_xyz test_quat2rot
## XYZ-LIE
# run_test test_xyz test_lie_Exp_Log
## XYZ-GNUPLOT
# run_test test_xyz test_gnuplot_xyplot
# run_test test_xyz test_gnuplot_multiplot
## XYZ-CONTROL
# run_test test_xyz test_pid_ctrl
## XYZ-MAV
# run_test test_xyz test_mav_att_ctrl
# run_test test_xyz test_mav_vel_ctrl
# run_test test_xyz test_mav_pos_ctrl
# run_test test_xyz test_mav_waypoints
## XYZ-COMPUTER-VISION
# run_test test_xyz test_image_setup
# run_test test_xyz test_image_load
# run_test test_xyz test_image_print_properties
# run_test test_xyz test_image_free
# run_test test_xyz test_radtan4_distort
# run_test test_xyz test_radtan4_undistort
# run_test test_xyz test_radtan4_point_jacobian
# run_test test_xyz test_radtan4_params_jacobian
# run_test test_xyz test_equi4_distort
# run_test test_xyz test_equi4_undistort
# run_test test_xyz test_equi4_point_jacobian
# run_test test_xyz test_equi4_params_jacobian
# run_test test_xyz test_pinhole_focal
# run_test test_xyz test_pinhole_K
# run_test test_xyz test_pinhole_projection_matrix
# run_test test_xyz test_pinhole_project
# run_test test_xyz test_pinhole_point_jacobian
# run_test test_xyz test_pinhole_params_jacobian
# run_test test_xyz test_pinhole_radtan4_project
# run_test test_xyz test_pinhole_radtan4_project_jacobian
# run_test test_xyz test_pinhole_radtan4_params_jacobian
# run_test test_xyz test_pinhole_equi4_project
# run_test test_xyz test_pinhole_equi4_project_jacobian
# run_test test_xyz test_pinhole_equi4_params_jacobian
# run_test test_xyz test_linear_triangulation
# run_test test_xyz test_homography_find
# run_test test_xyz test_homography_pose
# run_test test_xyz test_p3p_kneip
# run_test test_xyz test_solvepnp
## XYZ-APRILGRID
# run_test test_xyz test_aprilgrid_malloc_and_free
# run_test test_xyz test_aprilgrid_center
# run_test test_xyz test_aprilgrid_grid_index
# run_test test_xyz test_aprilgrid_object_point
# run_test test_xyz test_aprilgrid_add_and_remove_corner
# run_test test_xyz test_aprilgrid_add_and_remove_tag
# run_test test_xyz test_aprilgrid_save_and_load
# run_test test_xyz test_aprilgrid_detector_detect
## XYZ-STATE-ESTIMATION
# run_test test_xyz test_pose
# run_test test_xyz test_extrinsics
# run_test test_xyz test_fiducial
# run_test test_xyz test_fiducial_buffer
# run_test test_xyz test_imu_biases
# run_test test_xyz test_feature
# run_test test_xyz test_features
# run_test test_xyz test_time_delay
# run_test test_xyz test_joint
# run_test test_xyz test_camera_params
# run_test test_xyz test_triangulate_batch
# run_test test_xyz test_pose_factor
# run_test test_xyz test_ba_factor
# run_test test_xyz test_camera_factor
# run_test test_xyz test_imu_buffer_setup
# run_test test_xyz test_imu_buffer_add
# run_test test_xyz test_imu_buffer_clear
# run_test test_xyz test_imu_buffer_copy
# run_test test_xyz test_imu_propagate
# run_test test_xyz test_imu_initial_attitude
# run_test test_xyz test_imu_factor_form_F_matrix
# run_test test_xyz test_imu_factor
# run_test test_xyz test_joint_factor
# run_test test_xyz test_marg_factor
## XYZ-TIMELINE
# run_test test_xyz test_timeline
## XYZ-OCTREE
# run_test test_xyz test_octree_node
# run_test test_xyz test_octree
# run_test test_xyz test_octree_points
# run_test test_xyz test_octree_downsample
## XYZ-KD-TREE
# run_test test_xyz test_sort
# run_test test_xyz test_kdtree_node
# run_test test_xyz test_kdtree
# run_test test_xyz test_kdtree_nn
## XYZ-SIMULATION
# run_test test_xyz test_sim_features_load
# run_test test_xyz test_sim_imu_data_load
# run_test test_xyz test_sim_camera_frame_load
# run_test test_xyz test_sim_camera_data_load
# run_test test_xyz test_sim_camera_circle_trajectory
## XYZ-EUROC
# run_test test_xyz test_euroc_imu_load
# run_test test_xyz test_euroc_camera_load
# run_test test_xyz test_euroc_ground_truth_load
# run_test test_xyz test_euroc_data_load
# run_test test_xyz test_euroc_calib_target_load
# run_test test_xyz test_euroc_calib_load
## XYZ-KITTI
# run_test test_xyz test_kitti_camera_load
# run_test test_xyz test_kitti_oxts_load
# run_test test_xyz test_kitti_velodyne_load
# run_test test_xyz test_kitti_calib_load
# run_test test_xyz test_kitti_raw_load




# XYZ-GUI
# run_test test_gui
# run_test test_gui test_gl_zeros
# run_test test_gui test_gl_ones
# run_test test_gui test_gl_eye
# run_test test_gui test_gl_matf_set
# run_test test_gui test_gl_matf_val
# run_test test_gui test_gl_transpose
# run_test test_gui test_gl_equals
# run_test test_gui test_gl_vec3_cross
# run_test test_gui test_gl_dot
# run_test test_gui test_gl_norm
# run_test test_gui test_gl_normalize
# run_test test_gui test_gl_perspective
# run_test test_gui test_gl_lookat
# run_test test_gui test_gl_shader_compile
# run_test test_gui test_gl_shaders_link
# run_test test_gui test_gl_prog_setup
# run_test test_gui test_gl_camera_setup
# run_test test_gui test_gui
# run_test test_gui test_components
# run_test test_imshow
# XYZ-VOXEL
# run_test test_voxel
