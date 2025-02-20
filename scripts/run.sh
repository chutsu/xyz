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
      && time make tests -j \
      && cd build && ./$1 --target $2 \
  " C-m C-m
  exit
}

# tmux send-keys -t dev -R C-l C-m
# tmux send-keys -t dev -R "cd ~/code/xyz && make libxyz -j" C-m
# tmux send-keys -t dev -R "cd ~/code/xyz && make ci" C-m
# exit

# XYZ
# run_all_tests
# run_test test_xyz
# run_test test_xyz test_tic
# run_test test_xyz test_toc
# run_test test_xyz test_mtoc
# run_test test_xyz test_time_now
# run_test test_xyz test_debug
# run_test test_xyz test_log_error
# run_test test_xyz test_log_warn
# run_test test_xyz test_path_file_name
# run_test test_xyz test_path_file_ext
# run_test test_xyz test_path_dir_name
# run_test test_xyz test_path_join
# run_test test_xyz test_list_files
# run_test test_xyz test_list_files_free
# run_test test_xyz test_file_read
# run_test test_xyz test_file_copy
# run_test test_xyz test_malloc_string
# run_test test_xyz test_dsv_rows
# run_test test_xyz test_dsv_cols
# run_test test_xyz test_dsv_fields
# run_test test_xyz test_dsv_data
# run_test test_xyz test_dsv_free
# run_test test_xyz test_tcp_server_setup
# run_test test_xyz test_http_parse_request
# run_test test_xyz test_websocket_hash
# run_test test_xyz test_ws_handshake_respond
# run_test test_xyz test_ws_server
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
# run_test test_xyz test_eye
# run_test test_xyz test_ones
# run_test test_xyz test_xyzs
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
# run_test test_xyz test_skew
# run_test test_xyz test_check_jacobian
# run_test test_xyz test_svd
# run_test test_xyz test_pinv
# run_test test_xyz test_svd_det
# run_test test_xyz test_eig_sym
# run_test test_xyz test_eig_inv
# run_test test_xyz test_chol
# run_test test_xyz test_chol_solve
# run_test test_xyz test_qr
# run_test test_xyz test_suitesparse_chol_solve
# run_test test_xyz test_tf_set_rot
# run_test test_xyz test_tf_set_trans
# run_test test_xyz test_tf_trans
# run_test test_xyz test_tf_rot
# run_test test_xyz test_tf_quat
# run_test test_xyz test_tf_inv
# run_test test_xyz test_tf_point
# run_test test_xyz test_tf_hpoint
# run_test test_xyz test_tf_perturb_rot
# run_test test_xyz test_tf_perturb_trans
# run_test test_xyz test_tf_chain
# run_test test_xyz test_quat2rot
# run_test test_xyz test_pose_init
# run_test test_xyz test_pose_set_get_quat
# run_test test_xyz test_pose_set_get_trans
# run_test test_xyz test_pose2tf
# run_test test_xyz test_load_poses
# XYZ - DATA STRUCTURE
# run_test test_ds
# run_test test_ds test_darray_new_and_destroy
# run_test test_ds test_darray_push_pop
# run_test test_ds test_darray_contains
# run_test test_ds test_darray_copy
# run_test test_ds test_darray_new_element
# run_test test_ds test_darray_set_and_get
# run_test test_ds test_darray_update
# run_test test_ds test_darray_remove
# run_test test_ds test_darray_expand_and_contract
# run_test test_ds test_list_new_and_destroy
# run_test test_ds test_list_push_pop
# run_test test_ds test_list_shift
# run_test test_ds test_list_unshift
# run_test test_ds test_list_remove
# run_test test_ds test_list_remove_destroy
# run_test test_ds test_stack_new_and_destroy
# run_test test_ds test_stack_push
# run_test test_ds test_stack_pop
# run_test test_ds test_queue_new_and_destroy
# run_test test_ds test_queue_enqueue_dequeue
# run_test test_ds test_hashmap_new_destroy
# run_test test_ds test_hashmap_clear_destroy
# run_test test_ds test_hashmap_get_set
# run_test test_ds test_hashmap_delete
# run_test test_ds test_hashmap_traverse
# XYZ-CV
# run_test test_linear_triangulation
# run_test test_homography_find
# run_test test_homography_pose
# run_test test_p3p_kneip
# run_test test_solvepnp
# run_test test_radtan4_distort
# run_test test_radtan4_undistort
# run_test test_radtan4_point_jacobian
# run_test test_radtan4_params_jacobian
# run_test test_equi4_distort
# run_test test_equi4_undistort
# run_test test_equi4_point_jacobian
# run_test test_equi4_params_jacobian
# run_test test_pinhole_focal
# run_test test_pinhole_K
# run_test test_pinhole_project
# run_test test_pinhole_projection_matrix
# run_test test_pinhole_point_jacobian
# run_test test_pinhole_params_jacobian
# run_test test_pinhole_radtan4_project
# run_test test_pinhole_radtan4_project_jacobian
# run_test test_pinhole_radtan4_params_jacobian
# run_test test_pinhole_equi4_project
# run_test test_pinhole_equi4_project_jacobian
# run_test test_pinhole_equi4_params_jacobian
# XYZ-SIM
# memcheck run_test test_load_sim_features
# memcheck run_test test_load_sim_imu_data
# memcheck run_test test_load_sim_cam_frame
# memcheck run_test test_load_sim_cam_data
# XYZ-CONTROL
# run_test test_control test_pid_ctrl
# XYZ-GIMBAL
# run_test test_gimbal test_gimbal
# XYZ-MAV
# run_test test_mav test_mav_att_ctrl
# run_test test_mav test_mav_vel_ctrl
# run_test test_mav test_mav_pos_ctrl
# run_test test_mav test_mav_waypoints
# XYZ - STATE ESTIMATION
# run_test test_se test_schur_complement
# run_test test_se test_timeline
# run_test test_se test_pose
# run_test test_se test_speed_bias
# run_test test_se test_extrinsics
# run_test test_se test_fiducial
# run_test test_se test_fiducial_buffer
# run_test test_se test_camera
# run_test test_se test_triangulation_batch
# run_test test_se test_feature
# run_test test_se test_idf
# run_test test_se test_features
# run_test test_se test_pose_factor
# run_test test_se test_ba_factor
# run_test test_se test_camera_factor
# run_test test_se test_idf_factor
# run_test test_se test_imu_buf_setup
# run_test test_se test_imu_buf_add
# run_test test_se test_imu_buf_clear
# run_test test_se test_imu_buf_copy
# run_test test_se test_imu_buf_print
# run_test test_se test_imu_propagate
# run_test test_se test_imu_initial_attitude
# run_test test_se test_imu_factor_form_F_matrix
# run_test test_se test_imu_factor
# run_test test_se test_joint_angle_factor
# run_test test_se test_calib_camera_factor
# run_test test_se test_calib_imucam_factor
# run_test test_se test_calib_gimbal_factor
# run_test test_se test_marg
# run_test test_se test_visual_odometry_batch
# run_test test_se test_inertial_odometry_batch
# run_test test_se test_visual_inertial_odometry_batch
# run_test test_se test_tsf
# run_test test_se test_se test_assoc_pose_data
# run_test test_se test_ceres_example
# run_test test_se test_invert_block_diagonal
# run_test test_se test_solver_setup
# run_test test_se test_solver_print
# run_test test_se test_solver_eval
# XYZ - CALIBRATION
run_test test_calib test_camchain
# run_test test_calib test_calib_camera_mono_batch
# run_test test_calib_camera_mono_ceres
# run_test test_calib_camera_mono_incremental
# run_test test_calib_camera_stereo_batch
# run_test test_calib_camera_stereo_ceres
# run_test test_calib_imucam_view
# run_test test_calib_imucam_add_imu
# run_test test_calib_imucam_add_camera
# run_test test_calib_imucam_add_imu_event
# run_test test_calib_imucam_add_fiducial_event
# run_test test_calib_imucam_update
# run_test test_calib_imucam_batch
# run_test test_calib_imucam_batch_ceres
# XYZ-GNUPLOT
# run_test test_gnuplot test_gnuplot_xyplot
# run_test test_gnuplot test_gnuplot_multiplot
# XYZ-KITTI
# run_test test_kitti test_kitti_raw_load
# XYZ-SIM
# run_test test_sim_features_load
# run_test test_sim_imu_data_load
# run_test test_sim_camera_frame_load
# run_test test_sim_camera_data_load
# run_test test_sim_camera_circle_trajectory
# run_test test_sim_gimbal_malloc_free
# run_test test_sim_gimbal_view
# run_test test_sim_gimbal_solve
# XYZ-GUI
# run_test test_gl_zeros
# run_test test_gl_ones
# run_test test_gl_eye
# run_test test_gl_matf_set
# run_test test_gl_matf_val
# run_test test_gl_transpose
# run_test test_gl_equals
# run_test test_gl_vec3_cross
# run_test test_gl_dot
# run_test test_gl_norm
# run_test test_gl_normalize
# run_test test_gl_perspective
# run_test test_gl_lookat
# run_test test_gl_shader_compile
# run_test test_gl_shaders_link
# run_test test_gl_prog_setup
# run_test test_gl_camera_setup
# run_test test_gui
# run_test test_imshow
