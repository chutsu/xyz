This folder features the following state-estimation algorithms as Docker
images:

- [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)
- [VINS-Fusion](https://github.com/HKUST-Aerial-Robotics/VINS-fusion)
- [ORBSLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3)
- [Stereo-MSCKF](https://github.com/KumarRobotics/msckf_vio)
- [OKVIS](https://github.com/ethz-asl/okvis)

as well as:

- [Kalibr](https://github.com/ethz-asl/kalibr) (to calibrate your own sensors,
  namely cameras and IMUs)


# Usage

1. Install Docker or alternatively assuming you're using Ubuntu you can install
   docker via

   ```make install_docker```

2. Create a data directory to store SLAM datasets and state-estimation
   algorithm configuration files. By default `slam_bench` assumes `/data`

3. Build the docker containers using one of the following make targets:

   ```
   make build_all          # Build all
   make build_kalibr       # Build Kalibr
   make build_msckf        # Build MSCKF-VIO
   make build_okvis        # Build OKVIS
   make build_orbslam3     # Build ORBSLAM3
   make build_vins_mono    # Build VINS-Mono
   make build_vins_fusion  # Build VINS-Fusion
   ```

4. To run one of the docker containers in interactive mode, issue one of the
   following commands:

   ```
   make run_kalibr          # Run Kalibr
   make run_msckf           # Run MSCKF-VIO
   make run_okvis           # Run OKVIS
   make run_orbslam3        # Run ORBSLAM3
   make run_vins_mono       # Run VINS-Mono
   make run_vins_fusion     # Run VINS-Fusion
   ```
