#!/bin/bash
set -eu
# sudo apt-get install python-numpy python-scipy python-matplotlib -yyq

function source_kalibr() {
  KALIBR_WS=$HOME/kalibr_ws
  cd "$KALIBR_WS";
  # catkin build -DCMAKE_BUILD_TYPE=RELEASE -j1 kalibr
  source "devel/setup.bash";
  cd -
}

function kalibr_calibrate_euroc() {
  source_kalibr

  cd /data/euroc/rosbags
  time rosrun kalibr kalibr_calibrate_cameras \
    --bag "cam_april.bag" \
    --topics /cam0/image_raw /cam1/image_raw \
    --models pinhole-radtan pinhole-radtan \
    --target /data/yac_experiments/euroc_results/configs/kalibr/april_6x6.yaml \
    --dont-show-report > /data/yac_experiments/euroc_results/euroc_calib_cameras.log
  mv camchain-cam_april.yaml /data/yac_experiments/euroc_results/configs/kalibr/cam_april-camchain.yaml
  mv report-cam-cam_april.pdf /data/yac_experiments/euroc_results/configs/kalibr/cam_april-report.pdf
  mv results-cam-cam_april.txt /data/yac_experiments/euroc_results/configs/kalibr/cam_april-results.txt
  mv /tmp/kalibr_info.csv /data/yac_experiments/euroc_results/configs/kalibr/cam_april-info.csv
  cd -

  cd /data/euroc/rosbags
  time rosrun kalibr kalibr_calibrate_imu_camera \
    --bag "imu_april.bag" \
    --cam /data/yac_experiments/euroc_results/configs/kalibr/cam_april-camchain.yaml \
    --imu /data/yac_experiments/euroc_results/configs/kalibr/imu.yaml \
    --target /data/yac_experiments/euroc_results/configs/kalibr/april_6x6.yaml \
    --dont-show-report > /data/yac_experiments/euroc_results/euroc_calib_imu.log
  mv camchain-imucam-imu_april.yaml /data/yac_experiments/euroc_results/configs/kalibr/imu_april-camchain.yaml
  mv imu-imu_april.yaml /data/yac_experiments/euroc_results/configs/kalibr/imu_april-imu.yaml
  mv report-imucam-imu_april.pdf /data/yac_experiments/euroc_results/configs/kalibr/imu_april-report.pdf
  mv results-imucam-imu_april.txt /data/yac_experiments/euroc_results/configs/kalibr/imu_april-results.txt
  cd -
}

function kalibr_calibrate() {
  source_kalibr

  DATA_DIRS="./human_trials/kalibr/calib*";
  for DATA_DIR in $DATA_DIRS
  do
    cd "$DATA_DIR";

    if [ ! -f calib_camera-camchain.yaml ]; then
      echo "Calibrating cameras in [$DATA_DIR]";
      /usr/bin/time -o calib_camera-time.txt \
        rosrun kalibr kalibr_calibrate_cameras \
          --bag "calib_camera.bag" \
          --topics /rs/ir0/image /rs/ir1/image \
          --models pinhole-radtan pinhole-radtan \
          --target ../april_6x6.yaml \
          --dont-show-report > calib_camera.log

      mv camchain-calib_camera.yaml calib_camera-camchain.yaml
      mv report-cam-calib_camera.pdf calib_camera-report.pdf
      mv results-cam-calib_camera.txt calib_camera-results.txt
      mv /tmp/kalibr_info.csv calib_camera-info.csv
    fi

    if [ ! -f calib_imu-camchain.yaml ]; then
      echo "Calibrating camera-imu in [$DATA_DIR]";
      /usr/bin/time -o calib_imu-time.txt \
        rosrun kalibr kalibr_calibrate_imu_camera \
          --bag "calib_imu.bag" \
          --cam calib_camera-camchain.yaml \
          --imu ../imu.yaml \
          --target ../april_6x6.yaml \
          --dont-show-report > calib_imu.log

        rm imu-calib_imu.yaml
        mv camchain-imucam-calib_imu.yaml calib_imu-camchain.yaml
        mv report-imucam-calib_imu.pdf calib_imu-report.pdf
        mv results-imucam-calib_imu.txt calib_imu-results.txt
    fi

    if [ ! -f calib_imu-info.csv ]; then
      echo "Extracting rosbag data [$DATA_DIR/calib_imu.bag]"
      python3 ../../../scripts/bag2euroc.py calib_imu.bag

      echo "Convert kalibr camchain to YAC format..."
      python3 ../../../scripts/kalibr2yac.py calib_imu-camchain.yaml

      echo "Evaluating calib IMU information..."
      ~/projects/yac/build/calib_info \
        camera-imu \
        calib_imu-camchain-yac.yaml \
        calib_imu/mav0 \
        ./calib_imu-info.csv
      rm -rf calib_imu
    fi

    cd -
  done
}

function yac_calibrate_euroc() {
  export LD_LIBRARY_PATH=/opt/yac/lib

  /usr/bin/time -o calib-rerun-time.txt \
    /opt/yac/bin/calib_vi \
    /tmp/calib_camera-results.yaml \
    /data/euroc/imu_april/mav0

  # mv /tmp/calib-results.yaml calib-imu-rerun.yaml
}

function yac_calibrate() {
  DATA_DIRS="./human_trials/yac/calib*";
  for DATA_DIR in $DATA_DIRS
  do
    echo "Calibrating [$DATA_DIR]"
    cd "$DATA_DIR";
    export LD_LIBRARY_PATH=/opt/yac/lib

    /usr/bin/time -o calib-rerun-time.txt \
      /opt/yac/bin/calib_vi \
      ./calib_imu/calib-results.yaml \
      ./calib_imu

    mv /tmp/calib-results.yaml calib-imu-rerun.yaml

    cd -
  done

  # DATA_DIR="/data/yac_experiments/human_trials/yac/calib0"
  # # echo "Calibrating [$DATA_DIR]"
  # cd "$DATA_DIR";
  # export LD_LIBRARY_PATH=/opt/yac/lib

  # time /opt/yac/bin/calib_vi \
  #   ./calib_imu/calib-results.yaml \
  #   ./calib_imu

  # /usr/bin/time -o calib-rerun-time.txt \
  #   /opt/yac/bin/calib_vi \
  #   ./calib_imu/calib-results.yaml \
  #   ./calib_imu

  # mv /tmp/calib-results.yaml calib-imu-rerun.yaml
}

function yac_eval_info() {
  DATA_DIRS="./human_trials/yac/calib*";
  CALIB_INFO_EXEC="";
  for DATA_DIR in $DATA_DIRS
  do
    cd "$DATA_DIR/calib_imu";

    if [ ! -f calib_imu-info.csv ]; then
      echo "Evaluating calib IMU information..."
      ~/projects/yac/build/calib_info \
        camera-imu \
        calib-results.yaml \
        ./ \
        ./calib_imu-info.csv
    fi

    cd -
  done
}

############################ KALIBR CALIBRATE ################################

# kalibr_calibrate_euroc
# kalibr_calibrate

############################# YAC CALIBRATE ##################################

# yac_calibrate_euroc
# yac_calibrate
# yac_eval_info

########################### ORB-SLAM3 Benchmark ##############################

# python3 scripts/bag2euroc.py experiments/2022-06-06-15-09-43.bag

# ./Examples/Stereo-Inertial/stereo_inertial_euroc
#   ./Vocabulary/ORBvoc.txt
#   /data/yac_results/kalibr/orb_config.yaml
#   /data/yac_results/experiments/2022-06-06-15-09-43
#   /data/yac_results/experiments/2022-06-06-15-09-43/camera_timestamps.txt
#   results0

########################### VINS-FUSION Benchmark ############################

# make run_vins_fusion-stereo_imu

################################ Benchmark ###################################

# time python3 scripts/eval_runs.py
# time python3 scripts/plot_nbvs.py ./scripts/data/calib_poses.csv
# time python3 scripts/lissajous.py

# python3 slam_bench.py \
#   --algo "orbslam3" \
#   --mode "stereo" \
#   --dataset "/data/euroc/MH_01" \
#   --run_name "test" \
#   --config "/data/yac_experiments/euroc_results/configs/euroc/orbslam3-stereo-euroc.yaml" \
#   --output "/data"

# python3 slam_bench.py \
#   --algo "vins-fusion" \
#   --mode "stereo_imu" \
#   --dataset "/data/euroc/rosbags/MH_01.bag" \
#   --run_name "test" \
#   --config "/home/slam_bench/configs/vins-fusion/euroc/euroc_stereo_imu_config.yaml" \
#   --output "/data/test.csv"

python3 slam_bench.py \
  --algo "okvis" \
  --mode "stereo_imu" \
  --dataset "/data/euroc/MH_01/mav0" \
  --run_name "test" \
  --config "/home/slam_bench/configs/okvis/euroc/config_fpga_p2_euroc.yaml" \
  --output "/data/test.csv"
