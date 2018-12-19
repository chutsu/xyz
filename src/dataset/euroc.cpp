#include "prototype/dataset/euroc.hpp"

namespace prototype {

/*****************************************************************************
 * euroc_imu_t
 ****************************************************************************/

euroc_imu_t::euroc_imu_t() {}

euroc_imu_t::euroc_imu_t(const std::string &data_dir_) : data_dir{data_dir_} {}

euroc_imu_t::~euroc_imu_t() {}

int euroc_imu_load(euroc_imu_t &data, const std::string &data_dir) {
  data.data_dir = data_dir;
  const std::string data_path = data.data_dir + "/data.csv";
  const std::string sensor_path = data.data_dir + "/sensor.yaml";

  // Open file for loading
  int nb_rows = 0;
  FILE *fp = file_open(data_path, "r", &nb_rows);
  if (fp == nullptr) {
    LOG_ERROR("Failed to open [%s]!", data_path.c_str());
    return -1;
  }

  // Parse file
  std::string str_format = "%ld,%lf,%lf,%lf,%lf,%lf,%lf";
  for (int i = 0; i < nb_rows; i++) {
    // Skip first line
    if (i == 0) {
      skip_line(fp);
      continue;
    }

    // Parse line
    long ts = 0;
    double w_x, w_y, w_z = 0.0;
    double a_x, a_y, a_z = 0.0;
    fscanf(fp, str_format.c_str(), &ts, &w_x, &w_y, &w_z, &a_x, &a_y, &a_z);
    data.timestamps.push_back(ts);
    data.w_B.emplace_back(w_x, w_y, w_z);
    data.a_B.emplace_back(a_x, a_y, a_z);
  }
  fclose(fp);

  // Load calibration data
  config_t config{sensor_path};
  if (config.ok != true) {
    LOG_ERROR("Failed to load sensor file [%s]!", sensor_path.c_str());
    return -1;
  }
  parse(config, "sensor_type", data.sensor_type);
  parse(config, "comment", data.comment);
  parse(config, "T_BS", data.T_BS);
  parse(config, "rate_hz", data.rate_hz);
  parse(config, "gyroscope_noise_density", data.gyro_noise_density);
  parse(config, "gyroscope_random_walk", data.gyro_random_walk);
  parse(config, "accelerometer_noise_density", data.accel_noise_density);
  parse(config, "accelerometer_random_walk", data.accel_random_walk);

  return 0;
}

std::ostream &operator<<(std::ostream &os, const euroc_imu_t &data) {
  // clang-format off
  os << "sensor_type: " << data.sensor_type << std::endl;
  os << "comment: " << data.comment << std::endl;
  os << "T_BS:\n" << data.T_BS << std::endl;
  os << "rate_hz: " << data.rate_hz << std::endl;
  os << "gyroscope_noise_density: " << data.gyro_noise_density << std::endl;
  os << "gyroscope_random_walk: " << data.gyro_random_walk << std::endl;
  os << "accelerometer_noise_density: " << data.accel_noise_density << std::endl;
  os << "accelerometer_random_walk: " << data.accel_random_walk << std::endl;
  // clang-format on

  return os;
}

/*****************************************************************************
 * euroc_camera_t
 ****************************************************************************/

euroc_camera_t::euroc_camera_t() {}

euroc_camera_t::euroc_camera_t(const std::string &data_dir_)
    : data_dir{data_dir_} {}

euroc_camera_t::~euroc_camera_t() {}

int euroc_camera_load(euroc_camera_t &data,
                      const std::string &data_dir,
                      const bool is_calib_data) {
  data.data_dir = data_dir;
  const std::string data_path = data.data_dir + "/data.csv";
  const std::string sensor_path = data.data_dir + "/sensor.yaml";

  // Open file for loading
  int nb_rows = 0;
  FILE *fp = file_open(data_path, "r", &nb_rows);
  if (fp == nullptr) {
    LOG_ERROR("Failed to open [%s]!", data_path.c_str());
    return -1;
  }

  // Parse file
  for (int i = 0; i < nb_rows; i++) {
    // Skip first line
    if (i == 0) {
      skip_line(fp);
      continue;
    }

    // Parse line
    long ts = 0;
    char filename[50] = {0};
    fscanf(fp, "%ld,%s", &ts, filename);
    const std::string image_file{filename};
    const auto image_path = data.data_dir + "/data/" + image_file;

    // Check if file exists
    if (file_exists(image_path) == false) {
      LOG_ERROR("File [%s] does not exist!", image_path.c_str());
      return -1;
    }

    // Add data
    data.timestamps.emplace_back(ts);
    data.image_paths.emplace_back(image_path);
  }
  fclose(fp);

  // Load calibration data
  config_t config{sensor_path};
  if (config.ok == false) {
    LOG_ERROR("Failed to load senor file [%s]!", sensor_path.c_str());
    return -1;
  }
  parse(config, "sensor_type", data.sensor_type);
  parse(config, "comment", data.comment);
  parse(config, "T_BS", data.T_BS);
  parse(config, "rate_hz", data.rate_hz);
  parse(config, "resolution", data.resolution);
  parse(config, "camera_model", data.camera_model, is_calib_data);
  parse(config, "intrinsics", data.intrinsics, is_calib_data);
  parse(config, "distortion_model", data.distortion_model, is_calib_data);
  parse(config,
        "distortion_coefficients",
        data.distortion_coefficients,
        is_calib_data);

  return 0;
}

std::ostream &operator<<(std::ostream &os, const euroc_camera_t &data) {
  // clang-format off
  os << "sensor_type: " << data.sensor_type << std::endl;
  os << "comment: " << data.comment << std::endl;
  os << "T_BS:\n" << data.T_BS << std::endl;
  os << "rate_hz: " << data.rate_hz << std::endl;
  os << "resolution: " << data.resolution.transpose() << std::endl;
  os << "camera_model: " << data.camera_model << std::endl;
  os << "intrinsics: " << data.intrinsics.transpose() << std::endl;
  os << "distortion_model: " << data.distortion_model << std::endl;
  os << "distortion_coefficients: " << data.distortion_coefficients.transpose() << std::endl;
  // clang-format on

  return os;
}

/*****************************************************************************
 * euroc_ground_truth_t
 ****************************************************************************/

euroc_ground_truth_t::euroc_ground_truth_t() {}

euroc_ground_truth_t::euroc_ground_truth_t(const std::string data_dir_)
    : data_dir{data_dir_} {}

euroc_ground_truth_t::~euroc_ground_truth_t() {}

int euroc_ground_truth_load(euroc_ground_truth_t &data,
                            const std::string &data_dir) {
  data.data_dir = data_dir;

  // Open file for loading
  const std::string data_path = data.data_dir + "/data.csv";
  int nb_rows = 0;
  FILE *fp = file_open(data_path, "r", &nb_rows);
  if (fp == nullptr) {
    LOG_ERROR("Failed to open [%s]!", data_path.c_str());
    return -1;
  }

  // Parse file
  std::string str_format;
  str_format += "%ld,";              // Timestamp
  str_format += "%lf,%lf,%lf,";      // Position
  str_format += "%lf,%lf,%lf,%lf,";  // Quaternion
  str_format += "%lf,%lf,%lf,";      // Velocity
  str_format += "%lf,%lf,%lf,";      // Gyro bias
  str_format += "%lf,%lf,%lf";       // Accel bias

  for (int i = 0; i < nb_rows; i++) {
    // Skip first line
    if (i == 0) {
      skip_line(fp);
      continue;
    }

    // Parse line
    long ts = 0;
    double p_x, p_y, p_z = 0.0;
    double q_x, q_y, q_z, q_w = 0.0;
    double v_x, v_y, v_z = 0.0;
    double b_w_x, b_w_y, b_w_z = 0.0;
    double b_a_x, b_a_y, b_a_z = 0.0;
    fscanf(fp, str_format.c_str(),
      &ts,
      &p_x, &p_y, &p_z,
      &q_x, &q_y, &q_z, &q_w,
      &v_x, &v_y, &v_z,
      &b_w_x, &b_w_y, &b_w_z,
      &b_a_x, &b_a_y, &b_a_z
    );
    data.timestamps.push_back(ts);
    data.p_RS_R.emplace_back(p_x, p_y, p_z);
    data.q_RS.emplace_back(q_w, q_x, q_y, q_z);
    data.v_RS_R.emplace_back(v_x, v_y, v_z);
    data.b_w_RS_S.emplace_back(b_w_x, b_w_y, b_w_z);
    data.b_a_RS_S.emplace_back(b_a_x, b_a_y, b_a_z);
  }
  fclose(fp);

  return 0;
}

/*****************************************************************************
 * euroc_data_t
 ****************************************************************************/

euroc_data_t::euroc_data_t() {}

euroc_data_t::euroc_data_t(const std::string &data_path)
    : data_path{strip_end(data_path, "/")} {}

euroc_data_t::~euroc_data_t() {}

int euroc_data_load(euroc_data_t &data) {
  // Load IMU data
  const std::string imu_data_dir = data.data_path + "/mav0/imu0";
  if (euroc_imu_load(data.imu_data, imu_data_dir) != 0) {
    LOG_ERROR("Failed to load IMU data [%s]!", data.imu_data.data_dir.c_str());
    return -1;
  }
  for (size_t i = 0; i < data.imu_data.timestamps.size(); i++) {
    const long ts = data.imu_data.timestamps[i];
    const vec3_t a_B = data.imu_data.a_B[i];
    const vec3_t w_B = data.imu_data.w_B[i];
    const auto imu_event = timeline_event_t{ts, a_B, w_B};
    data.timeline.insert({ts, imu_event});
  }

  // Load camera data
  const std::string cam0_data_dir = data.data_path + "/mav0/cam0";
  if (euroc_camera_load(data.cam0_data, cam0_data_dir) != 0) {
    LOG_ERROR("Failed to load cam0 data [%s]!", data.cam0_data.data_dir.c_str());
    return -1;
  }
  const std::string cam1_data_dir = data.data_path + "/mav0/cam1";
  if (euroc_camera_load(data.cam1_data, cam1_data_dir) != 0) {
    LOG_ERROR("Failed to load cam1 data [%s]!", data.cam1_data.data_dir.c_str());
    return -1;
  }
  for (size_t i = 0; i < data.cam0_data.timestamps.size(); i++) {
    const long ts = data.cam0_data.timestamps[i];
    const auto image_path = data.cam0_data.image_paths[i];
    const auto cam0_event = timeline_event_t(ts, 0, image_path);
    data.timeline.insert({ts, cam0_event});
  }
  for (size_t i = 0; i < data.cam1_data.timestamps.size(); i++) {
    const long ts = data.cam1_data.timestamps[i];
    const auto image_path = data.cam1_data.image_paths[i];
    const auto cam1_event = timeline_event_t(ts, 1, image_path);
    data.timeline.insert({ts, cam1_event});
  }
  cv::Mat image = cv::imread(data.cam0_data.image_paths[0]);
  data.image_size = cv::Size(image.size());

  // Load ground truth
  const std::string gnd_path =
      data.data_path + "/mav0/state_groundtruth_estimate0";
  if (euroc_ground_truth_load(data.ground_truth, gnd_path) != 0) {
    LOG_ERROR("Failed to load ground truth!");
    return -1;
  }

  // Process timestamps
  data.ts_start = euroc_data_min_timestamp(data);
  data.ts_end = euroc_data_max_timestamp(data);
  data.ts_now = data.ts_start;

  // Get timestamps and calculate relative time
  auto it = data.timeline.begin();
  auto it_end = data.timeline.end();
  while (it != it_end) {
    const long ts = it->first;
    data.timestamps.insert(ts);
    data.time[ts] = ((double) ts - data.ts_start) * 1e-9;

    // Advance to next non-duplicate entry.
    do {
      ++it;
    } while (ts == it->first);
  }

  data.ok = true;
  return 0;
}

void euroc_data_reset(euroc_data_t &data) {
  data.ts_start = euroc_data_min_timestamp(data);
  data.ts_end = euroc_data_max_timestamp(data);
  data.ts_now = data.ts_start;
  data.time_index = 0;
  data.imu_index = 0;
  data.frame_index = 0;
}

long euroc_data_min_timestamp(const euroc_data_t &data) {
  const long cam0_first_ts = data.cam0_data.timestamps.front();
  const long imu_first_ts = data.imu_data.timestamps.front();

  std::vector<long> first_ts{cam0_first_ts, imu_first_ts};
  auto first_result = std::min_element(first_ts.begin(), first_ts.end());
  const long first_ts_index = std::distance(first_ts.begin(), first_result);
  const long min_ts = first_ts[first_ts_index];

  return min_ts;
}

long euroc_data_max_timestamp(const euroc_data_t &data) {
  const long cam0_last_ts = data.cam0_data.timestamps.back();
  const long imu_last_ts = data.imu_data.timestamps.back();

  std::vector<long> last_ts{cam0_last_ts, imu_last_ts};
  auto last_result = std::max_element(last_ts.begin(), last_ts.end());
  const long last_ts_index = std::distance(last_ts.begin(), last_result);
  const long max_ts = last_ts[last_ts_index];

  return max_ts;
}

/*****************************************************************************
 * euroc_target_t
 ****************************************************************************/

euroc_target_t::euroc_target_t() {}

euroc_target_t::euroc_target_t(const std::string &file_path)
    : file_path{file_path} {
  if (euroc_target_load(*this, file_path) == 0) {
    ok = true;
  }
}

euroc_target_t::~euroc_target_t() {}

int euroc_target_load(euroc_target_t &target, const std::string &target_file) {
  config_t config{target_file};
  if (config.ok != true) {
    LOG_ERROR("Failed to load target file [%s]!", target_file.c_str());
    return -1;
  }
  parse(config, "target_type", target.type);
  parse(config, "tagRows", target.tag_rows);
  parse(config, "tagCols", target.tag_cols);
  parse(config, "tagSize", target.tag_size);
  parse(config, "tagSpacing", target.tag_spacing);

  return 0;
}

std::ostream &operator<<(std::ostream &os, const euroc_target_t &target) {
  os << "target_type: " << target.type << std::endl;
  os << "tag_rows: " << target.tag_rows << std::endl;
  os << "tag_cols: " << target.tag_cols << std::endl;
  os << "tag_size: " << target.tag_size << std::endl;
  os << "tag_spacing: " << target.tag_spacing << std::endl;
  return os;
}

/*****************************************************************************
 * euroc_calib_t
 ****************************************************************************/

euroc_calib_t::euroc_calib_t() {}

euroc_calib_t::euroc_calib_t(const std::string &data_path)
    : data_path{strip_end(data_path, "/")} {}

euroc_calib_t::~euroc_calib_t() {}

int euroc_calib_load(euroc_calib_t &data, const std::string &data_path) {
  // Load IMU data
  data.data_path = data_path;
  const std::string imu_data_dir = data_path + "/mav0/imu0";
  if (euroc_imu_load(data.imu_data, imu_data_dir) != 0) {
    LOG_ERROR("Failed to load IMU data [%s]!", imu_data_dir.c_str());
    return -1;
  }

  // Load cam0 data
  const std::string cam0_dir = data.data_path + "/mav0/cam0";
  if (euroc_camera_load(data.cam0_data, cam0_dir, true) != 0) {
    LOG_ERROR("Failed to load cam0 data [%s]!", cam0_dir.c_str());
    return -1;
  }

  // Load cam1 data
  const std::string cam1_dir = data.data_path + "/mav0/cam1";
  if (euroc_camera_load(data.cam1_data, cam1_dir, true) != 0) {
    LOG_ERROR("Failed to load cam1 data [%s]!", cam1_dir.c_str());
    return -1;
  }

  // Check if cam0 has same amount of images as cam1
  const size_t cam0_nb_images = data.cam0_data.image_paths.size();
  const size_t cam1_nb_images = data.cam1_data.image_paths.size();
  if (cam0_nb_images != cam1_nb_images) {
    if (cam0_nb_images > cam1_nb_images) {
      data.cam0_data.timestamps.pop_back();
      data.cam0_data.image_paths.pop_back();
    } else if (cam0_nb_images < cam1_nb_images) {
      data.cam1_data.timestamps.pop_back();
      data.cam1_data.image_paths.pop_back();
    }
  }

  // Get image size
  const cv::Mat image = cv::imread(data.cam0_data.image_paths[0]);
  data.image_size = cv::Size(image.size());

  // Load calibration target data
  const std::string target_path = data.data_path + "/april_6x6.yaml";
  if (euroc_target_load(data.calib_target, target_path) != 0) {
    LOG_ERROR("Failed to load target [%s]!", target_path.c_str());
    return -1;
  }

  return 0;
}

} // namespace prototype