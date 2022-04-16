#pragma once
#include "core.h"
#include "comms.h"

// HCSR04 ////////////////////////////////////////////////////////////////////

typedef struct hcsr04_t {
  uint8_t pin_trigger;
  uint8_t pin_echo;
  unsigned short max_dist_cm;
  unsigned long max_timeout_ms;

} hcsr04_t;

void hcsr04_setup(hcsr04_t *sensor,
                  uint8_t pin_trigger,
                  uint8_t pin_echo,
                  unsigned short max_dist_cm,
                  unsigned long max_timeout_ms) {
  sensor->pin_trigger = pin_trigger;
  sensor->pin_echo = pin_echo;
  sensor->max_dist_cm = max_dist_cm;
  sensor->max_timeout_ms = max_timeout_ms;

  pinMode(pin_trigger, OUTPUT);
  pinMode(pin_echo, INPUT);
}

float hcsr04_measure(hcsr04_t *sensor, float temperature) {
  // Ensure trigger pin is LOW.
  digitalWrite(sensor->pin_trigger, LOW);
  delayMicroseconds(2);

  // Hold trigger for 10ms (signal for sensor to measure distance)
  digitalWrite(sensor->pin_trigger, HIGH);
  delayMicroseconds(10);
  digitalWrite(sensor->pin_trigger, LOW);

  // Speed of sound [cm / ms]
  const float sos = 0.03313 + 0.0000606 * temperature;
  // Cair ≈ (331.3 + 0.606 * theta) m/s

  // Compute max delay based on max distance with 25% margin in ms
  unsigned long max_dist = 2.5 * sensor->max_dist_cm / sos;
  if (sensor->max_timeout_ms > 0) {
    max_dist = min(max_dist, sensor->max_timeout_ms);
  }

  // Measure the length of echo signal, which is equal to the time needed for
  // sound to go there and back. Cannot measure beyond max distance.
  unsigned long dt_ms = pulseIn(sensor->pin_echo, HIGH, max_dist);

  // Return distance in cm
  float dist_cm = dt_ms / 2.0 * sos;
  if (dist_cm == 0 || dist_cm > sensor->max_dist_cm) {
    return -1.0;
  } else {
    return dist_cm;
  }
}

// BMP280 ////////////////////////////////////////////////////////////////////

// BMP280 Addresses
#define BMP280_ADDR 0x77     // Primary I2C Address
#define BMP280_ADDR_ALT 0x76 // Alternate Address

// BMP280 Registers
#define BMP280_REG_DIG_T1 0x88
#define BMP280_REG_DIG_T2 0x8A
#define BMP280_REG_DIG_T3 0x8C
#define BMP280_REG_DIG_P1 0x8E
#define BMP280_REG_DIG_P2 0x90
#define BMP280_REG_DIG_P3 0x92
#define BMP280_REG_DIG_P4 0x94
#define BMP280_REG_DIG_P5 0x96
#define BMP280_REG_DIG_P6 0x98
#define BMP280_REG_DIG_P7 0x9A
#define BMP280_REG_DIG_P8 0x9C
#define BMP280_REG_DIG_P9 0x9E
#define BMP280_REG_DIG_H1 0xA1
#define BMP280_REG_DIG_H2 0xE1
#define BMP280_REG_DIG_H3 0xE3
#define BMP280_REG_DIG_H4 0xE4
#define BMP280_REG_DIG_H5 0xE5
#define BMP280_REG_DIG_H6 0xE7
#define BMP280_REG_CHIPID 0xD0
#define BMP280_REG_VERSION 0xD1
#define BMP280_REG_SOFTRESET 0xE0
#define BMP280_REG_CAL26 0xE1 // R calibration stored in 0xE1-0xF0
#define BMP280_REG_CONTROLHUMID 0xF2
#define BMP280_REG_STATUS 0XF3
#define BMP280_REG_CONTROL 0xF4
#define BMP280_REG_CONFIG 0xF5
#define BMP280_REG_PRESSUREDATA 0xF7
#define BMP280_REG_TEMPDATA 0xFA
#define BMP280_REG_HUMIDDATA 0xFD

// BMP280 Sampling rates
#define BMP280_SAMPLING_NONE 0b000
#define BMP280_SAMPLING_X1 0b001
#define BMP280_SAMPLING_X2 0b010
#define BMP280_SAMPLING_X4 0b011
#define BMP280_SAMPLING_X8 0b100
#define BMP280_SAMPLING_X16 0b101

// BMP280 Power modes
#define BMP280_MODE_SLEEP 0b00
#define BMP280_MODE_FORCED 0b01
#define BMP280_MODE_NORMAL 0b11

// BMP280 Filter values
#define BMP280_FILTER_OFF 0b000
#define BMP280_FILTER_X2 0b001
#define BMP280_FILTER_X4 0b010
#define BMP280_FILTER_X8 0b011
#define BMP280_FILTER_X16 0b10

// BMP280 Standby duration in ms
#define BMP280_STANDBY_MS_0_5 0b000
#define BMP280_STANDBY_MS_62_5 0b001
#define BMP280_STANDBY_MS_125 0b010
#define BMP280_STANDBY_MS_250 0b011
#define BMP280_STANDBY_MS_500 0b100
#define BMP280_STANDBY_MS_1000 0b101
#define BMP280_STANDBY_MS_2000 0b110
#define BMP280_STANDBY_MS_4000 0b111

typedef struct bmp280_t {
  // Temperature compensation value
  uint16_t dig_T1;
  int16_t dig_T2;
  int16_t dig_T3;

  // Pressure compenstation value
  uint16_t dig_P1;
  int16_t dig_P2;
  int16_t dig_P3;
  int16_t dig_P4;
  int16_t dig_P5;
  int16_t dig_P6;
  int16_t dig_P7;
  int16_t dig_P8;
  int16_t dig_P9;

  // State
  uint8_t previous_measuring;
} bmp280_t;

void bmp280_setup(bmp280_t *bmp280);
void bmp280_get_calib_data(bmp280_t *bmp280);
int bmp280_data_ready(bmp280_t *bmp280);
void bmp280_get_data(const bmp280_t *bmp280,
                     float *temperature,
                     float *pressure);

void bmp280_setup(bmp280_t *bmp280) {
  uint8_t sensor_addr = BMP280_ADDR_ALT;

  // Reset the device using soft-reset
  i2c_write_byte(sensor_addr, BMP280_REG_SOFTRESET, 0xB6);

  // Wait for chip to wake up.
  delay(10);

  // Get calibration data
  bmp280_get_calib_data(bmp280);

  // Configure sensor
  uint8_t t_sb = BMP280_STANDBY_MS_0_5; // Standby time in normal mode
  uint8_t filter = BMP280_FILTER_X16;   // Filter settings
  uint8_t spi3w_en = 0;                 // Enable 3-wire SPI
  uint8_t config = (t_sb << 5) | (filter << 2) | spi3w_en;
  i2c_write_byte(sensor_addr, BMP280_REG_CONFIG, config);

  // Configure measurement
  uint8_t mode = BMP280_MODE_NORMAL;    // Device mode
  uint8_t osrs_p = BMP280_SAMPLING_X16; // Pressure oversampling
  uint8_t osrs_t = BMP280_SAMPLING_X2;  // Temperature oversampling
  uint8_t meas_config = (osrs_t << 5) | (osrs_p << 2) | mode;
  i2c_write_byte(sensor_addr, BMP280_REG_CONTROL, meas_config);

  // Wait
  delay(100);
}

void bmp280_get_calib_data(bmp280_t *bmp280) {
  uint8_t sensor_addr = BMP280_ADDR_ALT;

  bmp280->dig_T1 = i2c_read_u16(sensor_addr, BMP280_REG_DIG_T1);
  bmp280->dig_T2 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_T2);
  bmp280->dig_T3 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_T3);

  bmp280->dig_P1 = i2c_read_u16(sensor_addr, BMP280_REG_DIG_P1);
  bmp280->dig_P2 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P2);
  bmp280->dig_P3 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P3);
  bmp280->dig_P4 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P4);
  bmp280->dig_P5 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P5);
  bmp280->dig_P6 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P6);
  bmp280->dig_P7 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P7);
  bmp280->dig_P8 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P8);
  bmp280->dig_P9 = i2c_read_s16(sensor_addr, BMP280_REG_DIG_P9);
}

int bmp280_data_ready(bmp280_t *bmp280) {
  uint8_t sensor_addr = BMP280_ADDR_ALT;

  const uint8_t reg = i2c_read_byte(sensor_addr, BMP280_REG_STATUS);
  const uint8_t measuring = (reg & 0b00001000) >> 3;

  // Has measuring bit flipped?
  if (measuring ^ bmp280->previous_measuring) {
    bmp280->previous_measuring = measuring;
    if (measuring == 0) {
      return 1;
    }
  }

  return 0;
}

/** Returns temperature in Celcius degrees and pressure in Pa **/
void bmp280_get_data(const bmp280_t *bmp280,
                     float *temperature,
                     float *pressure) {
  uint8_t sensor_addr = BMP280_ADDR_ALT;

  // Get data
  uint8_t data[6] = {0};
  i2c_read_bytes(sensor_addr, BMP280_REG_PRESSUREDATA, 6, data);

  // Compensate for temperature
  int32_t t_fine = 0;
  {
    int32_t T = (data[3] << 12) | (data[4] << 4) | (data[5] >> 4);
    int32_t T1 = bmp280->dig_T1;
    int32_t T2 = bmp280->dig_T2;
    int32_t T3 = bmp280->dig_T3;
    int32_t var1 = ((((T >> 3) - (T1 << 1))) * T2) >> 11;
    int32_t var2 = (((((T >> 4) - T1) * ((T >> 4) - T1)) >> 12) * (T3)) >> 14;
    t_fine = var1 + var2;
    *temperature = ((t_fine * 5 + 128) >> 8) / 100.0f;
  }

  // Compensate for pressure
  {
    int32_t P = (data[0] << 12) | (data[1] << 4) | (data[2] >> 4);
    int64_t P1 = bmp280->dig_P1;
    int64_t P2 = bmp280->dig_P2;
    int64_t P3 = bmp280->dig_P3;
    int64_t P4 = bmp280->dig_P4;
    int64_t P5 = bmp280->dig_P5;
    int64_t P6 = bmp280->dig_P6;
    int64_t P7 = bmp280->dig_P7;
    int64_t P8 = bmp280->dig_P8;
    int64_t P9 = bmp280->dig_P9;

    int64_t var1, var2 = 0;
    var1 = ((int64_t) t_fine) - 128000;
    var2 = var1 * var1 * P6;
    var2 = var2 + ((var1 * P5) << 17);
    var2 = var2 + (P4 << 35);
    var1 = ((var1 * var1 * P3) >> 8) + ((var1 * P2) << 12);
    var1 = (((((int64_t) 1) << 47) + var1)) * P1 >> 33;

    // Avoid exception caused by division by zero
    if (var1 == 0) {
      *pressure = -1;
      return;
    }

    int64_t p = 1048576 - P;
    p = (((p << 31) - var2) * 3125) / var1;
    var1 = (P9 * (p >> 13) * (p >> 13)) >> 25;
    var2 = (P8 * p) >> 19;
    *pressure = ((float) (((p + var1 + var2) >> 8) + (P7 << 4))) / 256.0f;
  }
}

//////////////////////////////////////////////////////////////////////////////

// MPU6050 ///////////////////////////////////////////////////////////////////

// GENERAL
#define MPU6050_ADDRESS 0x68
#define MPU6050_ADDRESS_AD0_LOW 0x68  // addr pin low (GND) [default]
#define MPU6050_ADDRESS_AD0_HIGH 0x69 // addr pin high (VCC)

// REGISTER ADDRESSES
#define MPU6050_REG_XG_OFFS_TC 0x00
#define MPU6050_REG_YG_OFFS_TC 0x01
#define MPU6050_REG_ZG_OFFS_TC 0x02
#define MPU6050_REG_X_FINE_GAIN 0x03
#define MPU6050_REG_Y_FINE_GAIN 0x04
#define MPU6050_REG_Z_FINE_GAIN 0x05
#define MPU6050_REG_XA_OFFS_H 0x06
#define MPU6050_REG_XA_OFFS_L_TC 0x07
#define MPU6050_REG_YA_OFFS_H 0x08
#define MPU6050_REG_YA_OFFS_L_TC 0x09
#define MPU6050_REG_ZA_OFFS_H 0x0A
#define MPU6050_REG_ZA_OFFS_L_TC 0x0B
#define MPU6050_REG_XG_OFFS_USRH 0x13
#define MPU6050_REG_XG_OFFS_USRL 0x14
#define MPU6050_REG_YG_OFFS_USRH 0x15
#define MPU6050_REG_YG_OFFS_USRL 0x16
#define MPU6050_REG_ZG_OFFS_USRH 0x17
#define MPU6050_REG_ZG_OFFS_USRL 0x18
#define MPU6050_REG_SMPLRT_DIV 0x19
#define MPU6050_REG_CONFIG 0x1A
#define MPU6050_REG_GYRO_CONFIG 0x1B
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_FF_THR 0x1D
#define MPU6050_REG_FF_DUR 0x1E
#define MPU6050_REG_MOT_THR 0x1F
#define MPU6050_REG_MOT_DUR 0x20
#define MPU6050_REG_ZRMOT_THR 0x21
#define MPU6050_REG_ZRMOT_DUR 0x22
#define MPU6050_REG_FIFO_EN 0x23
#define MPU6050_REG_I2C_MST_CTRL 0x24
#define MPU6050_REG_I2C_SLV0_ADDR 0x25
#define MPU6050_REG_I2C_SLV0_REG 0x26
#define MPU6050_REG_I2C_SLV0_CTRL 0x27
#define MPU6050_REG_I2C_SLV1_ADDR 0x28
#define MPU6050_REG_I2C_SLV1_REG 0x29
#define MPU6050_REG_I2C_SLV1_CTRL 0x2A
#define MPU6050_REG_I2C_SLV2_ADDR 0x2B
#define MPU6050_REG_I2C_SLV2_REG 0x2C
#define MPU6050_REG_I2C_SLV2_CTRL 0x2D
#define MPU6050_REG_I2C_SLV3_ADDR 0x2E
#define MPU6050_REG_I2C_SLV3_REG 0x2F
#define MPU6050_REG_I2C_SLV3_CTRL 0x30
#define MPU6050_REG_I2C_SLV4_ADDR 0x31
#define MPU6050_REG_I2C_SLV4_REG 0x32
#define MPU6050_REG_I2C_SLV4_DO 0x33
#define MPU6050_REG_I2C_SLV4_CTRL 0x34
#define MPU6050_REG_I2C_SLV4_DI 0x35
#define MPU6050_REG_I2C_MST_STATUS 0x36
#define MPU6050_REG_INT_PIN_CFG 0x37
#define MPU6050_REG_INT_ENABLE 0x38
#define MPU6050_REG_DMP_INT_STATUS 0x39
#define MPU6050_REG_INT_STATUS 0x3A
#define MPU6050_REG_ACCEL_XOUT_H 0x3B
#define MPU6050_REG_ACCEL_XOUT_L 0x3C
#define MPU6050_REG_ACCEL_YOUT_H 0x3D
#define MPU6050_REG_ACCEL_YOUT_L 0x3E
#define MPU6050_REG_ACCEL_ZOUT_H 0x3F
#define MPU6050_REG_ACCEL_ZOUT_L 0x40
#define MPU6050_REG_TEMP_OUT_H 0x41
#define MPU6050_REG_TEMP_OUT_L 0x42
#define MPU6050_REG_GYRO_XOUT_H 0x43
#define MPU6050_REG_GYRO_XOUT_L 0x44
#define MPU6050_REG_GYRO_YOUT_H 0x45
#define MPU6050_REG_GYRO_YOUT_L 0x46
#define MPU6050_REG_GYRO_ZOUT_H 0x47
#define MPU6050_REG_GYRO_ZOUT_L 0x48
#define MPU6050_REG_EXT_SENS_DATA_00 0x49
#define MPU6050_REG_EXT_SENS_DATA_01 0x4A
#define MPU6050_REG_EXT_SENS_DATA_02 0x4B
#define MPU6050_REG_EXT_SENS_DATA_03 0x4C
#define MPU6050_REG_EXT_SENS_DATA_04 0x4D
#define MPU6050_REG_EXT_SENS_DATA_05 0x4E
#define MPU6050_REG_EXT_SENS_DATA_06 0x4F
#define MPU6050_REG_EXT_SENS_DATA_07 0x50
#define MPU6050_REG_EXT_SENS_DATA_08 0x51
#define MPU6050_REG_EXT_SENS_DATA_09 0x52
#define MPU6050_REG_EXT_SENS_DATA_10 0x53
#define MPU6050_REG_EXT_SENS_DATA_11 0x54
#define MPU6050_REG_EXT_SENS_DATA_12 0x55
#define MPU6050_REG_EXT_SENS_DATA_13 0x56
#define MPU6050_REG_EXT_SENS_DATA_14 0x57
#define MPU6050_REG_EXT_SENS_DATA_15 0x58
#define MPU6050_REG_EXT_SENS_DATA_16 0x59
#define MPU6050_REG_EXT_SENS_DATA_17 0x5A
#define MPU6050_REG_EXT_SENS_DATA_18 0x5B
#define MPU6050_REG_EXT_SENS_DATA_19 0x5C
#define MPU6050_REG_EXT_SENS_DATA_20 0x5D
#define MPU6050_REG_EXT_SENS_DATA_21 0x5E
#define MPU6050_REG_EXT_SENS_DATA_22 0x5F
#define MPU6050_REG_EXT_SENS_DATA_23 0x60
#define MPU6050_REG_MOT_DETECT_STATUS 0x61
#define MPU6050_REG_I2C_SLV0_DO 0x63
#define MPU6050_REG_I2C_SLV1_DO 0x64
#define MPU6050_REG_I2C_SLV2_DO 0x65
#define MPU6050_REG_I2C_SLV3_DO 0x66
#define MPU6050_REG_I2C_MST_DELAY_CTRL 0x67
#define MPU6050_REG_SIGNAL_PATH_RESET 0x68
#define MPU6050_REG_MOT_DETECT_CTRL 0x69
#define MPU6050_REG_USER_CTRL 0x6A
#define MPU6050_REG_PWR_MGMT_1 0x6B
#define MPU6050_REG_PWR_MGMT_2 0x6C
#define MPU6050_REG_BANK_SEL 0x6D
#define MPU6050_REG_MEM_START_ADDR 0x6E
#define MPU6050_REG_MEM_R_W 0x6F
#define MPU6050_REG_DMP_CFG_1 0x70
#define MPU6050_REG_DMP_CFG_2 0x71
#define MPU6050_REG_FIFO_COUNTH 0x72
#define MPU6050_REG_FIFO_COUNTL 0x73
#define MPU6050_REG_FIFO_R_W 0x74
#define MPU6050_REG_WHO_AM_I 0x75

typedef struct mpu6050_t {
  int8_t ok;

  float accel_sensitivity;
  float gyro_sensitivity;
  float accel[3];
  float gyro[3];

  float temperature;
  float sample_rate;
  int8_t dplf_config;
} mpu6050_t;

int8_t mpu6050_setup(mpu6050_t *imu);
uint8_t mpu6050_ping(const mpu6050_t *imu);
int8_t mpu6050_get_data(mpu6050_t *imu);
int8_t mpu6050_set_dplf(const mpu6050_t *imu, const uint8_t setting);
int8_t mpu6050_get_dplf(const mpu6050_t *imu);
int8_t mpu6050_set_sample_rate_div(const mpu6050_t *imu, const int8_t div);
int8_t mpu6050_get_sample_rate_div(const mpu6050_t *imu);
int8_t mpu6050_get_sample_rate(const mpu6050_t *imu);
int8_t mpu6050_get_gyro_range(const mpu6050_t *imu, int8_t *range);
int8_t mpu6050_set_gyro_range(const mpu6050_t *imu, const int8_t range);
int8_t mpu6050_get_accel_range(const mpu6050_t *imu, int8_t *range);
int8_t mpu6050_set_accel_range(const mpu6050_t *imu, const int8_t range);

int8_t mpu6050_setup(mpu6050_t *imu) {
  int8_t retval = 0;
  int8_t dplf = 1;
  int8_t accel_range = 0;
  int8_t gyro_range = 0;

  // Set dplf
  mpu6050_set_dplf(imu, dplf);
  retval = mpu6050_get_dplf(imu);
  if (retval > 7 || retval < 0) {
    return -1;
  } else {
    imu->dplf_config = retval;
  }

  // Set power management register
  i2c_write_byte(MPU6050_ADDRESS, MPU6050_REG_PWR_MGMT_1, 0x00);

  // Configure gyro range
  if (mpu6050_set_gyro_range(imu, gyro_range) != 0) {
    return -1;
  }
  switch (gyro_range) {
    case 0:
      imu->gyro_sensitivity = 131.0;
      break;
    case 1:
      imu->gyro_sensitivity = 65.5;
      break;
    case 2:
      imu->gyro_sensitivity = 32.8;
      break;
    case 3:
      imu->gyro_sensitivity = 16.4;
      break;
    default:
      goto error;
  }

  // Configure accel range
  if (mpu6050_set_accel_range(imu, accel_range) != 0) {
    goto error;
  }
  switch (accel_range) {
    case 0:
      imu->accel_sensitivity = 16384.0;
      break;
    case 1:
      imu->accel_sensitivity = 8192.0;
      break;
    case 2:
      imu->accel_sensitivity = 4096.0;
      break;
    case 3:
      imu->accel_sensitivity = 2048.0;
      break;
    default:
      return -3;
  }

  // Get sample rate
  imu->sample_rate = mpu6050_get_sample_rate(imu);

  return 0;
error:
  return -1;
}

uint8_t mpu6050_ping(const mpu6050_t *imu) {
  UNUSED(imu);
  return i2c_read_byte(MPU6050_ADDRESS, MPU6050_REG_WHO_AM_I);
}

int8_t mpu6050_get_data(mpu6050_t *imu) {
  // Read data
  uint8_t raw_data[14] = {0};
  i2c_read_bytes(MPU6050_ADDRESS, MPU6050_REG_ACCEL_XOUT_H, 14, raw_data);

  // Accelerometer
  const float g = 9.81; // Gravitational constant
  const int16_t raw_x = (raw_data[0] << 8) | (raw_data[1]);
  const int16_t raw_y = (raw_data[2] << 8) | (raw_data[3]);
  const int16_t raw_z = (raw_data[4] << 8) | (raw_data[5]);
  imu->accel[0] = (raw_x / imu->accel_sensitivity) * g;
  imu->accel[1] = (raw_y / imu->accel_sensitivity) * g;
  imu->accel[2] = (raw_z / imu->accel_sensitivity) * g;

  // Temperature
  const int8_t raw_temp = (raw_data[6] << 8) | (raw_data[7]);
  imu->temperature = raw_temp / 340.0 + 36.53;

  // Gyroscope
  const int16_t gyro_raw_x = (raw_data[8] << 8) | (raw_data[9]);
  const int16_t gyro_raw_y = (raw_data[10] << 8) | (raw_data[11]);
  const int16_t gyro_raw_z = (raw_data[12] << 8) | (raw_data[13]);
  imu->gyro[0] = deg2rad(gyro_raw_x / imu->gyro_sensitivity);
  imu->gyro[1] = deg2rad(gyro_raw_y / imu->gyro_sensitivity);
  imu->gyro[2] = deg2rad(gyro_raw_z / imu->gyro_sensitivity);

  return 0;
}

int8_t mpu6050_set_dplf(const mpu6050_t *imu, const uint8_t setting) {
  UNUSED(imu);

  /*
      DPLF_CFG    Accelerometer
      ----------------------------------------
                  Bandwidth(Hz) | Delay(ms)
      0           260             0
      1           184             2.0
      2           94              3.0
      3           44              4.9
      4           21              8.5
      5           10              13.8
      6           5               19.0
      7           RESERVED        RESERVED

      DPLF_CFG    Gyroscope
      ----------------------------------------------
                  Bandwidth(Hz) | Delay(ms) | Fs(kHz)
      0           256             0.98        8
      1           188             1.9         1
      2           98              2.8         1
      3           42              4.8         1
      4           20              8.3         1
      5           10              13.4        1
      6           5               18.5        1
      7           RESERVED        RESERVED    8
  */

  // Check setting range
  if (setting > 7) {
    return -2;
  }

  // Set DPLF
  i2c_write_byte(MPU6050_ADDRESS, MPU6050_REG_CONFIG, (char) setting);

  return 0;
}

int8_t mpu6050_get_dplf(const mpu6050_t *imu) {
  UNUSED(imu);

  uint8_t data[1] = {0x00};
  i2c_read_bytes(MPU6050_ADDRESS, MPU6050_REG_CONFIG, 1, data);
  data[0] = data[0] & 0b00000111;
  return data[0];
}

int8_t mpu6050_set_sample_rate_div(const mpu6050_t *imu, const int8_t div) {
  UNUSED(imu);
  i2c_write_byte(MPU6050_ADDRESS, MPU6050_REG_SMPLRT_DIV, div);
  return 0;
}

int8_t mpu6050_get_sample_rate_div(const mpu6050_t *imu) {
  UNUSED(imu);
  return i2c_read_byte(MPU6050_ADDRESS, MPU6050_REG_SMPLRT_DIV);
}

int8_t mpu6050_get_sample_rate(const mpu6050_t *imu) {
  // Get sample rate divider
  uint16_t sample_div = 0;
  int8_t rate_div = mpu6050_get_sample_rate_div(imu);
  if (rate_div != -1 || rate_div != -2) {
    sample_div = (float) rate_div;
  } else {
    return -1;
  }

  // Get gyro sample rate
  uint16_t gyro_rate = 0;
  uint8_t dlpf_cfg = mpu6050_get_sample_rate_div(imu);
  if (dlpf_cfg == 0 || dlpf_cfg == 7) {
    gyro_rate = 8000;
  } else if (dlpf_cfg >= 1 || dlpf_cfg <= 6) {
    gyro_rate = 1000;
  } else {
    return -2;
  }

  // Calculate sample rate
  return gyro_rate / (1 + sample_div);
}

int8_t mpu6050_get_gyro_range(const mpu6050_t *imu, int8_t *range) {
  UNUSED(imu);

  // Get gyro config
  const uint8_t data = i2c_read_byte(MPU6050_ADDRESS, MPU6050_REG_GYRO_CONFIG);

  // Get gyro range bytes
  *range = (data >> 3) & 0b00000011;

  return 0;
}

int8_t mpu6050_set_gyro_range(const mpu6050_t *imu, const int8_t range) {
  UNUSED(imu);

  // Pre-check
  if (range > 3 || range < 0) {
    return -2;
  }

  // Set sample rate
  uint8_t data = range << 3;
  i2c_write_byte(MPU6050_ADDRESS, MPU6050_REG_GYRO_CONFIG, data);

  // Double check
  int8_t gyro_range = 0;
  int8_t retval = mpu6050_get_gyro_range(imu, &gyro_range);
  if (retval != 0 || (gyro_range != range)) {
    return -1;
  }

  return 0;
}

int8_t mpu6050_get_accel_range(const mpu6050_t *imu, int8_t *range) {
  UNUSED(imu);

  // Get accel config
  uint8_t data = i2c_read_byte(MPU6050_ADDRESS, MPU6050_REG_ACCEL_CONFIG);

  // Get accel range bytes
  *range = (data >> 3) & 0b00000011;

  return 0;
}

int8_t mpu6050_set_accel_range(const mpu6050_t *imu, const int8_t range) {
  UNUSED(imu);

  // Pre-check
  if (range > 3 || range < 0) {
    return -2;
  }

  // Set sample rate
  uint8_t data = range << 3;
  i2c_write_byte(MPU6050_ADDRESS, MPU6050_REG_ACCEL_CONFIG, data);

  // Double check
  int8_t accel_range = 0;
  int8_t retval = mpu6050_get_accel_range(imu, &accel_range);
  if (retval != 0 || (accel_range != range)) {
    return -1;
  }

  return 0;
}

//////////////////////////////////////////////////////////////////////////////
