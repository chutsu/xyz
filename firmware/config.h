#define GRAVITY 9.81
#define MAX_TILT_RAD 1.0472

#define I2C_MAX_BUF_LEN 1024

#define PWM_PIN_0 2
#define PWM_PIN_1 3
#define PWM_PIN_2 4
#define PWM_PIN_3 5
#define PWM_NUM_PINS 4
#define PWM_PERIOD_MIN 0.055
#define PWM_PERIOD_MAX 0.095
#define PWM_PERIOD_DIFF (0.1 - 0.05)
#define PWM_VALUE_MAX 1800.0
#define PWM_VALUE_MIN 180.0
#define PWM_RESOLUTION_BITS 15
#define PWM_FREQUENCY_HZ 50
#if PWM_RESOLUTION_BITS == 16
#define PWM_RANGE_MAX 65535.0f
#elif PWM_RESOLUTION_BITS == 15
#define PWM_RANGE_MAX 32767.0f
#elif PWM_RESOLUTION_BITS == 14
#define PWM_RANGE_MAX 16383.0f
#elif PWM_RESOLUTION_BITS == 13
#define PWM_RANGE_MAX 8191.0f
#elif PWM_RESOLUTION_BITS == 12
#define PWM_RANGE_MAX 4095.0f
#elif PWM_RESOLUTION_BITS == 11
#define PWM_RANGE_MAX 2047.0f
#elif PWM_RESOLUTION_BITS == 10
#define PWM_RANGE_MAX 1023.0f
#elif PWM_RESOLUTION_BITS == 9
#define PWM_RANGE_MAX 511.0f
#elif PWM_RESOLUTION_BITS == 8
#define PWM_RANGE_MAX 255.0f
#elif PWM_RESOLUTION_BITS == 7
#define PWM_RANGE_MAX 127.0f
#elif PWM_RESOLUTION_BITS == 6
#define PWM_RANGE_MAX 63.0f
#elif PWM_RESOLUTION_BITS == 5
#define PWM_RANGE_MAX 31.0f
#elif PWM_RESOLUTION_BITS == 4
#define PWM_RANGE_MAX 15.0f
#elif PWM_RESOLUTION_BITS == 3
#define PWM_RANGE_MAX 7.0f
#elif PWM_RESOLUTION_BITS == 2
#define PWM_RANGE_MAX 3.0f
#endif

#define HCSR04_PIN_TRIGGER 20
#define HCSR04_PIN_ECHO 21
#define HCSR04_MAX_DIST_CM 400
#define HCSR04_MAX_TIMEOUT_MS 0

#define IMU_SAMPLE_PERIOD_S 0.00166668 // 600Hz
#define TELEM_SAMPLE_PERIOD_S 0.5      // 100Hz
#define ROLL_PID_KP 0.05
#define ROLL_PID_KI 0.0
#define ROLL_PID_KD 0.005
#define PITCH_PID_KP 0.05
#define PITCH_PID_KI 0.0
#define PITCH_PID_KD 0.001
#define YAW_PID_KP -0.05
#define YAW_PID_KI 0.0
#define YAW_PID_KD 0.0
