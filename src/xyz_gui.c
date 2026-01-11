#include "xyz_gui.h"

#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#endif

// GLOBAL VARIABLES
char _window_title[100] = {0};
int _window_loop = 1;
int _window_width = 0;
int _window_height = 0;
float _frame_dt = 0.0f;
float _frame_last = 0.0f;

gl_camera_t _camera;
float _camera_speed = 0.001f;

float _mouse_sensitivity = 0.02f;
int _mouse_button_left = 0;
int _mouse_button_right = 0;
double _cursor_x = 0.0;
double _cursor_y = 0.0;
double _cursor_dx = 0.0;
double _cursor_dy = 0.0;
double _cursor_last_x = 0.0;
double _cursor_last_y = 0.0;
int _cursor_is_dragging = 0;
int _ui_engaged = 0;

int _key_q = 0;
int _key_w = 0;
int _key_a = 0;
int _key_s = 0;
int _key_d = 0;
int _key_n = 0;
int _key_esc = 0;
int _key_equal = 0;
int _key_minus = 0;

/******************************************************************************
 * OPENGL UTILS
 *****************************************************************************/

#define GL_DEL_VERTEX_ARRAY(X)                                                 \
  if (glIsVertexArray(X) == GL_TRUE) {                                         \
    glDeleteVertexArrays(1, &X);                                               \
  }

#define GL_DEL_BUFFER(X)                                                       \
  if (glIsBuffer(X) == GL_TRUE) {                                              \
    glDeleteBuffers(1, &X);                                                    \
  }

/**
 * Read file contents in file path `fp`.
 * @returns
 * - Success: File contents
 * - Failure: NULL
 */
char *load_file(const char *fp) {
  assert(fp != NULL);
  FILE *f = fopen(fp, "rb");
  if (f == NULL) {
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long int len = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *buf = malloc(sizeof(char) * len + 1);
  if (buf == NULL) {
    fclose(f);
    return NULL;
  }
  const ssize_t read = fread(buf, 1, len, f);
  if (read != len) {
    FATAL("Failed to read file [%s]\n", fp);
  }
  buf[len] = '\0';
  fclose(f);

  return buf;
}

gl_enum_t gl_check_error(const char *file, const int line) {
  gl_enum_t error_code;

  while ((error_code = glGetError()) != GL_NO_ERROR) {
    char error[1000] = {0};
    switch (error_code) {
      case GL_INVALID_ENUM:
        strcpy(error, "INVALID_ENUM");
        break;
      case GL_INVALID_VALUE:
        strcpy(error, "INVALID_VALUE");
        break;
      case GL_INVALID_OPERATION:
        strcpy(error, "INVALID_OPERATION");
        break;
      case GL_STACK_OVERFLOW:
        strcpy(error, "STACK_OVERFLOW");
        break;
      case GL_STACK_UNDERFLOW:
        strcpy(error, "STACK_UNDERFLOW");
        break;
      case GL_OUT_OF_MEMORY:
        strcpy(error, "OUT_OF_MEMORY");
        break;
      case GL_INVALID_FRAMEBUFFER_OPERATION:
        strcpy(error, "INVALID_FRAMEBUFFER_OPERATION");
        break;
    }
    printf("%s | %s:%d\n", error, file, line);
  }

  return error_code;
}

/**
 * Generate random number between a and b from a uniform distribution.
 * @returns Random number
 */
gl_float_t gl_randf(const gl_float_t a, const gl_float_t b) {
  float random = ((float) rand()) / (float) RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

gl_float_t gl_deg2rad(const gl_float_t d) { return d * M_PI / 180.0f; }

gl_float_t gl_rad2deg(const gl_float_t r) { return r * 180.0f / M_PI; }

void gl_print_vector(const char *prefix,
                     const gl_float_t *x,
                     const int length) {
  printf("%s: [", prefix);
  for (int i = 0; i < length; i++) {
    printf("%f", x[i]);
    if ((i + 1) != length) {
      printf(", ");
    }
  }
  printf("]\n");
}

void gl_print_matrix(const char *prefix,
                     const gl_float_t *A,
                     const int num_rows,
                     const int num_cols) {
  printf("%s:\n", prefix);
  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      printf("%f", A[i + (j * num_rows)]);
      if ((j + 1) != num_cols) {
        printf(", ");
      }
    }
    printf("\n");
  }
  printf("\n");
}

void gl_zeros(gl_float_t *A, const int num_rows, const int num_cols) {
  for (int i = 0; i < (num_rows * num_cols); i++) {
    A[i] = 0.0f;
  }
}

void gl_ones(gl_float_t *A, const int num_rows, const int num_cols) {
  for (int i = 0; i < (num_rows * num_cols); i++) {
    A[i] = 1.0f;
  }
}

void gl_eye(gl_float_t *A, const int num_rows, const int num_cols) {
  int idx = 0;
  for (int j = 0; j < num_cols; j++) {
    for (int i = 0; i < num_rows; i++) {
      A[idx++] = (i == j) ? 1.0f : 0.0f;
    }
  }
}

void gl_vec2(gl_float_t *v, const gl_float_t x, const gl_float_t y) {
  v[0] = x;
  v[1] = y;
}

void gl_vec3(gl_float_t *v,
             const gl_float_t x,
             const gl_float_t y,
             const gl_float_t z) {
  v[0] = x;
  v[1] = y;
  v[2] = z;
}

void gl_vec4(gl_float_t *v,
             const gl_float_t x,
             const gl_float_t y,
             const gl_float_t z,
             const gl_float_t w) {
  v[0] = x;
  v[1] = y;
  v[2] = z;
  v[3] = w;
}

int gl_equals(const gl_float_t *A,
              const gl_float_t *B,
              const int num_rows,
              const int num_cols,
              const gl_float_t tol) {
  for (int i = 0; i < (num_rows * num_cols); i++) {
    if (fabs(A[i] - B[i]) > tol) {
      return 0;
    }
  }

  return 1;
}

void gl_mat_set(gl_float_t *A,
                const int m,
                const int n,
                const int i,
                const int j,
                const gl_float_t val) {
  UNUSED(n);
  A[i + (j * m)] = val;
}

gl_float_t gl_mat_val(const gl_float_t *A,
                      const int m,
                      const int n,
                      const int i,
                      const int j) {
  UNUSED(n);
  return A[i + (j * m)];
}

void gl_copy(const gl_float_t *src,
             const int m,
             const int n,
             gl_float_t *dest) {
  for (int i = 0; i < (m * n); i++) {
    dest[i] = src[i];
  }
}

void gl_transpose(const gl_float_t *A, size_t m, size_t n, gl_float_t *A_t) {
  assert(A != NULL && A != A_t);
  assert(m > 0 && n > 0);

  int idx = 0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A_t[idx++] = gl_mat_val(A, m, n, i, j);
    }
  }
}

void gl_vec3_cross(const gl_float_t u[3],
                   const gl_float_t v[3],
                   gl_float_t n[3]) {
  assert(u);
  assert(v);
  assert(n);

  n[0] = u[1] * v[2] - u[2] * v[1];
  n[1] = u[2] * v[0] - u[0] * v[2];
  n[2] = u[0] * v[1] - u[1] * v[0];
}

void gl_add(const gl_float_t *A,
            const gl_float_t *B,
            const int num_rows,
            const int num_cols,
            gl_float_t *C) {
  for (int i = 0; i < (num_rows * num_cols); i++) {
    C[i] = A[i] + B[i];
  }
}

void gl_sub(const gl_float_t *A,
            const gl_float_t *B,
            const int num_rows,
            const int num_cols,
            gl_float_t *C) {
  for (int i = 0; i < (num_rows * num_cols); i++) {
    C[i] = A[i] - B[i];
  }
}

void gl_dot(const gl_float_t *A,
            const int A_m,
            const int A_n,
            const gl_float_t *B,
            const int B_m,
            const int B_n,
            gl_float_t *C) {
  assert(A != C && B != C);
  assert(A_n == B_m);

  int m = A_m;
  int n = B_n;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < A_n; k++) {
        C[i + (j * n)] += A[i + (k * A_n)] * B[k + (j * B_n)];
      }
    }
  }
}

void gl_scale(gl_float_t factor,
              gl_float_t *A,
              const int num_rows,
              const int num_cols) {
  for (int i = 0; i < (num_rows * num_cols); i++) {
    A[i] *= factor;
  }
}

gl_float_t gl_norm(const gl_float_t *x, const int size) {
  gl_float_t sum_sq = 0.0f;
  for (int i = 0; i < size; i++) {
    sum_sq += x[i] * x[i];
  }

  return sqrt(sum_sq);
}

void gl_normalize(gl_float_t *x, const int size) {
  const gl_float_t n = gl_norm(x, size);
  for (int i = 0; i < size; i++) {
    x[i] /= n;
  }
}

void gl_perspective(const gl_float_t fov,
                    const gl_float_t aspect,
                    const gl_float_t near,
                    const gl_float_t far,
                    gl_float_t P[4 * 4]) {
  const gl_float_t f = 1.0f / tan(fov * 0.5f);

  gl_zeros(P, 4, 4);
  P[0] = f / aspect;
  P[1] = 0.0f;
  P[2] = 0.0f;
  P[3] = 0.0f;

  P[4] = 0.0f;
  P[5] = f;
  P[6] = 0.0f;
  P[7] = 0.0f;

  P[8] = 0.0f;
  P[9] = 0.0f;
  P[10] = (far + near) / (near - far);
  P[11] = -1;

  P[12] = 0.0f;
  P[13] = 0.0f;
  P[14] = (2 * far * near) / (near - far);
  P[15] = 0.0f;
}

void gl_ortho(const gl_float_t w, const gl_float_t h, gl_float_t P[4 * 4]) {
  const gl_float_t left = 0.0f;
  const gl_float_t right = w;
  const gl_float_t top = 0.0f;
  const gl_float_t bottom = h;
  const gl_float_t znear = -1.0f;
  const gl_float_t zfar = 1.0f;

  gl_zeros(P, 4, 4);
  P[0] = 2.0f / (right - left);
  P[1] = 0.0f;
  P[2] = 0.0f;
  P[3] = 0.0f;

  P[4] = 0.0f;
  P[5] = 2.0f / (top - bottom);
  P[6] = 0.0f;
  P[7] = 0.0f;

  P[8] = 0.0f;
  P[9] = 0.0f;
  P[10] = -2.0f / (zfar - znear);
  P[11] = 0.0f;

  P[12] = -(right + left) / (right - left);
  P[13] = -(top + bottom) / (top - bottom);
  P[14] = -(zfar + znear) / (zfar - znear);
  P[15] = 1.0f;
}

void gl_lookat(const gl_float_t eye[3],
               const gl_float_t at[3],
               const gl_float_t up[3],
               gl_float_t V[4 * 4]) {
  // Z-axis: Camera forward
  gl_float_t z[3] = {0};
  gl_sub(at, eye, 3, 1, z);
  gl_normalize(z, 3);

  // X-axis: Camera right
  gl_float_t x[3] = {0};
  gl_vec3_cross(z, up, x);
  gl_normalize(x, 3);

  // Y-axis: Camera up
  gl_float_t y[3] = {0};
  gl_vec3_cross(x, z, y);

  // Negate z-axis
  gl_scale(-1.0f, z, 3, 1);

  // Form rotation component
  gl_float_t R[4 * 4] = {0};
  R[0] = x[0];
  R[1] = y[0];
  R[2] = z[0];
  R[3] = 0.0f;

  R[4] = x[1];
  R[5] = y[1];
  R[6] = z[1];
  R[7] = 0.0f;

  R[8] = x[2];
  R[9] = y[2];
  R[10] = z[2];
  R[11] = 0.0f;

  R[12] = 0.0f;
  R[13] = 0.0f;
  R[14] = 0.0f;
  R[15] = 1.0f;

  // Form translation component
  gl_float_t T[4 * 4] = {0};
  T[0] = 1.0f;
  T[1] = 0.0f;
  T[2] = 0.0f;
  T[3] = 0.0f;

  T[4] = 0.0f;
  T[5] = 1.0f;
  T[6] = 0.0f;
  T[7] = 0.0f;

  T[8] = 0.0f;
  T[9] = 0.0f;
  T[10] = 1.0f;
  T[11] = 0.0f;

  T[12] = -eye[0];
  T[13] = -eye[1];
  T[14] = -eye[2];
  T[15] = 1.0f;

  // Form view matrix
  gl_zeros(V, 4, 4);
  gl_dot(R, 4, 4, T, 4, 4, V);
}

/**
 * Convert Euler angles `euler` in degrees to a 3x3 rotation
 * matrix `C`.
 */
void gl_euler321(const gl_float_t euler[3], gl_float_t C[3 * 3]) {
  assert(euler != NULL);
  assert(C != NULL);

  const gl_float_t psi = euler[2];
  const gl_float_t theta = euler[1];
  const gl_float_t phi = euler[0];

  const gl_float_t cpsi = cos(psi);
  const gl_float_t spsi = sin(psi);

  const gl_float_t ctheta = cos(theta);
  const gl_float_t stheta = sin(theta);

  const gl_float_t cphi = cos(phi);
  const gl_float_t sphi = sin(phi);

  // 1st row
  C[0] = cpsi * ctheta;
  C[3] = cpsi * stheta * sphi - spsi * cphi;
  C[6] = cpsi * stheta * cphi + spsi * sphi;

  // 2nd row
  C[1] = spsi * ctheta;
  C[4] = spsi * stheta * sphi + cpsi * cphi;
  C[7] = spsi * stheta * cphi - cpsi * sphi;

  // 3rd row
  C[2] = -stheta;
  C[5] = ctheta * sphi;
  C[8] = ctheta * cphi;
}

/**
 * Convert Euler angles `euler` in radians to a Hamiltonian Quaternion.
 */
void gl_euler2quat(const gl_float_t euler[3], gl_quat_t *q) {
  const gl_float_t psi = euler[2];
  const gl_float_t theta = euler[1];
  const gl_float_t phi = euler[0];

  const gl_float_t cphi = cos(phi / 2.0);
  const gl_float_t ctheta = cos(theta / 2.0);
  const gl_float_t cpsi = cos(psi / 2.0);
  const gl_float_t sphi = sin(phi / 2.0);
  const gl_float_t stheta = sin(theta / 2.0);
  const gl_float_t spsi = sin(psi / 2.0);

  const gl_float_t qx = sphi * ctheta * cpsi - cphi * stheta * spsi;
  const gl_float_t qy = cphi * stheta * cpsi + sphi * ctheta * spsi;
  const gl_float_t qz = cphi * ctheta * spsi - sphi * stheta * cpsi;
  const gl_float_t qw = cphi * ctheta * cpsi + sphi * stheta * spsi;

  const gl_float_t mag = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  q->w = qw / mag;
  q->x = qx / mag;
  q->y = qy / mag;
  q->z = qz / mag;
}

/**
 * Convert Quaternion `q` to 3x3 rotation matrix `C`.
 */
void gl_quat2rot(const gl_quat_t *q, gl_float_t C[3 * 3]) {
  assert(q != NULL);
  assert(C != NULL);

  const gl_float_t qx2 = q->x * q->x;
  const gl_float_t qy2 = q->y * q->y;
  const gl_float_t qz2 = q->z * q->z;
  const gl_float_t qw2 = q->w * q->w;

  // Homogeneous form
  // -- 1st row
  C[0] = qw2 + qx2 - qy2 - qz2;
  C[3] = 2 * (q->x * q->y - q->w * q->z);
  C[6] = 2 * (q->x * q->z + q->w * q->y);
  // -- 2nd row
  C[1] = 2 * (q->x * q->y + q->w * q->z);
  C[4] = qw2 - qx2 + qy2 - qz2;
  C[7] = 2 * (q->y * q->z - q->w * q->x);
  // -- 3rd row
  C[2] = 2 * (q->x * q->z - q->w * q->y);
  C[5] = 2 * (q->y * q->z + q->w * q->x);
  C[8] = qw2 - qx2 - qy2 + qz2;
}

/**
 * Convert 3x3 rotation matrix `C` to Quaternion `q`.
 */
void gl_rot2quat(const gl_float_t C[3 * 3], gl_quat_t *q) {
  assert(C != NULL);
  assert(q != NULL);

  const gl_float_t C00 = C[0];
  const gl_float_t C01 = C[3];
  const gl_float_t C02 = C[6];
  const gl_float_t C10 = C[1];
  const gl_float_t C11 = C[4];
  const gl_float_t C12 = C[7];
  const gl_float_t C20 = C[2];
  const gl_float_t C21 = C[5];
  const gl_float_t C22 = C[8];
  const gl_float_t tr = C00 + C11 + C22;

  if (tr > 0) {
    const gl_float_t S = sqrt(tr + 1.0) * 2; // S=4*qw
    q->w = 0.25 * S;
    q->x = (C21 - C12) / S;
    q->y = (C02 - C20) / S;
    q->z = (C10 - C01) / S;
  } else if ((C00 > C11) && (C[0] > C22)) {
    const gl_float_t S = sqrt(1.0 + C[0] - C11 - C22) * 2; // S=4*qx
    q->w = (C21 - C12) / S;
    q->x = 0.25 * S;
    q->y = (C01 + C10) / S;
    q->z = (C02 + C20) / S;
  } else if (C11 > C22) {
    const gl_float_t S = sqrt(1.0 + C11 - C[0] - C22) * 2; // S=4*qy
    q->w = (C02 - C20) / S;
    q->x = (C01 + C10) / S;
    q->y = 0.25 * S;
    q->z = (C12 + C21) / S;
  } else {
    const gl_float_t S = sqrt(1.0 + C22 - C[0] - C11) * 2; // S=4*qz
    q->w = (C10 - C01) / S;
    q->x = (C02 + C20) / S;
    q->y = (C12 + C21) / S;
    q->z = 0.25 * S;
  }
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a 7x1 pose vector
 * `params`.
 *
 *    pose = (translation, rotation)
 *    pose = (rx, ry, rz, qw, qx, qy, qz)
 */
void gl_tf(const gl_float_t params[7], gl_float_t T[4 * 4]) {
  assert(params != NULL);
  assert(T != NULL);

  const gl_float_t r[3] = {params[0], params[1], params[2]};
  const gl_quat_t q = {.w = params[3],
                       .x = params[4],
                       .y = params[5],
                       .z = params[6]};

  gl_float_t C[3 * 3] = {0};
  gl_quat2rot(&q, C);

  T[0] = C[0];
  T[1] = C[1];
  T[2] = C[2];
  T[3] = 0.0f;

  T[4] = C[3];
  T[5] = C[4];
  T[6] = C[5];
  T[7] = 0.0f;

  T[8] = C[6];
  T[9] = C[7];
  T[10] = C[8];
  T[11] = 0.0f;

  T[12] = r[0];
  T[13] = r[1];
  T[14] = r[2];
  T[15] = 1.0f;
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a rotation matrix `C`
 * and translation vector `r`.
 */
void gl_tf_cr(const gl_float_t C[3 * 3],
              const gl_float_t r[3],
              gl_float_t T[4 * 4]) {
  T[0] = C[0];
  T[1] = C[1];
  T[2] = C[2];
  T[3] = 0.0f;

  T[4] = C[3];
  T[5] = C[4];
  T[6] = C[5];
  T[7] = 0.0f;

  T[8] = C[6];
  T[9] = C[7];
  T[10] = C[8];
  T[11] = 0.0f;

  T[12] = r[0];
  T[13] = r[1];
  T[14] = r[2];
  T[15] = 1.0f;
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a quaternion `q` and
 * translation vector `r`.
 */
void gl_tf_qr(const gl_quat_t *q, const gl_float_t r[3], gl_float_t T[4 * 4]) {
  gl_float_t C[3 * 3] = {0};
  gl_quat2rot(q, C);
  gl_tf_cr(C, r, T);
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a euler-angles `euler`
 * (yaw-pitch-roll) and translation vector `r`.
 */
void gl_tf_er(const gl_float_t euler[3],
              const gl_float_t r[3],
              gl_float_t T[4 * 4]) {
  gl_float_t C[3 * 3] = {0};
  gl_euler321(euler, C);
  gl_tf_cr(C, r, T);
}

void gl_flu2rub(const gl_float_t T_flu[4 * 4], gl_float_t *T_rub) {
  // T_rub = T * T_flu * T'
  //
  // FLU X (Forward) -> RUB -Z (Back)
  // FLU Y (Left)    -> RUB -X (Right)
  // FLU Z (Up)      -> RUB  Y (Up)
  //
  // [row major]
  // T = [ 0 -1  0  0
  //       0  0  1  0
  //      -1  0  0  0
  //       0  0  0  1]

  // clang-format off
  gl_float_t T[4* 4] = {0};
  gl_float_t T_[4* 4] = {0};
  T[0] =  0.0f; T[4] = -1.0f; T[8]  = 0.0f; T[12] = 0.0f;
  T[1] =  0.0f; T[5] =  0.0f; T[9]  = 1.0f; T[13] = 0.0f;
  T[2] = -1.0f; T[6] =  0.0f; T[10] = 0.0f; T[14] = 0.0f;
  T[3] =  0.0f; T[7] =  0.0f; T[11] = 0.0f; T[15] = 1.0f;
  gl_transpose(T, 4, 4, T_);
  // clang-format on

  gl_float_t tmp[4 * 4] = {0};
  gl_dot(T, 4, 4, T_flu, 4, 4, tmp);
  gl_dot(tmp, 4, 4, T_, 4, 4, T_rub);
}

void gl_tf2pose(const gl_float_t T[4 * 4], gl_pose_t *pose) {
  pose->pos.x = T[12];
  pose->pos.y = T[13];
  pose->pos.z = T[14];

  // clang-format off
  gl_float_t C[3 * 3] = {0};
  C[0] = T[0]; C[1] = T[1]; C[2] = T[2];
  C[3] = T[4]; C[4] = T[5]; C[5] = T[6];
  C[6] = T[8]; C[7] = T[9]; C[8] = T[10];
  gl_rot2quat(C, &pose->quat);
  // clang-format on
}

void gl_pose2tf(const gl_pose_t *pose, gl_float_t T[4 * 4]) {
  gl_float_t C[3 * 3] = {0};
  gl_quat2rot(&pose->quat, C);

  // clang-format off
  T[0] = C[0];  T[1]  = C[1]; T[2]  = C[2]; T[3]  = pose->pos.x;
  T[4] = C[3];  T[5]  = C[4]; T[6]  = C[5]; T[7]  = pose->pos.y;
  T[8] = C[6];  T[9]  = C[7]; T[10] = C[8]; T[11] = pose->pos.z;
  T[12] = 0.0f; T[13] = 0.0f; T[14] = 0.0f; T[15] = 1.0f;
  // clang-format on
}

int gl_save_frame_buffer(const int width, const int height, const char *fp) {
  // Malloc pixels
  const int num_channels = 3;
  const size_t num_pixels = num_channels * width * height;
  GLubyte *pixels = malloc(sizeof(GLubyte) * num_pixels);

  // Read pixels
  const gl_int_t x = 0;
  const gl_int_t y = 0;
  glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

  // Write to file
  GLsizei stride = num_channels * width;
  stride += (stride % 4) ? (4 - stride % 4) : 0;
  stbi_flip_vertically_on_write(1);
  stbi_write_png(fp, width, height, num_channels, pixels, stride);

  // Clean up
  if (pixels) {
    free(pixels);
  }

  return 0;
}

void gl_jet_colormap(const gl_float_t value,
                     gl_float_t *r,
                     gl_float_t *g,
                     gl_float_t *b) {
  // Clamp value to [0,1]
  gl_float_t value_ = (value < 0.0f) ? 0.0f : value;
  value_ = (value_ > 1.0f) ? 1.0f : value_;

  gl_float_t four_value = value_ * 4.0f;      // Scale to 0-4
  int region = (int) four_value;              // Get integer part
  gl_float_t remainder = four_value - region; // Fractional part

  switch (region) {
    case 0:
      *r = 0.0f;
      *g = remainder;
      *b = 1.0f;
      break; // Blue to Cyan
    case 1:
      *r = 0.0f;
      *g = 1.0f;
      *b = 1.0f - remainder;
      break; // Cyan to Green
    case 2:
      *r = remainder;
      *g = 1.0f;
      *b = 0.0f;
      break; // Green to Yellow
    case 3:
      *r = 1.0f;
      *g = 1.0f - remainder;
      *b = 0.0f;
      break; // Yellow to Red
    default:
      *r = 1.0f;
      *g = 0.0f;
      *b = 0.0f;
      break; // Red (value >= 1)
  }
}

/******************************************************************************
 * GL-SHADER
 *****************************************************************************/

// void gl_shader_cleanup(gl_shader_t *shader) {
//   if (glIsProgram(shader->program_id) == GL_TRUE) {
//     glDeleteProgram(shader->program_id);
//   }
// }

gl_uint_t gl_compile(const char *src, const int type) {
  if (src == NULL) {
    LOG_ERROR("Shader source is NULL!");
    return GL_FALSE;
  }

  const gl_uint_t shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, NULL);
  glCompileShader(shader);

  gl_int_t retval = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &retval);
  if (retval == GL_FALSE) {
    char log[9046] = {0};
    glGetShaderInfoLog(shader, 9046, NULL, log);
    LOG_ERROR("Failed to compile shader:\n%s", log);
    // LOG_ERROR("source:\n%s", src);
    return retval;
  }

  return shader;
}

void gl_shader_status(gl_uint_t shader) {
  gl_int_t success = 0;
  GLchar infoLog[1024];

  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(shader, 1024, NULL, infoLog);
    printf("ERROR::SHADER_COMPILATION_ERROR:\n");
    printf("%s\n", infoLog);
    printf("\n -- --------------------------------------------------- -- ");
    printf("\n");
  }

  glGetProgramiv(shader, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shader, 1024, NULL, infoLog);
    printf("ERROR::PROGRAM_LINKING_ERROR:\n");
    printf("%s\n", infoLog);
    printf("\n -- --------------------------------------------------- -- ");
    printf("\n");
  }
}

gl_uint_t gl_link(const gl_uint_t vs, const gl_uint_t fs, const gl_uint_t gs) {
  // Attach shaders to link
  gl_uint_t program = glCreateProgram();
  glAttachShader(program, vs);
  if (gs != GL_FALSE) {
    glAttachShader(program, gs);
  }
  glAttachShader(program, fs);
  glLinkProgram(program);

  // Link program
  gl_int_t success = 0;
  char log[9046];
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (success == GL_FALSE) {
    glGetProgramInfoLog(program, 9046, NULL, log);
    LOG_ERROR("Failed to link shaders:\nReason: %s\n", log);
    return GL_FALSE;
  }

  // Delete shaders
  glDeleteShader(vs);
  if (gs == GL_FALSE) {
    glDeleteShader(gs);
  }
  glDeleteShader(fs);

  return program;
}

gl_uint_t gl_shader(const char *vs_src,
                    const char *fs_src,
                    const char *gs_src) {
  gl_uint_t vs = GL_FALSE;
  gl_uint_t fs = GL_FALSE;
  gl_uint_t gs = GL_FALSE;

  if (vs_src) {
    vs = gl_compile(vs_src, GL_VERTEX_SHADER);
  }

  if (gs_src) {
    gs = gl_compile(gs_src, GL_GEOMETRY_SHADER);
  }

  if (fs_src) {
    fs = gl_compile(fs_src, GL_FRAGMENT_SHADER);
  }

  const gl_uint_t program_id = gl_link(vs, fs, gs);
  return program_id;
}

int gl_set_color(const gl_int_t id, const char *k, const gl_color_t v) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform3f(location, v.r, v.g, v.b);
  return 0;
}

int gl_set_int(const gl_int_t id, const char *k, const gl_int_t v) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform1i(location, v);
  return 0;
}

int gl_set_float(const gl_int_t id, const char *k, const gl_float_t v) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform1f(location, v);
  return 0;
}

int gl_set_vec2i(const gl_int_t id, const char *k, const gl_int_t v[2]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform2i(location, v[0], v[1]);
  return 0;
}

int gl_set_vec3i(const gl_int_t id, const char *k, const gl_int_t v[3]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform3i(location, v[0], v[1], v[2]);
  return 0;
}

int gl_set_vec4i(const gl_int_t id, const char *k, const gl_int_t v[4]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform4i(location, v[0], v[1], v[2], v[3]);
  return 0;
}

int gl_set_vec2(const gl_int_t id, const char *k, const gl_float_t v[2]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform2f(location, v[0], v[1]);
  return 0;
}

int gl_set_vec3(const gl_int_t id, const char *k, const gl_float_t v[3]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform3f(location, v[0], v[1], v[2]);
  return 0;
}

int gl_set_vec4(const gl_int_t id, const char *k, const gl_float_t v[4]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniform4f(location, v[0], v[1], v[2], v[3]);
  return 0;
}

int gl_set_mat2(const gl_int_t id, const char *k, const gl_float_t v[2 * 2]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniformMatrix2fv(location, 1, GL_FALSE, v);
  return 0;
}

int gl_set_mat3(const gl_int_t id, const char *k, const gl_float_t v[3 * 3]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniformMatrix3fv(location, 1, GL_FALSE, v);
  return 0;
}

int gl_set_mat4(const gl_int_t id, const char *k, const gl_float_t v[4 * 4]) {
  const gl_int_t location = glGetUniformLocation(id, k);
  if (location == -1) {
    return -1;
  }

  glUniformMatrix4fv(location, 1, GL_FALSE, v);
  return 0;
}

/******************************************************************************
 * GL-CAMERA
 *****************************************************************************/

void gl_camera_setup(gl_camera_t *camera,
                     int *window_width,
                     int *window_height) {
  assert(camera);
  assert(window_width);
  assert(window_height);

  camera->view_mode = FPS;

  gl_zeros(camera->focal, 3, 1);
  gl_vec3(camera->world_up, 0.0f, 1.0f, 0.0f);
  gl_vec3(camera->position, 0.0f, 0.0f, 0.0f);
  gl_vec3(camera->right, -1.0f, 0.0f, 0.0f);
  gl_vec3(camera->up, 0.0f, 1.0f, 0.0f);
  gl_vec3(camera->front, 0.0f, 0.0f, -1.0f);
  camera->yaw = gl_deg2rad(180.0f);
  camera->pitch = gl_deg2rad(-45.0f);
  camera->radius = 1.0f;

  camera->fov = gl_deg2rad(60.0f);
  camera->fov_min = gl_deg2rad(10.0f);
  camera->fov_max = gl_deg2rad(120.0f);
  camera->near = 0.01f;
  camera->far = 500.0f;

  gl_camera_update(camera);
}

void gl_camera_update(gl_camera_t *camera) {
  assert(camera);

  // Front vector
  camera->front[0] = sin(camera->yaw) * cos(camera->pitch);
  camera->front[1] = sin(camera->pitch);
  camera->front[2] = cos(camera->yaw) * cos(camera->pitch);
  gl_normalize(camera->front, 3);

  // Right vector
  gl_vec3_cross(camera->front, camera->world_up, camera->right);
  gl_normalize(camera->right, 3);

  // Up vector
  gl_vec3_cross(camera->right, camera->front, camera->up);
  gl_normalize(camera->up, 3);

  // Projection matrix
  const float aspect = (float) _window_width / _window_height;
  gl_perspective(camera->fov, aspect, camera->near, camera->far, camera->P);

  // View matrix (Orbit mode)
  if (camera->view_mode == ORBIT) {
    camera->position[0] =
        camera->radius * sin(camera->pitch) * sin(camera->yaw);
    camera->position[1] = camera->radius * cos(camera->pitch);
    camera->position[2] =
        camera->radius * sin(camera->pitch) * cos(camera->yaw);

    gl_float_t eye[3] = {0};
    eye[0] = camera->position[0];
    eye[1] = camera->position[1];
    eye[2] = camera->position[2];
    gl_lookat(eye, camera->focal, camera->world_up, camera->V);
  }

  // View matrix (FPS mode)
  if (camera->view_mode == FPS) {
    gl_float_t eye[3] = {0};
    eye[0] = camera->position[0];
    eye[1] = camera->position[1];
    eye[2] = camera->position[2];

    gl_float_t carrot[3] = {0};
    carrot[0] = camera->position[0] + camera->front[0];
    carrot[1] = camera->position[1] + camera->front[1];
    carrot[2] = camera->position[2] + camera->front[2];

    gl_lookat(eye, carrot, camera->world_up, camera->V);
  }
}

void gl_camera_rotate(gl_camera_t *camera,
                      const float factor,
                      const float dx,
                      const float dy) {
  assert(camera);

  // Update yaw and pitch
  float pitch = camera->pitch;
  float yaw = camera->yaw;
  yaw -= dx * factor;
  pitch -= dy * factor;

  // Constrain pitch and yaw
  pitch = (pitch > gl_deg2rad(89.99f)) ? gl_deg2rad(89.99f) : pitch;
  pitch = (pitch < gl_deg2rad(-89.99f)) ? gl_deg2rad(-89.99f) : pitch;

  // Update camera attitude
  camera->pitch = pitch;
  camera->yaw = yaw;

  // Update camera forward
  float direction[3] = {0};
  direction[0] = cos(yaw) * cos(pitch);
  direction[1] = sin(pitch);
  direction[2] = sin(yaw) * cos(pitch);
  gl_normalize(direction, 3);

  camera->front[0] = direction[0];
  camera->front[1] = direction[1];
  camera->front[2] = direction[2];
}

void gl_camera_pan(gl_camera_t *camera,
                   const float factor,
                   const float dx,
                   const float dy) {
  assert(camera);

  // camera->focal -= (dy * _mouse_sensitivity) * camera->front;
  // camera->focal += (dx * _mouse_sensitivity) * camera->right;
  const gl_float_t dx_scaled = dx * factor;
  const gl_float_t dy_scaled = dy * factor;
  gl_float_t front[3] = {camera->front[0], camera->front[1], camera->front[2]};
  gl_float_t right[3] = {camera->right[0], camera->right[1], camera->right[2]};
  gl_scale(dy_scaled, front, 3, 1);
  gl_scale(dx_scaled, right, 3, 1);
  gl_sub(camera->focal, front, 3, 1, camera->focal);
  gl_add(camera->focal, right, 3, 1, camera->focal);

  // limit focal point y-axis
  camera->focal[1] = (camera->focal[1] < 0) ? 0 : camera->focal[1];
}

void gl_camera_zoom(gl_camera_t *camera,
                    const float factor,
                    const float dx,
                    const float dy) {
  assert(camera);

  UNUSED(factor);
  UNUSED(dx);
  gl_float_t fov = camera->fov + dy;
  fov = (fov <= camera->fov_min) ? camera->fov_min : fov;
  fov = (fov >= camera->fov_max) ? camera->fov_max : fov;
  camera->fov = fov;
}

/******************************************************************************
 * GUI
 *****************************************************************************/

void window_callback(GLFWwindow *window, int width, int height) {
  assert(window);
  assert(width > 0);
  assert(height > 0);

  _window_width = width;
  _window_height = height;

  // Maintain aspect ratio
  const float aspect = 16.0f / 9.0f;
  int new_width = 0;
  int new_height = 0;
  if (width / (float) height > aspect) {
    new_width = (int) (height * aspect);
    new_height = height;
  } else {
    new_width = width;
    new_height = (int) (width / aspect);
  }

  // Center the viewport
  int x_offset = (width - new_width) / 2;
  int y_offset = (height - new_height) / 2;
  glViewport(x_offset, y_offset, new_width, new_height);
}

void gui_process_input(GLFWwindow *window) {
  assert(window);

  // Handle keyboard events
  // -- Key press
  _key_esc = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
  _key_q = glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS;
  _key_w = glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS;
  _key_a = glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS;
  _key_s = glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS;
  _key_d = glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS;
  _key_n = glfwGetKey(window, GLFW_KEY_N) == GLFW_PRESS;
  _key_equal = glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS;
  _key_minus = glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS;
  if (_key_esc || _key_q) {
    _window_loop = 0;
  }

  // -- FPS MODE
  if (_camera.view_mode == FPS) {
    if (_key_w) {
      _camera.position[0] += _camera.front[0] * _camera_speed * _frame_dt;
      _camera.position[1] += _camera.front[1] * _camera_speed * _frame_dt;
      _camera.position[2] += _camera.front[2] * _camera_speed * _frame_dt;
    } else if (_key_s) {
      _camera.position[0] -= _camera.front[0] * _camera_speed * _frame_dt;
      _camera.position[1] -= _camera.front[1] * _camera_speed * _frame_dt;
      _camera.position[2] -= _camera.front[2] * _camera_speed * _frame_dt;
    } else if (_key_a) {
      gl_float_t camera_left[3] = {0};
      gl_vec3_cross(_camera.front, _camera.up, camera_left);
      gl_normalize(camera_left, 3);
      _camera.position[0] -= camera_left[0] * _camera_speed * _frame_dt;
      _camera.position[1] -= camera_left[1] * _camera_speed * _frame_dt;
      _camera.position[2] -= camera_left[2] * _camera_speed * _frame_dt;
    } else if (_key_d) {
      gl_float_t camera_left[3] = {0};
      gl_vec3_cross(_camera.front, _camera.up, camera_left);
      gl_normalize(camera_left, 3);
      _camera.position[0] += camera_left[0] * _camera_speed * _frame_dt;
      _camera.position[1] += camera_left[1] * _camera_speed * _frame_dt;
      _camera.position[2] += camera_left[2] * _camera_speed * _frame_dt;
    } else if (_key_equal) {
      gl_camera_zoom(&_camera, 1.0, 0, _camera_speed * _frame_dt);
    } else if (_key_minus) {
      gl_camera_zoom(&_camera, 1.0, 0, -_camera_speed * _frame_dt);
    }
  }

  // -- ORBIT MODE
  if (_camera.view_mode == ORBIT) {
    if (_key_w) {
      _camera.pitch += 0.01;
      _camera.pitch = (_camera.pitch >= M_PI) ? M_PI : _camera.pitch;
      _camera.pitch = (_camera.pitch <= 0.0f) ? 0.0f : _camera.pitch;
    } else if (_key_s) {
      _camera.pitch -= 0.01;
      _camera.pitch = (_camera.pitch >= M_PI) ? M_PI : _camera.pitch;
      _camera.pitch = (_camera.pitch <= 0.0f) ? 0.0f : _camera.pitch;
    } else if (_key_a) {
      _camera.yaw -= 0.01;
      _camera.yaw = (_camera.yaw >= M_PI) ? M_PI : _camera.yaw;
      _camera.yaw = (_camera.yaw <= -M_PI) ? -M_PI : _camera.yaw;
    } else if (_key_d) {
      _camera.yaw += 0.01;
      _camera.yaw = (_camera.yaw >= M_PI) ? M_PI : _camera.yaw;
      _camera.yaw = (_camera.yaw <= -M_PI) ? -M_PI : _camera.yaw;
    } else if (_key_equal) {
      _camera.radius += 0.1;
      _camera.radius = (_camera.radius <= 0.01) ? 0.01 : _camera.radius;
    } else if (_key_minus) {
      _camera.radius -= 0.1;
      _camera.radius = (_camera.radius <= 0.01) ? 0.01 : _camera.radius;
    }
  }

  // Handle mouse events
  // -- Mouse button press
  _mouse_button_left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT);
  _mouse_button_right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT);
  if (_mouse_button_left == GLFW_PRESS) {
    _cursor_is_dragging = 1;
  } else if (_mouse_button_left == GLFW_RELEASE) {
    _cursor_is_dragging = 0;
    _ui_engaged = 0;
  }

  // -- Mouse cursor position
  glfwGetCursorPos(window, &_cursor_x, &_cursor_y);
  if (_cursor_is_dragging) {
    _cursor_dx = _cursor_x - _cursor_last_x;
    _cursor_dy = _cursor_y - _cursor_last_y;
    _cursor_last_x = _cursor_x;
    _cursor_last_y = _cursor_y;
  } else {
    _cursor_last_x = _cursor_x;
    _cursor_last_y = _cursor_y;
  }

  // Check if UI element has been selected
  if (_ui_engaged) {
    return;
  }

  // Rotate camera
  if (_cursor_is_dragging) {
    gl_camera_rotate(&_camera, _mouse_sensitivity, _cursor_dx, _cursor_dy);
  }

  // Pan camera
  if (_cursor_is_dragging) {
    gl_camera_pan(&_camera, _mouse_sensitivity, _cursor_dx, _cursor_dy);
  }

  // Update camera
  gl_camera_update(&_camera);
}

gui_t *gui_malloc(const char *window_title,
                  const int window_width,
                  const int window_height) {
  assert(window_title);
  assert(window_width);
  assert(window_height);

  _window_loop = 1;
  gui_t *gui = malloc(sizeof(gui_t));
  gui->window = NULL;
  gui->fps_limit = 1.0 / 60.0;
  gui->last_time = 0;
  gui->last_frame = 0;

  // gui->key_q = &_key_q;
  // gui->key_w = &_key_w;
  // gui->key_a = &_key_a;
  // gui->key_s = &_key_s;
  // gui->key_d = &_key_d;
  gui->key_n = &_key_n;
  // gui->key_esc = &_key_esc;
  // gui->key_equal = &_key_equal;
  // gui->key_minus = &_key_minus;

  strcpy(_window_title, window_title);
  _window_width = window_width;
  _window_height = window_height;

  // GLFW
  if (!glfwInit()) {
    printf("Failed to initialize glfw!\n");
    exit(EXIT_FAILURE);
  }

  // Window settings
  glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, 1);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  gui->window = glfwCreateWindow(_window_width,
                                 _window_height,
                                 _window_title,
                                 NULL,
                                 NULL);
  if (!gui->window) {
    printf("Failed to create glfw window!\n");
    glfwTerminate();
    exit(EXIT_FAILURE);
  }
  glfwMakeContextCurrent(gui->window);
  glfwSetWindowSizeCallback(gui->window, window_callback);
  glfwSetInputMode(gui->window, GLFW_RAW_MOUSE_MOTION, GLFW_TRUE);

  // GLAD
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    printf("Failed to load GL functions!\n");
    free(gui);
    return NULL;
  }

  // OpenGL functions
  glEnable(GL_DEBUG_OUTPUT);
  glEnable(GL_PROGRAM_POINT_SIZE);
  glEnable(GL_LINE_SMOOTH);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  assert(glIsEnabled(GL_PROGRAM_POINT_SIZE));
  assert(glIsEnabled(GL_LINE_SMOOTH));
  assert(glIsEnabled(GL_DEPTH_TEST));
  assert(glIsEnabled(GL_CULL_FACE));
  assert(glIsEnabled(GL_BLEND));

  // Camera
  gl_camera_setup(&_camera, &_window_width, &_window_height);
  _camera.position[0] = 0.0f;
  _camera.position[1] = 4.0f;
  _camera.position[2] = 5.0f;
  // _camera.position[1] = 200.0f;
  // _camera.position[2] = 200.0f;
  _mouse_sensitivity = 0.02f;

  // UI event
  _ui_engaged = 0;

  // GUI
  glfwMakeContextCurrent(gui->window);
  glfwSwapInterval(0);

  return gui;
}

void gui_free(gui_t *gui) {
  assert(gui);
  glfwTerminate();
  free(gui);
}

double gui_time(void) {
  struct timespec time;
  clock_gettime(CLOCK_REALTIME, &time);
  return time.tv_sec + time.tv_nsec * 1e-9;
}

int gui_poll(gui_t *gui) {
  assert(gui);

  // Process keyboard and mouse input
  glfwPollEvents();
  gui_process_input(gui->window);

  // Clear screen
  glClear(GL_DEPTH_BUFFER_BIT);
  glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  return _window_loop;
}

void gui_update(gui_t *gui) {
  assert(gui);
  glfwSwapBuffers(gui->window);

  const double time_now = gui_time();
  _frame_dt = time_now - _frame_last;
  _frame_last = time_now;
  // printf("fps: %f\n", 1.0 / _frame_dt);
  // const double time_now = glfwGetTime();
  // const double dt = time_now - gui->last_frame;
  // if (dt >= gui->fps_limit) {
  //   glfwSwapBuffers(gui->window);
  //   gui->last_frame = time_now;
  // }
  // gui->last_time = time_now;
}

// RECT //////////////////////////////////////////////////////////////////////

#define GL_RECT_VS                                                             \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec2 in_pos;\n"                                    \
  "uniform float w;\n"                                                         \
  "uniform float h;\n"                                                         \
  "uniform float x;\n"                                                         \
  "uniform float y;\n"                                                         \
  "uniform mat4 ortho;\n"                                                      \
  "void main() {\n"                                                            \
  "  float x = in_pos.x * w + x;\n"                                            \
  "  float y = in_pos.y * h + y;\n"                                            \
  "  gl_Position = ortho * vec4(x, y, 0.0f, 1.0f);\n"                          \
  "}\n"

#define GL_RECT_FS                                                             \
  "#version 330 core\n"                                                        \
  "uniform vec3 color;\n"                                                      \
  "out vec4 frag_color;\n"                                                     \
  "void main() {\n"                                                            \
  "  frag_color = vec4(color, 1.0f);\n"                                        \
  "}\n"

gl_rect_t *gl_rect_malloc(const gl_bounds_t bounds, const gl_color_t color) {
  gl_rect_t *rect = malloc(sizeof(gl_rect_t));
  rect->program_id = 0;
  rect->VAO = 0;
  rect->VBO = 0;
  rect->EBO = 0;
  rect->bounds = bounds;
  rect->color = color;

  // Shader
  rect->program_id = gl_shader(GL_RECT_VS, GL_RECT_FS, NULL);
  if (rect->program_id == GL_FALSE) {
    FATAL("Failed to create shader!");
  }

  // Vertices
  // clang-format off
  const float vertices[2 * 4] = {
    1.0f, 0.0f,  // Top-right
    1.0f, 1.0f,  // Bottom-right
    0.0f, 1.0f,  // Bottom-left
    0.0f, 0.0f,  // Top-left
  };
  const gl_int_t indices[6] = {
    0, 3, 1, // First triangle
    2, 1, 3  // Second triangle
  };
  const size_t vertex_size = sizeof(gl_float_t) * 2;
  const size_t vbo_size = sizeof(vertices);
  const size_t ebo_size = sizeof(indices);
  // clang-format on

  // VAO
  glGenVertexArrays(1, &rect->VAO);
  glBindVertexArray(rect->VAO);
  assert(rect->VAO != 0);

  // VBO
  glGenBuffers(1, &rect->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, rect->VBO);
  glBufferData(GL_ARRAY_BUFFER, vbo_size, vertices, GL_STATIC_DRAW);
  assert(rect->VBO != 0);

  // EBO
  glGenBuffers(1, &rect->EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, rect->EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebo_size, indices, GL_STATIC_DRAW);
  assert(rect->EBO != 0);

  // Position attribute
  void *pos_offset = (void *) 0;
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, vertex_size, pos_offset);
  glEnableVertexAttribArray(0);

  // Unbind VBO and VAO
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  return rect;
}

void gl_rect_free(gl_rect_t *rect) {
  if (rect == NULL) {
    return;
  }
  GL_DEL_VERTEX_ARRAY(rect->VAO);
  GL_DEL_BUFFER(rect->VBO);
  GL_DEL_BUFFER(rect->EBO);
  free(rect);
}

void draw_rect(gl_rect_t *rect) {
  assert(rect);

  // Use shader
  gl_float_t ortho[16] = {0};
  gl_ortho(_window_width, _window_height, ortho);

  // Draw
  glDepthMask(GL_FALSE);
  glUseProgram(rect->program_id);
  gl_set_mat4(rect->program_id, "ortho", ortho);
  gl_set_color(rect->program_id, "color", rect->color);
  gl_set_float(rect->program_id, "w", rect->bounds.w);
  gl_set_float(rect->program_id, "h", rect->bounds.h);
  gl_set_float(rect->program_id, "x", rect->bounds.x);
  gl_set_float(rect->program_id, "y", rect->bounds.y);
  glBindVertexArray(rect->VAO);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glDepthMask(GL_TRUE);
}

// POINTS 3D /////////////////////////////////////////////////////////////////

#define GL_POINTS3D_VS                                                         \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec3 in_pos;\n"                                    \
  "layout (location = 1) in vec3 in_color;\n"                                  \
  "out vec3 color;\n"                                                          \
  "uniform mat4 view;\n"                                                       \
  "uniform mat4 projection;\n"                                                 \
  "uniform float point_size;\n"                                                \
  "void main() {\n"                                                            \
  "  gl_Position = projection * view * vec4(in_pos, 1.0);\n"                   \
  "  gl_PointSize = point_size;\n"                                             \
  "  color = in_color;\n"                                                      \
  "}\n"

#define GL_POINTS3D_FS                                                         \
  "#version 330 core\n"                                                        \
  "in vec3 color;\n"                                                           \
  "out vec4 frag_color;\n"                                                     \
  "uniform float alpha;\n"                                                     \
  "void main() {\n"                                                            \
  "  frag_color = vec4(color, alpha);\n"                                       \
  "}\n"

#define GL_POINTS3D_MAX_POINTS 1000000

gl_points3d_t *gl_points3d_malloc(gl_float_t *points_data,
                                  const size_t num_points,
                                  const gl_float_t point_size) {
  glEnable(GL_PROGRAM_POINT_SIZE); // Need this for setting point size
  assert(num_points >= 0);
  assert(point_size >= 0);

  gl_points3d_t *points = malloc(sizeof(gl_points3d_t));
  points->program_id = 0;
  points->VAO = 0;
  points->VBO = 0;
  points->points_data = points_data;
  points->num_points = (num_points == 0) ? GL_POINTS3D_MAX_POINTS : num_points;
  points->point_size = point_size;

  // Shader
  points->program_id = gl_shader(GL_POINTS3D_VS, GL_POINTS3D_FS, NULL);
  if (points->program_id == GL_FALSE) {
    FATAL("Failed to create shaders to draw points!");
  }

  // VAO
  glGenVertexArrays(1, &points->VAO);
  glBindVertexArray(points->VAO);

  // VBO
  const size_t vbo_size = sizeof(float) * 3 * points->num_points;
  glGenBuffers(1, &points->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, points->VBO);
  glBufferData(GL_ARRAY_BUFFER, vbo_size, points->points_data, GL_STATIC_DRAW);
  // -- Position attribute
  size_t vertex_size = 6 * sizeof(float);
  void *pos_offset = (void *) 0;
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, pos_offset);
  glEnableVertexAttribArray(0);
  // -- Color attribute
  void *color_offset = (void *) (3 * sizeof(float));
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, color_offset);
  glEnableVertexAttribArray(1);

  // Unbind VBO and VAO
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
  glBindVertexArray(0);             // Unbind VAO

  return points;
}

void gl_points3d_free(gl_points3d_t *points) {
  if (points == NULL) {
    return;
  }
  GL_DEL_VERTEX_ARRAY(points->VAO);
  GL_DEL_BUFFER(points->VBO);
  free(points);
}

void gl_points3d_update(gl_points3d_t *points,
                        gl_float_t *points_data,
                        size_t num_points,
                        const gl_float_t point_size) {
  // Check if we have data to upload
  if (num_points == 0) {
    return;
  }

  points->points_data = points_data;
  points->num_points = num_points;
  points->point_size = point_size;

  const size_t vbo_size = sizeof(gl_float_t) * 6 * points->num_points;
  glBindBuffer(GL_ARRAY_BUFFER, points->VBO);
  glBufferSubData(GL_ARRAY_BUFFER, 0, vbo_size, points->points_data);
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
}

void draw_points3d(gl_points3d_t *points) {
  assert(points);
  if (points->num_points == 0) {
    return;
  }

  // Use shader program
  glUseProgram(points->program_id);
  gl_set_mat4(points->program_id, "view", _camera.V);
  gl_set_mat4(points->program_id, "projection", _camera.P);
  gl_set_float(points->program_id, "point_size", points->point_size);
  gl_set_float(points->program_id, "alpha", 1.0f);

  // Draw
  glBindVertexArray(points->VAO);
  glDrawArrays(GL_POINTS, 0, points->num_points);
  glBindVertexArray(0); // Unbind VAO
}

// LINE 3D ///////////////////////////////////////////////////////////////////

#define GL_LINE3D_VS                                                           \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec3 in_pos;\n"                                    \
  "uniform mat4 projection;\n"                                                 \
  "uniform mat4 view;\n"                                                       \
  "void main() {\n"                                                            \
  "  gl_Position = projection * view * vec4(in_pos, 1.0);\n"                   \
  "}\n"

#define GL_LINE3D_GS                                                           \
  "#version 330 core\n"                                                        \
  "layout (lines) in;\n"                                                       \
  "layout (triangle_strip, max_vertices = 4) out;\n"                           \
  "\n"                                                                         \
  "uniform vec2 viewport_size;\n"                                              \
  "uniform float linewidth;\n"                                                 \
  "\n"                                                                         \
  "bool within_clipspace(vec4 v) {\n"                                          \
  "  if (v.x < -v.w || v.x > v.w ||\n"                                         \
  "      v.y < -v.w || v.y > v.w ||\n"                                         \
  "      v.z < -v.w || v.z > v.w) {\n"                                         \
  "    return false;\n"                                                        \
  "  }\n"                                                                      \
  "\n"                                                                         \
  "  return true;\n"                                                           \
  "}\n"                                                                        \
  "\n"                                                                         \
  "void main() {\n"                                                            \
  "  // Transform from clip -> NCD -> screen space\n"                          \
  "  vec4 p1_clip = gl_in[0].gl_Position;\n"                                   \
  "  vec4 p2_clip = gl_in[1].gl_Position;\n"                                   \
  "  vec2 p1_ndc = p1_clip.xy / p1_clip.w;\n"                                  \
  "  vec2 p2_ndc = p2_clip.xy / p2_clip.w;\n"                                  \
  "  vec2 p1_screen = 0.5 * (p1_ndc + 1.0) * viewport_size;\n"                 \
  "  vec2 p2_screen = 0.5 * (p2_ndc + 1.0) * viewport_size;\n"                 \
  "\n"                                                                         \
  "  // Form thick line four vertices\n"                                       \
  "  vec2 line = p2_screen - p1_screen;\n"                                     \
  "  vec2 normal = normalize(vec2(-line.y, line.x));\n"                        \
  "  vec2 a_screen = p1_screen - 0.5 * linewidth * normal;\n"                  \
  "  vec2 b_screen = p1_screen + 0.5 * linewidth * normal;\n"                  \
  "  vec2 c_screen = p2_screen - 0.5 * linewidth * normal;\n"                  \
  "  vec2 d_screen = p2_screen + 0.5 * linewidth * normal;\n"                  \
  "\n"                                                                         \
  "  // Convert back from screen space -> NDC -> clip space\n"                 \
  "  vec2 a_ndc = (a_screen / viewport_size) * 2.0 - 1.0;\n"                   \
  "  vec2 b_ndc = (b_screen / viewport_size) * 2.0 - 1.0;\n"                   \
  "  vec2 c_ndc = (c_screen / viewport_size) * 2.0 - 1.0;\n"                   \
  "  vec2 d_ndc = (d_screen / viewport_size) * 2.0 - 1.0;\n"                   \
  "  float z1 = p1_clip.z / p1_clip.w;\n"                                      \
  "  float z2 = p2_clip.z / p2_clip.w;\n"                                      \
  "  vec4 a_clip = vec4(a_ndc.x, a_ndc.y, z1, 1.0f);\n"                        \
  "  vec4 b_clip = vec4(b_ndc.x, b_ndc.y, z1, 1.0f);\n"                        \
  "  vec4 c_clip = vec4(c_ndc.x, c_ndc.y, z2, 1.0f);\n"                        \
  "  vec4 d_clip = vec4(d_ndc.x, d_ndc.y, z2, 1.0f);\n"                        \
  "\n"                                                                         \
  "  // Emit the for quad vertices\n"                                          \
  "  // IMPORTANT!: Vertices are ordered CCW, assert glFrontFace is CCW\n"     \
  "  gl_Position = a_clip;\n"                                                  \
  "  EmitVertex();\n"                                                          \
  "  gl_Position = c_clip;\n"                                                  \
  "  EmitVertex();\n"                                                          \
  "  gl_Position = b_clip;\n"                                                  \
  "  EmitVertex();\n"                                                          \
  "  gl_Position = d_clip;\n"                                                  \
  "  EmitVertex();\n"                                                          \
  "  EndPrimitive();\n"                                                        \
  "}\n"

#define GL_LINE3D_FS                                                           \
  "#version 330 core\n"                                                        \
  "uniform vec3 color;\n"                                                      \
  "uniform float alpha;\n"                                                     \
  "out vec4 frag_color;\n"                                                     \
  "void main() {\n"                                                            \
  "  frag_color = vec4(color, alpha);\n"                                       \
  "}\n"

void world2clip(const gl_float_t p_world[3],
                const gl_float_t proj[4 * 4],
                const gl_float_t view[4 * 4],
                gl_float_t hp_clip[4]) {
  // proj_view = proj * view
  gl_float_t proj_view[4 * 4] = {0};
  gl_dot(proj, 4, 4, view, 4, 4, proj_view);

  // p_clip = proj * view * p_world
  gl_float_t hp_world[4] = {p_world[0], p_world[1], p_world[2], 1.0f};
  gl_dot(proj_view, 4, 4, hp_world, 4, 1, hp_clip);
}

void world2screen(const gl_float_t p_world[3],
                  const gl_float_t proj[4 * 4],
                  const gl_float_t view[4 * 4],
                  const gl_float_t viewport[2],
                  gl_float_t p_screen[2]) {
  // World to clip space
  gl_float_t hp_clip[4] = {0};
  world2clip(p_world, proj, view, hp_clip);

  // Clip space to NDC
  gl_float_t p_ndc[2] = {0};
  p_ndc[0] = hp_clip[0] / hp_clip[3];
  p_ndc[1] = hp_clip[1] / hp_clip[3];

  // NDC to screen
  const gl_float_t w = viewport[0];
  const gl_float_t h = viewport[1];
  p_screen[0] = 0.5 * (p_ndc[0] + 1.0) * w;
  p_screen[1] = 0.5 * (p_ndc[1] + 1.0) * h;
}

void line_vertex(const gl_float_t p1[2],
                 const gl_float_t p2[2],
                 const gl_float_t thickness,
                 gl_float_t *data,
                 size_t offset) {
  // Calculate perpendicular vector
  const gl_float_t dx = p2[0] - p1[0];
  const gl_float_t dy = p2[1] - p1[1];
  const gl_float_t length = sqrt(dx * dx + dy * dy);

  // Normalize and get perpendicular
  const gl_float_t nx = -dy / length * thickness * 0.5f;
  const gl_float_t ny = dx / length * thickness * 0.5f;

  // Create quad vertices
  // clang-format off
  data[offset + 0] = p1[0] + nx; data[offset + 1] = p1[1] + ny; // Top-left
  data[offset + 2] = p1[0] - nx; data[offset + 3] = p1[1] - ny; // Bottom-left
  data[offset + 4] = p2[0] - nx; data[offset + 5] = p2[1] - ny; // Bottom-right
  data[offset + 6] = p2[0] + nx; data[offset + 7] = p2[1] + ny; // Top-right
  // clang-format on
}

void gl_line3d_setup(gl_line3d_t *line3d,
                     const gl_color_t color,
                     const gl_float_t lw) {
  // Setup
  line3d->program_id = 0;
  line3d->VAO = 0;
  line3d->VBO = 0;
  line3d->num_points = 0;
  line3d->color = color;
  line3d->alpha = 1.0f;
  line3d->lw = lw;

  // Shader
  line3d->program_id = gl_shader(GL_LINE3D_VS, GL_LINE3D_FS, GL_LINE3D_GS);
  if (line3d->program_id == GL_FALSE) {
    FATAL("Failed to create shaders!");
  }

  // VAO
  glGenVertexArrays(1, &line3d->VAO);
  glBindVertexArray(line3d->VAO);

  // VBO
  const size_t max_size = sizeof(gl_float_t) * 3 * 10000;
  glGenBuffers(1, &line3d->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, line3d->VBO);
  glBufferData(GL_ARRAY_BUFFER, max_size, NULL, GL_DYNAMIC_DRAW);

  // Position attribute
  const size_t vertex_size = sizeof(gl_float_t) * 3;
  const void *pos_offset = (void *) 0;
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, pos_offset);
  glEnableVertexAttribArray(0);

  // Unbind VBO and VAO
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
  glBindVertexArray(0);             // Unbind VAO
}

gl_line3d_t *gl_line3d_malloc(const gl_color_t color, const gl_float_t lw) {
  gl_line3d_t *line3d = malloc(sizeof(gl_line3d_t));
  gl_line3d_setup(line3d, color, lw);
  return line3d;
}

void gl_line3d_update(gl_line3d_t *line3d,
                      const size_t offset,
                      const gl_float_t *data,
                      const size_t num_verts) {
  line3d->num_points = num_verts;
  glBindBuffer(GL_ARRAY_BUFFER, line3d->VBO);
  size_t data_size = sizeof(float) * 3 * num_verts;
  glBufferSubData(GL_ARRAY_BUFFER, offset, data_size, data);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void gl_line3d_free(gl_line3d_t *line) {
  if (line == NULL) {
    return;
  }
  GL_DEL_VERTEX_ARRAY(line->VAO);
  GL_DEL_BUFFER(line->VBO);
  free(line);
}

void draw_line3d(gl_line3d_t *line3d) {
  // Get viewport
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT, viewport);

  gl_float_t viewport_wh[2] = {0};
  viewport_wh[0] = viewport[2];
  viewport_wh[1] = viewport[3];

  // Use shader program
  glUseProgram(line3d->program_id);
  gl_set_mat4(line3d->program_id, "projection", _camera.P);
  gl_set_mat4(line3d->program_id, "view", _camera.V);
  gl_set_vec2(line3d->program_id, "viewport_size", viewport_wh);
  gl_set_float(line3d->program_id, "linewidth", line3d->lw);
  gl_set_color(line3d->program_id, "color", line3d->color);
  gl_set_float(line3d->program_id, "alpha", line3d->alpha);

  // Draw frame
  glBindVertexArray(line3d->VAO);
  glDrawArrays(GL_LINE_STRIP, 0, line3d->num_points);
  glBindVertexArray(0);
}

// CUBE 3D ///////////////////////////////////////////////////////////////////

#define GL_CUBE3D_VS                                                           \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec3 in_pos;\n"                                    \
  "layout (location = 1) in vec3 in_normal;\n"                                 \
  ""                                                                           \
  "out vec3 frag_pos;\n"                                                       \
  "out vec3 normal;\n"                                                         \
  ""                                                                           \
  "uniform float size;\n"                                                      \
  "uniform mat4 model;\n"                                                      \
  "uniform mat4 view;\n"                                                       \
  "uniform mat4 projection;\n"                                                 \
  ""                                                                           \
  "void main() {\n"                                                            \
  "  gl_Position = projection * view * model * vec4(in_pos * size, 1.0);\n"    \
  "  frag_pos = vec3(model * vec4(in_pos, 1.0));\n"                            \
  "  normal = in_normal;\n"                                                    \
  "}\n"

#define GL_CUBE3D_FS                                                           \
  "#version 330 core\n"                                                        \
  "in vec3 frag_pos;\n"                                                        \
  "in vec3 normal;\n"                                                          \
  "out vec4 frag_color;\n"                                                     \
  ""                                                                           \
  "uniform vec3 view_pos;\n"                                                   \
  "uniform vec3 light_pos;\n"                                                  \
  "uniform vec3 light_color;\n"                                                \
  "uniform vec3 object_color;\n"                                               \
  ""                                                                           \
  "void main() {\n"                                                            \
  "  // Ambient \n"                                                            \
  "  float ambient_strength = 0.5;\n"                                          \
  "  vec3 ambient = ambient_strength * light_color;\n"                         \
  ""                                                                           \
  "  // Diffuse \n"                                                            \
  "  vec3 norm = normalize(normal);\n"                                         \
  "  vec3 light_dir = normalize(light_pos - frag_pos);\n"                      \
  "  float diff = max(dot(norm, light_dir), 0.0);\n"                           \
  "  vec3 diffuse = diff * light_color;\n"                                     \
  ""                                                                           \
  " // Specular \n"                                                            \
  " float specular_strenght = 0.5; \n"                                         \
  " vec3 view_dir = normalize(view_pos - frag_pos); \n"                        \
  " vec3 reflect_dir = reflect(-light_dir, norm); \n"                          \
  " float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32); \n"            \
  " vec3 specular = specular_strenght * spec * light_color; \n"                \
  ""                                                                           \
  "  vec3 result = (ambient + diffuse + specular) * object_color;\n"           \
  "  frag_color = vec4(result, 1.0f);\n"                                       \
  "}\n"

gl_cube3d_t *gl_cube3d_malloc(void) {
  gl_cube3d_t *cube = malloc(sizeof(gl_cube3d_t));
  cube->program_id = 0;
  cube->VAO = 0;
  cube->VBO = 0;

  // Shader
  cube->program_id = gl_shader(GL_CUBE3D_VS, GL_CUBE3D_FS, NULL);
  if (cube->program_id == GL_FALSE) {
    FATAL("Failed to create shaders!");
  }

  // clang-format off
  // Vertices
  gl_float_t vertices[] = {
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
     1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
    -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f,
    -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f,

    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
     1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f,
    -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f,

    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f,

     1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f,
     1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f,

    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f,
     1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f,
     1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f,
    -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f,
    -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f,

    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f,
     1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
     1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f,
    -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f
  };
  const size_t vertex_size = sizeof(gl_float_t) * 6;
  // clang-format on

  // VAO
  glGenVertexArrays(1, &cube->VAO);
  glBindVertexArray(cube->VAO);

  // VBO
  glGenBuffers(1, &cube->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, cube->VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  // -- Position attribute
  const void *pos_offset = (void *) 0;
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, pos_offset);
  glEnableVertexAttribArray(0);
  // -- Normal attribute
  const void *normal_offset = (void *) (3 * sizeof(gl_float_t));
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertex_size, normal_offset);
  glEnableVertexAttribArray(1);

  // Clean up
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
  glBindVertexArray(0);             // Unbind VAO

  return cube;
}

void gl_cube3d_free(gl_cube3d_t *cube) {
  if (cube == NULL) {
    return;
  }
  GL_DEL_VERTEX_ARRAY(cube->VAO);
  GL_DEL_BUFFER(cube->VBO);
  free(cube);
}

void draw_cube(gl_cube3d_t *cube,
               const gl_float_t T[4 * 4],
               const gl_float_t size,
               const gl_color_t color) {
  assert(cube);

  // Disable cull face
  int cull_face_mode = 0;
  if (glIsEnabled(GL_CULL_FACE)) {
    glDisable(GL_CULL_FACE);
    cull_face_mode = 1;
  }

  // Use shader
  glUseProgram(cube->program_id);

  // Draw cube
  gl_float_t *view_pos = _camera.position;
  gl_float_t light_pos[3] = {0.0, 20.0, 2.0};
  gl_color_t light_color = (gl_color_t){1.0, 1.0, 1.0};

  gl_set_mat4(cube->program_id, "projection", _camera.P);
  gl_set_mat4(cube->program_id, "view", _camera.V);
  gl_set_mat4(cube->program_id, "model", T);
  gl_set_float(cube->program_id, "size", size);
  gl_set_vec3(cube->program_id, "view_pos", view_pos);
  gl_set_vec3(cube->program_id, "light_pos", light_pos);
  gl_set_color(cube->program_id, "light_color", light_color);
  gl_set_color(cube->program_id, "object_color", color);

  glBindVertexArray(cube->VAO);
  glDrawArrays(GL_TRIANGLES, 0, 36); // 36 Vertices

  // Unbind VAO
  glBindVertexArray(0);

  // Renable cull face
  if (cull_face_mode) {
    glEnable(GL_CULL_FACE);
  }
}

// FRUSTUM /////////////////////////////////////////////////////////////////////

#define GL_FRUSTUM_VS                                                          \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec3 in_pos;\n"                                    \
  "uniform mat4 model;\n"                                                      \
  "uniform mat4 view;\n"                                                       \
  "uniform mat4 projection;\n"                                                 \
  "void main() {\n"                                                            \
  "  gl_Position = projection * view * model * vec4(in_pos, 1.0);\n"           \
  "}\n"

#define GL_FRUSTUM_FS                                                          \
  "#version 150 core\n"                                                        \
  "out vec4 frag_color;\n"                                                     \
  "void main() {\n"                                                            \
  "  frag_color = vec4(1.0f, 1.0f, 1.0f, 1.0f);\n"                             \
  "}\n"

gl_frustum_t *gl_frustum_malloc(const gl_float_t hfov,
                                const gl_float_t aspect,
                                const gl_float_t znear,
                                const gl_float_t zfar,
                                const gl_float_t T[4 * 4],
                                const gl_float_t size,
                                const gl_color_t color,
                                const gl_float_t lw) {
  assert(T);
  assert(size > 0);
  assert(lw > 0);

  gl_frustum_t *frustum = malloc(sizeof(gl_frustum_t));
  frustum->hfov = hfov;
  frustum->aspect = aspect;
  frustum->znear = znear;
  frustum->zfar = zfar;

  frustum->program_id = 0;
  frustum->VAO = 0;
  frustum->VBO = 0;
  for (int i = 0; i < 16; ++i) {
    frustum->T[i] = T[i];
  }
  frustum->size = size;
  frustum->color = color;
  frustum->lw = lw;

  // Shader
  frustum->program_id = gl_shader(GL_FRUSTUM_VS, GL_FRUSTUM_FS, NULL);
  if (frustum->program_id == GL_FALSE) {
    FATAL("Failed to create shaders!");
  }

  // Form the camera fov frame
  // gl_float_t fov = gl_deg2rad(60.0);
  // gl_float_t hfov = fov / 2.0f;
  gl_float_t scale = 1.0f;
  gl_float_t z = scale;
  gl_float_t hwidth = z * tan(hfov);
  const gl_float_t lb[3] = {-hwidth, hwidth, z};  // Left bottom
  const gl_float_t lt[3] = {-hwidth, -hwidth, z}; // Left top
  const gl_float_t rt[3] = {hwidth, -hwidth, z};  // Right top
  const gl_float_t rb[3] = {hwidth, hwidth, z};   // Right bottom

  // Rectangle frame
  // clang-format off
  const size_t vertex_size = sizeof(gl_float_t) * 3;
  const gl_float_t vertices[8 * 6] = {
    // -- Left bottom to left top
    lb[0], lb[1], lb[2], lt[0], lt[1], lt[2],
    // -- Left top to right top
    lt[0], lt[1], lt[2], rt[0], rt[1], rt[2],
    // -- Right top to right bottom
    rt[0], rt[1], rt[2], rb[0], rb[1], rb[2],
    // -- Right bottom to left bottom
    rb[0], rb[1], rb[2], lb[0], lb[1], lb[2],
    // Rectangle frame to origin
    // -- Origin to left bottom
    0.0f, 0.0f, 0.0f, lb[0], lb[1], lb[2],
    // -- Origin to left top
    0.0f, 0.0f, 0.0f, lt[0], lt[1], lt[2],
    // -- Origin to right top
    0.0f, 0.0f, 0.0f, rt[0], rt[1], rt[2],
    // -- Origin to right bottom
    0.0f, 0.0f, 0.0f, rb[0], rb[1], rb[2]
  };
  // clang-format on

  // VAO
  glGenVertexArrays(1, &frustum->VAO);
  glBindVertexArray(frustum->VAO);

  // VBO
  glGenBuffers(1, &frustum->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, frustum->VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertex_size, (void *) 0);
  glEnableVertexAttribArray(0);

  // Clean up
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
  glBindVertexArray(0);             // Unbind VAO

  return frustum;
}

void gl_frustum_free(gl_frustum_t *frustum) {
  if (frustum == NULL) {
    return;
  }
  GL_DEL_VERTEX_ARRAY(frustum->VAO);
  GL_DEL_BUFFER(frustum->VBO);
  free(frustum);
}

void draw_frustum(gl_frustum_t *frustum) {
  assert(frustum);

  // Use shader program
  glUseProgram(frustum->program_id);
  gl_set_mat4(frustum->program_id, "projection", _camera.P);
  gl_set_mat4(frustum->program_id, "view", _camera.V);
  gl_set_mat4(frustum->program_id, "model", frustum->T);

  // Store original line width
  gl_float_t lw_bak = 0.0f;
  glGetFloatv(GL_LINE_WIDTH, &lw_bak);

  // Set line width
  glLineWidth(frustum->lw);

  // Draw frame
  const size_t num_lines = 8;
  const size_t num_vertices = num_lines * 2;
  glBindVertexArray(frustum->VAO);
  glDrawArrays(GL_LINES, 0, num_vertices);
  glBindVertexArray(0); // Unbind VAO

  // Restore original line width
  glLineWidth(lw_bak);
}

// AXES 3D ///////////////////////////////////////////////////////////////////

gl_axes3d_t *gl_axes3d_malloc(const gl_float_t T[4 * 4],
                              const gl_float_t size,
                              const gl_float_t lw) {
  assert(T);
  assert(size > 0);
  assert(lw > 0);

  gl_color_t red = (gl_color_t){1.0, 0.0, 0.0};
  gl_color_t green = (gl_color_t){0.0, 1.0, 0.0};
  gl_color_t blue = (gl_color_t){0.0, 0.0, 1.0};

  gl_axes3d_t *axes = malloc(sizeof(gl_axes3d_t));
  gl_line3d_setup(&axes->x_axis, red, lw);
  gl_line3d_setup(&axes->y_axis, green, lw);
  gl_line3d_setup(&axes->z_axis, blue, lw);

  gl_float_t x_data[3 * 2] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
  gl_float_t y_data[3 * 2] = {0.0, 0.0, 0.0, 0.0, 1.0, 0.0};
  gl_float_t z_data[3 * 2] = {0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  gl_line3d_update(&axes->x_axis, 0, x_data, 2);
  gl_line3d_update(&axes->y_axis, 0, y_data, 2);
  gl_line3d_update(&axes->z_axis, 0, z_data, 2);

  for (int i = 0; i < 16; ++i) {
    axes->T[i] = T[i];
  }
  axes->size = size;
  axes->lw = lw;

  return axes;
}

void gl_axes3d_free(gl_axes3d_t *axes) {
  if (axes == NULL) {
    return;
  }
  free(axes);
}

void draw_axes3d(gl_axes3d_t *axes) {
  assert(axes);
  draw_line3d(&axes->x_axis);
  draw_line3d(&axes->y_axis);
  draw_line3d(&axes->z_axis);
}

// GRID 3D ///////////////////////////////////////////////////////////////////

#define GL_GRID3D_VS                                                           \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec3 in_pos;\n"                                    \
  "uniform mat4 model;\n"                                                      \
  "uniform mat4 view;\n"                                                       \
  "uniform mat4 projection;\n"                                                 \
  "void main() {\n"                                                            \
  "  gl_Position = projection * view * model * vec4(in_pos, 1.0);\n"           \
  "}\n"

#define GL_GRID3D_FS                                                           \
  "#version 150 core\n"                                                        \
  "out vec4 frag_color;\n"                                                     \
  "void main() {\n"                                                            \
  "  frag_color = vec4(0.8f, 0.8f, 0.8f, 1.0f);\n"                             \
  "}\n"

float *generate_grid_vertices(int width, int depth, float size, int *count) {
  *count = (width + 1 + depth + 1) * 2;
  float *v = malloc(*count * 3 * sizeof(float));
  float step = size / width, half = size * 0.5f;
  int i = 0;

  for (int x = 0; x <= width; x++) {
    float pos = -half + x * step;
    v[i++] = pos;
    v[i++] = 0;
    v[i++] = -half;

    v[i++] = pos;
    v[i++] = 0;
    v[i++] = half;
  }

  for (int z = 0; z <= depth; z++) {
    float pos = -half + z * step;
    v[i++] = -half;
    v[i++] = 0;
    v[i++] = pos;

    v[i++] = half;
    v[i++] = 0;
    v[i++] = pos;
  }
  return v;
}

gl_grid3d_t *gl_grid3d_malloc(const gl_int_t num_rows,
                              const gl_int_t num_cols,
                              const gl_float_t grid_size,
                              const gl_color_t color,
                              const gl_float_t lw) {
  assert(num_rows > 0);
  assert(num_cols > 0);
  assert(grid_size > 0);
  assert(lw > 0);

  gl_grid3d_t *grid = malloc(sizeof(gl_grid3d_t));
  grid->num_rows = num_rows;
  grid->num_cols = num_cols;
  grid->grid_size = grid_size;
  grid->color = color;
  grid->lw = lw;

  // Allocate memory for vertices
  const float row_step_size = grid_size / grid->num_rows;
  const float col_step_size = grid_size / grid->num_cols;
  const float half_size = grid_size * 0.5f;
  grid->row_lines = malloc(sizeof(gl_line3d_t) * (grid->num_rows + 1));
  grid->col_lines = malloc(sizeof(gl_line3d_t) * (grid->num_cols + 1));

  // -- Setup row lines
  for (int x = 0; x <= grid->num_rows; x++) {
    float pos = -half_size + x * row_step_size;
    float vertices[3 * 2] = {0};
    vertices[0] = pos;
    vertices[1] = 0;
    vertices[2] = -half_size;
    vertices[3] = pos;
    vertices[4] = 0;
    vertices[5] = half_size;

    gl_line3d_setup(&grid->row_lines[x], grid->color, grid->lw);
    gl_line3d_update(&grid->row_lines[x], 0, vertices, 2);
  }

  // -- Setup column lines
  for (int z = 0; z <= grid->num_cols; z++) {
    float pos = -half_size + z * col_step_size;
    float vertices[3 * 2] = {0};
    vertices[0] = -half_size;
    vertices[1] = 0;
    vertices[2] = pos;
    vertices[3] = half_size;
    vertices[4] = 0;
    vertices[5] = pos;

    gl_line3d_setup(&grid->col_lines[z], grid->color, grid->lw);
    gl_line3d_update(&grid->col_lines[z], 0, vertices, 2);
  }

  return grid;
}

void gl_grid3d_free(gl_grid3d_t *grid) {
  if (grid == NULL) {
    return;
  }

  // Row lines
  for (int i = 0; i <= grid->num_rows; ++i) {
    GL_DEL_VERTEX_ARRAY(grid->row_lines[i].VAO);
    GL_DEL_BUFFER(grid->row_lines[i].VBO);
  }
  free(grid->row_lines);

  // Colum lines
  for (int i = 0; i <= grid->num_cols; ++i) {
    GL_DEL_VERTEX_ARRAY(grid->col_lines[i].VAO);
    GL_DEL_BUFFER(grid->col_lines[i].VBO);
  }
  free(grid->col_lines);

  free(grid);
}

void draw_grid3d(gl_grid3d_t *grid) {
  assert(grid);

  for (int i = 0; i <= grid->num_rows; ++i) {
    draw_line3d(&grid->row_lines[i]);
  }
  for (int i = 0; i <= grid->num_cols; ++i) {
    draw_line3d(&grid->col_lines[i]);
  }
}

// IMAGE /////////////////////////////////////////////////////////////////////

#define GL_IMAGE_VS                                                            \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec2 in_pos;\n"                                    \
  "layout (location = 1) in vec2 in_tex_coord;\n"                              \
  "uniform float w;\n"                                                         \
  "uniform float h;\n"                                                         \
  "uniform float x;\n"                                                         \
  "uniform float y;\n"                                                         \
  "uniform mat4 ortho;\n"                                                      \
  "out vec2 tex_coord;\n"                                                      \
  "void main() {\n"                                                            \
  "  float x = in_pos.x * w + x;\n"                                            \
  "  float y = in_pos.y * h + y;\n"                                            \
  "  gl_Position = ortho * vec4(x, y, 0.0f, 1.0f);\n"                          \
  "  tex_coord = in_tex_coord;\n"                                              \
  "}\n"

#define GL_IMAGE_FS                                                            \
  "#version 330 core\n"                                                        \
  "in vec2 tex_coord;\n"                                                       \
  "out vec4 frag_color;\n"                                                     \
  "uniform sampler2D texture1;\n"                                              \
  "void main() {\n"                                                            \
  "  frag_color = texture(texture1, tex_coord);\n"                             \
  "}\n"

gl_image_t *gl_image_malloc(const int x,
                            const int y,
                            const uint8_t *data,
                            const int width,
                            const int height,
                            const int channels) {
  assert(glIsEnabled(GL_BLEND)); // Need this for text-rendering
  assert(x >= 0 && y >= 0);
  assert(data);
  assert(width > 0);
  assert(height > 0);
  assert(channels > 0);

  gl_image_t *image = malloc(sizeof(gl_image_t));
  image->program_id = 0;
  image->VAO = 0;
  image->VBO = 0;
  image->EBO = 0;
  image->texture_id = 0;
  image->x = x;
  image->y = y;
  image->data = data;
  image->width = width;
  image->height = height;
  image->channels = channels;

  // Rectangle vertices and texture coordinates
  // clang-format off
  const gl_float_t vertices[4 * 4] = {
     // Positions // Texture coords
     1.0f,  0.0f, 1.0f,  1.0f, // Top-right
     1.0f,  1.0f, 1.0f,  0.0f, // Bottom-right
     0.0f,  1.0f, 0.0f,  0.0f, // Bottom-left
     0.0f,  0.0f, 0.0f,  1.0f  // Top-left
  };
  const gl_uint_t indices[2 * 3] = {
    0, 3, 1, // First Triangle
    2, 1, 3  // Second Triangle
  };
  const size_t num_vertices = 4;
  const size_t vertex_size = sizeof(gl_float_t) * 4;
  const size_t vbo_size = sizeof(vertices);
  const size_t ebo_size = sizeof(indices);
  // clang-format on

  // Shader
  image->program_id = gl_shader(GL_IMAGE_VS, GL_IMAGE_FS, NULL);
  if (image->program_id == GL_FALSE) {
    FATAL("Failed to create shader!");
  }

  // VAO
  glGenVertexArrays(1, &image->VAO);
  glBindVertexArray(image->VAO);
  assert(image->VAO != 0);

  // VBO
  glGenBuffers(1, &image->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, image->VBO);
  glBufferData(GL_ARRAY_BUFFER, vbo_size, vertices, GL_STATIC_DRAW);
  assert(image->VBO != 0);

  // EBO
  glGenBuffers(1, &image->EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image->EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, ebo_size, indices, GL_STATIC_DRAW);
  assert(image->EBO != 0);

  // Position attribute
  const void *pos_offset = (void *) (sizeof(gl_float_t) * 0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, vertex_size, pos_offset);
  glEnableVertexAttribArray(0);

  // Texture coordinate attribute
  const void *tex_offset = (void *) (sizeof(gl_float_t) * 2);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vertex_size, tex_offset);
  glEnableVertexAttribArray(1);

  // Load texture
  glGenTextures(1, &image->texture_id);
  glBindTexture(GL_TEXTURE_2D, image->texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D,
               0,
               GL_RGB,
               image->width,
               image->height,
               0,
               GL_RGB,
               GL_UNSIGNED_BYTE,
               image->data);
  glGenerateMipmap(GL_TEXTURE_2D);

  // Unbind VBO and VAO
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  return image;
}

void gl_image_free(gl_image_t *image) {
  if (image == NULL) {
    return;
  }
  GL_DEL_VERTEX_ARRAY(image->VAO);
  GL_DEL_BUFFER(image->VBO);
  GL_DEL_BUFFER(image->EBO);
  free(image);
}

void draw_image(gl_image_t *image) {
  assert(image);

  // Draw
  gl_float_t ortho[16] = {0};
  gl_ortho(_window_width, _window_height, ortho);

  glUseProgram(image->program_id);
  gl_set_mat4(image->program_id, "ortho", ortho);
  gl_set_float(image->program_id, "w", image->width);
  gl_set_float(image->program_id, "h", image->height);
  gl_set_float(image->program_id, "x", image->x);
  gl_set_float(image->program_id, "y", image->y);
  glBindVertexArray(image->VAO);
  glBindTexture(GL_TEXTURE_2D, image->texture_id);
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0); // Unbind VAO
}

// TEXT //////////////////////////////////////////////////////////////////////

#define GL_TEXT_VS                                                             \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec4 vertex;\n"                                    \
  "out vec2 tex_coords;\n"                                                     \
  "uniform mat4 ortho;\n"                                                      \
  "void main() {\n"                                                            \
  "  gl_Position = ortho * vec4(vertex.xy, 0.0, 1.0);\n"                       \
  "  tex_coords = vertex.zw;\n"                                                \
  "}\n"

#define GL_TEXT_FS                                                             \
  "#version 330 core\n"                                                        \
  "in vec2 tex_coords;\n"                                                      \
  "out vec4 frag_color;\n"                                                     \
  "uniform sampler2D text;\n"                                                  \
  "uniform vec3 text_color;\n"                                                 \
  "void main() {\n"                                                            \
  "  float alpha = texture(text, tex_coords).r;\n"                             \
  "  frag_color = vec4(text_color, alpha);\n"                                  \
  "}\n"

void gl_char_print(const gl_char_t *ch) {
  assert(ch);
  printf("texture_id: %d\n", ch->texture_id);
  printf("width:      %d\n", ch->size[0]);
  printf("height:     %d\n", ch->size[1]);
  printf("bearing_x:  %d\n", ch->bearing[0]);
  printf("bearing_y:  %d\n", ch->bearing[1]);
  printf("offset:     %d\n", ch->offset);
  printf("\n");
}

gl_text_t *gl_text_malloc(const int text_size) {
  gl_text_t *text = malloc(sizeof(gl_text_t));

  // Shader
  text->program_id = gl_shader(GL_TEXT_VS, GL_TEXT_FS, NULL);
  if (text->program_id == GL_FALSE) {
    FATAL("Failed to create shader!");
  }

  // VAO
  glGenVertexArrays(1, &text->VAO);
  glBindVertexArray(text->VAO);

  // VBO
  glGenBuffers(1, &text->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, text->VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
  glEnableVertexAttribArray(0);

  // Clean up
  glBindBuffer(GL_ARRAY_BUFFER, 0); // Unbind VBO
  glBindVertexArray(0);             // Unbind VAO

  // Initialize FreeType library
  FT_Library ft;
  if (FT_Init_FreeType(&ft)) {
    FATAL("Error: Could not initialize FreeType library\n");
  }

  const char *font_path = "./fonts/Inconsolata-Regular.ttf";
  if (access(font_path, F_OK) == -1) {
    printf("Font file not found!\n");
  }

  // Load text
  FT_Face face;
  FT_Error error = FT_New_Face(ft, font_path, 0, &face);
  if (error) {
    FATAL("Error: Failed to load text [0x%X]\n", error);
  }

  // Set the text size (width and height in pixels)
  FT_Set_Pixel_Sizes(face, 0, text_size);

  // Disable byte-alignment restriction
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // Setup the standard 128 ASCII characters
  for (unsigned char c = 0; c < 128; c++) {
    // Load character glyph
    if (FT_Load_Char(face, c, FT_LOAD_RENDER)) {
      FATAL("ERROR::FREETYTPE: Failed to load Glyph");
      continue;
    }

    // text details
    const gl_int_t ft_width = face->glyph->bitmap.width;
    const gl_int_t ft_height = face->glyph->bitmap.rows;
    const void *ft_data = face->glyph->bitmap.buffer;

    // Generate texture
    unsigned int texture_id;
    const gl_enum_t target = GL_TEXTURE_2D;
    const gl_enum_t ifmt = GL_RED;
    const gl_enum_t fmt = GL_RED;
    const gl_enum_t type = GL_UNSIGNED_BYTE;

    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(target, 0, ifmt, ft_width, ft_height, 0, fmt, type, ft_data);

    // Set texture options
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Store character for later use
    text->chars[c].texture_id = texture_id;
    text->chars[c].size[0] = face->glyph->bitmap.width;
    text->chars[c].size[1] = face->glyph->bitmap.rows;
    text->chars[c].bearing[0] = face->glyph->bitmap_left;
    text->chars[c].bearing[1] = face->glyph->bitmap_top;
    text->chars[c].offset = face->glyph->advance.x;
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  FT_Done_Face(face);
  FT_Done_FreeType(ft);

  return text;
}

void gl_text_free(gl_text_t *text) {
  if (text == NULL) {
    return;
  }
  GL_DEL_VERTEX_ARRAY(text->VAO);
  GL_DEL_BUFFER(text->VBO);
  GL_DEL_BUFFER(text->EBO);
  free(text);
}

void text_width_height(gl_text_t *text,
                       const char *s,
                       gl_float_t *w,
                       gl_float_t *h) {
  assert(s);
  assert(w);
  assert(h);

  float x = 0.0f;
  gl_char_t *hch = &text->chars[(int) 'H'];
  gl_char_t *ch = &text->chars[(int) s[0]];

  for (size_t i = 0; i < strlen(s); ++i) {
    ch = &text->chars[(int) s[i]];
    x += (ch->offset >> 6);
  }

  *w = x + ch->bearing[0];
  *h = (hch->bearing[1] - ch->bearing[1]) + ch->size[1];
}

void draw_text(gl_text_t *text,
               const char *s,
               const float x,
               const float y,
               const gl_color_t c) {
  assert(text);
  assert(s);

  // Setup projection matrix
  gl_float_t ortho[4 * 4];
  gl_ortho(_window_width, _window_height, ortho);

  // Activate shader
  const gl_float_t scale = 1.0f;
  glDepthMask(GL_FALSE);
  glUseProgram(text->program_id);
  gl_set_mat4(text->program_id, "ortho", ortho);
  gl_set_color(text->program_id, "text_color", c);
  gl_set_int(text->program_id, "text", 0);
  glActiveTexture(GL_TEXTURE0);
  glBindVertexArray(text->VAO);

  // Render text
  float x_ = x;
  gl_char_t *hch = &text->chars[(int) 'H'];
  for (size_t i = 0; i < strlen(s); ++i) {
    gl_char_t *ch = &text->chars[(int) s[i]];
    const float xpos = x_ + ch->bearing[0] * scale;
    const float ypos = y + (hch->bearing[1] - ch->bearing[1]) * scale;
    const float w = ch->size[0] * scale;
    const float h = ch->size[1] * scale;

    // Update VBO for each character
    // clang-format off
    float vertices[6][4] = {
        {xpos,     ypos + h, 0.0f, 1.0f},
        {xpos + w, ypos,     1.0f, 0.0f},
        {xpos,     ypos,     0.0f, 0.0f},
        {xpos,     ypos + h, 0.0f, 1.0f},
        {xpos + w, ypos + h, 1.0f, 1.0f},
        {xpos + w, ypos,     1.0f, 0.0f},
    };
    // clang-format on

    // Render glyph texture over quad
    glBindTexture(GL_TEXTURE_2D, ch->texture_id);

    // Update content of VBO memory
    glBindBuffer(GL_ARRAY_BUFFER, text->VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Render quad
    glDrawArrays(GL_TRIANGLES, 0, 6);
    // Offset cursors for next glyph (Note: advance is number of 1/64 pixels)

    // Bitshift by 6 to get value in pixels (2^6 = 64)
    x_ += (ch->offset >> 6) * scale;
  }

  // Clean up
  glBindVertexArray(0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glDepthMask(GL_TRUE);
}

// MESH //////////////////////////////////////////////////////////////////////

void gl_mesh_setup(gl_mesh_t *mesh,
                   gl_vertex_t *vertices,
                   const int num_vertices,
                   unsigned int *indices,
                   const int num_indices,
                   gl_texture_t *textures,
                   const int num_textures) {
  assert(mesh);
  assert(vertices);
  assert(indices);
  assert(textures);

  // Setup
  mesh->vertices = vertices;
  mesh->indices = indices;
  mesh->textures = textures;
  mesh->num_vertices = num_vertices;
  mesh->num_indices = num_indices;
  mesh->num_textures = num_textures;

  // VAO
  glGenVertexArrays(1, &mesh->VAO);
  glBindVertexArray(mesh->VAO);

  // VBO
  glGenBuffers(1, &mesh->VBO);
  glBindBuffer(GL_ARRAY_BUFFER, mesh->VBO);
  glBufferData(GL_ARRAY_BUFFER,
               sizeof(gl_vertex_t) * num_vertices,
               &vertices[0],
               GL_STATIC_DRAW);

  // EBO
  glGenBuffers(1, &mesh->EBO);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh->EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER,
               sizeof(unsigned int) * num_indices,
               &indices[0],
               GL_STATIC_DRAW);

  // Vertex positions
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0,
                        3,
                        GL_FLOAT,
                        GL_FALSE,
                        sizeof(gl_vertex_t),
                        (void *) 0);

  // Vertex normals
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1,
                        3,
                        GL_FLOAT,
                        GL_FALSE,
                        sizeof(gl_vertex_t),
                        (void *) offsetof(gl_vertex_t, normal));

  // Vertex texture coords
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2,
                        2,
                        GL_FLOAT,
                        GL_FALSE,
                        sizeof(gl_vertex_t),
                        (void *) offsetof(gl_vertex_t, tex_coords));

  // Clean up
  glBindVertexArray(0);
}

void gl_mesh_draw(const gl_mesh_t *mesh, const gl_uint_t shader) {
  assert(mesh);

  // bind appropriate textures
  unsigned int num_diffuse = 1;
  unsigned int num_specular = 1;
  unsigned int num_normal = 1;
  unsigned int num_height = 1;

  for (int i = 0; i < mesh->num_textures; i++) {
    // Active proper texture unit before binding
    glActiveTexture(GL_TEXTURE0 + i);

    // Form texture unit (the N in diffuse_textureN)
    char texture_unit[120] = {0};
    if (strcmp(mesh->textures[i].type, "texture_diffuse") == 0) {
      sprintf(texture_unit, "%s%d", mesh->textures[i].type, num_diffuse++);
    } else if (strcmp(mesh->textures[i].type, "texture_specular") == 0) {
      sprintf(texture_unit, "%s%d", mesh->textures[i].type, num_specular++);
    } else if (strcmp(mesh->textures[i].type, "texture_normal") == 0) {
      sprintf(texture_unit, "%s%d", mesh->textures[i].type, num_normal++);
    } else if (strcmp(mesh->textures[i].type, "texture_height") == 0) {
      sprintf(texture_unit, "%s%d", mesh->textures[i].type, num_height++);
    }

    // Set the sampler to the correct texture unit and bind the texture
    glUniform1i(glGetUniformLocation(shader, texture_unit), i);
    glBindTexture(GL_TEXTURE_2D, mesh->textures[i].id);
  }

  // Draw mesh
  glBindVertexArray(mesh->VAO);
  glDrawElements(GL_TRIANGLES, mesh->num_indices, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  // Set everything back to defaults once configured
  glActiveTexture(GL_TEXTURE0);
}

// MODEL /////////////////////////////////////////////////////////////////////

#define GL_MODEL_VS                                                            \
  "#version 330 core\n"                                                        \
  "layout (location = 0) in vec3 in_pos;\n"                                    \
  "layout (location = 1) in vec3 in_normal;\n"                                 \
  "layout (location = 2) in vec2 in_tex_coords;\n"                             \
  "out vec2 tex_coords;\n"                                                     \
  "out vec3 frag_pos;\n"                                                       \
  "out vec3 normal;\n"                                                         \
  "uniform mat4 model;\n"                                                      \
  "uniform mat4 view;\n"                                                       \
  "uniform mat4 projection;\n"                                                 \
  "void main() {\n"                                                            \
  "  tex_coords = in_tex_coords;\n"                                            \
  "  frag_pos = vec3(model * vec4(in_pos, 1.0));\n"                            \
  "  normal = mat3(transpose(inverse(model))) * in_normal;\n"                  \
  "  gl_Position = projection * view * model * vec4(in_pos, 1.0);\n"           \
  "}\n"

#define GL_MODEL_FS                                                            \
  "#version 330 core\n"                                                        \
  "in vec2 tex_coords;\n"                                                      \
  "out vec4 frag_color;\n"                                                     \
  "uniform sampler2D texture_diffuse1;\n"                                      \
  "void main() {\n"                                                            \
  "  frag_color = texture(texture_diffuse1, tex_coords);\n"                    \
  "}\n"

static unsigned int gl_texture_load(const char *model_dir,
                                    const char *texture_fname) {
  assert(model_dir);
  assert(texture_fname);

  // File fullpath
  char filepath[9046] = {0};
  strcat(filepath, model_dir);
  strcat(filepath, "/");
  strcat(filepath, texture_fname);

  // Generate texture ID
  unsigned int texture_id;
  glGenTextures(1, &texture_id);

  // Load image
  stbi_set_flip_vertically_on_load(1);
  int width = 0;
  int height = 0;
  int channels = 0;
  unsigned char *data = stbi_load(filepath, &width, &height, &channels, 0);
  if (data) {
    // Image format
    gl_enum_t format;
    if (channels == 1) {
      format = GL_RED;
    } else if (channels == 3) {
      format = GL_RGB;
    } else if (channels == 4) {
      format = GL_RGBA;
    } else {
      printf("Invalid number of channels: %d\n", channels);
      return -1;
    }

    // Load image to texture ID
    // clang-format off
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // clang-format on

  } else {
    printf("Texture failed to load: [%s]\n", filepath);
    return -1;
  }

  // Clean up
  stbi_image_free(data);

  return texture_id;
}

static void assimp_load_textures(const struct aiMaterial *material,
                                 const enum aiTextureType type,
                                 const char *model_dir,
                                 gl_texture_t *textures,
                                 int *textures_length) {
  assert(material);
  assert(model_dir);
  assert(textures);
  assert(textures_length);

  // Setup
  int texture_index = MAX(*textures_length - 1, 0);
  const int num_textures = aiGetMaterialTextureCount(material, type);

  // Type name
  char type_name[30] = {0};
  switch (type) {
    case aiTextureType_DIFFUSE:
      strcpy(type_name, "texture_diffuse");
      break;
    case aiTextureType_SPECULAR:
      strcpy(type_name, "texture_specular");
      break;
    case aiTextureType_HEIGHT:
      strcpy(type_name, "texture_height");
      break;
    case aiTextureType_AMBIENT:
      strcpy(type_name, "texture_ambient");
      break;
    default:
      FATAL("Not Implemented!");
      break;
  }

  // Load texture
  for (int index = 0; index < num_textures; index++) {
    struct aiString texture_fname;
    enum aiTextureMapping *mapping = NULL;
    unsigned int *uvindex = NULL;
    ai_real *blend = NULL;
    enum aiTextureOp *op = NULL;
    enum aiTextureMapMode *mapmode = NULL;
    unsigned int *flags = NULL;
    aiGetMaterialTexture(material,
                         type,
                         index,
                         &texture_fname,
                         mapping,
                         uvindex,
                         blend,
                         op,
                         mapmode,
                         flags);

    // Check if texture was loaded before and if so, continue to next iteration
    // int load_texture = 1;
    // for (unsigned int j = 0; j < textures_loaded.size(); j++) {
    //   if (strcmp(textures_loaded[j].path.data(), str.C_Str()) == 0) {
    //     // textures.push_back(textures_loaded[j]);
    //     load_texture = 0;
    //     break;
    //   }
    // }

    // Load texture
    // if (load_texture) {
    //   Texture texture;
    //   texture.id = TextureFromFile(str.C_Str(), this->directory);
    //   texture.type = type_name;
    //   texture.path = str.C_Str();
    //   textures.push_back(texture);
    //   textures_loaded.push_back(texture);
    // }

    textures[texture_index].id = gl_texture_load(model_dir, texture_fname.data);
    strcpy(textures[texture_index].type, type_name);
    strcpy(textures[texture_index].path, texture_fname.data);
    texture_index++;
    (*textures_length)++;
  }
}

static void assimp_load_mesh(const struct aiMesh *mesh,
                             const struct aiMaterial *material,
                             gl_model_t *model) {
  assert(mesh);
  assert(material);
  assert(model);

  // For each mesh vertices
  const int num_vertices = mesh->mNumVertices;
  gl_vertex_t *vertices = malloc(sizeof(gl_vertex_t) * num_vertices);
  for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
    // Position
    vertices[i].position[0] = mesh->mVertices[i].x;
    vertices[i].position[1] = mesh->mVertices[i].y;
    vertices[i].position[2] = mesh->mVertices[i].z;

    // Normal
    if (mesh->mNormals != NULL && mesh->mNumVertices > 0) {
      vertices[i].normal[0] = mesh->mNormals[i].x;
      vertices[i].normal[1] = mesh->mNormals[i].y;
      vertices[i].normal[2] = mesh->mNormals[i].z;
    }

    // Texture coordinates
    if (mesh->mTextureCoords[0]) {
      // Texture coordinates
      vertices[i].tex_coords[0] = mesh->mTextureCoords[0][i].x;
      vertices[i].tex_coords[1] = mesh->mTextureCoords[0][i].y;
      // Note: A vertex can contain up to 8 different texture coordinates. We
      // thus make the assumption that we won't use models where a vertex can
      // have multiple texture coordinates so we always take the first set (0).
    } else {
      // Default Texture coordinates
      vertices[i].tex_coords[0] = 0.0f;
      vertices[i].tex_coords[1] = 0.0f;
    }

    // Tangent
    if (mesh->mTangents) {
      vertices[i].tangent[0] = mesh->mTangents[i].x;
      vertices[i].tangent[1] = mesh->mTangents[i].y;
      vertices[i].tangent[2] = mesh->mTangents[i].z;
    }

    // Bitangent
    if (mesh->mBitangents) {
      vertices[i].bitangent[0] = mesh->mBitangents[i].x;
      vertices[i].bitangent[1] = mesh->mBitangents[i].y;
      vertices[i].bitangent[2] = mesh->mBitangents[i].z;
    }
  }

  // For each mesh face
  // -- Determine number of indices
  size_t num_indices = 0;
  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    num_indices += mesh->mFaces[i].mNumIndices;
  }
  // -- Form indices array
  unsigned int *indices = malloc(sizeof(unsigned int) * num_indices);
  int index_counter = 0;
  for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
    for (unsigned int j = 0; j < mesh->mFaces[i].mNumIndices; j++) {
      indices[index_counter] = mesh->mFaces[i].mIndices[j];
      index_counter++;
    }
  }

  // Process texture materials
  // struct aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
  // Note: we assume a convention for sampler names in the shaders. Each
  // diffuse texture should be named as 'texture_diffuseN' where N is a
  // sequential number ranging from 1 to MAX_SAMPLER_NUMBER. Same applies to
  // other texture as the following list summarizes:
  // diffuse: texture_diffuseN
  // specular: texture_specularN
  // normal: texture_normalN

  // -- Get total number of textures
  int num_textures = 0;
  num_textures += aiGetMaterialTextureCount(material, aiTextureType_DIFFUSE);
  num_textures += aiGetMaterialTextureCount(material, aiTextureType_SPECULAR);
  num_textures += aiGetMaterialTextureCount(material, aiTextureType_HEIGHT);
  num_textures += aiGetMaterialTextureCount(material, aiTextureType_AMBIENT);

  // -- Load textures
  const char *model_dir = model->model_dir;
  int textures_length = 0;
  gl_texture_t *textures = malloc(sizeof(gl_texture_t) * num_textures);

  assimp_load_textures(material,
                       aiTextureType_DIFFUSE,
                       model_dir,
                       textures,
                       &textures_length);
  assimp_load_textures(material,
                       aiTextureType_SPECULAR,
                       model_dir,
                       textures,
                       &textures_length);
  assimp_load_textures(material,
                       aiTextureType_HEIGHT,
                       model_dir,
                       textures,
                       &textures_length);
  assimp_load_textures(material,
                       aiTextureType_AMBIENT,
                       model_dir,
                       textures,
                       &textures_length);

  // Form Mesh
  const int mesh_index = model->num_meshes;
  gl_mesh_setup(&model->meshes[mesh_index],
                vertices,
                num_vertices,
                indices,
                num_indices,
                textures,
                num_textures);
  model->num_meshes++;
}

static void assimp_load_model(const struct aiScene *scene,
                              const struct aiNode *node,
                              gl_model_t *model) {
  // Process each mesh located at the current node
  for (unsigned int i = 0; i < node->mNumMeshes; i++) {
    // The node object only contains indices to index the actual objects in the
    // scene. The scene contains all the data, node is just to keep stuff
    // organized (like relations between nodes).
    struct aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
    struct aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];
    assimp_load_mesh(mesh, material, model);
  }

  // After processing all of the meshes (if any) we then recursively process
  // each of the children nodes
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    assimp_load_model(scene, node->mChildren[i], model);
  }
}

static int assimp_num_meshes(const struct aiNode *node) {
  assert(node);
  int num_meshes = node->mNumMeshes;
  for (unsigned int i = 0; i < node->mNumChildren; i++) {
    num_meshes += assimp_num_meshes(node->mChildren[i]);
  }
  return num_meshes;
}

gl_model_t *gl_model_load(const char *model_path) {
  assert(model_path);

  // Check model file
  if (access(model_path, F_OK) != 0) {
    return NULL;
  }

  // Malloc
  gl_model_t *model = malloc(sizeof(gl_model_t) * 1);

  // Entity transform
  gl_eye(model->T, 4, 4);
  model->T[12] = 0.0;
  model->T[13] = 0.0;
  model->T[14] = 0.0;

  // Shader program
  model->program_id = gl_shader(GL_MODEL_VS, GL_MODEL_FS, NULL);
  if (model->program_id == GL_FALSE) {
    FATAL("Failed to create shader!");
  }

  // Using assimp to load model
  const struct aiScene *scene =
      aiImportFile(model_path, aiProcessPreset_TargetRealtime_MaxQuality);
  if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE ||
      !scene->mRootNode) {
    printf("Failed to load model: %s\n", model_path);
    free(model);
    return NULL;
  }

  // Get model directory
  char path[9046] = {0};
  strcpy(path, model_path);
  char *model_dir = dirname(path);
  if (model_dir == NULL) {
    printf("Failed to get directory name of [%s]!", model_path);
    free(model);
    return NULL;
  }
  strcpy(model->model_dir, model_dir);

  // Load model
  const int num_meshes = assimp_num_meshes(scene->mRootNode);
  model->meshes = malloc(sizeof(gl_mesh_t) * num_meshes);
  model->num_meshes = 0;
  assimp_load_model(scene, scene->mRootNode, model);

  // Clean up
  aiReleaseImport(scene);

  return model;
}

void gl_model_free(gl_model_t *model) {
  if (model == NULL) {
    return;
  }

  for (int i = 0; i < model->num_meshes; i++) {
    free(model->meshes[i].vertices);
    free(model->meshes[i].indices);
    free(model->meshes[i].textures);
  }
  free(model->meshes);
  free(model);
  model = NULL;
}

void gl_model_draw(const gl_model_t *model, const gl_camera_t *camera) {
  glUseProgram(model->program_id);
  gl_set_mat4(model->program_id, "projection", _camera.P);
  gl_set_mat4(model->program_id, "view", _camera.V);
  gl_set_mat4(model->program_id, "model", model->T);

  float light_pos[3] = {0, 10, 0};
  float light_color[3] = {1, 1, 1};
  float object_color[3] = {1, 1, 1};
  gl_set_vec3(model->program_id, "lightPos", light_pos);
  gl_set_vec3(model->program_id, "viewPos", _camera.position);
  gl_set_vec3(model->program_id, "lightColor", light_color);
  gl_set_vec3(model->program_id, "objectColor", object_color);

  for (int i = 0; i < model->num_meshes; i++) {
    gl_mesh_draw(&model->meshes[i], model->program_id);
  }
}
