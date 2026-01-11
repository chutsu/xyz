#include "munit.h"
#include "xyz.h"
// #include "xyz_gui.h"

static GLFWwindow *test_setup(void) {
  // GLFW
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  const int win_w = 800;
  const int win_h = 800;
  const char *win_title = "Test";
  GLFWwindow *win = glfwCreateWindow(win_w, win_h, win_title, NULL, NULL);
  if (win == NULL) {
    FATAL("Failed to create GLFW window!\n!");
    glfwTerminate();
    return NULL;
  }
  glfwMakeContextCurrent(win);

  // GLAD
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    FATAL("Failed to load GL context!\n!");
    return NULL;
  }

  // OpenGL Features
  glEnable(GL_CULL_FACE);

  return win;
}

static void test_teardown(GLFWwindow *window) {
  glfwTerminate();
  // free(window);
}

// TEST GLFW /////////////////////////////////////////////////////////////////

int test_glfw(void) {
  if (!glfwInit()) {
    printf("Cannot initialize GLFW\n");
    exit(EXIT_FAILURE);
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  GLFWwindow *window = glfwCreateWindow(640, 480, "Simple example", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return -1;
  }

  while (!glfwWindowShouldClose(window)) {
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  glfwTerminate();

  return 0;
}

// TEST OPENGL UTILS /////////////////////////////////////////////////////////

int test_gl_zeros(void) {
  // clang-format off
  gl_float_t A[3*3] = {1.0, 4.0, 7.0,
                    2.0, 5.0, 8.0,
                    3.0, 6.0, 9.0};
  gl_float_t expected[3*3] = {0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0};
  // clang-format on
  gl_zeros(A, 3, 3);
  MU_ASSERT(gl_equals(A, expected, 3, 3, 1e-8));

  return 0;
}

int test_gl_ones(void) {
  // clang-format off
  gl_float_t A[3*3] = {1.0, 4.0, 7.0,
                    2.0, 5.0, 8.0,
                    3.0, 6.0, 9.0};
  gl_float_t expected[3*3] = {1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0};
  // clang-format on
  gl_ones(A, 3, 3);
  MU_ASSERT(gl_equals(A, expected, 3, 3, 1e-8));

  return 0;
}

int test_gl_eye(void) {
  /* Check 4x4 matrix */
  // clang-format off
  gl_float_t A[4*4] = {0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0};
  gl_float_t A_expected[4*4] = {1.0, 0.0, 0.0, 0.0,
                                0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 0.0, 1.0};
  // clang-format on
  gl_eye(A, 4, 4);
  MU_ASSERT(gl_equals(A, A_expected, 4, 4, 1e-8));

  /* Check 3x4 matrix */
  // clang-format off
  gl_float_t B[3*4] = {0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0};
  gl_float_t B_expected[3*4] = {1.0, 0.0, 0.0,
                             0.0, 1.0, 0.0,
                             0.0, 0.0, 1.0,
                             0.0, 0.0, 0.0};
  // clang-format on
  gl_eye(B, 3, 4);
  MU_ASSERT(gl_equals(B, B_expected, 3, 4, 1e-8));

  return 0;
}

int test_gl_equals(void) {
  // clang-format off
  gl_float_t A[3*3] = {1.0, 4.0, 7.0,
                       2.0, 5.0, 8.0,
                       3.0, 6.0, 9.0};
  gl_float_t B[3*3] = {1.0, 4.0, 7.0,
                       2.0, 5.0, 8.0,
                       3.0, 6.0, 9.0};
  gl_float_t C[3*3] = {1.0, 4.0, 7.0,
                       2.0, 5.0, 8.0,
                       3.0, 6.0, 10.0};
  // clang-format on

  /* Assert */
  MU_ASSERT(gl_equals(A, B, 3, 3, 1e-8) == 1);
  MU_ASSERT(gl_equals(A, C, 3, 3, 1e-8) == 0);

  return 0;
}

int test_gl_mat_set(void) {
  // clang-format off
  gl_float_t A[3*4] = {0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0};
  // clang-format on

  gl_mat_set(A, 3, 4, 0, 1, 1.0);
  gl_mat_set(A, 3, 4, 1, 0, 2.0);
  gl_mat_set(A, 3, 4, 0, 2, 3.0);
  gl_mat_set(A, 3, 4, 2, 0, 4.0);

  return 0;
}

int test_gl_mat_val(void) {
  // clang-format off
  gl_float_t A[3*4] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0,
                       10.0, 11.0, 12.0};
  // clang-format on

  const float tol = 1e-4;
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 0, 0) - 1.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 1, 0) - 2.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 2, 0) - 3.0) < tol);

  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 0, 1) - 4.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 1, 1) - 5.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 2, 1) - 6.0) < tol);

  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 0, 2) - 7.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 1, 2) - 8.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 2, 2) - 9.0) < tol);

  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 0, 3) - 10.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 1, 3) - 11.0) < tol);
  MU_ASSERT(fabs(gl_mat_val(A, 3, 4, 2, 3) - 12.0) < tol);

  return 0;
}

int test_gl_transpose(void) {
  /* Transpose a 3x3 matrix */
  // clang-format off
  gl_float_t A[3*3] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0};
  // clang-format on
  gl_float_t A_t[3 * 3] = {0};

  gl_transpose(A, 3, 3, A_t);

  /* Transpose a 3x4 matrix */
  // clang-format off
  gl_float_t B[3*4] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0,
                       10.0, 11.0, 12.0};
  // clang-format on
  gl_float_t B_t[3 * 4] = {0};
  gl_transpose(B, 3, 4, B_t);

  return 0;
}

int test_gl_vec3_cross(void) {
  const gl_float_t u[3] = {1.0f, 2.0f, 3.0f};
  const gl_float_t v[3] = {4.0f, 5.0f, 6.0f};
  gl_float_t z[3] = {0};
  gl_vec3_cross(u, v, z);

  /* Assert */
  gl_float_t expected[3] = {-3.0f, 6.0f, -3.0f};
  MU_ASSERT(gl_equals(z, expected, 3, 1, 1e-8));

  return 0;
}

int test_gl_dot(void) {
  // clang-format off
  gl_float_t A[3*3] = {1.0, 4.0, 7.0,
                       2.0, 5.0, 8.0,
                       3.0, 6.0, 9.0};
  gl_float_t B[3*3] = {1.0, 4.0, 7.0,
                       2.0, 5.0, 8.0,
                       3.0, 6.0, 9.0};
  // clang-format on
  gl_float_t C[3 * 3] = {0.0};
  gl_dot(A, 3, 3, B, 3, 3, C);

  /* Assert */
  // clang-format off
  gl_float_t expected[3*3] = {30.0f, 66.0f, 102.0f,
                              36.0f, 81.0f, 126.0f,
                              42.0f, 96.0f, 150.0f};
  // clang-format on
  MU_ASSERT(gl_equals(C, expected, 3, 3, 1e-8));

  return 0;
}

int test_gl_norm(void) {
  const gl_float_t x[3] = {1.0f, 2.0f, 3.0f};
  const gl_float_t n = gl_norm(x, 3);

  /* Assert */
  const gl_float_t expected = 3.741657f;
  MU_ASSERT(fabs(n - expected) < 1e-6);

  return 0;
}

int test_gl_normalize(void) {
  gl_float_t x[3] = {1.0f, 2.0f, 3.0f};
  gl_normalize(x, 3);

  /* Assert */
  const gl_float_t expected[3] = {0.26726f, 0.53452f, 0.80178f};
  MU_ASSERT(gl_equals(x, expected, 3, 1, 1e-5));

  return 0;
}

int test_gl_perspective(void) {
  const gl_float_t fov = gl_deg2rad(60.0);
  const gl_float_t window_width = 1000.0f;
  const gl_float_t window_height = 1000.0f;
  const gl_float_t ratio = window_width / window_height;
  const gl_float_t near = 0.1f;
  const gl_float_t far = 100.0f;

  gl_float_t P[4 * 4] = {0};
  gl_perspective(fov, ratio, near, far, P);

  // clang-format off
  const gl_float_t P_expected[4*4] = {
    1.732051, 0.000000, 0.000000, 0.000000,
    0.000000, 1.732051, 0.000000, 0.000000,
    0.000000, 0.000000, -1.002002, -1.000000,
    0.000000, 0.000000, -0.200200, 0.000000
  };
  // clang-format on
  MU_ASSERT(gl_equals(P, P_expected, 4, 4, 1e-4));

  return 0;
}

int test_gl_ortho(void) {
  const gl_float_t w = 800.0f;
  const gl_float_t h = 600.0f;
  gl_float_t P[4 * 4] = {0};
  gl_ortho(w, h, P);

  // clang-format off
  const gl_float_t P_expected[4*4] = {
    0.00250000,  0.00000000,  0.000000,  0.000000,
    0.00000000, -0.00333333,  0.000000,  0.000000,
    0.00000000,  0.00000000, -1.000000,  0.000000,
   -1.00000000,  1.00000000,  0.000000,  1.000000
  };
  // clang-format on
  MU_ASSERT(gl_equals(P, P_expected, 4, 4, 1e-4));

  return 0;
}

int test_gl_lookat(void) {
  const gl_float_t yaw = -0.785398;
  const gl_float_t pitch = 0.000000;
  const gl_float_t radius = 10.000000;
  const gl_float_t focal[3] = {0.000000, 0.000000, 0.000000};
  const gl_float_t world_up[3] = {0.000000, 1.000000, 0.000000};

  gl_float_t eye[3];
  eye[0] = focal[0] + radius * sin(yaw);
  eye[1] = focal[1] + radius * cos(pitch);
  eye[2] = focal[2] + radius * cos(yaw);

  gl_float_t V[4 * 4] = {0};
  gl_lookat(eye, focal, world_up, V);

  // clang-format off
  const gl_float_t V_expected[4*4] = {
    0.707107, 0.500000, -0.500000, 0.000000,
    -0.000000, 0.707107, 0.707107, 0.000000,
    0.707107, -0.500000, 0.500000, 0.000000,
    0.000000, 0.000000, -14.142136, 1.000000
  };
  // clang-format on
  MU_ASSERT(gl_equals(V, V_expected, 4, 4, 1e-4));

  return 0;
}

// TEST SHADER ///////////////////////////////////////////////////////////////

int test_gl_compile(void) {

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

  // Setup
  GLFWwindow *window = test_setup();

  // Vertex shader
  char *vs_str = GL_RECT_VS;
  const gl_uint_t vs = gl_compile(vs_str, GL_VERTEX_SHADER);
  MU_ASSERT(vs != GL_FALSE);

  // Fragment shader
  char *fs_str = GL_RECT_FS;
  const gl_uint_t fs = gl_compile(fs_str, GL_VERTEX_SHADER);
  MU_ASSERT(fs != GL_FALSE);

  // Cleanup
  test_teardown(window);

  return 0;
}

int test_gl_link(void) {

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

  // Setup
  GLFWwindow *window = test_setup();

  // Cube vertex shader
  char *vs_str = GL_RECT_VS;
  const gl_uint_t vs = gl_compile(vs_str, GL_VERTEX_SHADER);
  MU_ASSERT(vs != GL_FALSE);

  // Cube fragment shader
  char *fs_str = GL_RECT_FS;
  const gl_uint_t fs = gl_compile(fs_str, GL_FRAGMENT_SHADER);
  MU_ASSERT(fs != GL_FALSE);

  // Link shakders
  const gl_uint_t gs = GL_FALSE;
  const gl_uint_t prog = gl_link(vs, fs, gs);
  MU_ASSERT(prog != GL_FALSE);

  // Cleanup
  test_teardown(window);

  return 0;
}

// TEST GL PROGRAM ///////////////////////////////////////////////////////////

int test_gl_shader(void) {

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

  // Setup
  GLFWwindow *window = test_setup();

  // Shader program
  const gl_uint_t program_id = gl_shader(GL_RECT_VS, GL_RECT_FS, NULL);
  MU_ASSERT(program_id != GL_FALSE);

  // Cleanup
  test_teardown(window);

  return 0;
}

// TEST GL-CAMERA ////////////////////////////////////////////////////////////

int test_gl_camera_setup(void) {
  int window_width = 640;
  int window_height = 480;

  gl_camera_t camera;
  gl_camera_setup(&camera, &window_width, &window_height);

  return 0;
}

// TEST GL-MODEL /////////////////////////////////////////////////////////////

int test_gl_model_load(void) {
  gl_model_t *model = gl_model_load("/home/chutsu/monkey.obj");
  gl_model_free(model);
  MU_ASSERT(model == NULL);

  return 0;
}

// TEST GUI //////////////////////////////////////////////////////////////////

int test_gui(void) {
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  for (int i = 0; i < 10; ++i) {
    gui_poll(gui);
    gui_update(gui);
  }
  gui_free(gui);

  return 0;
}

int test_gl_rect(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Rect
  gl_bounds_t rect_bounds = (gl_bounds_t){10, 10, 100, 100};
  gl_color_t rect_color = (gl_color_t){1.0f, 0.0f, 1.0f};
  gl_rect_t *rect = gl_rect_malloc(rect_bounds, rect_color);

  // Render
  while (gui_poll(gui)) {
    draw_rect(rect);
    gui_update(gui);
  }

  // Clean up
  gl_rect_free(rect);
  gui_free(gui);

  return 0;
}

int test_gl_points3d(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Points3D
  gl_color_t points_color = (gl_color_t){1.0, 0.0, 0.0};
  gl_float_t point_size = 5.0;
  size_t num_points = 1e3;
  gl_float_t *points_data = malloc(sizeof(gl_float_t) * num_points * 6);
  for (size_t i = 0; i < num_points; ++i) {
    points_data[i * 6 + 0] = gl_randf(-1.0f, 1.0f);
    points_data[i * 6 + 1] = gl_randf(-1.0f, 1.0f);
    points_data[i * 6 + 2] = gl_randf(-1.0f, 1.0f);
    points_data[i * 6 + 3] = points_color.r;
    points_data[i * 6 + 4] = points_color.g;
    points_data[i * 6 + 5] = points_color.b;
  }
  gl_points3d_t *points3d = gl_points3d_malloc(NULL, 0, point_size);
  gl_points3d_update(points3d, points_data, num_points, point_size);
  free(points_data);

  // Render
  while (gui_poll(gui)) {
    draw_points3d(points3d);
    gui_update(gui);
  }

  // Clean up
  gl_points3d_free(points3d);
  gui_free(gui);

  return 0;
}

int test_gl_line3d(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Line3D
  gl_float_t line_lw = 5.0f;
  gl_color_t line_color = (gl_color_t){1.0, 0.0, 0.0};
  size_t line_length = 100;
  float radius = 3.0f;
  float dtheta = 2 * M_PI / (line_length - 1);
  float theta = 0.0f;
  float *line_data = malloc(sizeof(float) * line_length * 3);
  for (size_t i = 0; i < line_length; ++i) {
    line_data[i * 3 + 0] = radius * sin(theta);
    line_data[i * 3 + 1] = 0.0f;
    line_data[i * 3 + 2] = radius * cos(theta);
    theta += dtheta;
  }
  gl_line3d_t *line3d = gl_line3d_malloc(line_color, line_lw);
  gl_line3d_update(line3d, 0, line_data, line_length);

  // Draw
  while (gui_poll(gui)) {
    draw_line3d(line3d);
    gui_update(gui);
  }

  // Clean up
  gl_line3d_free(line3d);
  gui_free(gui);
  free(line_data);

  return 0;
}

int test_gl_cube3d(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Cube
  gl_float_t cube_T[4 * 4] = {0};

  gl_eye(cube_T, 4, 4);
  cube_T[12] = 0.0;
  cube_T[13] = 0.0;
  cube_T[14] = 1.0;
  const gl_float_t cube_size = 1.0f;
  const gl_color_t cube_color = (gl_color_t){1.0, 0.0, 0.0};
  gl_cube3d_t *cube = gl_cube3d_malloc();

  // Render
  while (gui_poll(gui)) {
    draw_cube(cube, cube_T, cube_size, cube_color);
    gui_update(gui);
  }

  // Clean up
  gl_cube3d_free(cube);
  gui_free(gui);

  return 0;
}

int test_gl_axes3d(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Axes
  gl_float_t axes_T[4 * 4];
  gl_eye(axes_T, 4, 4);
  gl_float_t axes_size = 0.5f;
  gl_float_t axes_lw = 3.0f;
  gl_axes3d_t *axes = gl_axes3d_malloc(axes_T, axes_size, axes_lw);

  // Render
  while (gui_poll(gui)) {
    draw_axes3d(axes);
    gui_update(gui);
  }

  // Clean up
  gl_axes3d_free(axes);
  gui_free(gui);

  return 0;
}

int test_gl_grid3d(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Grid
  gl_int_t num_rows = 10;
  gl_int_t num_cols = 10;
  gl_float_t size = 1.0f;
  gl_float_t lw = 2.0f;
  gl_color_t color = (gl_color_t){1.0, 1.0, 1.0};
  gl_grid3d_t *grid = gl_grid3d_malloc(num_rows, num_cols, size, color, lw);

  // Render
  while (gui_poll(gui)) {
    draw_grid3d(grid);
    gui_update(gui);
  }

  // Clean up
  gl_grid3d_free(grid);
  gui_free(gui);

  return 0;
}

int test_gl_image(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Load image
  int w = 0;
  int h = 0;
  int c = 0;
  const char *image_path = "test_data/images/awesomeface.png";
  stbi_set_flip_vertically_on_load(1);
  uint8_t *image_data = stbi_load(image_path, &w, &h, &c, 3);

  // Form gl_image
  const int x = window_width / 2.0 - w / 2.0;
  const int y = window_height / 2.0 - h / 2.0;
  gl_image_t *image = gl_image_malloc(x, y, image_data, w, h, c);

  // Render
  while (gui_poll(gui)) {
    draw_image(image);
    gui_update(gui);
  }

  // Clean up
  stbi_image_free(image_data);
  gl_image_free(image);
  gui_free(gui);

  return 0;
}

int test_gl_text(void) {
  // Setup
  const char *window_title = "viz";
  const int window_width = 1024;
  const int window_height = 768;
  gui_t *gui = gui_malloc(window_title, window_width, window_height);

  // Text
  gl_color_t text_color = (gl_color_t){1.0, 1.0, 1.0};
  int text_size = 18;
  gl_text_t *text = gl_text_malloc(text_size);

  // Render
  while (gui_poll(gui)) {
    // Get text width and height
    gl_float_t text_w = 0.0f;
    gl_float_t text_h = 0.0f;
    const char *text_str = "Hello World";
    text_width_height(text, text_str, &text_w, &text_h);

    // Center text and draw
    const int text_x = window_width / 2.0 - text_w / 2.0;
    const int text_y = window_height / 2.0 - text_h / 2.0;
    draw_text(text, text_str, text_x, text_y, text_color);

    gui_update(gui);
  }

  // Clean up
  gl_text_free(text);
  gui_free(gui);

  return 0;
}

// TEST SANDBOX //////////////////////////////////////////////////////////////

int test_sandbox(void) {
  GLFWwindow *window = test_setup();

  // // Render loop
  // while (!glfwWindowShouldClose(win)) {
  //   glClear(GL_DEPTH_BUFFER_BIT);
  //   glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
  //   glClear(GL_COLOR_BUFFER_BIT);
  //
  //   glfwSwapBuffers(win);
  //   glfwPollEvents();
  // }

  // Clean up
  test_teardown(window);

  return 0;
}

// TEST-SUITE ////////////////////////////////////////////////////////////////

void test_suite(void) {
  // TEST(test_glfw);

  MU_ADD_TEST(test_gl_zeros);
  MU_ADD_TEST(test_gl_ones);
  MU_ADD_TEST(test_gl_eye);
  MU_ADD_TEST(test_gl_equals);
  MU_ADD_TEST(test_gl_mat_set);
  MU_ADD_TEST(test_gl_mat_val);
  MU_ADD_TEST(test_gl_transpose);
  MU_ADD_TEST(test_gl_vec3_cross);
  MU_ADD_TEST(test_gl_dot);
  MU_ADD_TEST(test_gl_norm);
  MU_ADD_TEST(test_gl_normalize);
  MU_ADD_TEST(test_gl_perspective);
  MU_ADD_TEST(test_gl_ortho);
  MU_ADD_TEST(test_gl_lookat);
  MU_ADD_TEST(test_gl_compile);
  MU_ADD_TEST(test_gl_link);
  MU_ADD_TEST(test_gl_shader);
  MU_ADD_TEST(test_gl_camera_setup);
  MU_ADD_TEST(test_gl_model_load);
#if CI_MODE == 0
  MU_ADD_TEST(test_gui);
  MU_ADD_TEST(test_gl_rect);
  MU_ADD_TEST(test_gl_points3d);
  MU_ADD_TEST(test_gl_line3d);
  MU_ADD_TEST(test_gl_cube3d);
  MU_ADD_TEST(test_gl_axes3d);
  MU_ADD_TEST(test_gl_grid3d);
  MU_ADD_TEST(test_gl_image);
  MU_ADD_TEST(test_gl_text);
  MU_ADD_TEST(test_sandbox);
#endif
}
MU_RUN_TESTS(test_suite)
