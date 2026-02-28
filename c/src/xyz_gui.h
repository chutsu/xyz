#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <assert.h>
#include <libgen.h>

#include "stb_image.h"
#include "glad.h"
#include <GLFW/glfw3.h>

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/material.h>
#include <assimp/postprocess.h>

#include <ft2build.h>
#include FT_FREETYPE_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/******************************************************************************
 * MACROS
 *****************************************************************************/

/**
 * Mark variable unused.
 * @param[in] expr Variable to mark as unused
 */
#ifndef UNUSED
#define UNUSED(expr)                                                           \
  do {                                                                         \
    (void) (expr);                                                             \
  } while (0)
#endif

/**
 * Return max between a and b
 */
#ifndef MAX
#define MAX(a, b) a > b ? a : b
#endif

/**
 * Return min between a and b
 */
#ifndef MIN
#define MIN(a, b) a < b ? a : b;
#endif

/**
 * Fatal
 *
 * @param[in] M Message
 * @param[in] ... Varadic arguments
 */
#ifndef FATAL
#define FATAL(...)                                                             \
  do {                                                                         \
    fprintf(stderr, "[FATAL] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__);   \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0);                                                                 \
  exit(-1)
#endif

/**
 * Log error
 * @param[in] M Message
 * @param[in] ... Varadic arguments
 */
#ifndef LOG_ERROR
#define LOG_ERROR(...)                                                         \
  do {                                                                         \
    fprintf(stderr, "[ERROR] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__);   \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0)
#endif

/******************************************************************************
 * OPENGL UTILS
 *****************************************************************************/

typedef GLuint gl_uint_t;
typedef GLint gl_int_t;
typedef GLfloat gl_float_t;
typedef GLenum gl_enum_t;

typedef struct {
  gl_uint_t texture_id;
  gl_int_t size[2];    // Size of glyph
  gl_int_t bearing[2]; // Offset from baseline to left/top of glyph
  gl_uint_t offset;    // Offset to advance to next glyph
} gl_char_t;

typedef struct {
  gl_float_t x;
  gl_float_t y;
  gl_float_t z;
} gl_vec3_t;

typedef struct {
  gl_float_t w;
  gl_float_t x;
  gl_float_t y;
  gl_float_t z;
} gl_quat_t;

typedef struct {
  gl_float_t r;
  gl_float_t g;
  gl_float_t b;
} gl_color_t;

typedef struct {
  gl_int_t x;
  gl_int_t y;
  gl_int_t w;
  gl_int_t h;
} gl_bounds_t;

typedef struct {
  gl_vec3_t pos;
  gl_quat_t quat;
} gl_pose_t;

char *load_file(const char *fp);

gl_enum_t gl_check_error(const char *file, const int line);
#define GL_CHECK_ERROR() gl_check_error(__FILE__, __LINE__)

gl_float_t gl_randf(const gl_float_t a, const gl_float_t b);
gl_float_t gl_deg2rad(const gl_float_t d);
gl_float_t gl_rad2deg(const gl_float_t r);
gl_float_t gl_rad2deg(const gl_float_t r);
void gl_print_vector(const char *prefix, const gl_float_t *x, const int length);
void gl_print_matrix(const char *prefix,
                     const gl_float_t *A,
                     const int num_rows,
                     const int num_cols);
int gl_equals(const gl_float_t *A,
              const gl_float_t *B,
              const int num_rows,
              const int num_cols,
              const gl_float_t tol);
void gl_mat_set(gl_float_t *A,
                const int m,
                const int n,
                const int i,
                const int j,
                const gl_float_t val);
gl_float_t gl_mat_val(const gl_float_t *A,
                      const int m,
                      const int n,
                      const int i,
                      const int j);
void gl_copy(const gl_float_t *src, const int m, const int n, gl_float_t *dest);
void gl_transpose(const gl_float_t *A, size_t m, size_t n, gl_float_t *A_t);
void gl_zeros(gl_float_t *A, const int num_rows, const int num_cols);
void gl_ones(gl_float_t *A, const int num_rows, const int num_cols);
void gl_eye(gl_float_t *A, const int num_rows, const int num_cols);
void gl_vec2(gl_float_t *v, const gl_float_t x, const gl_float_t y);
void gl_vec3(gl_float_t *v,
             const gl_float_t x,
             const gl_float_t y,
             const gl_float_t z);
void gl_vec4(gl_float_t *v,
             const gl_float_t x,
             const gl_float_t y,
             const gl_float_t z,
             const gl_float_t w);
void gl_vec3_cross(const gl_float_t u[3],
                   const gl_float_t v[3],
                   gl_float_t n[3]);

void gl_add(const gl_float_t *A,
            const gl_float_t *B,
            const int num_rows,
            const int num_cols,
            gl_float_t *C);
void gl_sub(const gl_float_t *A,
            const gl_float_t *B,
            const int num_rows,
            const int num_cols,
            gl_float_t *C);
void gl_dot(const gl_float_t *A,
            const int A_m,
            const int A_n,
            const gl_float_t *B,
            const int B_m,
            const int B_n,
            gl_float_t *C);
void gl_scale(gl_float_t factor,
              gl_float_t *A,
              const int num_rows,
              const int num_cols);
gl_float_t gl_norm(const gl_float_t *x, const int size);
void gl_normalize(gl_float_t *x, const int size);

void gl_perspective(const gl_float_t fov,
                    const gl_float_t aspect,
                    const gl_float_t near,
                    const gl_float_t far,
                    gl_float_t P[4 * 4]);
void gl_ortho(const gl_float_t w, const gl_float_t h, gl_float_t P[4 * 4]);
void gl_lookat(const gl_float_t eye[3],
               const gl_float_t at[3],
               const gl_float_t up[3],
               gl_float_t V[4 * 4]);
void gl_euler321(const gl_float_t ypr[3], gl_float_t C[3 * 3]);
void gl_euler2quat(const gl_float_t ypr[3], gl_quat_t *q);
void gl_quat2rot(const gl_quat_t *q, gl_float_t C[3 * 3]);
void gl_rot2quat(const gl_float_t C[3 * 3], gl_quat_t *q);

void gl_tf(const gl_float_t params[7], gl_float_t T[4 * 4]);
void gl_tf_cr(const gl_float_t C[3 * 3],
              const gl_float_t r[3],
              gl_float_t T[4 * 4]);
void gl_tf_qr(const gl_quat_t *q, const gl_float_t r[3], gl_float_t T[4 * 4]);
void gl_tf_er(const gl_float_t ypr[3],
              const gl_float_t r[3],
              gl_float_t T[4 * 4]);
void gl_tf2pose(const gl_float_t T[4 * 4], gl_pose_t *pose);
int gl_save_frame_buffer(const int width, const int height, const char *fp);

void gl_jet_colormap(const gl_float_t value,
                     gl_float_t *r,
                     gl_float_t *g,
                     gl_float_t *b);

/******************************************************************************
 * GL-SHADER
 *****************************************************************************/

gl_uint_t gl_compile(const char *src, const int type);
gl_uint_t gl_link(const gl_uint_t vs, const gl_uint_t fs, const gl_uint_t gs);
gl_uint_t gl_shader(const char *vs_src, const char *fs_src, const char *gs_src);

int gl_set_color(const gl_int_t id, const char *k, const gl_color_t v);
int gl_set_int(const gl_int_t id, const char *k, const gl_int_t v);
int gl_set_float(const gl_int_t id, const char *k, const gl_float_t v);
int gl_set_vec2i(const gl_int_t id, const char *k, const gl_int_t v[2]);
int gl_set_vec3i(const gl_int_t id, const char *k, const gl_int_t v[3]);
int gl_set_vec4i(const gl_int_t id, const char *k, const gl_int_t v[4]);
int gl_set_vec2(const gl_int_t id, const char *k, const gl_float_t v[2]);
int gl_set_vec3(const gl_int_t id, const char *k, const gl_float_t v[3]);
int gl_set_vec4(const gl_int_t id, const char *k, const gl_float_t v[4]);
int gl_set_mat2(const gl_int_t id, const char *k, const gl_float_t v[2 * 2]);
int gl_set_mat3(const gl_int_t id, const char *k, const gl_float_t v[3 * 3]);
int gl_set_mat4(const gl_int_t id, const char *k, const gl_float_t v[4 * 4]);

/******************************************************************************
 * GL-CAMERA
 *****************************************************************************/

typedef enum { ORBIT, FPS } gl_view_mode_t;

typedef struct gl_camera_t {
  gl_view_mode_t view_mode;

  gl_float_t focal[3];
  gl_float_t world_up[3];
  gl_float_t position[3];
  gl_float_t right[3];
  gl_float_t up[3];
  gl_float_t front[3];
  gl_float_t yaw;
  gl_float_t pitch;
  gl_float_t radius;

  gl_float_t fov;
  gl_float_t fov_min;
  gl_float_t fov_max;
  gl_float_t near;
  gl_float_t far;

  gl_float_t P[4 * 4]; // Projection matrix
  gl_float_t V[4 * 4]; // View matrix
} gl_camera_t;

void gl_camera_setup(gl_camera_t *camera,
                     int *screen_width,
                     int *screen_height);
void gl_camera_update(gl_camera_t *camera);
void gl_camera_rotate(gl_camera_t *camera,
                      const float factor,
                      const float dx,
                      const float dy);
void gl_camera_pan(gl_camera_t *camera,
                   const float factor,
                   const float dx,
                   const float dy);
void gl_camera_zoom(gl_camera_t *camera,
                    const float factor,
                    const float dx,
                    const float dy);

/******************************************************************************
 * GUI
 *****************************************************************************/

typedef struct gui_t {
  GLFWwindow *window;
  float fps_limit;
  float last_time;
  float last_frame;

  int *key_q;
  int *key_w;
  int *key_a;
  int *key_s;
  int *key_d;
  int *key_n;
  int *key_esc;
  int *key_equal;
  int *key_minus;
} gui_t;

gui_t *gui_malloc(const char *window_title,
                  const int window_width,
                  const int window_height);
void gui_free(gui_t *gui);

double gui_time(void);
int gui_poll(gui_t *gui);
void gui_update(gui_t *gui);

// RECT //////////////////////////////////////////////////////////////////////

typedef struct gl_rect_t {
  gl_uint_t program_id;
  gl_uint_t VAO;
  gl_uint_t VBO;
  gl_uint_t EBO;
  gl_bounds_t bounds;
  gl_color_t color;
} gl_rect_t;

gl_rect_t *gl_rect_malloc(const gl_bounds_t bounds, const gl_color_t color);
void gl_rect_free(gl_rect_t *rect);
void draw_rect(gl_rect_t *rect);

// POINTS 3D /////////////////////////////////////////////////////////////////

typedef struct gl_points3d_t {
  gl_uint_t program_id;
  gl_uint_t VAO;
  gl_uint_t VBO;

  gl_float_t *points_data;
  size_t num_points;
  gl_float_t point_size;
} gl_points3d_t;

gl_points3d_t *gl_points3d_malloc(gl_float_t *points_data,
                                  const size_t num_points,
                                  const gl_float_t point_size);
void gl_points3d_free(gl_points3d_t *points);
void gl_points3d_update(gl_points3d_t *points,
                        gl_float_t *points_data,
                        size_t num_points,
                        const gl_float_t point_size);
void draw_points3d(gl_points3d_t *points);

// LINE 3D ///////////////////////////////////////////////////////////////////

typedef struct gl_line3d_t {
  gl_uint_t program_id;
  gl_uint_t VAO;
  gl_uint_t VBO;
  size_t num_points;
  gl_color_t color;
  gl_float_t alpha;
  gl_float_t lw;
} gl_line3d_t;

void gl_line3d_setup(gl_line3d_t *line3d,
                     const gl_color_t color,
                     const gl_float_t lw);
gl_line3d_t *gl_line3d_malloc(const gl_color_t color, const gl_float_t lw);
void gl_line3d_update(gl_line3d_t *line3d,
                      const size_t offset,
                      const gl_float_t *data,
                      const size_t num_points);
void gl_line3d_free(gl_line3d_t *line);
void draw_line3d(gl_line3d_t *line);

// CUBE 3D ///////////////////////////////////////////////////////////////////

typedef struct gl_cube3d_t {
  gl_uint_t program_id;
  gl_uint_t VAO;
  gl_uint_t VBO;
} gl_cube3d_t;

gl_cube3d_t *gl_cube3d_malloc(void);
void gl_cube3d_free(gl_cube3d_t *cube);
void draw_cube(gl_cube3d_t *cube,
               const gl_float_t T[4 * 4],
               const gl_float_t size,
               const gl_color_t color);

// PLANE /////////////////////////////////////////////////////////////////////

// typedef struct gl_plane_t {
//   gl_float_t normal[3];
//   gl_float_t p[3];
//   gl_float_t d;
//
//   gl_uint_t VAO;
//   gl_uint_t VBO;
//
//   gl_float_t size;
//   gl_color_t color;
//   gl_float_t lw;
// } gl_plane_t;
//
// void gl_plane_setup(plane_t *plane,
//                  const real_t normal[3],
//                  const real_t p[3],
//                  const real_t d);
// void gl_plane_vector(const plane_t *plane, real_t v[4]);
// void gl_plane_set_transform(plane_t *plane, const real_t T[4 * 4]);
// void gl_plane_get_transform(const plane_t *plane,
//                          const real_t world_up[3],
//                          real_t T[4 * 4]);
// real_t gl_plane_point_dist(const plane_t *plane, const real_t p[3]);

// FRUSTUM ///////////////////////////////////////////////////////////////////

typedef struct gl_frustum_t {
  gl_uint_t program_id;
  gl_float_t hfov;
  gl_float_t aspect;
  gl_float_t znear;
  gl_float_t zfar;

  // gl_plane_t near;
  // gl_plane_t far;
  // gl_plane_t left;
  // gl_plane_t right;
  // gl_plane_t top;
  // gl_plane_t bottom;

  gl_uint_t VAO;
  gl_uint_t VBO;

  gl_float_t T[4 * 4];
  gl_float_t size;
  gl_color_t color;
  gl_float_t lw;
} gl_frustum_t;

gl_frustum_t *gl_frustum_malloc(const gl_float_t hfov,
                                const gl_float_t aspect,
                                const gl_float_t znear,
                                const gl_float_t zfar,
                                const gl_float_t T[4 * 4],
                                const gl_float_t size,
                                const gl_color_t color,
                                const gl_float_t lw);
void gl_frustum_free(gl_frustum_t *frustum);
void draw_frustum(gl_frustum_t *frustum);

// AXES 3D ///////////////////////////////////////////////////////////////////

typedef struct gl_axes3d_t {
  gl_line3d_t x_axis;
  gl_line3d_t y_axis;
  gl_line3d_t z_axis;

  gl_float_t T[4 * 4];
  gl_float_t size;
  gl_float_t lw;
} gl_axes3d_t;

gl_axes3d_t *gl_axes3d_malloc(const gl_float_t T[4 * 4],
                              const gl_float_t size,
                              const gl_float_t lw);
void gl_axes3d_free(gl_axes3d_t *axes);
void draw_axes3d(gl_axes3d_t *axes);

// GRID 3D ///////////////////////////////////////////////////////////////////

typedef struct gl_grid3d_t {
  int num_rows;
  int num_cols;
  gl_float_t grid_size;
  gl_color_t color;
  gl_float_t lw;
  gl_line3d_t *row_lines;
  gl_line3d_t *col_lines;
} gl_grid3d_t;

gl_grid3d_t *gl_grid3d_malloc(const gl_int_t num_rows,
                              const gl_int_t num_cols,
                              const gl_float_t grid_size,
                              const gl_color_t color,
                              const gl_float_t lw);
void gl_grid3d_free(gl_grid3d_t *grid);
void draw_grid3d(gl_grid3d_t *grid);

// IMAGE /////////////////////////////////////////////////////////////////////

typedef struct gl_image_t {
  gl_uint_t program_id;
  gl_uint_t VAO;
  gl_uint_t VBO;
  gl_uint_t EBO;
  gl_uint_t texture_id;

  gl_int_t x;
  gl_int_t y;

  const uint8_t *data;
  gl_int_t width;
  gl_int_t height;
  gl_int_t channels;
} gl_image_t;

gl_image_t *gl_image_malloc(const int x,
                            const int y,
                            const uint8_t *data,
                            const int w,
                            const int h,
                            const int c);
void gl_image_free(gl_image_t *image);
void draw_image(gl_image_t *image);

// TEXT //////////////////////////////////////////////////////////////////////

typedef struct gl_text_t {
  gl_uint_t program_id;
  gl_uint_t VAO;
  gl_uint_t VBO;
  gl_uint_t EBO;
  gl_char_t chars[128];
} gl_text_t;

gl_text_t *gl_text_malloc(const int text_size);
void gl_text_free(gl_text_t *text);
void text_width_height(gl_text_t *text,
                       const char *s,
                       gl_float_t *w,
                       gl_float_t *h);
void draw_text(gl_text_t *text,
               const char *s,
               const float x,
               const float y,
               const gl_color_t c);

// MESH //////////////////////////////////////////////////////////////////////

typedef struct gl_vertex_t {
  float position[3];
  float normal[3];
  float tex_coords[2];
  float tangent[3];
  float bitangent[3];
} gl_vertex_t;

typedef struct gl_texture_t {
  unsigned int id;
  char type[100];
  char path[100];
} gl_texture_t;

typedef struct gl_mesh_t {
  gl_vertex_t *vertices;
  unsigned int *indices;
  gl_texture_t *textures;

  int num_vertices;
  int num_indices;
  int num_textures;

  gl_uint_t VAO;
  gl_uint_t VBO;
  gl_uint_t EBO;
} gl_mesh_t;

void gl_mesh_setup(gl_mesh_t *mesh,
                   gl_vertex_t *vertices,
                   const int num_vertices,
                   unsigned int *indices,
                   const int num_indices,
                   gl_texture_t *textures,
                   const int num_textures);
void gl_mesh_draw(const gl_mesh_t *mesh, const gl_uint_t shader);

// MODEL /////////////////////////////////////////////////////////////////////

typedef struct gl_model_t {
  char model_dir[100];

  gl_float_t T[4 * 4];
  gl_uint_t program_id;

  gl_mesh_t *meshes;
  int num_meshes;

  int enable_gamma_correction;
} gl_model_t;

gl_model_t *gl_model_load(const char *model_path);
void gl_model_free(gl_model_t *model);
void gl_model_draw(const gl_model_t *model, const gl_camera_t *camera);
