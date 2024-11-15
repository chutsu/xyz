#ifndef PROTO_H
#define PROTO_H

#ifdef __cplusplus
extern "C" {
#endif

// PROTO SETTINGS
#define PRECISION 2
#define MAX_LINE_LENGTH 9046

#define USE_CBLAS
#define USE_LAPACK
#define USE_SUITESPARSE
#define USE_CERES
#define USE_STB
// #define USE_GUI
#define USE_APRILGRID
// #define USE_TIS
// #define USE_SBGC

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <libgen.h>
#include <assert.h>
#include <sys/time.h>

#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

#ifdef USE_SUITESPARSE
#include <suitesparse/cholmod.h>
#endif

#ifdef USE_CERES
#include "ceres_bridge.h"
#endif

#ifdef USE_STB
#include <stb_image.h>
#endif

#ifdef USE_GUI
#include "gui.h"
#endif

#ifdef USE_APRILGRID
#include "aprilgrid.h"
#endif

#ifdef USE_TIS
#include "tis.h"
#endif

#ifdef USE_SBGC
#include "sbgc.h"
#endif

/******************************************************************************
 * LOGGING / MACROS
 ******************************************************************************/

#ifndef status_t
#define status_t __attribute__((warn_unused_result)) int
#endif

/** Terminal ANSI colors */
#define KRED "\x1B[1;31m"
#define KGRN "\x1B[1;32m"
#define KYEL "\x1B[1;33m"
#define KBLU "\x1B[1;34m"
#define KMAG "\x1B[1;35m"
#define KCYN "\x1B[1;36m"
#define KWHT "\x1B[1;37m"
#define KNRM "\x1B[1;0m"

/** Macro function that returns the caller's filename */
#define __FILENAME__                                                           \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

/** Macro that adds the ability to switch between C / C++ style mallocs */
#ifdef __cplusplus

#ifndef MALLOC
#define MALLOC(TYPE, N) (TYPE *) malloc(sizeof(TYPE) * (N));
#endif

#ifndef REALLOC
#define REALLOC(PTR, TYPE, N) (TYPE *) realloc(PTR, sizeof(TYPE) * (N));
#endif

#ifndef CALLOC
#define CALLOC(TYPE, N) (TYPE *) calloc((N), sizeof(TYPE));
#endif

#else

#ifndef MALLOC
#define MALLOC(TYPE, N) malloc(sizeof(TYPE) * (N));
#endif

#ifndef REALLOC
#define REALLOC(PTR, TYPE, N) realloc(PTR, sizeof(TYPE) * (N));
#endif

#ifndef CALLOC
#define CALLOC(TYPE, N) calloc((N), sizeof(TYPE));
#endif

#endif

/**
 * Free macro
 */
#ifndef FREE
#define FREE(X) free(X);
#endif

/**
 * Free memory
 */
#ifndef FREE_MEM
#define FREE_MEM(TARGET, FREE_FUNC)                                            \
  if (TARGET) {                                                                \
    FREE_FUNC((void *) TARGET);                                                \
  }
#endif

/**
 * Assert if condition is true
 */
#ifndef ASSERT_IF
#define ASSERT_IF(COND, ASSERT_COND)                                           \
  if (COND) {                                                                  \
    assert(ASSERT_COND);                                                       \
  }
#endif

/**
 * Debug
 * @param[in] M Message
 * @param[in] ... Varadic arguments
 */
#ifndef DEBUG
#define DEBUG(...)                                                             \
  do {                                                                         \
    fprintf(stderr, "[DEBUG] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__);   \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0);
#endif

/**
 * Log info
 * @param[in] M Message
 * @param[in] ... Varadic arguments
 */
#ifndef LOG_INFO
#define LOG_INFO(...)                                                          \
  do {                                                                         \
    fprintf(stderr, "[INFO] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__);    \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0)
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

/**
 * Log warn
 * @param[in] M Message
 * @param[in] ... Varadic arguments
 */
#ifndef LOG_WARN
#define LOG_WARN(...)                                                          \
  do {                                                                         \
    fprintf(stderr, "[WARN] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__);    \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0)
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
 * Check if condition is satisfied.
 *
 * If the condition is not satisfied a message M will be logged and a goto
 * error is called.
 *
 * @param[in] A Condition to be checked
 */
#ifndef CHECK
#define CHECK(A)                                                               \
  if (!(A)) {                                                                  \
    LOG_ERROR(#A " Failed!\n");                                                \
    goto error;                                                                \
  }
#endif

/**
 * Array copy
 */
#define ARRAY_COPY(SRC, N, DST)                                                \
  for (int i = 0; i < N; i++) {                                                \
    DST[i] = SRC[i];                                                           \
  }

/**
 * Median value in buffer
 */
#define MEDIAN_VALUE(DATA_TYPE, DATA_CMP, BUF, BUF_SIZE, MEDIAN_VAR)           \
  {                                                                            \
    DATA_TYPE VALUES[BUF_SIZE] = {0};                                          \
    for (size_t i = 0; i < BUF_SIZE; i++) {                                    \
      VALUES[i] = BUF[i];                                                      \
    }                                                                          \
                                                                               \
    qsort(VALUES, BUF_SIZE, sizeof(DATA_TYPE), DATA_CMP);                      \
    if ((BUF_SIZE % 2) == 0) {                                                 \
      const size_t bwd_idx = (size_t) (BUF_SIZE - 1) / 2.0;                    \
      const size_t fwd_idx = (size_t) (BUF_SIZE + 1) / 2.0;                    \
      MEDIAN_VAR = (VALUES[bwd_idx] + VALUES[fwd_idx]) / 2.0;                  \
    } else {                                                                   \
      const size_t mid_idx = (BUF_SIZE - 1) / 2;                               \
      MEDIAN_VAR = VALUES[mid_idx];                                            \
    }                                                                          \
  }

/**
 * Mean value in buffer
 */
#define MEAN_VALUE(DATA_TYPE, BUF, BUF_SIZE, MEAN_VAR)                         \
  {                                                                            \
    DATA_TYPE VALUE = 0;                                                       \
    for (size_t i = 0; i < BUF_SIZE; i++) {                                    \
      VALUE += BUF[i];                                                         \
    }                                                                          \
    MEAN_VAR = VALUE / (real_t) BUF_SIZE;                                      \
  }

/******************************************************************************
 * FILESYSTEM
 ******************************************************************************/

void path_file_name(const char *path, char *fname);
void path_file_ext(const char *path, char *fext);
void path_dir_name(const char *path, char *dir_name);
char *path_join(const char *x, const char *y);
char **list_files(const char *path, int *num_files);
void list_files_free(char **data, const int n);
char *file_read(const char *fp);
void skip_line(FILE *fp);
status_t file_exists(const char *fp);
status_t file_rows(const char *fp);
status_t file_copy(const char *src, const char *dest);

/*******************************************************************************
 * DATA
 ******************************************************************************/

#ifndef REAL_TYPE
#define REAL_TYPE
#if PRECISION == 1
typedef float real_t;
#elif PRECISION == 2
typedef double real_t;
#else
#error "Floating Point Precision not defined!"
#endif
#endif

size_t string_copy(char *dst, const char *src);
void string_subcopy(char *dst, const char *src, const int s, const int n);
void string_cat(char *dst, const char *src);
char *string_malloc(const char *s);
char *string_strip(char *s);
char *string_strip_char(char *s, const char c);
char **string_split(char *s, const char d, size_t *n);

int **load_iarrays(const char *csv_path, int *num_arrays);
real_t **load_darrays(const char *csv_path, int *num_arrays);

int *int_malloc(const int val);
float *float_malloc(const float val);
double *double_malloc(const double val);
real_t *vector_malloc(const real_t *vec, const real_t N);

int dsv_rows(const char *fp);
int dsv_cols(const char *fp, const char delim);
char **dsv_fields(const char *fp, const char delim, int *num_fields);
real_t **
dsv_data(const char *fp, const char delim, int *num_rows, int *num_cols);
void dsv_free(real_t **data, const int num_rows);

real_t **csv_data(const char *fp, int *num_rows, int *num_cols);
void csv_free(real_t **data, const int num_rows);

/*******************************************************************************
 * DATA-STRUCTURES
 ******************************************************************************/

////////////
// DARRAY //
////////////

#ifndef DEFAULT_EXPAND_RATE
#define DEFAULT_EXPAND_RATE 300
#endif

typedef struct darray_t {
  int end;
  int max;
  size_t element_size;
  size_t expand_rate;
  void **contents;
} darray_t;

darray_t *darray_new(size_t element_size, size_t initial_max);
void darray_destroy(darray_t *array);
void darray_clear(darray_t *array);
void darray_clear_destroy(darray_t *array);
int darray_push(darray_t *array, void *el);
void *darray_pop(darray_t *array);
int darray_contains(darray_t *array,
                    void *el,
                    int (*cmp)(const void *, const void *));
darray_t *darray_copy(darray_t *array);
void *darray_new_element(darray_t *array);
void *darray_first(darray_t *array);
void *darray_last(darray_t *array);
void darray_set(darray_t *array, int i, void *el);
void *darray_get(darray_t *array, int i);
void *darray_update(darray_t *array, int i, void *el);
void *darray_remove(darray_t *array, int i);
int darray_expand(darray_t *array);
int darray_contract(darray_t *array);

//////////
// LIST //
//////////

typedef struct list_node_t list_node_t;
struct list_node_t {
  list_node_t *next;
  list_node_t *prev;
  void *value;
};

typedef struct list_t {
  int length;
  list_node_t *first;
  list_node_t *last;
} list_t;

list_t *list_malloc(void);
void list_free(list_t *list);
void list_clear(list_t *list);
void list_clear_free(list_t *list);
void list_push(list_t *list, void *value);
void *list_pop(list_t *list);
void *list_pop_front(list_t *list);
void *list_shift(list_t *list);
void list_unshift(list_t *list, void *value);
void *list_remove(list_t *list,
                  void *target,
                  int (*cmp)(const void *, const void *));
int list_remove_destroy(list_t *list,
                        void *value,
                        int (*cmp)(const void *, const void *),
                        void (*free_func)(void *));

///////////
// STACK //
///////////

typedef struct mstack_node_t mstack_node_t;
struct mstack_node_t {
  void *value;
  mstack_node_t *next;
  mstack_node_t *prev;
};

typedef struct mstack_t {
  int size;
  mstack_node_t *root;
  mstack_node_t *end;
} mstack_t;

mstack_t *stack_new(void);
void mstack_destroy_traverse(mstack_node_t *n, void (*free_func)(void *));
void mstack_clear_destroy(mstack_t *s, void (*free_func)(void *));
void mstack_destroy(mstack_t *s);
int mstack_push(mstack_t *s, void *value);
void *mstack_pop(mstack_t *s);

///////////
// QUEUE //
///////////

typedef struct queue_t {
  int count;
  list_t *queue;
} queue_t;

queue_t *queue_malloc(void);
void queue_free(queue_t *q);
int queue_enqueue(queue_t *q, void *data);
void *queue_dequeue(queue_t *q);
int queue_count(queue_t *q);
int queue_empty(queue_t *q);
int queue_full(queue_t *q);
void *queue_first(queue_t *q);
void *queue_last(queue_t *q);

/////////////
// HASHMAP //
/////////////

#ifndef DEFEAULT_NUMBER_OF_BUCKETS
#define DEFAULT_NUMBER_OF_BUCKETS 10000
#endif

typedef struct hashmap_node_t {
  uint32_t hash;
  void *key;
  void *value;
} hashmap_node_t;

typedef struct hashmap_t {
  darray_t *buckets;
  int (*cmp)(void *, void *);
  uint32_t (*hash)(void *);

  int copy_kv;
  void *(*k_copy)(void *);
  void *(*v_copy)(void *);
  void (*k_free)(void *);
  void (*v_free)(void *);
} hashmap_t;

hashmap_t *hashmap_new(void);
void hashmap_clear_destroy(hashmap_t *map);
void hashmap_destroy(hashmap_t *map);
int hashmap_set(hashmap_t *map, void *key, void *data);
void *hashmap_get(hashmap_t *map, void *key);
int hashmap_traverse(hashmap_t *map,
                     int (*hashmap_traverse_cb)(hashmap_node_t *node));
void *hashmap_delete(hashmap_t *map, void *key);

/*******************************************************************************
 * TIME
 ******************************************************************************/

/** Timestamp Type */
#ifndef timestamp_t
typedef int64_t timestamp_t;
#endif

/** Tic toc macros */
#define TIC(X) struct timespec X = tic()
#define TOC(X) toc(&X)
#define MTOC(X) mtoc(&X)
#define PRINT_TOC(PREFIX, X) printf("[%s]: %.4fs\n", PREFIX, toc(&X))
#define PRINT_MTOC(PREFIX, X) printf("[%s]: %.4fms\n", PREFIX, mtoc(&X))

struct timespec tic(void);
float toc(struct timespec *tic);
float mtoc(struct timespec *tic);
timestamp_t time_now(void);

timestamp_t str2ts(const char *ts_str);
real_t ts2sec(const timestamp_t ts);
timestamp_t sec2ts(const real_t time_s);

/*******************************************************************************
 * NETWORK
 ******************************************************************************/

/**
 * TCP server
 */
typedef struct tcp_server_t {
  int port;
  int sockfd;
  int conn;
  void *(*conn_handler)(void *);
} tcp_server_t;

/**
 * TCP client
 */
typedef struct tcp_client_t {
  char server_ip[1024];
  int server_port;
  int sockfd;
  int (*loop_cb)(struct tcp_client_t *);
} tcp_client_t;

status_t ip_port_info(const int sockfd, char *ip, int *port);

status_t tcp_server_setup(tcp_server_t *server, const int port);
status_t tcp_server_loop(tcp_server_t *server);

status_t tcp_client_setup(tcp_client_t *client,
                        const char *server_ip,
                        const int server_port);
status_t tcp_client_loop(tcp_client_t *client);

/*******************************************************************************
 * MATHS
 ******************************************************************************/

/** Mathematical Pi constant (i.e. 3.1415..) */
#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

/** Real number comparison tolerance */
#ifndef CMP_TOL
#define CMP_TOL 1e-6
#endif

/** Min of two numbers, X or Y. */
#define MIN(x, y) ((x) < (y) ? (x) : (y))

/** Max of two numbers, X or Y. */
#define MAX(x, y) ((x) > (y) ? (x) : (y))

/** Based on sign of b, return +ve or -ve a. */
#define SIGN2(a, b) ((b) > 0.0 ? fabs(a) : -fabs(a))

float randf(float a, float b);
void randvec(const real_t a, const real_t b, const size_t n, real_t *v);
real_t deg2rad(const real_t d);
real_t rad2deg(const real_t r);
real_t wrap_180(const real_t d);
real_t wrap_360(const real_t d);
real_t wrap_pi(const real_t r);
real_t wrap_2pi(const real_t r);
int intcmp(const int x, const int y);
int intcmp2(const void *x, const void *y);
int fltcmp(const real_t x, const real_t y);
int fltcmp2(const void *x, const void *y);
int strcmp2(const void *x, const void *y);
int flteqs(const real_t x, const real_t y);
int streqs(const char *x, const char *y);
void cumsum(const real_t *x, const size_t n, real_t *s);
void logspace(const real_t a, const real_t b, const size_t n, real_t *x);
real_t pythag(const real_t a, const real_t b);
real_t clip_value(const real_t x, const real_t vmin, const real_t vmax);
void clip(real_t *x, const size_t n, const real_t vmin, const real_t vmax);
real_t lerp(const real_t a, const real_t b, const real_t t);
void lerp3(const real_t a[3], const real_t b[3], const real_t t, real_t x[3]);
real_t sinc(const real_t x);
real_t mean(const real_t *x, const size_t length);
real_t median(const real_t *x, const size_t length);
real_t var(const real_t *x, const size_t length);
real_t stddev(const real_t *x, const size_t length);

/*********************************************************************************
 * LINEAR ALGEBRA
 ******************************************************************************/

void print_matrix(const char *prefix,
                  const real_t *A,
                  const size_t m,
                  const size_t n);
void print_vector(const char *prefix, const real_t *v, const size_t n);
void vec2str(const real_t *v, const int n, char *s);
void vec2csv(const real_t *v, const int n, char *s);

void eye(real_t *A, const size_t m, const size_t n);
void ones(real_t *A, const size_t m, const size_t n);
void zeros(real_t *A, const size_t m, const size_t n);
void hat(const real_t x[3], real_t A[3 * 3]);
void vee(const real_t A[3 * 3], real_t x[3]);
void fwdsubs(const real_t *L, const real_t *b, real_t *y, const size_t n);
void bwdsubs(const real_t *U, const real_t *y, real_t *x, const size_t n);
void enforce_spd(real_t *A, const int m, const int n);

real_t *mat_malloc(const size_t m, const size_t n);
int mat_cmp(const real_t *A, const real_t *B, const size_t m, const size_t n);
int mat_equals(const real_t *A,
               const real_t *B,
               const size_t m,
               const size_t n,
               const real_t tol);
int mat_save(const char *save_path, const real_t *A, const int m, const int n);
real_t *mat_load(const char *save_path, int *num_rows, int *num_cols);
void mat_set(real_t *A,
             const size_t stride,
             const size_t i,
             const size_t j,
             const real_t val);
real_t
mat_val(const real_t *A, const size_t stride, const size_t i, const size_t j);
void mat_copy(const real_t *src, const int m, const int n, real_t *dest);
void mat_row_set(real_t *A,
                 const size_t stride,
                 const int row_idx,
                 const real_t *x);
void mat_col_set(real_t *A,
                 const size_t stride,
                 const int num_rows,
                 const int col_idx,
                 const real_t *x);
void mat_col_get(const real_t *A,
                 const int m,
                 const int n,
                 const int col_idx,
                 real_t *x);
void mat_block_get(const real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   real_t *block);
void mat_block_set(real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   const real_t *block);
void mat_block_add(real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   const real_t *block);
void mat_block_sub(real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   const real_t *block);
void mat_diag_get(const real_t *A, const int m, const int n, real_t *d);
void mat_diag_set(real_t *A, const int m, const int n, const real_t *d);
void mat_triu(const real_t *A, const size_t n, real_t *U);
void mat_tril(const real_t *A, const size_t n, real_t *L);
real_t mat_trace(const real_t *A, const size_t m, const size_t n);
void mat_transpose(const real_t *A, size_t m, size_t n, real_t *A_t);
void mat_add(const real_t *A, const real_t *B, real_t *C, size_t m, size_t n);
void mat_sub(const real_t *A, const real_t *B, real_t *C, size_t m, size_t n);
void mat_scale(real_t *A, const size_t m, const size_t n, const real_t scale);

void mat3_copy(const real_t src[3 * 3], real_t dst[3 * 3]);
void mat3_add(const real_t A[3 * 3], const real_t B[3 * 3], real_t C[3 * 3]);
void mat3_sub(const real_t A[3 * 3], const real_t B[3 * 3], real_t C[3 * 3]);

real_t *vec_malloc(const real_t *x, const size_t n);
void vec_copy(const real_t *src, const size_t n, real_t *dest);
int vec_equals(const real_t *x, const real_t *y, const size_t n);
real_t vec_min(const real_t *x, const size_t n);
real_t vec_max(const real_t *x, const size_t n);
void vec_range(const real_t *x,
               const size_t n,
               real_t *vmin,
               real_t *vmax,
               real_t *r);
real_t *vec_load(const char *save_path, int *num_rows, int *num_cols);
void vec_add(const real_t *x, const real_t *y, real_t *z, size_t n);
void vec_sub(const real_t *x, const real_t *y, real_t *z, size_t n);
void vec_scale(real_t *x, const size_t n, const real_t scale);
real_t vec_norm(const real_t *x, const size_t n);
void vec_normalize(real_t *x, const size_t n);

void vec3_copy(const real_t src[3], real_t dst[3]);
void vec3_add(const real_t a[3], const real_t b[3], real_t c[3]);
void vec3_sub(const real_t a[3], const real_t b[3], real_t c[3]);
void vec3_cross(const real_t a[3], const real_t b[3], real_t c[3]);
real_t vec3_norm(const real_t x[3]);
void vec3_normalize(real_t x[3]);

void dot(const real_t *A,
         const size_t A_m,
         const size_t A_n,
         const real_t *B,
         const size_t B_m,
         const size_t B_n,
         real_t *C);
void dot3(const real_t *A,
          const size_t A_m,
          const size_t A_n,
          const real_t *B,
          const size_t B_m,
          const size_t B_n,
          const real_t *C,
          const size_t C_m,
          const size_t C_n,
          real_t *D);
void dot_XtAX(const real_t *X,
              const size_t X_m,
              const size_t X_n,
              const real_t *A,
              const size_t A_m,
              const size_t A_n,
              real_t *Y);
void dot_XAXt(const real_t *X,
              const size_t X_m,
              const size_t X_n,
              const real_t *A,
              const size_t A_m,
              const size_t A_n,
              real_t *Y);

void bdiag_inv(const real_t *A, const int m, const int bs, real_t *A_inv);
void bdiag_inv_sub(const real_t *A,
                   const int stride,
                   const int m,
                   const int bs,
                   real_t *A_inv);
void bdiag_dot(const real_t *A,
               const int m,
               const int n,
               const int bs,
               const real_t *x,
               real_t *b);

#define MAT_TRANSPOSE(A, M, N, B)                                              \
  real_t B[N * M] = {0};                                                       \
  mat_transpose(A, M, N, B);

#define DOT(A, AM, AN, B, BM, BN, C)                                           \
  real_t C[AM * BN] = {0};                                                     \
  dot(A, AM, AN, B, BM, BN, C);

#define DOT3(A, AM, AN, B, BM, BN, C, CM, CN, D)                               \
  real_t D[AM * CN] = {0};                                                     \
  dot3(A, AM, AN, B, BM, BN, C, CM, CN, D);

#define DOT_XTAX(X, XM, XN, A, AM, AN, Y)                                      \
  real_t Y[XN * XN] = {0};                                                     \
  dot_XtAX(X, XM, XN, A, AM, AN, Y);

#define DOT_XAXt(X, XM, XN, A, AM, AN, Y)                                      \
  real_t Y[XM * XM] = {0};                                                     \
  dot_XAXt(X, XM, XN, A, AM, AN, Y);

#define HAT(X, X_HAT)                                                          \
  real_t X_HAT[3 * 3] = {0};                                                   \
  hat(X, X_HAT);

#define VEE(A, X)                                                              \
  real_t X[3] = {0};                                                           \
  vee(A, X);

int check_inv(const real_t *A, const real_t *A_inv, const int m);
real_t check_Axb(const real_t *A,
                 const real_t *x,
                 const real_t *b,
                 const int m,
                 const int n);
int check_jacobian(const char *jac_name,
                   const real_t *fdiff,
                   const real_t *jac,
                   const size_t m,
                   const size_t n,
                   const real_t tol,
                   const int verbose);

#define CHECK_JACOBIAN(JAC_IDX, FACTOR, FACTOR_EVAL, STEP_SIZE, TOL, VERBOSE)  \
  {                                                                            \
    const int r_size = FACTOR.r_size;                                          \
    const int p_size = param_global_size(FACTOR.param_types[JAC_IDX]);         \
    const int J_cols = param_local_size(FACTOR.param_types[JAC_IDX]);          \
                                                                               \
    real_t *param_copy = MALLOC(real_t, p_size);                               \
    real_t *r_fwd = MALLOC(real_t, r_size);                                    \
    real_t *r_bwd = MALLOC(real_t, r_size);                                    \
    real_t *r_diff = MALLOC(real_t, r_size);                                   \
    real_t *J_fdiff = MALLOC(real_t, r_size * J_cols);                         \
    real_t *J = MALLOC(real_t, r_size * J_cols);                               \
                                                                               \
    /* Evaluate factor to get analytical Jacobian */                           \
    FACTOR_EVAL((void *) &FACTOR);                                             \
    mat_copy(factor.jacs[JAC_IDX], r_size, J_cols, J);                         \
                                                                               \
    /* Calculate numerical differerntiated Jacobian */                         \
    for (int i = 0; i < J_cols; i++) {                                         \
      vec_copy(FACTOR.params[JAC_IDX], p_size, param_copy);                    \
                                                                               \
      FACTOR.params[JAC_IDX][i] += 0.5 * STEP_SIZE;                            \
      FACTOR_EVAL((void *) &FACTOR);                                           \
      vec_copy(FACTOR.r, r_size, r_fwd);                                       \
      vec_copy(param_copy, p_size, FACTOR.params[JAC_IDX]);                    \
                                                                               \
      FACTOR.params[JAC_IDX][i] -= 0.5 * STEP_SIZE;                            \
      FACTOR_EVAL((void *) &FACTOR);                                           \
      vec_copy(FACTOR.r, r_size, r_bwd);                                       \
      vec_copy(param_copy, p_size, FACTOR.params[JAC_IDX]);                    \
                                                                               \
      vec_sub(r_fwd, r_bwd, r_diff, r_size);                                   \
      vec_scale(r_diff, r_size, 1.0 / STEP_SIZE);                              \
      mat_col_set(J_fdiff, J_cols, r_size, i, r_diff);                         \
    }                                                                          \
                                                                               \
    char s[100] = {0};                                                         \
    sprintf(s, "J%d", JAC_IDX);                                                \
    int retval = check_jacobian(s, J_fdiff, J, r_size, J_cols, TOL, VERBOSE);  \
                                                                               \
    free(param_copy);                                                          \
    free(r_fwd);                                                               \
    free(r_bwd);                                                               \
    free(r_diff);                                                              \
    free(J_fdiff);                                                             \
    free(J);                                                                   \
                                                                               \
    MU_ASSERT(retval == 0);                                                    \
  }

#define CHECK_POSE_JACOBIAN(JAC_IDX,                                           \
                            FACTOR,                                            \
                            FACTOR_EVAL,                                       \
                            STEP_SIZE,                                         \
                            TOL,                                               \
                            VERBOSE)                                           \
  {                                                                            \
    const int r_size = FACTOR.r_size;                                          \
    const int J_cols = param_local_size(FACTOR.param_types[JAC_IDX]);          \
                                                                               \
    real_t *r = MALLOC(real_t, r_size);                                        \
    real_t *r_fwd = MALLOC(real_t, r_size);                                    \
    real_t *r_diff = MALLOC(real_t, r_size);                                   \
    real_t *J_fdiff = MALLOC(real_t, r_size * J_cols);                         \
    real_t *J = MALLOC(real_t, r_size * J_cols);                               \
                                                                               \
    /* Eval */                                                                 \
    FACTOR_EVAL(&FACTOR);                                                      \
    vec_copy(FACTOR.r, r_size, r);                                             \
    mat_copy(FACTOR.jacs[JAC_IDX], r_size, J_cols, J);                         \
                                                                               \
    /* Check pose position jacobian */                                         \
    for (int i = 0; i < 3; i++) {                                              \
      FACTOR.params[JAC_IDX][i] += STEP_SIZE;                                  \
      FACTOR_EVAL((void *) &FACTOR);                                           \
      vec_copy(FACTOR.r, r_size, r_fwd);                                       \
      FACTOR.params[JAC_IDX][i] -= STEP_SIZE;                                  \
                                                                               \
      vec_sub(r_fwd, r, r_diff, r_size);                                       \
      vec_scale(r_diff, r_size, 1.0 / STEP_SIZE);                              \
      mat_col_set(J_fdiff, J_cols, r_size, i, r_diff);                         \
    }                                                                          \
    for (int i = 0; i < 3; i++) {                                              \
      quat_perturb(FACTOR.params[JAC_IDX] + 3, i, STEP_SIZE);                  \
      FACTOR_EVAL((void *) &FACTOR);                                           \
      vec_copy(FACTOR.r, r_size, r_fwd);                                       \
      quat_perturb(FACTOR.params[JAC_IDX] + 3, i, -STEP_SIZE);                 \
                                                                               \
      vec_sub(r_fwd, r, r_diff, r_size);                                       \
      vec_scale(r_diff, r_size, 1.0 / STEP_SIZE);                              \
      mat_col_set(J_fdiff, J_cols, r_size, i + 3, r_diff);                     \
    }                                                                          \
                                                                               \
    char s[100] = {0};                                                         \
    sprintf(s, "J%d", JAC_IDX);                                                \
    int retval = check_jacobian(s, J_fdiff, J, r_size, J_cols, TOL, VERBOSE);  \
                                                                               \
    free(r);                                                                   \
    free(r_fwd);                                                               \
    free(r_diff);                                                              \
    free(J_fdiff);                                                             \
    free(J);                                                                   \
                                                                               \
    MU_ASSERT(retval == 0);                                                    \
  }

#define CHECK_FACTOR_J(PARAM_IDX,                                              \
                       FACTOR,                                                 \
                       FACTOR_EVAL,                                            \
                       STEP_SIZE,                                              \
                       TOL,                                                    \
                       VERBOSE)                                                \
  {                                                                            \
    int param_type = FACTOR.param_types[PARAM_IDX];                            \
    switch (param_type) {                                                      \
      case POSE_PARAM:                                                         \
      case EXTRINSIC_PARAM:                                                    \
      case FIDUCIAL_PARAM:                                                     \
        CHECK_POSE_JACOBIAN(PARAM_IDX,                                         \
                            FACTOR,                                            \
                            FACTOR_EVAL,                                       \
                            STEP_SIZE,                                         \
                            TOL,                                               \
                            VERBOSE)                                           \
        break;                                                                 \
      default:                                                                 \
        CHECK_JACOBIAN(PARAM_IDX,                                              \
                       FACTOR,                                                 \
                       FACTOR_EVAL,                                            \
                       STEP_SIZE,                                              \
                       TOL,                                                    \
                       VERBOSE)                                                \
        break;                                                                 \
    }                                                                          \
  }

/////////
// SVD //
/////////

int svd(const real_t *A,
        const int m,
        const int n,
        real_t *U,
        real_t *s,
        real_t *V);
void pinv(const real_t *A, const int m, const int n, real_t *A_inv);
int svd_det(const real_t *A, const int m, const int n, real_t *det);
int svd_rank(const real_t *A, const int m, const int n, real_t tol);

//////////
// CHOL //
//////////

void chol(const real_t *A, const size_t n, real_t *L);
void chol_solve(const real_t *A, const real_t *b, real_t *x, const size_t n);

////////
// QR //
////////

void qr(real_t *A, const int m, const int n, real_t *R);

/////////
// EIG //
/////////

#define EIG_V_SIZE(Am, An) (Am * An)
#define EIG_W_SIZE(Am, An) (An)

int eig_sym(const real_t *A, const int m, const int n, real_t *V, real_t *w);
int eig_inv(real_t *A, const int m, const int n, const int c, real_t *A_inv);
int eig_rank(const real_t *A, const int m, const int n, const real_t tol);

/******************************************************************************
 * SUITE-SPARSE
 *****************************************************************************/

cholmod_sparse *cholmod_sparse_malloc(cholmod_common *c,
                                      const real_t *A,
                                      const int m,
                                      const int n,
                                      const int stype);
cholmod_dense *cholmod_dense_malloc(cholmod_common *c,
                                    const real_t *x,
                                    const int n);
void cholmod_dense_raw(const cholmod_dense *src, real_t *dst, const int n);
real_t suitesparse_chol_solve(cholmod_common *c,
                              const real_t *A,
                              const int A_m,
                              const int A_n,
                              const real_t *b,
                              const int b_m,
                              real_t *x);

/*******************************************************************************
 * TRANSFORMS
 ******************************************************************************/

#define TF(PARAMS, T)                                                          \
  real_t T[4 * 4] = {0};                                                       \
  tf(PARAMS, T);

#define TF_TRANS(T, TRANS) real_t TRANS[3] = {T[3], T[7], T[11]};

#define TF_ROT(T, ROT)                                                         \
  real_t ROT[3 * 3] = {0};                                                     \
  ROT[0] = T[0];                                                               \
  ROT[1] = T[1];                                                               \
  ROT[2] = T[2];                                                               \
  ROT[3] = T[4];                                                               \
  ROT[4] = T[5];                                                               \
  ROT[5] = T[6];                                                               \
  ROT[6] = T[8];                                                               \
  ROT[7] = T[9];                                                               \
  ROT[8] = T[10];

#define TF_QUAT(T, QUAT)                                                       \
  real_t QUAT[4] = {0};                                                        \
  tf_quat_get(T, QUAT);

#define TF_CR(C, R, T)                                                         \
  real_t T[4 * 4] = {0};                                                       \
  tf_cr(C, R, T);

#define TF_ER(E, R, T)                                                         \
  real_t T[4 * 4] = {0};                                                       \
  tf_er(E, R, T);

#define TF_VECTOR(T, V)                                                        \
  real_t V[7] = {0};                                                           \
  tf_vector(T, V);

#define TF_DECOMPOSE(T, ROT, TRANS)                                            \
  real_t ROT[3 * 3] = {0};                                                     \
  real_t TRANS[3] = {0};                                                       \
  tf_decompose(T, ROT, TRANS);

#define TF_QR(Q, R, T)                                                         \
  real_t T[4 * 4] = {0};                                                       \
  tf_qr(Q, R, T);

#define TF_INV(T, T_INV)                                                       \
  real_t T_INV[4 * 4] = {0};                                                   \
  tf_inv(T, T_INV);

#define TF_POINT(T, P_IN, P_OUT)                                               \
  real_t P_OUT[3] = {0};                                                       \
  tf_point(T, P_IN, P_OUT);

#define TF_CHAIN(T, N, ...)                                                    \
  real_t T[4 * 4] = {0};                                                       \
  tf_chain2(N, __VA_ARGS__, T);

#define EULER321(YPR, C)                                                       \
  real_t C[3 * 3] = {0};                                                       \
  euler321(YPR, C);

#define EULER2QUAT(YPR, Q)                                                     \
  real_t Q[4] = {0};                                                           \
  euler2quat(YPR, Q);

#define ROT2QUAT(C, Q)                                                         \
  real_t Q[4] = {0};                                                           \
  rot2quat(C, Q);

#define QUAT2ROT(Q, C)                                                         \
  real_t C[3 * 3] = {0};                                                       \
  quat2rot(Q, C);

// clang-format off
#define TF_IDENTITY(T)                                                         \
  real_t T[4 * 4] = {                                                          \
    1.0, 0.0, 0.0, 0.0,                                                        \
    0.0, 1.0, 0.0, 0.0,                                                        \
    0.0, 0.0, 1.0, 0.0,                                                        \
    0.0, 0.0, 0.0, 1.0                                                         \
  };
// clang-format on

#define POSE_ER(YPR, POS, POSE)                                                \
  real_t POSE[7] = {0};                                                        \
  POSE[0] = POS[0];                                                            \
  POSE[1] = POS[1];                                                            \
  POSE[2] = POS[2];                                                            \
  euler2quat(YPR, POSE + 3);

#define POSE2TF(POSE, TF)                                                      \
  real_t TF[4 * 4] = {0};                                                      \
  tf(POSE, TF);

void rotx(const real_t theta, real_t C[3 * 3]);
void roty(const real_t theta, real_t C[3 * 3]);
void rotz(const real_t theta, real_t C[3 * 3]);
void tf(const real_t params[7], real_t T[4 * 4]);
void tf_cr(const real_t C[3 * 3], const real_t r[3], real_t T[4 * 4]);
void tf_qr(const real_t q[4], const real_t r[3], real_t T[4 * 4]);
void tf_er(const real_t ypr[3], const real_t r[3], real_t T[4 * 4]);
void tf_vector(const real_t T[4 * 4], real_t params[7]);
void tf_decompose(const real_t T[4 * 4], real_t C[3 * 3], real_t r[3]);
void tf_rot_set(real_t T[4 * 4], const real_t C[3 * 3]);
void tf_rot_get(const real_t T[4 * 4], real_t C[3 * 3]);
void tf_quat_set(real_t T[4 * 4], const real_t q[4]);
void tf_quat_get(const real_t T[4 * 4], real_t q[4]);
void tf_euler_set(real_t T[4 * 4], const real_t ypr[3]);
void tf_euler_get(const real_t T[4 * 4], real_t ypr[3]);
void tf_trans_set(real_t T[4 * 4], const real_t r[3]);
void tf_trans_get(const real_t T[4 * 4], real_t r[3]);
void tf_inv(const real_t T[4 * 4], real_t T_inv[4 * 4]);
void tf_point(const real_t T[4 * 4], const real_t p[3], real_t retval[3]);
void tf_hpoint(const real_t T[4 * 4], const real_t p[4], real_t retval[4]);
void tf_perturb_rot(real_t T[4 * 4], const real_t step_size, const int i);
void tf_perturb_trans(real_t T[4 * 4], const real_t step_size, const int i);
void tf_chain(const real_t **tfs, const int num_tfs, real_t T_out[4 * 4]);
void tf_chain2(const int num_tfs, ...);
void tf_diff(const real_t Ti[4 * 4], const real_t Tj[4 * 4], real_t diff[6]);
void tf_diff2(const real_t Ti[4 * 4],
              const real_t Tj[4 * 4],
              real_t dr[3],
              real_t *dtheta);
void pose_get_trans(const real_t pose[7], real_t r[3]);
void pose_get_quat(const real_t pose[7], real_t q[4]);
void pose_get_rot(const real_t p[7], real_t C[3 * 3]);
void pose_diff(const real_t pose0[7], const real_t pose1[7], real_t diff[6]);
void pose_diff2(const real_t pose0[7],
                const real_t pose1[7],
                real_t dr[3],
                real_t *dangle);
void pose_update(real_t pose[7], const real_t dx[6]);
void pose_random_perturb(real_t pose[7],
                         const real_t dtrans,
                         const real_t drot);
void print_pose(const char *prefix, const real_t pose[7]);
void vecs2rot(const real_t acc[3], const real_t g[3], real_t *C);
void rvec2rot(const real_t *rvec, const real_t eps, real_t *R);
void euler321(const real_t ypr[3], real_t C[3 * 3]);
void euler2quat(const real_t ypr[3], real_t q[4]);
void rot2quat(const real_t C[3 * 3], real_t q[4]);
void rot2euler(const real_t C[3 * 3], real_t ypr[3]);
void quat2euler(const real_t q[4], real_t ypr[3]);
void quat2rot(const real_t q[4], real_t C[3 * 3]);
void print_quat(const char *prefix, const real_t q[4]);
real_t quat_norm(const real_t q[4]);
void quat_setup(real_t q[4]);
void quat_normalize(real_t q[4]);
void quat_normalize_copy(const real_t q[4], real_t q_normalized[4]);
void quat_inv(const real_t q[4], real_t q_inv[4]);
void quat_left(const real_t q[4], real_t left[4 * 4]);
void quat_left_xyz(const real_t q[4], real_t left_xyz[3 * 3]);
void quat_right(const real_t q[4], real_t right[4 * 4]);
void quat_lmul(const real_t p[4], const real_t q[4], real_t r[4]);
void quat_rmul(const real_t p[4], const real_t q[4], real_t r[4]);
void quat_mul(const real_t p[4], const real_t q[4], real_t r[4]);
void quat_delta(const real_t dalpha[3], real_t dq[4]);
void quat_update(real_t q[4], const real_t dalpha[3]);
void quat_update_dt(real_t q[4], const real_t w[3], const real_t dt);
void quat_perturb(real_t q[4], const int i, const real_t h);
void quat_transform(const real_t q[4], const real_t x[3], real_t y[3]);

/*******************************************************************************
 * LIE
 ******************************************************************************/

void lie_Exp(const real_t phi[3], real_t C[3 * 3]);
void lie_Log(const real_t C[3 * 3], real_t rvec[3]);
void box_plus(const real_t C[3 * 3],
              const real_t alpha[3],
              real_t C_new[3 * 3]);
void box_minus(const real_t Ca[3 * 3], const real_t Cb[3 * 3], real_t alpha[3]);

/*******************************************************************************
 * CV
 ******************************************************************************/

///////////
// IMAGE //
///////////

typedef struct image_t {
  int width;
  int height;
  int channels;
  uint8_t *data;
} image_t;

void image_setup(image_t *img,
                 const int width,
                 const int height,
                 uint8_t *data);
image_t *image_load(const char *file_path);
void image_print_properties(const image_t *img);
void image_free(image_t *img);

////////////
// RADTAN //
////////////

void radtan4_distort(const real_t params[4],
                     const real_t p_in[2],
                     real_t p_out[2]);
void radtan4_undistort(const real_t params[4],
                       const real_t p_in[2],
                       real_t p_out[2]);
void radtan4_point_jacobian(const real_t params[4],
                            const real_t p[2],
                            real_t J_point[2 * 2]);
void radtan4_params_jacobian(const real_t params[4],
                             const real_t p[2],
                             real_t J_param[2 * 4]);

//////////
// EQUI //
//////////

void equi4_distort(const real_t params[4],
                   const real_t p_in[2],
                   real_t p_out[2]);
void equi4_undistort(const real_t params[4],
                     const real_t p_in[2],
                     real_t p_out[2]);
void equi4_point_jacobian(const real_t params[4],
                          const real_t p[2],
                          real_t J_point[2 * 2]);
void equi4_params_jacobian(const real_t params[4],
                           const real_t p[2],
                           real_t J_param[2 * 4]);

/////////////
// PINHOLE //
/////////////

typedef void (*project_func_t)(const real_t *params,
                               const real_t p_C[3],
                               real_t z_out[2]);

typedef void (*back_project_func_t)(const real_t *params,
                                    const real_t z[2],
                                    real_t bearing[3]);

typedef void (*undistort_func_t)(const real_t *params,
                                 const real_t z_in[2],
                                 real_t z_out[2]);

real_t pinhole_focal(const int image_width, const real_t fov);
void pinhole_K(const real_t params[4], real_t K[3 * 3]);
void pinhole_projection_matrix(const real_t params[4],
                               const real_t T[4 * 4],
                               real_t P[3 * 4]);
void pinhole_project(const real_t params[4], const real_t p_C[3], real_t z[2]);
void pinhole_point_jacobian(const real_t params[4], real_t J_point[2 * 2]);
void pinhole_params_jacobian(const real_t params[4],
                             const real_t x[2],
                             real_t J[2 * 4]);

/////////////////////
// PINHOLE-RADTAN4 //
/////////////////////

void pinhole_radtan4_project(const real_t params[8],
                             const real_t p_C[3],
                             real_t z[2]);
void pinhole_radtan4_undistort(const real_t params[8],
                               const real_t z_in[2],
                               real_t z_out[2]);
void pinhole_radtan4_back_project(const real_t params[8],
                                  const real_t z[2],
                                  real_t ray[3]);
void pinhole_radtan4_project_jacobian(const real_t params[8],
                                      const real_t p_C[3],
                                      real_t J[2 * 3]);
void pinhole_radtan4_params_jacobian(const real_t params[8],
                                     const real_t p_C[3],
                                     real_t J[2 * 8]);

///////////////////
// PINHOLE-EQUI4 //
///////////////////

void pinhole_equi4_project(const real_t params[8],
                           const real_t p_C[3],
                           real_t z[2]);
void pinhole_equi4_undistort(const real_t params[8],
                             const real_t z_in[2],
                             real_t z_out[2]);
void pinhole_equi4_back_project(const real_t params[8],
                                const real_t z[2],
                                real_t ray[3]);
void pinhole_equi4_project_jacobian(const real_t params[8],
                                    const real_t p_C[3],
                                    real_t J[2 * 3]);
void pinhole_equi4_params_jacobian(const real_t params[8],
                                   const real_t p_C[3],
                                   real_t J[2 * 8]);

//////////////
// GEOMETRY //
//////////////

void linear_triangulation(const real_t P_i[3 * 4],
                          const real_t P_j[3 * 4],
                          const real_t z_i[2],
                          const real_t z_j[2],
                          real_t p[3]);

int homography_find(const real_t *pts_i,
                    const real_t *pts_j,
                    const int num_points,
                    real_t H[3 * 3]);

int homography_pose(const real_t *proj_params,
                    const real_t *img_pts,
                    const real_t *obj_pts,
                    const int N,
                    real_t T_CF[4 * 4]);

int p3p_kneip(const real_t features[3][3],
              const real_t points[3][3],
              real_t solutions[4][4 * 4]);

int solvepnp(const real_t proj_params[4],
             const real_t *img_pts,
             const real_t *obj_pts,
             const int N,
             real_t T_CO[4 * 4]);

/*******************************************************************************
 * CONTROL
 ******************************************************************************/

typedef struct pid_ctrl_t {
  real_t error_prev;
  real_t error_sum;

  real_t error_p;
  real_t error_i;
  real_t error_d;

  real_t k_p;
  real_t k_i;
  real_t k_d;
} pid_ctrl_t;

void pid_ctrl_setup(pid_ctrl_t *pid,
                    const real_t kp,
                    const real_t ki,
                    const real_t kd);
real_t pid_ctrl_update(pid_ctrl_t *pid,
                       const real_t setpoint,
                       const real_t input,
                       const real_t dt);
void pid_ctrl_reset(pid_ctrl_t *pid);

/*******************************************************************************
 * GIMBAL MODEL
 ******************************************************************************/

typedef struct gimbal_model_t {
  real_t x[6];
} gimbal_model_t;

typedef struct gimbal_ctrl_t {
  real_t dt;
  pid_ctrl_t roll;
  pid_ctrl_t pitch;
  pid_ctrl_t yaw;

  real_t setpoints[3];
  real_t outputs[3];
} gimbal_ctrl_t;

void gimbal_model_setup(gimbal_model_t *gimbal);
void gimbal_model_update(gimbal_model_t *gimbal,
                         const real_t u[3],
                         const real_t dt);

void gimbal_ctrl_setup(gimbal_ctrl_t *ctrl);
void gimbal_ctrl_update(gimbal_ctrl_t *ctrl,
                        const real_t setpoints[3],
                        const real_t actual[3],
                        const real_t dt,
                        real_t outputs[3]);

/*******************************************************************************
 * MAV MODEL
 ******************************************************************************/

/** MAV Model **/
typedef struct mav_model_t {
  real_t x[12];      // State
  real_t inertia[3]; // Moment of inertia
  real_t kr;         // Rotation drag constant
  real_t kt;         // Translation drag constant
  real_t l;          // Arm length
  real_t d;          // Drag
  real_t m;          // Mass
  real_t g;          // Gravitational constant
} mav_model_t;

void mav_model_setup(mav_model_t *mav,
                     const real_t x[12],
                     const real_t inertia[3],
                     const real_t kr,
                     const real_t kt,
                     const real_t l,
                     const real_t d,
                     const real_t m,
                     const real_t g);
void mav_model_attitude(const mav_model_t *mav, real_t rpy[3]);
void mav_model_angular_velocity(const mav_model_t *mav, real_t pqr[3]);
void mav_model_position(const mav_model_t *mav, real_t pos[3]);
void mav_model_velocity(const mav_model_t *mav, real_t vel[3]);
void mav_model_print_state(const mav_model_t *mav, const real_t time);
void mav_model_update(mav_model_t *mav, const real_t u[4], const real_t dt);

/** MAV Model Telemetry **/
typedef struct mav_model_telem_t {
  int num_events;
  real_t *time;

  real_t *roll;
  real_t *pitch;
  real_t *yaw;

  real_t *wx;
  real_t *wy;
  real_t *wz;

  real_t *x;
  real_t *y;
  real_t *z;

  real_t *vx;
  real_t *vy;
  real_t *vz;

} mav_model_telem_t;

mav_model_telem_t *mav_model_telem_malloc(void);
void mav_model_telem_free(mav_model_telem_t *telem);
void mav_model_telem_update(mav_model_telem_t *telem,
                            const mav_model_t *mav,
                            const real_t time);
void mav_model_telem_plot(const mav_model_telem_t *telem);
void mav_model_telem_plot_xy(const mav_model_telem_t *telem);

/** MAV Attitude Controller **/
typedef struct mav_att_ctrl_t {
  real_t dt;
  pid_ctrl_t roll;
  pid_ctrl_t pitch;
  pid_ctrl_t yaw;
  real_t u[4];
} mav_att_ctrl_t;

void mav_att_ctrl_setup(mav_att_ctrl_t *ctrl);
void mav_att_ctrl_update(mav_att_ctrl_t *ctrl,
                         const real_t sp[4],
                         const real_t pv[3],
                         const real_t dt,
                         real_t u[4]);

/** MAV Velocity Controller **/
typedef struct mav_vel_ctrl_t {
  real_t dt;
  pid_ctrl_t vx;
  pid_ctrl_t vy;
  pid_ctrl_t vz;
  real_t u[4];
} mav_vel_ctrl_t;

void mav_vel_ctrl_setup(mav_vel_ctrl_t *ctrl);
void mav_vel_ctrl_update(mav_vel_ctrl_t *ctrl,
                         const real_t sp[4],
                         const real_t pv[4],
                         const real_t dt,
                         real_t u[4]);

/** MAV Position Controller **/
typedef struct mav_pos_ctrl_t {
  real_t dt;
  pid_ctrl_t x;
  pid_ctrl_t y;
  pid_ctrl_t z;
  real_t u[4];
} mav_pos_ctrl_t;

void mav_pos_ctrl_setup(mav_pos_ctrl_t *ctrl);
void mav_pos_ctrl_update(mav_pos_ctrl_t *ctrl,
                         const real_t sp[4],
                         const real_t pv[4],
                         const real_t dt,
                         real_t u[4]);

/** MAV Waypoints **/
typedef struct mav_waypoints_t {
  int num_waypoints;
  real_t *waypoints;
  int index;

  int wait_mode;
  real_t wait_time;

  real_t threshold_dist;
  real_t threshold_yaw;
  real_t threshold_wait;
} mav_waypoints_t;

mav_waypoints_t *mav_waypoints_malloc(void);
void mav_waypoints_free(mav_waypoints_t *ctrl);
void mav_waypoints_print(const mav_waypoints_t *wps);
int mav_waypoints_done(const mav_waypoints_t *wps);
void mav_waypoints_add(mav_waypoints_t *wps, real_t wp[4]);
void mav_waypoints_target(const mav_waypoints_t *wps, real_t wp[4]);
int mav_waypoints_update(mav_waypoints_t *wps,
                         const real_t state[4],
                         const real_t dt,
                         real_t wp[4]);

/*******************************************************************************
 * SENSOR FUSION
 ******************************************************************************/

#define POSITION_PARAM 1
#define ROTATION_PARAM 2
#define POSE_PARAM 3
#define EXTRINSIC_PARAM 4
#define FIDUCIAL_PARAM 5
#define VELOCITY_PARAM 6
#define IMU_BIASES_PARAM 7
#define FEATURE_PARAM 8
#define IDF_BEARING_PARAM 9
#define IDF_POSITION_PARAM 10
#define JOINT_PARAM 11
#define CAMERA_PARAM 12
#define TIME_DELAY_PARAM 13

///////////
// UTILS //
///////////

int schur_complement(const real_t *H,
                     const real_t *b,
                     const int H_size,
                     const int m,
                     const int r,
                     real_t *H_marg,
                     real_t *b_marg);

int shannon_entropy(const real_t *covar, const int m, real_t *entropy);

//////////////
// TIMELINE //
//////////////

#define CAMERA_EVENT 1
#define IMU_EVENT 2
#define FIDUCIAL_EVENT 3

typedef struct camera_event_t {
  timestamp_t ts;
  int cam_idx;
  char *image_path;

  int num_features;
  size_t *feature_ids;
  real_t *keypoints;
} camera_event_t;

typedef struct imu_event_t {
  timestamp_t ts;
  real_t acc[3];
  real_t gyr[3];
} imu_event_t;

typedef struct fiducial_event_t {
  timestamp_t ts;
  int cam_idx;
  int num_corners;
  int *tag_ids;
  int *corner_indices;
  real_t *object_points;
  real_t *keypoints;
} fiducial_event_t;

union event_data_t {
  camera_event_t camera;
  imu_event_t imu;
  fiducial_event_t fiducial;
};

typedef struct timeline_event_t {
  int type;
  timestamp_t ts;
  union event_data_t data;
} timeline_event_t;

typedef struct timeline_t {
  // Stats
  int num_cams;
  int num_imus;
  int num_event_types;

  // Events
  timeline_event_t **events;
  timestamp_t **events_timestamps;
  int *events_lengths;
  int *events_types;

  // Timeline
  size_t timeline_length;
  timestamp_t *timeline_timestamps;
  timeline_event_t ***timeline_events;
  int *timeline_events_lengths;
} timeline_t;

void print_camera_event(const camera_event_t *event);
void print_imu_event(const imu_event_t *event);
void print_fiducial_event(const fiducial_event_t *event);

timeline_t *timeline_malloc(void); void timeline_free(timeline_t *timeline);
timeline_t *timeline_load_data(const char *data_dir,
                               const int num_cams,
                               const int num_imus);

//////////////
// POSITION //
//////////////

typedef struct pos_t {
  int marginalize;
  int fix;
  real_t data[3];
} pos_t;

void pos_setup(pos_t *pos, const real_t *data);
void pos_copy(const pos_t *src, pos_t *dst);
void pos_fprint(const char *prefix, const pos_t *pos, FILE *f);
void pos_print(const char *prefix, const pos_t *pos);

//////////////
// ROTATION //
//////////////

typedef struct rot_t {
  int marginalize;
  int fix;
  real_t data[4];
} rot_t;

void rot_setup(rot_t *rot, const real_t *data);
void rot_fprint(const char *prefix, const rot_t *rot, FILE *f);
void rot_print(const char *prefix, const rot_t *rot);

//////////
// POSE //
//////////

typedef struct pose_t {
  int marginalize;
  int fix;
  timestamp_t ts;
  real_t data[7];
} pose_t;

void pose_init(real_t *pose);
void pose_setup(pose_t *pose, const timestamp_t ts, const real_t *param);
void pose_copy(const pose_t *src, pose_t *dst);
void pose_fprint(const char *prefix, const pose_t *pose, FILE *f);
void pose_print(const char *prefix, const pose_t *pose);

///////////////
// EXTRINSIC //
///////////////

typedef struct extrinsic_t {
  int marginalize;
  int fix;
  real_t data[7];
} extrinsic_t;

void extrinsic_setup(extrinsic_t *extrinsic, const real_t *param);
void extrinsic_copy(const extrinsic_t *src, extrinsic_t *dst);
void extrinsic_fprint(const char *prefix, const extrinsic_t *exts, FILE *f);
void extrinsic_print(const char *prefix, const extrinsic_t *exts);

//////////////
// FIDUCIAL //
//////////////

/** Fiducial **/
typedef struct fiducial_t {
  int marginalize;
  int fix;
  real_t data[7];
} fiducial_t;

void fiducial_setup(fiducial_t *fiducial, const real_t *param);
void fiducial_copy(const fiducial_t *src, fiducial_t *dst);
void fiducial_fprint(const char *prefix, const fiducial_t *exts, FILE *f);
void fiducial_print(const char *prefix, const fiducial_t *exts);

/** Fiducial Buffer **/
typedef struct fiducial_buffer_t {
  fiducial_event_t **data;
  int size;
  int capacity;
} fiducial_buffer_t;

fiducial_buffer_t *fiducial_buffer_malloc(void); void fiducial_buffer_clear(fiducial_buffer_t *buf);
void fiducial_buffer_free(fiducial_buffer_t *buf);
int fiducial_buffer_total_corners(const fiducial_buffer_t *buf);
void fiducial_buffer_add(fiducial_buffer_t *buf,
                         const timestamp_t ts,
                         const int cam_idx,
                         const int num_corners,
                         const int *tag_ids,
                         const int *corner_indices,
                         const real_t *object_points,
                         const real_t *keypoints);

///////////////////////
// CAMERA-PARAMETERS //
///////////////////////

typedef struct camera_params_t {
  int marginalize;
  int fix;

  int cam_idx;
  int resolution[2];
  char proj_model[30];
  char dist_model[30];
  real_t data[8];

  project_func_t proj_func;
  back_project_func_t back_proj_func;
  undistort_func_t undistort_func;
} camera_params_t;

void camera_params_setup(camera_params_t *camera,
                         const int cam_idx,
                         const int cam_res[2],
                         const char *proj_model,
                         const char *dist_model,
                         const real_t *data);
void camera_params_copy(const camera_params_t *src, camera_params_t *dst);
void camera_params_fprint(const camera_params_t *cam, FILE *f);
void camera_params_print(const camera_params_t *camera);
void camera_project(const camera_params_t *camera,
                    const real_t p_C[3],
                    real_t z[2]);
void camera_back_project(const camera_params_t *camera,
                         const real_t z[2],
                         real_t bearing[3]);
void camera_undistort_points(const camera_params_t *camera,
                             const real_t *kps,
                             const int num_points,
                             real_t *kps_und);
int solvepnp_camera(const camera_params_t *cam_params,
                    const real_t *img_pts,
                    const real_t *obj_pts,
                    const int N,
                    real_t T_CO[4 * 4]);
void triangulate_batch(const camera_params_t *cam_i,
                       const camera_params_t *cam_j,
                       const real_t T_CiCj[4 * 4],
                       const real_t *kps_i,
                       const real_t *kps_j,
                       const int n,
                       real_t *points,
                       int *status);
void stereo_triangulate(const camera_params_t *cam_i,
                        const camera_params_t *cam_j,
                        const real_t T_WCi[4 * 4],
                        const real_t T_CiCj[4 * 4],
                        const real_t *kps_i,
                        const real_t *kps_j,
                        const int n,
                        real_t *points,
                        int *status);

//////////////
// VELOCITY //
//////////////

typedef struct velocity_t {
  int marginalize;
  int fix;

  timestamp_t ts;
  real_t data[3];
} velocity_t;

void velocity_setup(velocity_t *vel, const timestamp_t ts, const real_t v[3]);
void velocity_copy(const velocity_t *src, velocity_t *dst);

////////////////
// IMU-BIASES //
////////////////

typedef struct imu_biases_t {
  int marginalize;
  int fix;

  timestamp_t ts;
  real_t data[6];
} imu_biases_t;

void imu_biases_setup(imu_biases_t *sb,
                      const timestamp_t ts,
                      const real_t ba[3],
                      const real_t bg[3]);
void imu_biases_copy(const imu_biases_t *src, imu_biases_t *dst);
void imu_biases_get_accel_bias(const imu_biases_t *biases, real_t ba[3]);
void imu_biases_get_gyro_bias(const imu_biases_t *biases, real_t bg[3]);

/////////////
// FEATURE //
/////////////

#define FEATURE_XYZ 0
#define FEATURE_INVERSE_DEPTH 1
#define FEATURE_MAX_LENGTH 20

#define FEATURES_CAPACITY_INITIAL 10000
#define FEATURES_CAPACITY_GROWTH_FACTOR 2

/** Feature **/
typedef struct feature_t {
  int marginalize;
  int fix;
  int type;

  // Feature data
  size_t feature_id;
  int status;
  real_t data[3];
} feature_t;

typedef struct feature_map_t {
  size_t key;
  feature_t feature;
} feature_map_t;

void feature_setup(feature_t *f, const size_t feature_id);
void feature_init(feature_t *f, const size_t feature_id, const real_t *data);
void feature_print(const feature_t *feature);

// void idf_setup(feature_t *f,
//                const size_t feature_id,
//                const size_t pos_id,
//                const camera_params_t *cam_params,
//                const real_t C_WC[3 * 3],
//                const real_t z[2]);
// void idf_point(const feature_t *f, const real_t r_WC[3], real_t p_W[3]);

// /** Features **/
// typedef struct features_t {
//   feature_t **data;
//   size_t num_features;
//   size_t feature_capacity;

//   pos_t **pos_data;
//   size_t num_positions;
//   size_t position_capacity;
// } features_t;

// features_t *features_malloc(void); void features_free(features_t *features);
// int features_exists(const features_t *features, const size_t feature_id);
// void features_add_xyzs(features_t *features,
//                        const size_t *feature_ids,
//                        const real_t *params,
//                        const size_t num_features);
// void features_add_idfs(features_t *features,
//                        const size_t *feature_ids,
//                        const camera_params_t *cam_params,
//                        const real_t T_WC[4 * 4],
//                        const real_t *keypoints,
//                        const size_t num_keypoints);
// void features_get_xyz(const features_t *features,
//                       const size_t feature_id,
//                       feature_t **feature);
// void features_get_idf(const features_t *features,
//                       const size_t feature_id,
//                       feature_t **feature,
//                       pos_t **pos);
// int features_point(const features_t *features,
//                    const size_t feature_id,
//                    real_t p_W[3]);

////////////////
// TIME-DELAY //
////////////////

typedef struct time_delay_t {
  int marginalize;
  int fix;
  real_t data[1];
} time_delay_t;

void time_delay_setup(time_delay_t *time_delay, const real_t param);
void time_delay_copy(const time_delay_t *src, time_delay_t *dst);
void time_delay_print(const char *prefix, const time_delay_t *exts);

///////////
// JOINT //
///////////

typedef struct joint_t {
  int marginalize;
  int fix;

  timestamp_t ts;
  int joint_idx;
  real_t data[1];
} joint_t;

void joint_setup(joint_t *joint,
                 const timestamp_t ts,
                 const int joint_idx,
                 const real_t theta);
void joint_copy(const joint_t *src, joint_t *dst);
void joint_print(const char *prefix, const joint_t *joint);

////////////////
// PARAMETERS //
////////////////

typedef struct param_hash_t {
  int64_t key;
  int param_type;
  void *param_ptr;
} param_hash_t;

#define PARAM_HASH(HASH_NAME, KEY_TYPE, VALUE_TYPE)                            \
  typedef struct HASH_NAME {                                                   \
    KEY_TYPE key;                                                              \
    VALUE_TYPE *value;                                                         \
  } HASH_NAME;

PARAM_HASH(pos_hash_t, timestamp_t, pos_t)
PARAM_HASH(rot_hash_t, timestamp_t, rot_t)
PARAM_HASH(pose_hash_t, timestamp_t, pose_t)
PARAM_HASH(velocity_hash_t, timestamp_t, velocity_t)
PARAM_HASH(imu_biases_hash_t, timestamp_t, imu_biases_t)
PARAM_HASH(feature_hash_t, size_t, feature_t)
PARAM_HASH(joint_hash_t, size_t, joint_t)
PARAM_HASH(extrinsic_hash_t, size_t, extrinsic_t)
PARAM_HASH(fiducial_hash_t, size_t, fiducial_t)
PARAM_HASH(camera_params_hash_t, size_t, camera_params_t)
PARAM_HASH(time_delay_hash_t, size_t, time_delay_t)

typedef struct param_order_t {
  void *key;
  int idx;
  int type;
  int fix;
} param_order_t;

void param_order_free(param_order_t *hash);
void param_type_string(const int param_type, char *s);
size_t param_global_size(const int param_type);
size_t param_local_size(const int param_type);
void param_order_print(const param_order_t *hash);
int param_order_exists(param_order_t **hash, real_t *data);
void param_order_add(param_order_t **hash,
                     const int param_type,
                     const int fix,
                     real_t *data,
                     int *col_idx);
void param_order_add_position(param_order_t **h, pos_t *p, int *c);
void param_order_add_rotation(param_order_t **h, rot_t *p, int *c);
void param_order_add_pose(param_order_t **h, pose_t *p, int *c);
void param_order_add_extrinsic(param_order_t **h, extrinsic_t *p, int *c);
void param_order_add_fiducial(param_order_t **h, fiducial_t *p, int *c);
void param_order_add_velocity(param_order_t **h, velocity_t *p, int *c);
void param_order_add_imu_biases(param_order_t **h, imu_biases_t *p, int *c);
void param_order_add_feature(param_order_t **h, feature_t *p, int *c);
void param_order_add_joint(param_order_t **h, joint_t *p, int *c);
void param_order_add_camera(param_order_t **h, camera_params_t *p, int *c);
void param_order_add_time_delay(param_order_t **h, time_delay_t *p, int *c);

////////////
// FACTOR //
////////////

typedef struct factor_hash_t {
  int64_t key;
  int factor_type;
  void *factor_ptr;
} factor_hash_t;

#define FACTOR_EVAL_PTR                                                        \
  int (*factor_eval)(const void *factor,                                       \
                     real_t **params,                                          \
                     real_t *residuals,                                        \
                     real_t **jacobians)

#define CERES_FACTOR_EVAL(FACTOR_TYPE,                                         \
                          FACTOR,                                              \
                          FACTOR_EVAL,                                         \
                          PARAMS,                                              \
                          R_OUT,                                               \
                          J_OUT)                                               \
  {                                                                            \
    assert(FACTOR);                                                            \
    assert(PARAMS);                                                            \
    assert(R_OUT);                                                             \
                                                                               \
    /* Copy parameters */                                                      \
    for (int i = 0; i < FACTOR->num_params; i++) {                             \
      const int global_size = param_global_size(FACTOR->param_types[i]);       \
      vec_copy(PARAMS[i], global_size, FACTOR->params[i]);                     \
    }                                                                          \
                                                                               \
    /* Evaluate factor */                                                      \
    FACTOR_EVAL(factor_ptr);                                                   \
                                                                               \
    /* Residuals */                                                            \
    vec_copy(FACTOR->r, FACTOR->r_size, r_out);                                \
                                                                               \
    /* Jacobians */                                                            \
    if (J_OUT == NULL) {                                                       \
      return 1;                                                                \
    }                                                                          \
                                                                               \
    const int r_size = FACTOR->r_size;                                         \
    for (int jac_idx = 0; jac_idx < FACTOR->num_params; jac_idx++) {           \
      if (J_OUT[jac_idx]) {                                                    \
        const int gs = param_global_size(FACTOR->param_types[jac_idx]);        \
        const int ls = param_local_size(FACTOR->param_types[jac_idx]);         \
        const int rs = 0;                                                      \
        const int re = r_size - 1;                                             \
        const int cs = 0;                                                      \
        const int ce = ls - 1;                                                 \
        zeros(J_OUT[jac_idx], r_size, gs);                                     \
        mat_block_set(J_OUT[jac_idx],                                          \
                      gs,                                                      \
                      rs,                                                      \
                      re,                                                      \
                      cs,                                                      \
                      ce,                                                      \
                      FACTOR->jacs[jac_idx]);                                  \
      }                                                                        \
    }                                                                          \
    return 1;                                                                  \
  }

int check_factor_jacobian(const void *factor,
                          FACTOR_EVAL_PTR,
                          real_t **params,
                          real_t **jacobians,
                          const int r_size,
                          const int param_size,
                          const int param_idx,
                          const real_t step_size,
                          const real_t tol,
                          const int verbose);

int check_factor_so3_jacobian(const void *factor,
                              FACTOR_EVAL_PTR,
                              real_t **params,
                              real_t **jacobians,
                              const int r_size,
                              const int param_idx,
                              const real_t step_size,
                              const real_t tol,
                              const int verbose);

/////////////////
// POSE FACTOR //
/////////////////

typedef struct pose_factor_t {
  real_t pos_meas[3];
  real_t quat_meas[4];
  pose_t *pose_est;

  real_t covar[6 * 6];
  real_t sqrt_info[6 * 6];

  int r_size;
  int num_params;
  int param_types[1];

  real_t *params[1];
  real_t r[6];
  real_t *jacs[1];
  real_t J_pose[6 * 6];
} pose_factor_t;

void pose_factor_setup(pose_factor_t *factor,
                       pose_t *pose,
                       const real_t var[6]);
int pose_factor_eval(void *factor);

///////////////
// BA FACTOR //
///////////////

typedef struct ba_factor_t {
  pose_t *pose;
  feature_t *feature;
  camera_params_t *camera;

  real_t covar[2 * 2];
  real_t sqrt_info[2 * 2];
  real_t z[2];

  int r_size;
  int num_params;
  int param_types[3];

  real_t *params[3];
  real_t r[2];
  real_t *jacs[3];
  real_t J_pose[2 * 6];
  real_t J_feature[2 * 3];
  real_t J_camera[2 * 8];
} ba_factor_t;

void ba_factor_setup(ba_factor_t *factor,
                     pose_t *pose,
                     feature_t *feature,
                     camera_params_t *camera,
                     const real_t z[2],
                     const real_t var[2]);
int ba_factor_eval(void *factor_ptr);

///////////////////
// CAMERA FACTOR //
///////////////////

typedef struct camera_factor_t {
  pose_t *pose;
  extrinsic_t *extrinsic;
  camera_params_t *camera;
  feature_t *feature;

  real_t covar[2 * 2];
  real_t sqrt_info[2 * 2];
  real_t z[2];

  int r_size;
  int num_params;
  int param_types[4];

  real_t *params[4];
  real_t r[2];
  real_t *jacs[4];
  real_t J_pose[2 * 6];
  real_t J_extrinsic[2 * 6];
  real_t J_feature[2 * 3];
  real_t J_camera[2 * 8];
} camera_factor_t;

void camera_factor_setup(camera_factor_t *factor,
                         pose_t *pose,
                         extrinsic_t *extrinsic,
                         feature_t *feature,
                         camera_params_t *camera,
                         const real_t z[2],
                         const real_t var[2]);
int camera_factor_eval(void *factor_ptr);

////////////////////////////////////////
// INVERSE-DEPTH FEATURE (IDF) FACTOR //
////////////////////////////////////////

// typedef struct idf_factor_t {
//   timestamp_t ts;
//   int cam_idx;
//   size_t feature_id;

//   pose_t *pose;
//   extrinsic_t *extrinsic;
//   camera_params_t *camera;
//   pos_t *idf_pos;
//   feature_t *idf_param;

//   real_t covar[2 * 2];
//   real_t sqrt_info[2 * 2];
//   real_t z[2];

//   int r_size;
//   int num_params;
//   int param_types[5];

//   real_t *params[5];
//   real_t r[2];
//   real_t *jacs[5];
//   real_t J_pose[2 * 6];
//   real_t J_extrinsic[2 * 6];
//   real_t J_camera[2 * 8];
//   real_t J_idf_pos[2 * 3];
//   real_t J_idf_param[2 * 3];
// } idf_factor_t;

// void idf_factor_setup(idf_factor_t *factor,
//                       pose_t *pose,
//                       extrinsic_t *extrinsic,
//                       camera_params_t *camera,
//                       pos_t *idf_pos,
//                       feature_t *idf_param,
//                       const timestamp_t ts,
//                       const int cam_idx,
//                       const size_t feature_id,
//                       const real_t z[2],
//                       const real_t var[2]);
// int idf_factor_eval(void *factor_ptr);

////////////////
// IMU FACTOR //
////////////////

/** IMU Parameters **/
typedef struct imu_params_t {
  int imu_idx;
  real_t rate;

  real_t sigma_aw;
  real_t sigma_gw;
  real_t sigma_a;
  real_t sigma_g;
  real_t g;
} imu_params_t;

/** IMU Buffer **/
#define IMU_BUFFER_MAX_SIZE 1000

typedef struct imu_buffer_t {
  timestamp_t ts[IMU_BUFFER_MAX_SIZE];
  real_t acc[IMU_BUFFER_MAX_SIZE][3];
  real_t gyr[IMU_BUFFER_MAX_SIZE][3];
  int size;
} imu_buffer_t;

void imu_buffer_setup(imu_buffer_t *imu_buf);
void imu_buffer_add(imu_buffer_t *imu_buf,
                    const timestamp_t ts,
                    const real_t acc[3],
                    const real_t gyr[3]);
timestamp_t imu_buffer_first_ts(const imu_buffer_t *imu_buf);
timestamp_t imu_buffer_last_ts(const imu_buffer_t *imu_buf);
void imu_buffer_clear(imu_buffer_t *imu_buf);
void imu_buffer_copy(const imu_buffer_t *from, imu_buffer_t *to);
void imu_buffer_print(const imu_buffer_t *imu_buf);

/** IMU Factor **/
typedef struct imu_factor_t {
  // IMU parameters and buffer
  const imu_params_t *imu_params;
  imu_buffer_t imu_buf;

  // Parameters
  pose_t *pose_i;
  velocity_t *vel_i;
  imu_biases_t *biases_i;
  pose_t *pose_j;
  velocity_t *vel_j;
  imu_biases_t *biases_j;
  int num_params;
  real_t *params[6];
  int param_types[6];

  // Residuals
  int r_size;
  real_t r[15];

  // Jacobians
  real_t *jacs[6];
  real_t J_pose_i[15 * 6];
  real_t J_vel_i[15 * 3];
  real_t J_biases_i[15 * 6];
  real_t J_pose_j[15 * 6];
  real_t J_vel_j[15 * 3];
  real_t J_biases_j[15 * 6];

  // Preintegration variables
  real_t Dt;         // Time difference between pose_i and pose_j in seconds
  real_t F[15 * 15]; // State jacobian
  real_t P[15 * 15]; // State covariance
  real_t Q[18 * 18]; // Noise matrix
  real_t dr[3];      // Relative position
  real_t dv[3];      // Relative velocity
  real_t dq[4];      // Relative rotation
  real_t ba[3];      // Accel biase
  real_t bg[3];      // Gyro biase
  real_t ba_ref[3];
  real_t bg_ref[3];

  // Preintegration step variables
  real_t r_i[3];
  real_t v_i[3];
  real_t q_i[4];
  real_t ba_i[3];
  real_t bg_i[3];

  real_t r_j[3];
  real_t v_j[3];
  real_t q_j[4];
  real_t ba_j[3];
  real_t bg_j[3];

  // Covariance and square-root info
  real_t covar[15 * 15];
  real_t sqrt_info[15 * 15];
} imu_factor_t;

void imu_state_vector(const real_t r[3],
                      const real_t q[4],
                      const real_t v[3],
                      const real_t ba[3],
                      const real_t bg[3],
                      real_t x[16]);
void imu_propagate(const real_t pose_k[7],
                   const real_t vel_k[3],
                   const imu_buffer_t *imu_buf,
                   real_t pose_kp1[7],
                   real_t vel_kp1[3]);
void imu_initial_attitude(const imu_buffer_t *imu_buf, real_t q_WS[4]);
void imu_factor_propagate_step(imu_factor_t *factor,
                               const real_t a_i[3],
                               const real_t w_i[3],
                               const real_t a_j[3],
                               const real_t w_j[3],
                               const real_t dt);
void imu_factor_F_matrix(const real_t q_i[4],
                         const real_t q_j[4],
                         const real_t ba_i[3],
                         const real_t bg_i[3],
                         const real_t a_i[3],
                         const real_t w_i[3],
                         const real_t a_j[3],
                         const real_t w_j[3],
                         const real_t dt,
                         real_t F_dt[15 * 15]);
void imu_factor_form_G_matrix(const imu_factor_t *factor,
                              const real_t a_i[3],
                              const real_t a_j[3],
                              const real_t dt,
                              real_t G_dt[15 * 18]);
void imu_factor_setup(imu_factor_t *factor,
                      const imu_params_t *imu_params,
                      const imu_buffer_t *imu_buf,
                      pose_t *pose_i,
                      velocity_t *v_i,
                      imu_biases_t *biases_i,
                      pose_t *pose_j,
                      velocity_t *v_j,
                      imu_biases_t *biases_j);
void imu_factor_reset(imu_factor_t *factor);
void imu_factor_preintegrate(imu_factor_t *factor);
int imu_factor_residuals(imu_factor_t *factor, real_t **params, real_t *r_out);
int imu_factor_eval(void *factor_ptr);
int imu_factor_ceres_eval(void *factor_ptr,
                          real_t **params,
                          real_t *r_out,
                          real_t **J_out);

////////////////////////
// JOINT-ANGLE FACTOR //
////////////////////////

typedef struct joint_factor_t {
  joint_t *joint;

  real_t z[1];
  real_t covar[1];
  real_t sqrt_info[1];

  int r_size;
  int num_params;
  int param_types[1];

  real_t *params[1];
  real_t r[1];
  real_t *jacs[1];
  real_t J_joint[1 * 1];
} joint_factor_t;

void joint_factor_setup(joint_factor_t *factor,
                        joint_t *joint0,
                        const real_t z,
                        const real_t var);
void joint_factor_copy(const joint_factor_t *src, joint_factor_t *dst);
int joint_factor_eval(void *factor_ptr);
int joint_factor_equals(const joint_factor_t *j0, const joint_factor_t *j1);

/////////////////////////
// CALIB-CAMERA FACTOR //
/////////////////////////

typedef struct calib_camera_factor_t {
  pose_t *pose;
  extrinsic_t *cam_ext;
  camera_params_t *cam_params;

  timestamp_t ts;
  int cam_idx;
  int tag_id;
  int corner_idx;
  real_t p_FFi[3];
  real_t z[2];

  real_t covar[2 * 2];
  real_t sqrt_info[2 * 2];

  int r_size;
  int num_params;
  int param_types[3];

  real_t *params[3];
  real_t r[2];
  real_t *jacs[3];
  real_t J_pose[2 * 6];
  real_t J_cam_ext[2 * 6];
  real_t J_cam_params[2 * 8];
} calib_camera_factor_t;

void calib_camera_factor_setup(calib_camera_factor_t *factor,
                               pose_t *pose,
                               extrinsic_t *cam_ext,
                               camera_params_t *cam_params,
                               const int cam_idx,
                               const int tag_id,
                               const int corner_idx,
                               const real_t p_FFi[3],
                               const real_t z[2],
                               const real_t var[2]);
int calib_camera_factor_eval(void *factor_ptr);
int calib_camera_factor_ceres_eval(void *factor_ptr,
                                   real_t **params,
                                   real_t *r_out,
                                   real_t **J_out);

/////////////////////////
// CALIB-IMUCAM FACTOR //
/////////////////////////

typedef struct calib_imucam_factor_t {
  fiducial_t *fiducial;        // fiducial pose: T_WF
  pose_t *imu_pose;            // IMU pose: T_WS
  extrinsic_t *imu_ext;        // IMU extrinsic: T_SC0
  extrinsic_t *cam_ext;        // Camera extrinsic: T_C0Ci
  camera_params_t *cam_params; // Camera parameters
  time_delay_t *time_delay;    // Time delay

  timestamp_t ts;
  int cam_idx;
  int tag_id;
  int corner_idx;
  real_t p_FFi[3];
  real_t z[2];
  real_t v[2];

  real_t covar[2 * 2];
  real_t sqrt_info[2 * 2];

  int r_size;
  int num_params;
  int param_types[6];

  real_t *params[6];
  real_t r[2];
  real_t *jacs[6];
  real_t J_fiducial[2 * 6];
  real_t J_imu_pose[2 * 6];
  real_t J_imu_ext[2 * 6];
  real_t J_cam_ext[2 * 6];
  real_t J_cam_params[2 * 8];
  real_t J_time_delay[2 * 1];
} calib_imucam_factor_t;

void calib_imucam_factor_setup(calib_imucam_factor_t *factor,
                               fiducial_t *fiducial,
                               pose_t *pose,
                               extrinsic_t *imu_ext,
                               extrinsic_t *cam_ext,
                               camera_params_t *cam_params,
                               time_delay_t *time_delay,
                               const int cam_idx,
                               const int tag_id,
                               const int corner_idx,
                               const real_t p_FFi[3],
                               const real_t z[2],
                               const real_t v[2],
                               const real_t var[2]);
int calib_imucam_factor_eval(void *factor_ptr);
int calib_imucam_factor_ceres_eval(void *factor_ptr,
                                   real_t **params,
                                   real_t *r_out,
                                   real_t **J_out);

/////////////////////////
// CALIB-GIMBAL FACTOR //
/////////////////////////

typedef struct calib_gimbal_factor_t {
  fiducial_t *fiducial_ext;
  extrinsic_t *gimbal_ext;
  pose_t *pose;
  extrinsic_t *link0;
  extrinsic_t *link1;
  joint_t *joint0;
  joint_t *joint1;
  joint_t *joint2;
  extrinsic_t *cam_ext;
  camera_params_t *cam;

  timestamp_t ts;
  int cam_idx;
  int tag_id;
  int corner_idx;
  real_t p_FFi[3];
  real_t z[2];

  real_t covar[2 * 2];
  real_t sqrt_info[2 * 2];

  int r_size;
  int num_params;
  int param_types[10];

  real_t *params[10];
  real_t r[2];
  real_t *jacs[10];
  real_t J_fiducial_ext[2 * 6];
  real_t J_gimbal_ext[2 * 6];
  real_t J_pose[2 * 6];
  real_t J_link0[2 * 6];
  real_t J_link1[2 * 6];
  real_t J_joint0[2 * 1];
  real_t J_joint1[2 * 1];
  real_t J_joint2[2 * 1];
  real_t J_cam_ext[2 * 6];
  real_t J_cam_params[2 * 8];
} calib_gimbal_factor_t;

void gimbal_setup_extrinsic(const real_t ypr[3],
                            const real_t r[3],
                            real_t T[4 * 4],
                            extrinsic_t *link);
void gimbal_setup_joint(const timestamp_t ts,
                        const int joint_idx,
                        const real_t theta,
                        real_t T_joint[4 * 4],
                        joint_t *joint);

void calib_gimbal_factor_setup(calib_gimbal_factor_t *factor,
                               fiducial_t *fiducial_ext,
                               extrinsic_t *gimbal_ext,
                               pose_t *pose,
                               extrinsic_t *link0,
                               extrinsic_t *link1,
                               joint_t *joint0,
                               joint_t *joint1,
                               joint_t *joint2,
                               extrinsic_t *cam_ext,
                               camera_params_t *cam,
                               const timestamp_t ts,
                               const int cam_idx,
                               const int tag_id,
                               const int corner_idx,
                               const real_t p_FFi[3],
                               const real_t z[2],
                               const real_t var[2]);
int calib_gimbal_factor_eval(void *factor);
int calib_gimbal_factor_ceres_eval(void *factor_ptr,
                                   real_t **params,
                                   real_t *r_out,
                                   real_t **J_out);
int calib_gimbal_factor_equals(const calib_gimbal_factor_t *c0,
                               const calib_gimbal_factor_t *c1);
//////////////////
// MARGINALIZER //
//////////////////

#define MARG_FACTOR 1
#define BA_FACTOR 2
#define CAMERA_FACTOR 3
#define IDF_FACTOR 4
#define IMU_FACTOR 5
#define CALIB_CAMERA_FACTOR 6
#define CALIB_IMUCAM_FACTOR 7

#define MARG_TRACK(RHASH, MHASH, PARAM)                                        \
  if (PARAM->marginalize == 0) {                                               \
    hmput(RHASH, PARAM, PARAM);                                                \
  } else {                                                                     \
    hmput(MHASH, PARAM, PARAM);                                                \
  }

#define MARG_TRACK_FACTOR(PARAM, PARAM_TYPE)                                   \
  switch (PARAM_TYPE) {                                                        \
    case POSITION_PARAM:                                                       \
      MARG_TRACK(marg->r_positions, marg->m_positions, ((pos_t *) PARAM));     \
      break;                                                                   \
    case ROTATION_PARAM:                                                       \
      MARG_TRACK(marg->r_rotations, marg->m_rotations, ((rot_t *) PARAM));     \
      break;                                                                   \
    case POSE_PARAM:                                                           \
      MARG_TRACK(marg->r_poses, marg->m_poses, ((pose_t *) PARAM));            \
      break;                                                                   \
    case VELOCITY_PARAM:                                                       \
      MARG_TRACK(marg->r_velocities,                                           \
                 marg->m_velocities,                                           \
                 ((velocity_t *) PARAM));                                      \
      break;                                                                   \
    case IMU_BIASES_PARAM:                                                     \
      MARG_TRACK(marg->r_imu_biases,                                           \
                 marg->m_imu_biases,                                           \
                 ((imu_biases_t *) PARAM));                                    \
      break;                                                                   \
    case FEATURE_PARAM:                                                        \
      MARG_TRACK(marg->r_features, marg->m_features, ((feature_t *) PARAM));   \
      break;                                                                   \
    case FIDUCIAL_PARAM:                                                       \
      MARG_TRACK(marg->r_fiducials,                                            \
                 marg->m_fiducials,                                            \
                 ((fiducial_t *) PARAM));                                      \
      break;                                                                   \
    case EXTRINSIC_PARAM:                                                      \
      MARG_TRACK(marg->r_extrinsics,                                           \
                 marg->m_extrinsics,                                           \
                 ((extrinsic_t *) PARAM));                                     \
      break;                                                                   \
    case JOINT_PARAM:                                                          \
      MARG_TRACK(marg->r_joints, marg->m_joints, ((joint_t *) PARAM));         \
      break;                                                                   \
    case CAMERA_PARAM:                                                         \
      MARG_TRACK(marg->r_cam_params,                                           \
                 marg->m_cam_params,                                           \
                 ((camera_params_t *) PARAM));                                 \
      break;                                                                   \
    case TIME_DELAY_PARAM:                                                     \
      MARG_TRACK(marg->r_time_delays,                                          \
                 marg->m_time_delays,                                          \
                 ((time_delay_t *) PARAM));                                    \
      break;                                                                   \
    default:                                                                   \
      FATAL("Implementation Error!\n");                                        \
      break;                                                                   \
  }

#define MARG_INDEX(HASH, PARAM_TYPE, PARAM_ORDER, COL_IDX, SZ, GZ, N)          \
  for (size_t i = 0; i < hmlen(HASH); i++) {                                   \
    real_t *data = HASH[i].value->data;                                        \
    const int fix = HASH[i].value->fix;                                        \
    if (fix == 0) {                                                            \
      SZ += param_local_size(PARAM_TYPE);                                      \
      GZ += param_global_size(PARAM_TYPE);                                     \
      N += 1;                                                                  \
    }                                                                          \
    param_order_add(&PARAM_ORDER, PARAM_TYPE, fix, data, COL_IDX);             \
  }

#define MARG_PARAMS(MARG, HASH, PARAM_TYPE, PARAM_IDX, X0_IDX)                 \
  for (size_t i = 0; i < hmlen(HASH); i++) {                                   \
    const size_t param_size = param_global_size(PARAM_TYPE);                   \
    if (HASH[i].value->fix) {                                                  \
      continue;                                                                \
    }                                                                          \
    void *param = HASH[i].value;                                               \
    real_t *data = HASH[i].value->data;                                        \
                                                                               \
    MARG->param_types[PARAM_IDX] = PARAM_TYPE;                                 \
    MARG->param_ptrs[PARAM_IDX] = param;                                       \
    MARG->params[PARAM_IDX] = data;                                            \
    PARAM_IDX++;                                                               \
                                                                               \
    vec_copy(data, param_size, MARG->x0 + X0_IDX);                             \
    X0_IDX += param_size;                                                      \
  }

#define MARG_H(MARG, FACTOR_TYPE, FACTORS, H, G, LOCAL_SIZE)                   \
  {                                                                            \
    list_node_t *node = FACTORS->first;                                        \
    while (node != NULL) {                                                     \
      FACTOR_TYPE *factor = (FACTOR_TYPE *) node->value;                       \
      solver_fill_hessian(marg->hash,                                          \
                          factor->num_params,                                  \
                          factor->params,                                      \
                          factor->jacs,                                        \
                          factor->r,                                           \
                          factor->r_size,                                      \
                          LOCAL_SIZE,                                          \
                          H,                                                   \
                          G);                                                  \
      node = node->next;                                                       \
    }                                                                          \
  }

#define MARG_PARAM_HASH(PARAM_TYPE, HASH_NAME)                                 \
  typedef struct HASH_NAME {                                                   \
    void *key;                                                                 \
    PARAM_TYPE *value;                                                         \
  } HASH_NAME;

MARG_PARAM_HASH(pos_t, marg_pos_t)
MARG_PARAM_HASH(rot_t, marg_rot_t)
MARG_PARAM_HASH(pose_t, marg_pose_t)
MARG_PARAM_HASH(velocity_t, marg_velocity_t)
MARG_PARAM_HASH(imu_biases_t, marg_imu_biases_t)
MARG_PARAM_HASH(feature_t, marg_feature_t)
MARG_PARAM_HASH(joint_t, marg_joint_t)
MARG_PARAM_HASH(extrinsic_t, marg_extrinsic_t)
MARG_PARAM_HASH(fiducial_t, marg_fiducial_t)
MARG_PARAM_HASH(camera_params_t, marg_camera_params_t)
MARG_PARAM_HASH(time_delay_t, marg_time_delay_t)

typedef struct marg_factor_t {
  // Settings
  int debug;
  int cond_hessian;

  // Flags
  int marginalized;
  int schur_complement_ok;
  int eigen_decomp_ok;

  // parameters
  // -- Remain parameters
  marg_pos_t *r_positions;
  marg_rot_t *r_rotations;
  marg_pose_t *r_poses;
  marg_velocity_t *r_velocities;
  marg_imu_biases_t *r_imu_biases;
  marg_fiducial_t *r_fiducials;
  marg_joint_t *r_joints;
  marg_extrinsic_t *r_extrinsics;
  marg_feature_t *r_features;
  marg_camera_params_t *r_cam_params;
  marg_time_delay_t *r_time_delays;
  // -- Marginal parameters
  marg_pos_t *m_positions;
  marg_rot_t *m_rotations;
  marg_pose_t *m_poses;
  marg_velocity_t *m_velocities;
  marg_imu_biases_t *m_imu_biases;
  marg_feature_t *m_features;
  marg_fiducial_t *m_fiducials;
  marg_extrinsic_t *m_extrinsics;
  marg_joint_t *m_joints;
  marg_camera_params_t *m_cam_params;
  marg_time_delay_t *m_time_delays;

  // Factors
  list_t *ba_factors;
  list_t *camera_factors;
  list_t *idf_factors;
  list_t *imu_factors;
  list_t *calib_camera_factors;
  list_t *calib_imucam_factors;
  struct marg_factor_t *marg_factor;

  // Hessian, Jacobians and residuals
  param_order_t *hash;
  int m_size;
  int r_size;

  real_t *x0;
  real_t *r0;
  real_t *J0;
  real_t *J0_inv;
  real_t *dchi;
  real_t *J0_dchi;

  real_t *H;
  real_t *b;
  real_t *H_marg;
  real_t *b_marg;

  // Parameters, residuals and Jacobians (needed by the solver)
  int num_params;
  int *param_types;
  void **param_ptrs;
  real_t **params;
  real_t *r;
  real_t **jacs;

  // Profiling
  real_t time_hessian_form;
  real_t time_schur_complement;
  real_t time_hessian_decomp;
  real_t time_fejs;
  real_t time_total;
} marg_factor_t;

marg_factor_t *marg_factor_malloc(void); void marg_factor_free(marg_factor_t *marg);
void marg_factor_print_stats(const marg_factor_t *marg);
void marg_factor_add(marg_factor_t *marg, int factor_type, void *factor_ptr);
void marg_factor_marginalize(marg_factor_t *marg);
int marg_factor_eval(void *marg_ptr);

////////////
// SOLVER //
////////////

#define SOLVER_USE_SUITESPARSE

#define SOLVER_EVAL_FACTOR_COMPACT(HASH,                                       \
                                   SV_SIZE,                                    \
                                   H,                                          \
                                   G,                                          \
                                   FACTOR_EVAL,                                \
                                   FACTOR_PTR,                                 \
                                   R,                                          \
                                   R_IDX)                                      \
  FACTOR_EVAL(FACTOR_PTR);                                                     \
  vec_copy(FACTOR_PTR->r, FACTOR_PTR->r_size, &R[R_IDX]);                      \
  R_IDX += FACTOR_PTR->r_size;                                                 \
  solver_fill_hessian(HASH,                                                    \
                      FACTOR_PTR->num_params,                                  \
                      FACTOR_PTR->params,                                      \
                      FACTOR_PTR->jacs,                                        \
                      FACTOR_PTR->r,                                           \
                      FACTOR_PTR->r_size,                                      \
                      SV_SIZE,                                                 \
                      H,                                                       \
                      G);

typedef struct solver_t {
  // Settings
  int verbose;
  int max_iter;
  real_t lambda;
  real_t lambda_factor;

  // Data
  param_order_t *hash;
  int linearize;
  int r_size;
  int sv_size;
  real_t *H_damped;
  real_t *H;
  real_t *g;
  real_t *r;
  real_t *dx;

  // SuiteSparse
#ifdef SOLVER_USE_SUITESPARSE
  cholmod_common *common;
#endif

  // Callbacks
  param_order_t *(*param_order_func)(const void *data,
                                     int *sv_size,
                                     int *r_size);
  void (*cost_func)(const void *data, real_t *r);
  void (*linearize_func)(const void *data,
                         const int sv_size,
                         param_order_t *hash,
                         real_t *H,
                         real_t *g,
                         real_t *r);
  void (*linsolve_func)(const void *data,
                        const int sv_size,
                        param_order_t *hash,
                        real_t *H,
                        real_t *g,
                        real_t *dx);
} solver_t;

void solver_setup(solver_t *solver);
void solver_print_param_order(const solver_t *solver);
real_t solver_cost(const solver_t *solver, const void *data);
void solver_fill_jacobian(param_order_t *hash,
                          int num_params,
                          real_t **params,
                          real_t **jacs,
                          real_t *r,
                          int r_size,
                          int sv_size,
                          int J_row_idx,
                          real_t *J,
                          real_t *g);
void solver_fill_hessian(param_order_t *hash,
                         int num_params,
                         real_t **params,
                         real_t **jacs,
                         real_t *r,
                         int r_size,
                         int sv_size,
                         real_t *H,
                         real_t *g);
real_t **solver_params_copy(const solver_t *solver);
void solver_params_restore(solver_t *solver, real_t **x);
void solver_params_free(const solver_t *solver, real_t **x);
void solver_update(solver_t *solver, real_t *dx, int sv_size);
int solver_solve(solver_t *solver, void *data);

/////////////////////
// IMU CALIBRATION //
/////////////////////

void avar(const real_t *x, const real_t dt, const real_t *tau, const size_t n);

//////////////
// CAMCHAIN //
//////////////

typedef struct camchain_pose_hash_t {
  timestamp_t key;
  real_t *value;
} camchain_pose_hash_t;

typedef struct camchain_t {
  int analyzed;
  int num_cams;

  int **adj_list;
  real_t **adj_exts;
  camchain_pose_hash_t **cam_poses;
} camchain_t;

camchain_t *camchain_malloc(const int num_cams);
void camchain_free(camchain_t *cc);
void camchain_add_pose(camchain_t *cc,
                       const int cam_idx,
                       const timestamp_t ts,
                       const real_t T_CiF[4 * 4]);
void camchain_adjacency(camchain_t *cc);
void camchain_adjacency_print(const camchain_t *cc);
int camchain_find(camchain_t *cc,
                  const int idx_i,
                  const int idx_j,
                  real_t T_CiCj[4 * 4]);

////////////////////////
// CAMERA CALIBRATION //
////////////////////////

typedef struct calib_camera_view_t {
  timestamp_t ts;
  int view_idx;
  int cam_idx;
  int num_corners;

  int *tag_ids;
  int *corner_indices;
  real_t *object_points;
  real_t *keypoints;

  calib_camera_factor_t *factors;
} calib_camera_view_t;

typedef struct calib_camera_viewset_t {
  timestamp_t key;
  calib_camera_view_t **value;
} calib_camera_viewset_t;

typedef struct calib_camera_t {
  // Settings
  int fix_cam_params;
  int fix_cam_exts;
  int verbose;
  int max_iter;

  // Flags
  int cams_ok;

  // Counters
  int num_cams;
  int num_views;
  int num_factors;

  // Variables
  timestamp_t *timestamps;
  pose_hash_t *poses;
  extrinsic_t *cam_exts;
  camera_params_t *cam_params;

  // Factors
  calib_camera_viewset_t *view_sets;
  marg_factor_t *marg;
} calib_camera_t;

calib_camera_view_t *calib_camera_view_malloc(const timestamp_t ts,
                                              const int view_idx,
                                              const int cam_idx,
                                              const int num_corners,
                                              const int *tag_ids,
                                              const int *corner_indices,
                                              const real_t *object_points,
                                              const real_t *keypoints,
                                              pose_t *pose,
                                              extrinsic_t *cam_ext,
                                              camera_params_t *cam_params);
void calib_camera_view_free(calib_camera_view_t *view);

calib_camera_t *calib_camera_malloc(void); void calib_camera_free(calib_camera_t *calib);
void calib_camera_print(calib_camera_t *calib);
void calib_camera_add_camera(calib_camera_t *calib,
                             const int cam_idx,
                             const int cam_res[2],
                             const char *proj_model,
                             const char *dist_model,
                             const real_t *cam_params,
                             const real_t *cam_ext);
void calib_camera_add_view(calib_camera_t *calib,
                           const timestamp_t ts,
                           const int view_idx,
                           const int cam_idx,
                           const int num_corners,
                           const int *tag_ids,
                           const int *corner_indices,
                           const real_t *object_points,
                           const real_t *keypoints);
void calib_camera_marginalize(calib_camera_t *calib);
int calib_camera_add_data(calib_camera_t *calib,
                          const int cam_idx,
                          const char *data_path);
void calib_camera_errors(calib_camera_t *calib,
                         real_t *reproj_rmse,
                         real_t *reproj_mean,
                         real_t *reproj_median);
int calib_camera_shannon_entropy(calib_camera_t *calib, real_t *entropy);

param_order_t *calib_camera_param_order(const void *data,
                                        int *sv_size,
                                        int *r_size);
void calib_camera_cost(const void *data, real_t *r);
void calib_camera_linearize_compact(const void *data,
                                    const int sv_size,
                                    param_order_t *hash,
                                    real_t *H,
                                    real_t *g,
                                    real_t *r);
void calib_camera_linsolve(const void *data,
                           const int sv_size,
                           param_order_t *hash,
                           real_t *H,
                           real_t *g,
                           real_t *dx);
void calib_camera_solve(calib_camera_t *calib);

////////////////////////////
// CAMERA-IMU CALIBRATION //
////////////////////////////

typedef struct calib_imucam_view_t {
  timestamp_t ts;
  int view_idx;
  int cam_idx;
  int num_corners;

  int *tag_ids;
  int *corner_indices;
  real_t *object_points;
  real_t *keypoints;

  calib_imucam_factor_t *cam_factors;
} calib_imucam_view_t;

typedef struct calib_imucam_viewset_t {
  timestamp_t key;
  calib_imucam_view_t **value;
} calib_imucam_viewset_t;

typedef struct imu_factor_hash_t {
  int64_t key;
  imu_factor_t *value;
} imu_factor_hash_t;

typedef struct calib_imucam_t {
  // Settings
  int fix_fiducial;
  int fix_poses;
  int fix_velocities;
  int fix_biases;
  int fix_cam_params;
  int fix_cam_exts;
  int fix_time_delay;
  int verbose;
  int max_iter;

  // Flags
  int imu_ok;
  int cams_ok;
  int state_initialized;

  // Counters
  int num_imus;
  int num_cams;
  int num_views;
  int num_cam_factors;
  int num_imu_factors;

  // Variables
  timestamp_t *timestamps;
  pose_hash_t *poses;
  velocity_hash_t *velocities;
  imu_biases_hash_t *imu_biases;
  fiducial_t *fiducial;
  extrinsic_t *cam_exts;
  camera_params_t *cam_params;
  extrinsic_t *imu_ext;
  time_delay_t *time_delay;

  // Data
  fiducial_buffer_t *fiducial_buffer;
  imu_params_t imu_params;
  imu_buffer_t imu_buf;

  // Views
  calib_imucam_viewset_t *view_sets;
  imu_factor_hash_t *imu_factors;
} calib_imucam_t;

calib_imucam_view_t *calib_imucam_view_malloc(const timestamp_t ts,
                                              const int view_idx,
                                              const int cam_idx,
                                              const int num_corners,
                                              const int *tag_ids,
                                              const int *corner_indices,
                                              const real_t *object_points,
                                              const real_t *keypoints,
                                              fiducial_t *fiducial,
                                              pose_t *imu_pose,
                                              extrinsic_t *imu_ext,
                                              extrinsic_t *cam_ext,
                                              camera_params_t *cam_params,
                                              time_delay_t *time_delay);
void calib_imucam_view_free(calib_imucam_view_t *view);

calib_imucam_t *calib_imucam_malloc(void); void calib_imucam_free(calib_imucam_t *calib);
void calib_imucam_print(calib_imucam_t *calib);

void calib_imucam_add_imu(calib_imucam_t *calib,
                          const real_t imu_rate,
                          const real_t sigma_aw,
                          const real_t sigma_gw,
                          const real_t sigma_a,
                          const real_t sigma_g,
                          const real_t g,
                          const real_t *imu_ext);
void calib_imucam_add_camera(calib_imucam_t *calib,
                             const int cam_idx,
                             const int cam_res[2],
                             const char *proj_model,
                             const char *dist_model,
                             const real_t *cam_params,
                             const real_t *cam_ext);

void calib_imucam_add_imu_event(calib_imucam_t *calib,
                                const timestamp_t ts,
                                const real_t acc[3],
                                const real_t gyr[3]);
void calib_imucam_add_fiducial_event(calib_imucam_t *calib,
                                     const timestamp_t ts,
                                     const int cam_idx,
                                     const int num_corners,
                                     const int *tag_ids,
                                     const int *corner_indices,
                                     const real_t *object_points,
                                     const real_t *keypoints);
void calib_imucam_marginalize(calib_imucam_t *calib);
int calib_imucam_update(calib_imucam_t *calib);
void calib_imucam_errors(calib_imucam_t *calib,
                         real_t *reproj_rmse,
                         real_t *reproj_mean,
                         real_t *reproj_median);
param_order_t *calib_imucam_param_order(const void *data,
                                        int *sv_size,
                                        int *r_size);
void calib_imucam_cost(const void *data, real_t *r);
void calib_imucam_linearize_compact(const void *data,
                                    const int sv_size,
                                    param_order_t *hash,
                                    real_t *H,
                                    real_t *g,
                                    real_t *r);
void calib_imucam_save_estimates(calib_imucam_t *calib);
void calib_imucam_solve(calib_imucam_t *calib);

////////////////////////
// GIMBAL CALIBRATION //
////////////////////////

typedef struct calib_gimbal_view_t {
  timestamp_t ts;
  int view_idx;
  int cam_idx;
  int num_corners;

  int *tag_ids;
  int *corner_indices;
  real_t *object_points;
  real_t *keypoints;
  calib_gimbal_factor_t *calib_factors;
} calib_gimbal_view_t;

typedef struct calib_gimbal_t {
  // Settings
  int fix_fiducial_ext;
  int fix_gimbal_ext;
  int fix_poses;
  int fix_cam_params;
  int fix_cam_exts;
  int fix_links;
  int fix_joints;

  int num_rows;
  int num_cols;
  double tag_size;
  double tag_spacing;

  // Flags
  int fiducial_ext_ok;
  int gimbal_ext_ok;
  int poses_ok;
  int cams_ok;
  int links_ok;
  int joints_ok;

  // Counters
  int num_cams;
  int num_views;
  int num_poses;
  int num_links;
  int num_joints;
  int num_calib_factors;
  int num_joint_factors;

  // Variables
  timestamp_t *timestamps;
  fiducial_t fiducial_ext;
  extrinsic_t gimbal_ext;
  extrinsic_t *cam_exts;
  camera_params_t *cam_params;
  extrinsic_t *links;
  joint_t **joints;
  pose_t *poses;

  // Factors
  calib_gimbal_view_t ***views;
  joint_factor_t **joint_factors;
} calib_gimbal_t;

void calib_gimbal_view_setup(calib_gimbal_view_t *calib);
calib_gimbal_view_t *calib_gimbal_view_malloc(const timestamp_t ts,
                                              const int view_idx,
                                              const int cam_idx,
                                              const int *tag_ids,
                                              const int *corner_indices,
                                              const real_t *object_points,
                                              const real_t *keypoints,
                                              const int N,
                                              fiducial_t *fiducial_ext,
                                              extrinsic_t *gimbal_ext,
                                              pose_t *pose,
                                              extrinsic_t *link0,
                                              extrinsic_t *link1,
                                              joint_t *joint0,
                                              joint_t *joint1,
                                              joint_t *joint2,
                                              extrinsic_t *cam_ext,
                                              camera_params_t *cam_params);
void calib_gimbal_view_free(calib_gimbal_view_t *calib);
int calib_gimbal_view_equals(const calib_gimbal_view_t *v0,
                             const calib_gimbal_view_t *v1);

void calib_gimbal_setup(calib_gimbal_t *calib);
calib_gimbal_t *calib_gimbal_malloc(void); void calib_gimbal_free(calib_gimbal_t *calib);
int calib_gimbal_equals(const calib_gimbal_t *calib0,
                        const calib_gimbal_t *calib1);
calib_gimbal_t *calib_gimbal_copy(const calib_gimbal_t *src);
void calib_gimbal_print(const calib_gimbal_t *calib);
void calib_gimbal_add_fiducial(calib_gimbal_t *calib,
                               const real_t fiducial_pose[7]);
void calib_gimbal_add_pose(calib_gimbal_t *calib,
                           const timestamp_t ts,
                           const real_t pose[7]);
void calib_gimbal_add_gimbal_extrinsic(calib_gimbal_t *calib,
                                       const real_t gimbal_ext[7]);
void calib_gimbal_add_gimbal_link(calib_gimbal_t *calib,
                                  const int link_idx,
                                  const real_t link[7]);
void calib_gimbal_add_camera(calib_gimbal_t *calib,
                             const int cam_idx,
                             const int cam_res[2],
                             const char *proj_model,
                             const char *dist_model,
                             const real_t *cam_params,
                             const real_t *cam_ext);
void calib_gimbal_add_view(calib_gimbal_t *calib,
                           const int pose_idx,
                           const int view_idx,
                           const timestamp_t ts,
                           const int cam_idx,
                           const int num_corners,
                           const int *tag_ids,
                           const int *corner_indices,
                           const real_t *object_points,
                           const real_t *keypoints,
                           const real_t *joints,
                           const int num_joints);
int calib_gimbal_remove_view(calib_gimbal_t *calib, const int view_idx);
calib_gimbal_t *calib_gimbal_load(const char *data_path);
void calib_gimbal_save(const calib_gimbal_t *calib, const char *data_path);
int calib_gimbal_validate(calib_gimbal_t *calib);
void calib_gimbal_nbv(calib_gimbal_t *calib, real_t nbv_joints[3]);
param_order_t *calib_gimbal_param_order(const void *data,
                                        int *sv_size,
                                        int *r_size);
void calib_gimbal_reproj_errors(const calib_gimbal_t *calib,
                                real_t *reproj_rmse,
                                real_t *reproj_mean,
                                real_t *reproj_median);
void calib_gimbal_cost(const void *data, real_t *r);
void calib_gimbal_linearize(const void *data,
                            const int J_rows,
                            const int J_cols,
                            param_order_t *hash,
                            real_t *J,
                            real_t *g,
                            real_t *r);
void calib_gimbal_linearize_compact(const void *data,
                                    const int sv_size,
                                    param_order_t *hash,
                                    real_t *H,
                                    real_t *g,
                                    real_t *r);

///////////////////////
// INERTIAL ODOMETRY //
///////////////////////

typedef struct inertial_odometry_t {
  // IMU Parameters
  imu_params_t imu_params;

  // Factors
  int num_factors;
  imu_factor_t *factors;
  marg_factor_t *marg;

  // Variables
  pose_t *poses;
  velocity_t *vels;
  imu_biases_t *biases;
} inertial_odometry_t;

inertial_odometry_t *inertial_odometry_malloc(void); void inertial_odometry_free(inertial_odometry_t *odom);
void inertial_odometry_save(const inertial_odometry_t *odom,
                            const char *save_path);
param_order_t *inertial_odometry_param_order(const void *data,
                                             int *sv_size,
                                             int *r_size);
void inertial_odometry_cost(const void *data, real_t *r);
void inertial_odometry_linearize_compact(const void *data,
                                         const int sv_size,
                                         param_order_t *hash,
                                         real_t *H,
                                         real_t *g,
                                         real_t *r);

/////////////////////////////
// RELATIVE POSE ESTIMATOR //
/////////////////////////////

int relpose_estimator(const int num_cams,
                      const camera_params_t **cam_params,
                      const real_t **cam_exts,
                      const size_t **fids,
                      const real_t **kps,
                      const int *num_kps,
                      const feature_map_t *feature_map,
                      const real_t T_WB_km1[4 * 4],
                      real_t T_WB_k[4 * 4]);

////////////////////////////
// TWO-STATE FILTER (TSF) //
////////////////////////////

#define TSF_FRAME_LIMIT 1000
#define TSF_EST_MODE 3

/** TSF Frameset **/
typedef struct tsf_frameset_t {
  timestamp_t ts;

  size_t cam0_fids[TSF_FRAME_LIMIT];
  real_t cam0_kps[TSF_FRAME_LIMIT * 3];
  int cam0_num_kps;

  size_t cam1_fids[TSF_FRAME_LIMIT];
  real_t cam1_kps[TSF_FRAME_LIMIT * 3];
  int cam1_num_kps;
} tsf_frameset_t;

/** Two-State Filter (TSF) **/
typedef struct tsf_t {
  // Flags
  int state;
  int num_imus;
  int num_cams;
  int imu_started;
  ssize_t frame_idx;

  // Settings
  int fix_cam_params;
  int fix_cam_exts;
  int fix_imu_ext;
  int fix_time_delay;

  // IMU
  imu_params_t imu_params;
  imu_buffer_t imu_buf;
  extrinsic_t imu_ext;
  time_delay_t time_delay;

  // Vision
  camera_params_t *cam_params;
  extrinsic_t *cam_exts;
  tsf_frameset_t fs_km1;
  tsf_frameset_t fs_k;
  feature_map_t *feature_map;

  // Factors
  imu_factor_t imu_factor;
  marg_factor_t *marg;

  // State
  real_t pose_init[7];
  real_t vel_init[3];
  real_t ba_init[3];
  real_t bg_init[3];
  timestamp_t ts_i;
  timestamp_t ts_j;
  pose_t pose_i;
  pose_t pose_j;
  velocity_t vel_i;
  velocity_t vel_j;
  imu_biases_t biases_i;
  imu_biases_t biases_j;
} tsf_t;

void tsf_frameset_setup(tsf_frameset_t *fs);
void tsf_frameset_reset(tsf_frameset_t *fs);

tsf_t *tsf_malloc(void); void tsf_free(tsf_t *tsf);
void tsf_print(const tsf_t *tsf);

void tsf_set_init_pose(tsf_t *tsf, real_t pose[7]);
void tsf_set_init_velocity(tsf_t *tsf, real_t vel[3]);
void tsf_add_camera(tsf_t *tsf,
                    const int cam_idx,
                    const int cam_res[2],
                    const char *proj_model,
                    const char *dist_model,
                    const real_t *cam_params,
                    const real_t *cam_ext);
void tsf_add_imu(tsf_t *tsf,
                 const real_t imu_rate,
                 const real_t sigma_aw,
                 const real_t sigma_gw,
                 const real_t sigma_a,
                 const real_t sigma_g,
                 const real_t g,
                 const real_t *imu_ext);

void tsf_imu_event(tsf_t *tsf,
                   const timestamp_t ts,
                   const real_t acc[3],
                   const real_t gyr[3]);
void tsf_camera_event(tsf_t *tsf,
                      const timestamp_t ts,
                      const size_t *cam0_fids,
                      const real_t *cam0_kps,
                      const int num_cam0_kps,
                      const size_t *cam1_fids,
                      const real_t *cam1_kps,
                      const int num_cam1_kps);

void tsf_cost(const void *data, real_t *r);
void tsf_errors(const tsf_t *tsf,
                real_t *reproj_rmse,
                real_t *reproj_mean,
                real_t *reproj_median);
param_order_t *tsf_param_order(const void *data, int *sv_size, int *r_size);
void tsf_linearize_compact(const void *data,
                           const int sv_size,
                           param_order_t *hash,
                           real_t *H,
                           real_t *g,
                           real_t *r);
void tsf_update(tsf_t *tsf, const timestamp_t ts);

/*******************************************************************************
 * DATASET
 ******************************************************************************/

pose_t *load_poses(const char *fp, int *num_poses);
int **assoc_pose_data(pose_t *gnd_poses,
                      size_t num_gnd_poses,
                      pose_t *est_poses,
                      size_t num_est_poses,
                      double threshold,
                      size_t *num_matches);

/*******************************************************************************
 * PLOTTING
 ******************************************************************************/

FILE *gnuplot_init(void); void gnuplot_close(FILE *pipe);
void gnuplot_multiplot(FILE *pipe, const int num_rows, const int num_cols);
void gnuplot_send(FILE *pipe, const char *cmd);
void gnuplot_xrange(FILE *pipe, const real_t xmin, const real_t xmax);
void gnuplot_yrange(FILE *pipe, const real_t ymin, const real_t ymax);
void gnuplot_send_xy(FILE *pipe,
                     const char *data_name,
                     const real_t *xvals,
                     const real_t *yvals,
                     const int n);
void gnuplot_send_matrix(FILE *pipe,
                         const char *data_name,
                         const real_t *A,
                         const int m,
                         const int n);
void gnuplot_matshow(const real_t *A, const int m, const int n);

/*******************************************************************************
 * SIMULATION
 ******************************************************************************/

/** Sim Circle Trajectory **/
typedef struct sim_circle_t {
  real_t imu_rate;
  real_t cam_rate;
  real_t circle_r;
  real_t circle_v;
  real_t theta_init;
  real_t yaw_init;
} sim_circle_t;

void sim_circle_defaults(sim_circle_t *conf);

//////////////////
// SIM FEATURES //
//////////////////

typedef struct sim_features_t {
  real_t **features;
  int num_features;
} sim_features_t;

sim_features_t *sim_features_load(const char *csv_path);
void sim_features_free(sim_features_t *features_data);

//////////////////
// SIM IMU DATA //
//////////////////

typedef struct sim_imu_data_t {
  size_t num_measurements;
  real_t *timestamps;
  real_t *poses;
  real_t *velocities;
  real_t *imu_acc;
  real_t *imu_gyr;
} sim_imu_data_t;

void sim_imu_data_setup(sim_imu_data_t *imu_data);
sim_imu_data_t *sim_imu_data_malloc(void); void sim_imu_data_free(sim_imu_data_t *imu_data);
sim_imu_data_t *sim_imu_data_load(const char *csv_path);
sim_imu_data_t *sim_imu_circle_trajectory(const sim_circle_t *conf);
void sim_imu_measurements(const sim_imu_data_t *data,
                          const int64_t ts_i,
                          const int64_t ts_j,
                          imu_buffer_t *imu_buf);

/////////////////////
// SIM CAMERA DATA //
/////////////////////

/** Simulation Utils **/
void sim_create_features(const real_t origin[3],
                         const real_t dim[3],
                         const int num_features,
                         real_t *features);

/** Sim Camera Frame **/
typedef struct sim_camera_frame_t {
  timestamp_t ts;
  int cam_idx;
  size_t *feature_ids;
  real_t *keypoints;
  int n;
} sim_camera_frame_t;

void sim_camera_frame_setup(sim_camera_frame_t *frame,
                            const timestamp_t ts,
                            const int cam_idx);
sim_camera_frame_t *sim_camera_frame_malloc(const timestamp_t ts,
                                            const int cam_idx);
void sim_camera_frame_free(sim_camera_frame_t *frame_data);
void sim_camera_frame_add_keypoint(sim_camera_frame_t *frame_data,
                                   const size_t feature_id,
                                   const real_t kp[2]);
sim_camera_frame_t *sim_camera_frame_load(const char *csv_path);
void sim_camera_frame_print(const sim_camera_frame_t *frame_data);

/** Sim Camera Data **/
typedef struct sim_camera_data_t {
  int cam_idx;
  sim_camera_frame_t **frames;
  int num_frames;

  timestamp_t *timestamps;
  real_t *poses;
} sim_camera_data_t;

void sim_camera_data_setup(sim_camera_data_t *data);
sim_camera_data_t *sim_camerea_data_malloc(void); void sim_camera_data_free(sim_camera_data_t *cam_data);
sim_camera_data_t *sim_camera_data_load(const char *dir_path);

sim_camera_data_t *
sim_camera_circle_trajectory(const sim_circle_t *conf,
                             const real_t T_BC[4 * 4],
                             const camera_params_t *cam_params,
                             const real_t *features,
                             const int num_features);

/////////////////////////
// SIM CAMERA IMU DATA //
/////////////////////////

/** Sim Circle Camera-IMU Data **/
typedef struct sim_circle_camera_imu_t {
  sim_circle_t conf;
  sim_imu_data_t *imu_data;
  sim_camera_data_t *cam0_data;
  sim_camera_data_t *cam1_data;

  real_t feature_data[3 * 1000];
  int num_features;

  camera_params_t cam0_params;
  camera_params_t cam1_params;
  real_t cam0_ext[7];
  real_t cam1_ext[7];
  real_t imu0_ext[7];

  timeline_t *timeline;
} sim_circle_camera_imu_t;

sim_circle_camera_imu_t *sim_circle_camera_imu(void); void sim_circle_camera_imu_free(sim_circle_camera_imu_t *sim_data);

/////////////////////
// SIM GIMBAL DATA //
/////////////////////

typedef struct sim_gimbal_view_t {
  int num_measurements;
  int *tag_ids;
  int *corner_indices;
  real_t *object_points;
  real_t *keypoints;
} sim_gimbal_view_t;

typedef struct sim_gimbal_t {
  aprilgrid_t *grid;

  int num_links;
  int num_joints;
  int num_cams;

  fiducial_t fiducial_ext;
  pose_t gimbal_pose;
  extrinsic_t gimbal_ext;
  extrinsic_t *gimbal_links;
  joint_t *gimbal_joints;
  extrinsic_t *cam_exts;
  camera_params_t *cam_params;
} sim_gimbal_t;

sim_gimbal_view_t *sim_gimbal_view_malloc(const int max_corners);
void sim_gimbal_view_free(sim_gimbal_view_t *view);
void sim_gimbal_view_print(const sim_gimbal_view_t *view);

sim_gimbal_t *sim_gimbal_malloc(void); void sim_gimbal_free(sim_gimbal_t *sim);
void sim_gimbal_print(const sim_gimbal_t *sim);
void sim_gimbal_set_joint(sim_gimbal_t *sim,
                          const int joint_idx,
                          const real_t angle);
void sim_gimbal_get_joints(sim_gimbal_t *sim,
                           const int num_joints,
                           real_t *angles);
sim_gimbal_view_t *sim_gimbal3_view(const aprilgrid_t *grid,
                                    const timestamp_t ts,
                                    const int view_idx,
                                    const real_t fiducial_pose[7],
                                    const real_t body_pose[7],
                                    const real_t gimbal_ext[7],
                                    const real_t gimbal_link0[7],
                                    const real_t gimbal_link1[7],
                                    const real_t gimbal_joint0,
                                    const real_t gimbal_joint1,
                                    const real_t gimbal_joint2,
                                    const int cam_idx,
                                    const int cam_res[2],
                                    const real_t cam_params[8],
                                    const real_t cam_ext[7]);
sim_gimbal_view_t *sim_gimbal_view(const sim_gimbal_t *sim,
                                   const timestamp_t ts,
                                   const int view_idx,
                                   const int cam_idx,
                                   const real_t body_pose[7]);

#ifdef __cplusplus
} // extern C
#endif
#endif // PROTO_H
