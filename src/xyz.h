#pragma once

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdarg.h>
#include <inttypes.h>
#include <time.h>
#include <float.h>
#include <math.h>
#include <string.h>
#include <dirent.h>
#include <ftw.h>
#include <unistd.h>
#include <errno.h>
#include <termios.h>
#include <poll.h>
#include <execinfo.h>

#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>

#include <yaml.h>
#include <cblas.h>
#include <suitesparse/cholmod.h>

#define PRECISION 2
#define MAX_LINE_LENGTH 9046
#define USE_LAPACK

/******************************************************************************
 * MACROS
 *****************************************************************************/

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

/*******************************************************************************
 * LOGGING
 ******************************************************************************/

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

/*******************************************************************************
 * SYSTEM
 ******************************************************************************/

void print_stacktrace(void);

/*******************************************************************************
 * DATA
 ******************************************************************************/

char wait_key(int delay);

size_t string_copy(char *dst, const char *src);
void string_subcopy(char *dst, const char *src, const int s, const int n);
void string_cat(char *dst, const char *src);
char *string_malloc(const char *s);
char *string_strip(char *s);
char *string_strip_char(char *s, const char c);
char **string_split(char *s, const char d, size_t *n);

int **load_iarrays(const char *csv_path, int *num_arrays);
double **load_darrays(const char *csv_path, int *num_arrays);

int *int_malloc(const int val);
float *float_malloc(const float val);
double *double_malloc(const double val);
double *vector_malloc(const double *vec, const double N);

int dsv_rows(const char *fp);
int dsv_cols(const char *fp, const char delim);
char **dsv_fields(const char *fp, const char delim, int *num_fields);
double **
dsv_data(const char *fp, const char delim, int *num_rows, int *num_cols);
void dsv_free(double **data, const int num_rows);

double **csv_data(const char *fp, int *num_rows, int *num_cols);
void csv_free(double **data, const int num_rows);

void path_file_name(const char *path, char *fname);
void path_file_ext(const char *path, char *fext);
void path_dir_name(const char *path, char *dir_name);
char *path_join(const char *x, const char *y);
char **list_files(const char *path, int *num_files);
void list_files_free(char **data, const int n);
int mkdir_p(const char *path, const mode_t mode);
int rmdir(const char *path);

size_t file_lines(const char *fp);
char *file_read(const char *fp);
void skip_line(FILE *fp);
status_t file_exists(const char *fp);
status_t file_rows(const char *fp);
status_t file_copy(const char *src, const char *dest);

/*******************************************************************************
 * TIME
 ******************************************************************************/

// Timestamp Type
#ifndef TIMESTAMP_TYPE
#define TIMESTAMP_TYPE
typedef int64_t timestamp_t;
#endif

timestamp_t *timestamp_malloc(timestamp_t ts);
void timestamp_free(timestamp_t *ts_ptr);

void tic(void);
double toc(void);
double mtoc(void);
timestamp_t time_now(void);

timestamp_t str2ts(const char *ts_str);
double ts2sec(const timestamp_t ts);
timestamp_t sec2ts(const double time_s);
timestamp_t path2ts(const char *file_path);

/*******************************************************************************
 * ARRAY
 ******************************************************************************/

typedef struct arr_t {
  void **data;
  size_t size;
  size_t capacity;
} arr_t;

arr_t *arr_malloc(const size_t capacity);
void arr_free(arr_t *arr);
void arr_push_back(arr_t *arr, void *data);

/*******************************************************************************
 * DARRAY
 ******************************************************************************/

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

/*******************************************************************************
 * LIST
 ******************************************************************************/

typedef struct list_node_t list_node_t;
struct list_node_t {
  list_node_t *next;
  list_node_t *prev;
  void *value;
};

typedef struct list_t {
  size_t length;
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

/*******************************************************************************
 * RED-BLACK-TREE
 ******************************************************************************/

#define RB_RED 1
#define RB_BLACK 0

typedef void *(*copy_func_t)(const void *);
typedef void (*free_func_t)(void *);
typedef int (*cmp_t)(const void *, const void *);
int default_cmp(const void *x, const void *y);
int int_cmp(const void *x, const void *y);
int float_cmp(const void *x, const void *y);
int double_cmp(const void *x, const void *y);
int string_cmp(const void *x, const void *y);

typedef struct rbt_node_t rbt_node_t;
struct rbt_node_t {
  bool color;
  void *key;
  void *value;
  rbt_node_t *child[2];
  size_t size;
};

typedef struct rbt_t {
  rbt_node_t *root;
  cmp_t cmp;
  copy_func_t kcopy;
  free_func_t kfree;
  size_t size;
} rbt_t;

rbt_node_t *rbt_node_malloc(const int color, void *key, void *value);
void rbt_node_free(rbt_node_t *n);
bool rbt_node_is_red(const rbt_node_t *n);
rbt_node_t *rbt_node_min(rbt_node_t *n);
rbt_node_t *rbt_node_max(rbt_node_t *n);
size_t rbt_node_height(const rbt_node_t *n);
size_t rbt_node_size(const rbt_node_t *n);
void rbt_node_keys(const rbt_node_t *n,
                   const void *lo,
                   const void *hi,
                   arr_t *keys,
                   cmp_t cmp);
void rbt_node_keys_values(const rbt_node_t *n,
                          const void *lo,
                          const void *hi,
                          arr_t *keys,
                          arr_t *values,
                          cmp_t cmp);
int rbt_node_rank(const rbt_node_t *n, const void *key, cmp_t cmp);
void *rbt_node_select(const rbt_node_t *n, const int rank);
void rbt_node_flip_colors(rbt_node_t *n);
rbt_node_t *rbt_node_rotate(rbt_node_t *n, const bool dir);
rbt_node_t *rbt_node_move_red_left(rbt_node_t *n);
rbt_node_t *rbt_node_move_red_right(rbt_node_t *n);
rbt_node_t *rbt_node_balance(rbt_node_t *n);
bool rbt_node_bst_check(const rbt_node_t *n, void *min, void *max, cmp_t cmp);
bool rbt_node_size_check(const rbt_node_t *n);
bool rbt_node_23_check(const rbt_node_t *root, rbt_node_t *n);
bool rbt_node_balance_check(rbt_node_t *n);
bool rbt_node_check(rbt_node_t *root, cmp_t cmp);
rbt_node_t *rbt_node_insert(rbt_node_t *n, void *key, void *value, cmp_t cmp);
rbt_node_t *rbt_node_delete_min(rbt_node_t *n);
rbt_node_t *rbt_node_delete_max(rbt_node_t *n);
rbt_node_t *
rbt_node_delete(rbt_node_t *n, void *key, cmp_t cmp, free_func_t kfree);
void *rbt_node_search(rbt_node_t *rbt, const void *key, cmp_t cmp);
bool rbt_node_contains(const rbt_node_t *rbt, const void *key, cmp_t cmp);

rbt_t *rbt_malloc(cmp_t cmp);
void rbt_free(rbt_t *rbt);
void rbt_insert(rbt_t *rbt, void *key, void *value);
void rbt_delete(rbt_t *rbt, void *key);
void *rbt_search(rbt_t *rbt, const void *key);
bool rbt_contains(const rbt_t *rbt, const void *key);
rbt_node_t *rbt_min(const rbt_t *rbt);
rbt_node_t *rbt_max(const rbt_t *rbt);
size_t rbt_height(const rbt_t *rbt);
size_t rbt_size(const rbt_t *rbt);
void rbt_keys(const rbt_t *rbt, arr_t *keys);
void rbt_keys_values(const rbt_t *rbt, arr_t *keys, arr_t *values);
int rbt_rank(const rbt_t *rbt, const void *key);
void *rbt_select(const rbt_t *rbt, const int rank);

/*******************************************************************************
 * HASHMAP
 ******************************************************************************/

typedef struct hm_entry_t {
  void *key;
  void *value;
  size_t key_size;
} hm_entry_t;

typedef struct hm_t {
  hm_entry_t *entries;
  size_t length;
  size_t capacity;
  size_t (*hash)(const void *);
  int (*cmp)(const void *, const void *);
} hm_t;

typedef struct hm_iter_t {
  void *key;
  void *value;
  hm_t *_hm;
  size_t _index;
} hm_iter_t;

size_t hm_default_hash(const void *key, const size_t key_size);
size_t hm_int_hash(const void *key);
size_t hm_float_hash(const void *key);
size_t hm_double_hash(const void *key);
size_t hm_string_hash(const void *key);

hm_t *hm_malloc(const size_t capacity,
                size_t (*hash)(const void *),
                int (*cmp)(const void *, const void *));
void hm_free(hm_t *hm, void (*free_key)(void *), void (*free_value)(void *));
void *hm_get(const hm_t *hm, const void *key);
int hm_expand(hm_t *hm);
int hm_set(hm_t *hm, void *key, void *value);
hm_iter_t hm_iterator(hm_t *hm);
int hm_next(hm_iter_t *it);

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
 * MATH
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

/*******************************************************************************
 * LINEAR ALGEBRA
 ******************************************************************************/

void print_matrix(const char *prefix,
                  const real_t *A,
                  const size_t m,
                  const size_t n);
void print_vector(const char *prefix, const real_t *v, const size_t n);
void print_float_array(const char *prefix, const float *arr, const size_t n);
void print_double_array(const char *prefix, const double *arr, const size_t n);
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

void eyef(float *A, const size_t m, const size_t n);
void onesf(float *A, const size_t m, const size_t n);
void zerosf(float *A, const size_t m, const size_t n);
void hatf(const float x[3], float A[3 * 3]);
void veef(const float A[3 * 3], float x[3]);
void fwdsubsf(const float *L, const float *b, float *y, const size_t n);
void bwdsubsf(const float *U, const float *y, float *x, const size_t n);
void enforce_spdf(float *A, const int m, const int n);

real_t *mat_malloc(const size_t m, const size_t n);
int mat_cmp(const real_t *A, const real_t *B, const size_t m, const size_t n);
int mat_equals(const real_t *A,
               const real_t *B,
               const size_t m,
               const size_t n,
               const real_t tol);
// int mat_save(const char *save_path, const real_t *A, const int m, const int n);
// real_t *mat_load(const char *save_path, int *num_rows, int *num_cols);
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
// real_t *vec_load(const char *save_path, int *num_rows, int *num_cols);
void vec_add(const real_t *x, const real_t *y, real_t *z, size_t n);
void vec_sub(const real_t *x, const real_t *y, real_t *z, size_t n);
void vec_scale(real_t *x, const size_t n, const real_t scale);
real_t vec_norm(const real_t *x, const size_t n);
void vec_normalize(real_t *x, const size_t n);

void vec3_copy(const real_t src[3], real_t dst[3]);
void vec3_add(const real_t a[3], const real_t b[3], real_t c[3]);
void vec3_sub(const real_t a[3], const real_t b[3], real_t c[3]);
void vec3_scale(const real_t a[3], const real_t s, real_t b[3]);
real_t vec3_dot(const real_t a[3], const real_t b[3]);
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
void dotf(const float *A,
          const size_t A_m,
          const size_t A_n,
          const float *B,
          const size_t B_m,
          const size_t B_n,
          float *C);
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

#define VEC_ADD(X, Y, Z, N)                                                    \
  real_t Z[N] = {0};                                                           \
  vec_add(X, Y, Z, N);

#define VEC_SUB(X, Y, Z, N)                                                    \
  real_t Z[N] = {0};                                                           \
  vec_add(X, Y, Z, N);

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

#define DOTF(A, AM, AN, B, BM, BN, C)                                          \
  float C[AM * BN] = {0};                                                      \
  dotf(A, AM, AN, B, BM, BN, C);

#define HATF(X, X_HAT)                                                         \
  float X_HAT[3 * 3] = {0};                                                    \
  hatf(X, X_HAT);

#define VEEF(A, X)                                                             \
  float X[3] = {0};                                                            \
  veef(A, X);

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
    real_t *param_copy = malloc(sizeof(real_t) * p_size);                      \
    real_t *r_fwd = malloc(sizeof(real_t) * r_size);                           \
    real_t *r_bwd = malloc(sizeof(real_t) * r_size);                           \
    real_t *r_diff = malloc(sizeof(real_t) * r_size);                          \
    real_t *J_fdiff = malloc(sizeof(real_t) * r_size * J_cols);                \
    real_t *J = malloc(sizeof(real_t) * r_size * J_cols);                      \
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
    real_t *r = malloc(sizeof(real_t) * r_size);                               \
    real_t *r_fwd = malloc(sizeof(real_t) * r_size);                           \
    real_t *r_diff = malloc(sizeof(real_t) * r_size);                          \
    real_t *J_fdiff = malloc(sizeof(real_t) * r_size * J_cols);                \
    real_t *J = malloc(sizeof(real_t) * r_size * J_cols);                      \
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

int schur_complement(const real_t *H,
                     const real_t *b,
                     const int H_size,
                     const int m,
                     const int r,
                     real_t *H_marg,
                     real_t *b_marg);

int shannon_entropy(const real_t *covar, const int m, real_t *entropy);

/*******************************************************************************
 * SUITE-SPARSE
 ******************************************************************************/

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
real_t rot_diff(const real_t R0[3 * 3], const real_t R1[3 * 3]);
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
 * GNUPLOT
 ******************************************************************************/

FILE *gnuplot_init(void);
void gnuplot_close(FILE *pipe);
void gnuplot_multiplot(FILE *pipe, const int num_rows, const int num_cols);
void gnuplot_send(FILE *pipe, const char *cmd);
void gnuplot_xrange(FILE *pipe, const double xmin, const double xmax);
void gnuplot_yrange(FILE *pipe, const double ymin, const double ymax);
void gnuplot_send_xy(FILE *pipe,
                     const char *data_name,
                     const double *xvals,
                     const double *yvals,
                     const int n);
void gnuplot_send_matrix(FILE *pipe,
                         const char *data_name,
                         const double *A,
                         const int m,
                         const int n);
void gnuplot_matshow(const double *A, const int m, const int n);

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

/******************************************************************************
 * MAV
 *****************************************************************************/

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

/******************************************************************************
 * COMPUTER-VISION
 *****************************************************************************/

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

int solvepnp(const real_t proj_params[4],
             const real_t *img_pts,
             const real_t *obj_pts,
             const int N,
             real_t T_CO[4 * 4]);

/*******************************************************************************
 * APRILGRID
 ******************************************************************************/

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#include "apriltag/apriltag.h"
#include "apriltag/common/image_u8.h"
#include "apriltag/common/pjpeg.h"
#include "apriltag/tag36h11.h"
#include "apriltag/tagStandard41h12.h"
#pragma GCC diagnostic pop

#ifndef APRILGRID_LOG
#define APRILGRID_LOG(...) printf(__VA_ARGS__);
#endif

#ifndef APRILGRID_FATAL
#define APRILGRID_FATAL(...)                                                   \
  printf(__VA_ARGS__);                                                         \
  exit(-1);
#endif

#ifndef APRILGRID_UNUSED
#define APRILGRID_UNUSED(expr)                                                 \
  do {                                                                         \
    (void) (expr);                                                             \
  } while (0)
#endif

#ifndef APRILGRID_CHECK
#define APRILGRID_CHECK(X)                                                     \
  if (!(X)) {                                                                  \
    APRILGRID_LOG(#X " Failed!\n");                                            \
    goto error;                                                                \
  }
#endif

// APRILGRID /////////////////////////////////////////////////////////////////

typedef struct aprilgrid_t {
  // Grid properties
  int num_rows;
  int num_cols;
  real_t tag_size;
  real_t tag_spacing;

  // Grid data
  timestamp_t timestamp;
  int corners_detected;
  real_t *data;
} aprilgrid_t;

aprilgrid_t *aprilgrid_malloc(const int num_rows,
                              const int num_cols,
                              const real_t tag_size,
                              const real_t tag_spacing);
void aprilgrid_free(aprilgrid_t *grid);
void aprilgrid_clear(aprilgrid_t *grid);
void aprilgrid_reset(aprilgrid_t *grid);
void aprilgrid_copy(const aprilgrid_t *src, aprilgrid_t *dst);
int aprilgrid_equals(const aprilgrid_t *grid0, const aprilgrid_t *grid1);
void aprilgrid_center(const aprilgrid_t *grid, real_t *cx, real_t *cy);
void aprilgrid_grid_index(const aprilgrid_t *grid,
                          const int tag_id,
                          int *i,
                          int *j);
void aprilgrid_object_point(const aprilgrid_t *grid,
                            const int tag_id,
                            const int corner_idx,
                            real_t object_point[3]);
void aprilgrid_add_corner(aprilgrid_t *grid,
                          const int tag_id,
                          const int corner_idx,
                          const real_t kp[2]);
void aprilgrid_remove_corner(aprilgrid_t *grid,
                             const int tag_id,
                             const int corner_idx);
void aprilgrid_add_tag(aprilgrid_t *grid,
                       const int tag_id,
                       const real_t kp[4][2]);
void aprilgrid_remove_tag(aprilgrid_t *grid, const int tag_id);
void aprilgrid_measurements(const aprilgrid_t *grid,
                            int *tag_ids,
                            int *corner_idxs,
                            real_t *tag_kps,
                            real_t *obj_pts);
int aprilgrid_save(const aprilgrid_t *grid, const char *save_path);
aprilgrid_t *aprilgrid_load(const char *data_path);

// APRILGRID DETECTOR ////////////////////////////////////////////////////////

typedef struct aprilgrid_detector_t {
  apriltag_family_t *tf;
  apriltag_detector_t *td;

  int num_rows;
  int num_cols;
  real_t tag_size;
  real_t tag_spacing;
} aprilgrid_detector_t;

aprilgrid_detector_t *aprilgrid_detector_malloc(int num_rows,
                                                int num_cols,
                                                real_t tag_size,
                                                real_t tag_spacing);
void aprilgrid_detector_free(aprilgrid_detector_t *det);
aprilgrid_t *aprilgrid_detector_detect(const aprilgrid_detector_t *det,
                                       const timestamp_t ts,
                                       const int32_t image_width,
                                       const int32_t image_height,
                                       const int32_t image_stride,
                                       uint8_t *image_data);

/*******************************************************************************
 * MORTON CODES
 ******************************************************************************/

uint32_t part1by1(uint32_t x);
uint32_t part1by2(uint32_t x);
uint32_t compact1by1(uint32_t x);
uint32_t compact1by2(uint32_t x);
uint32_t morton_encode_2d(uint32_t x, uint32_t y);
uint32_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z);
void morton_decode_2d(uint32_t code, uint32_t *x, uint32_t *y);
void morton_decode_3d(uint32_t code, uint32_t *x, uint32_t *y, uint32_t *z);

/*******************************************************************************
 * PLANE
 ******************************************************************************/

typedef struct plane_t {
  real_t normal[3];
  real_t p[3];
  real_t d;
} plane_t;

void plane_setup(plane_t *plane,
                 const real_t normal[3],
                 const real_t p[3],
                 const real_t d);
void plane_vector(const plane_t *plane, real_t v[4]);
void plane_set_transform(plane_t *plane, const real_t T[4 * 4]);
void plane_get_transform(const plane_t *plane,
                         const real_t world_up[3],
                         real_t T[4 * 4]);
real_t plane_point_dist(const plane_t *plane, const real_t p[3]);

/*******************************************************************************
 * FRUSTUM
 ******************************************************************************/

typedef struct frustum_t {
  real_t hfov;
  real_t aspect;
  real_t znear;
  real_t zfar;

  plane_t near;
  plane_t far;
  plane_t left;
  plane_t right;
  plane_t top;
  plane_t bottom;
} frustum_t;

void frustum_setup(frustum_t *frustum,
                   const real_t hfov,
                   const real_t aspect,
                   const real_t znear,
                   const real_t zfar);
bool frustum_check_point(const frustum_t *frustum, const real_t p[3]);

/*******************************************************************************
 * POINT CLOUD
 ******************************************************************************/

void umeyama(const float *X,
             const float *Y,
             const size_t n,
             real_t scale[1],
             real_t R[3 * 3],
             real_t t[3]);

/*******************************************************************************
 * VOXEL
 ******************************************************************************/

#define VOXEL_MAX_POINTS 100

typedef struct voxel_t {
  int32_t key[3];
  float *points;
  size_t length;
} voxel_t;

void voxel_setup(voxel_t *voxel, const int32_t key[3]);
voxel_t *voxel_malloc(const int32_t key[3]);
void voxel_free(voxel_t *voxel);
void voxel_reset(voxel_t *voxel);
void voxel_print(voxel_t *voxel);
void voxel_copy(const voxel_t *src, voxel_t *dst);
void voxel_add(voxel_t *voxel, const float p[3]);

float *voxel_grid_downsample(const float *points,
                             const int num_points,
                             const float voxel_size,
                             size_t *output_count);

/*******************************************************************************
 * OCTREE
 ******************************************************************************/

/////////////////
// OCTREE NODE //
/////////////////

typedef struct octree_node_t {
  float center[3];
  float size;
  int depth;
  int max_depth;
  int max_points;

  struct octree_node_t *children[8];
  float *points;
  size_t num_points;
  size_t capacity;
} octree_node_t;

octree_node_t *octree_node_malloc(const float center[3],
                                  const float size,
                                  const int depth,
                                  const int max_depth,
                                  const int max_points);
void octree_node_free(octree_node_t *node);
bool octree_node_check_point(const octree_node_t *node, const float point[3]);

////////////
// OCTREE //
////////////

typedef struct octree_data_t {
  float *points;
  size_t num_points;
  size_t capacity;
} octree_data_t;

typedef struct octree_t {
  float center[3];
  float map_size;
  octree_node_t *root;
} octree_t;

octree_t *octree_malloc(const float octree_center[3],
                        const float map_size,
                        const int octree_max_depth,
                        const int voxel_max_points,
                        const float *octree_points,
                        const size_t num_points);
void octree_free(octree_t *octree);
void octree_add_point(octree_node_t *node, const float point[3]);
void octree_get_points(const octree_node_t *node, octree_data_t *data);
float *octree_downsample(const float *octree_points,
                         const size_t n,
                         const float voxel_size,
                         const size_t voxel_limit,
                         size_t *n_out);

/*****************************************************************************
 * KD-TREE
 ****************************************************************************/

#define KDTREE_KDIM 3

//////////////////
// KD-TREE NODE //
//////////////////

typedef struct kdtree_node_t {
  float p[3];
  int k;
  struct kdtree_node_t *left;
  struct kdtree_node_t *right;
} kdtree_node_t;

kdtree_node_t *kdtree_node_malloc(const float p[3], const int k);
void kdtree_node_free(kdtree_node_t *node);

/////////////
// KD-TREE //
/////////////

typedef struct kdtree_data_t {
  float *points;
  size_t size;
  size_t capacity;
} kdtree_data_t;

typedef struct kdtree_t {
  kdtree_node_t *root;
} kdtree_t;

kdtree_node_t *kdtree_insert(kdtree_node_t *node,
                             const float p[3],
                             const int depth);

kdtree_t *kdtree_malloc(float *points, size_t num_points);
void kdtree_free(kdtree_t *kdtree);
void kdtree_points(const kdtree_t *kdtree, kdtree_data_t *data);
void kdtree_nn(const kdtree_t *kdtree,
               const float target[3],
               float *best_point,
               float *best_dist);
kdtree_data_t *kdtree_nns(const kdtree_t *kdtree,
                          const float *query_points,
                          const size_t n);

/*******************************************************************************
 * STATE-ESTIMATION
 ******************************************************************************/

//////////////
// FIDUCIAL //
//////////////

/** Fiducial Info **/
typedef struct fiducial_info_t {
  timestamp_t ts;
  int cam_idx;
  int num_corners;
  int capacity;
  int *tag_ids;
  int *corner_indices;
  real_t *pts;
  real_t *kps;
} fiducial_info_t;

fiducial_info_t *fiducial_info_malloc(const timestamp_t ts, const int cam_idx);
void fiducial_info_free(fiducial_info_t *finfo);
void fiducial_info_print(const fiducial_info_t *finfo);
void fiducial_info_add(fiducial_info_t *finfo,
                       const int tag_id,
                       const int corner_index,
                       const real_t p[3],
                       const real_t z[2]);

////////////
// CAMERA //
////////////

typedef struct camera_t {
  int cam_idx;
  int resolution[2];
  char proj_model[30];
  char dist_model[30];
  real_t data[8];

  project_func_t proj_func;
  back_project_func_t back_proj_func;
  undistort_func_t undistort_func;
} camera_t;

void camera_setup(camera_t *camera,
                  const int cam_idx,
                  const int cam_res[2],
                  const char *proj_model,
                  const char *dist_model,
                  const real_t *data);
void camera_copy(const camera_t *src, camera_t *dst);
void camera_fprint(const camera_t *cam, FILE *f);
void camera_print(const camera_t *camera);
void camera_project(const camera_t *camera, const real_t p_C[3], real_t z[2]);
void camera_back_project(const camera_t *camera,
                         const real_t z[2],
                         real_t bearing[3]);
void camera_undistort_points(const camera_t *camera,
                             const real_t *kps,
                             const int num_points,
                             real_t *kps_und);
int solvepnp_camera(const camera_t *cam_params,
                    const real_t *img_pts,
                    const real_t *obj_pts,
                    const int N,
                    real_t T_CO[4 * 4]);
void triangulate_batch(const camera_t *cam_i,
                       const camera_t *cam_j,
                       const real_t T_CiCj[4 * 4],
                       const real_t *kps_i,
                       const real_t *kps_j,
                       const int n,
                       real_t *points,
                       int *status);
void stereo_triangulate(const camera_t *cam_i,
                        const camera_t *cam_j,
                        const real_t T_WCi[4 * 4],
                        const real_t T_CiCj[4 * 4],
                        const real_t *kps_i,
                        const real_t *kps_j,
                        const int n,
                        real_t *points,
                        int *status);

////////////////
// IMU-PARAMS //
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

////////////////
// IMU-BUFFER //
////////////////

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

///////////
// POINT //
///////////

typedef struct point_t {
  real_t x;
  real_t y;
  real_t z;
} point_t;

////////////////
// PARAMETERS //
////////////////

#define POSITION_PARAM 1
#define ROTATION_PARAM 2
#define POSE_PARAM 3
#define EXTRINSIC_PARAM 4
#define FIDUCIAL_PARAM 5
#define VELOCITY_PARAM 6
#define IMU_BIASES_PARAM 7
#define FEATURE_PARAM 8
#define JOINT_PARAM 9
#define CAMERA_PARAM 10
#define TIME_DELAY_PARAM 11

typedef struct param_info_t {
  real_t *data;
  int idx;
  int type;
  int fix;
} param_info_t;

void param_type_string(const int param_type, char *s);
size_t param_global_size(const int param_type);
size_t param_local_size(const int param_type);

rbt_t *param_index_malloc(void);
void param_index_free(rbt_t *param_index);
void param_index_print(const rbt_t *param_index);
bool param_index_exists(rbt_t *param_index, real_t *key);
void param_index_add(rbt_t *param_index,
                     const int param_type,
                     const int fix,
                     real_t *data,
                     int *col_idx);

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
  real_t *pose_est;

  real_t covar[6 * 6];
  real_t sqrt_info[6 * 6];

  real_t r[6];
  int r_size;

  int param_types[1];
  real_t *params[1];
  int num_params;

  real_t *jacs[1];
  real_t J_pose[6 * 6];
} pose_factor_t;

void pose_factor_setup(pose_factor_t *factor,
                       real_t *pose,
                       const real_t var[6]);
int pose_factor_eval(void *factor);

///////////////
// BA FACTOR //
///////////////

typedef struct ba_factor_t {
  real_t *pose;
  real_t *feature;
  camera_t *camera;

  real_t covar[2 * 2];
  real_t sqrt_info[2 * 2];
  real_t z[2];

  real_t r[2];
  int r_size;

  int param_types[3];
  real_t *params[3];
  int num_params;

  real_t *jacs[3];
  real_t J_pose[2 * 6];
  real_t J_feature[2 * 3];
  real_t J_camera[2 * 8];
} ba_factor_t;

void ba_factor_setup(ba_factor_t *factor,
                     real_t *pose,
                     real_t *feature,
                     camera_t *camera,
                     const real_t z[2],
                     const real_t var[2]);
int ba_factor_eval(void *factor_ptr);

///////////////////
// CAMERA FACTOR //
///////////////////

typedef struct camera_factor_t {
  real_t *pose;
  real_t *extrinsic;
  camera_t *camera;
  real_t *feature;

  real_t covar[2 * 2];
  real_t sqrt_info[2 * 2];
  real_t z[2];

  real_t r[2];
  int r_size;

  int num_params;
  int param_types[4];
  real_t *params[4];

  real_t *jacs[4];
  real_t J_pose[2 * 6];
  real_t J_extrinsic[2 * 6];
  real_t J_feature[2 * 3];
  real_t J_camera[2 * 8];
} camera_factor_t;

void camera_factor_setup(camera_factor_t *factor,
                         real_t *pose,
                         real_t *extrinsic,
                         real_t *feature,
                         camera_t *camera,
                         const real_t z[2],
                         const real_t var[2]);
int camera_factor_eval(void *factor_ptr);

////////////////
// IMU FACTOR //
////////////////

/** IMU Factor **/
typedef struct imu_factor_t {
  // IMU parameters and buffer
  const imu_params_t *imu_params;
  imu_buffer_t imu_buf;

  // Parameters
  timestamp_t ts_i;
  real_t *pose_i;
  real_t *vel_i;
  real_t *biases_i;

  timestamp_t ts_j;
  real_t *pose_j;
  real_t *vel_j;
  real_t *biases_j;

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
                      const timestamp_t ts_i,
                      const timestamp_t ts_j,
                      real_t *pose_i,
                      real_t *v_i,
                      real_t *biases_i,
                      real_t *pose_j,
                      real_t *v_j,
                      real_t *biases_j);
void imu_factor_reset(imu_factor_t *factor);
void imu_factor_preintegrate(imu_factor_t *factor);
int imu_factor_residuals(imu_factor_t *factor, real_t **params, real_t *r_out);
int imu_factor_eval(void *factor_ptr);
int imu_factor_ceres_eval(void *factor_ptr,
                          real_t **params,
                          real_t *r_out,
                          real_t **J_out);

//////////////////
// LIDAR FACTOR //
//////////////////

typedef struct pcd_t {
  timestamp_t ts_start;
  timestamp_t ts_end;
  float *data;
  float *time_diffs;
  size_t num_points;
  kdtree_t *kdtree;
} pcd_t;

pcd_t *pcd_malloc(const timestamp_t ts_start,
                  const timestamp_t ts_end,
                  const float *data,
                  const float *time_diffs,
                  const size_t num_points);
void pcd_free(pcd_t *pcd);
void pcd_deskew(pcd_t *points,
                const real_t T_WL_km1[4 * 4],
                const real_t T_WL_km2[4 * 4]);

typedef struct lidar_factor_t {
  pcd_t *pcd;
  kdtree_t *kdtree;

  real_t *pose;
  real_t *extrinsic;

  real_t covar[3 * 3];
  real_t sqrt_info[3 * 3];

  real_t *r;
  int r_size;

  int param_types[2];
  real_t *params[2];
  int num_params;

  real_t *jacs[1];
  real_t *J_pose;
} lidar_factor_t;

// void lidar_factor_setup(lidar_factor_t *factor,
//                         pcd_t *pcd,
//                         pose_t *pose_k,
//                         const real_t var[3]);
// void lidar_factor_eval(void *factor);

////////////////////////
// JOINT-ANGLE FACTOR //
////////////////////////

typedef struct joint_factor_t {
  real_t *joint;

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
                        real_t *joint0,
                        const real_t z,
                        const real_t var);
void joint_factor_copy(const joint_factor_t *src, joint_factor_t *dst);
int joint_factor_eval(void *factor_ptr);
int joint_factor_equals(const joint_factor_t *j0, const joint_factor_t *j1);

//////////////
// CAMCHAIN //
//////////////

typedef struct {
  int analyzed;
  int num_cams;

  int **adj_list;
  real_t **adj_exts;
  rbt_t **cam_poses;
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

/////////////////////////
// CALIB-CAMERA FACTOR //
/////////////////////////

typedef struct calib_camera_factor_t {
  real_t *pose;
  real_t *cam_ext;
  camera_t *cam_params;

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
                               real_t *pose,
                               real_t *cam_ext,
                               camera_t *cam_params,
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
  real_t *fiducial;   // fiducial pose: T_WF
  real_t *imu_pose;   // IMU pose: T_WS
  real_t *imu_ext;    // IMU extrinsic: T_SC0
  real_t *cam_ext;    // Camera extrinsic: T_C0Ci
  camera_t *camera;   // Camera parameters
  real_t *time_delay; // Time delay

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
  real_t J_camera[2 * 8];
  real_t J_time_delay[2 * 1];
} calib_imucam_factor_t;

void calib_imucam_factor_setup(calib_imucam_factor_t *factor,
                               real_t *fiducial,
                               real_t *imu_pose,
                               real_t *imu_ext,
                               real_t *cam_ext,
                               camera_t *camera,
                               real_t *time_delay,
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
  list_t *r_positions;
  list_t *r_rotations;
  list_t *r_poses;
  list_t *r_velocities;
  list_t *r_imu_biases;
  list_t *r_fiducials;
  list_t *r_joints;
  list_t *r_extrinsics;
  list_t *r_features;
  list_t *r_cam_params;
  list_t *r_time_delays;
  // -- Marginal parameters
  list_t *m_positions;
  list_t *m_rotations;
  list_t *m_poses;
  list_t *m_velocities;
  list_t *m_imu_biases;
  list_t *m_features;
  list_t *m_fiducials;
  list_t *m_extrinsics;
  list_t *m_joints;
  list_t *m_cam_params;
  list_t *m_time_delays;

  // Marginal pointers
  const rbt_t *marg_params;
  const rbt_t *fix_params;
  size_t num_marg_params;

  // Factors
  list_t *ba_factors;
  list_t *camera_factors;
  list_t *imu_factors;
  list_t *calib_camera_factors;
  list_t *calib_imucam_factors;
  struct marg_factor_t *marg_factor;

  // Hessian, Jacobians and residuals
  rbt_t *param_seen;
  rbt_t *param_index;

  int m_lsize; // Marginal local parameter length
  int r_lsize; // Remain local parameter length
  int m_gsize; // Marginal global parameter length
  int r_gsize; // Remain global parameter length

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

marg_factor_t *marg_factor_malloc(void);
void marg_factor_free(marg_factor_t *marg);
void marg_factor_print_stats(const marg_factor_t *marg);
void marg_factor_add(marg_factor_t *marg, int factor_type, void *factor_ptr);
void marg_factor_marginalize(marg_factor_t *marg,
                             const rbt_t *marg_params,
                             const rbt_t *fix_params);
int marg_factor_eval(void *marg_ptr);

////////////////
// DATA UTILS //
////////////////

int save_poses(const char *save_path,
               const timestamp_t *timestamps,
               const real_t *poses,
               const int num_poses);
int load_poses(const char *data_path,
               timestamp_t **timestamps,
               real_t **poses,
               int *num_poses);
// int **assoc_pose_data(pose_t *gnd_poses,
//                       size_t num_gnd_poses,
//                       pose_t *est_poses,
//                       size_t num_est_poses,
//                       double threshold,
//                       size_t *num_matches);

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
  rbt_t *param_index;
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
  rbt_t *(*param_index_func)(const void *data, int *sv_size, int *r_size);
  void (*cost_func)(const void *data, real_t *r);
  void (*linearize_func)(const void *data,
                         const int sv_size,
                         rbt_t *hash,
                         real_t *H,
                         real_t *g,
                         real_t *r);
  void (*linsolve_func)(const void *data,
                        const int sv_size,
                        rbt_t *hash,
                        real_t *H,
                        real_t *g,
                        real_t *dx);
} solver_t;

void solver_setup(solver_t *solver);
void solver_print_param_order(const solver_t *solver);
real_t solver_cost(const solver_t *solver, const void *data);
void solver_fill_jacobian(rbt_t *param_index,
                          int num_params,
                          real_t **params,
                          real_t **jacs,
                          real_t *r,
                          int r_size,
                          int sv_size,
                          int J_row_idx,
                          real_t *J,
                          real_t *g);
void solver_fill_hessian(rbt_t *param_index,
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

/*******************************************************************************
 * TIMELINE
 ******************************************************************************/

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

timeline_t *timeline_malloc(void);
void timeline_free(timeline_t *timeline);
void timeline_form_timeline(timeline_t *tl);
timeline_t *timeline_load_data(const char *data_dir,
                               const int num_cams,
                               const int num_imus);

/*******************************************************************************
 * SIMULATION
 ******************************************************************************/

////////////////
// TORUS KNOT //
////////////////

void torus_knot(real_t t, int p, int q, real_t R, real_t r, real_t out[3]);
void torus_knot_deriv(real_t t,
                      int p,
                      int q,
                      real_t R,
                      real_t r,
                      real_t out[3]);
float *torus_knot_points(size_t *num_points);
int torus_knot_save(const char *csv_path);

////////////////
// SIM CIRCLE //
////////////////

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

void sim_features_save(sim_features_t *features, const char *csv_path);
sim_features_t *sim_features_load(const char *csv_path);
void sim_features_free(sim_features_t *features);

//////////////////
// SIM IMU DATA //
//////////////////

typedef struct sim_imu_data_t {
  size_t num_measurements;
  timestamp_t *timestamps;
  real_t *poses;
  real_t *velocities;
  real_t *imu_acc;
  real_t *imu_gyr;
} sim_imu_data_t;

void sim_imu_data_setup(sim_imu_data_t *imu_data);
sim_imu_data_t *sim_imu_data_malloc(void);
void sim_imu_data_free(sim_imu_data_t *imu_data);
void sim_imu_data_save(sim_imu_data_t *imu_data, const char *csv_path);
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
  int camera_index;
  size_t *feature_ids;
  real_t *keypoints;
  int n;
} sim_camera_frame_t;

void sim_camera_frame_setup(sim_camera_frame_t *frame,
                            const timestamp_t ts,
                            const int camera_index);
sim_camera_frame_t *sim_camera_frame_malloc(const timestamp_t ts,
                                            const int camera_index);
void sim_camera_frame_free(sim_camera_frame_t *frame_data);
void sim_camera_frame_add_keypoint(sim_camera_frame_t *frame_data,
                                   const size_t feature_id,
                                   const real_t kp[2]);
void sim_camera_frame_save(const sim_camera_frame_t *frame_data,
                           const char *csv_path);
sim_camera_frame_t *sim_camera_frame_load(const char *csv_path);
void sim_camera_frame_print(const sim_camera_frame_t *frame_data);

/** Sim Camera Data **/
typedef struct sim_camera_data_t {
  int camera_index;
  int num_frames;
  timestamp_t *timestamps;
  real_t *poses;
  sim_camera_frame_t **frames;
} sim_camera_data_t;

void sim_camera_data_setup(sim_camera_data_t *data);
sim_camera_data_t *sim_camerea_data_malloc(void);
void sim_camera_data_free(sim_camera_data_t *cam_data);
void sim_camera_data_save(sim_camera_data_t *cam_data, const char *dir_path);
sim_camera_data_t *sim_camera_data_load(const char *dir_path);

sim_camera_data_t *sim_camera_circle_trajectory(const sim_circle_t *conf,
                                                const real_t T_BC[4 * 4],
                                                const camera_t *cam_params,
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

  camera_t cam0_params;
  camera_t cam1_params;
  real_t cam0_ext[7];
  real_t cam1_ext[7];
  real_t imu0_ext[7];

  timeline_t *timeline;
} sim_circle_camera_imu_t;

sim_circle_camera_imu_t *sim_circle_camera_imu(void);
void sim_circle_camera_imu_free(sim_circle_camera_imu_t *sim_data);

/******************************************************************************
 * EUROC
 ******************************************************************************/

/**
 * Fatal
 *
 * @param[in] M Message
 * @param[in] ... Varadic arguments
 */
#ifndef EUROC_FATAL
#define EUROC_FATAL(...)                                                       \
  do {                                                                         \
    fprintf(stderr,                                                            \
            "[EUROC_FATAL] [%s:%d:%s()]: ",                                    \
            __FILE__,                                                          \
            __LINE__,                                                          \
            __func__);                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0);                                                                 \
  exit(-1)
#endif

#ifndef EUROC_LOG
#define EUROC_LOG(...)                                                         \
  do {                                                                         \
    fprintf(stderr,                                                            \
            "[EUROC_LOG] [%s:%d:%s()]: ",                                      \
            __FILE__,                                                          \
            __LINE__,                                                          \
            __func__);                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0);
#endif

/////////////////
// euroc_imu_t //
/////////////////

/**
 * EuRoC IMU data
 */
typedef struct euroc_imu_t {
  // Data
  int num_timestamps;
  timestamp_t *timestamps;
  double **w_B;
  double **a_B;

  // Sensor properties
  char sensor_type[100];
  char comment[9046];
  double T_BS[4 * 4];
  double rate_hz;
  double gyro_noise_density;
  double gyro_random_walk;
  double accel_noise_density;
  double accel_random_walk;
} euroc_imu_t;

euroc_imu_t *euroc_imu_load(const char *data_dir);
void euroc_imu_free(euroc_imu_t *data);
void euroc_imu_print(const euroc_imu_t *data);

////////////////////
// euroc_camera_t //
////////////////////

/**
 * EuRoC camera data
 */
typedef struct euroc_camera_t {
  // Data
  int is_calib_data;
  int num_timestamps;
  timestamp_t *timestamps;
  char **image_paths;

  // Sensor properties
  char sensor_type[100];
  char comment[9046];
  double T_BS[4 * 4];
  double rate_hz;
  int resolution[2];
  char camera_model[100];
  double intrinsics[4];
  char distortion_model[100];
  double distortion_coefficients[4];
} euroc_camera_t;

euroc_camera_t *euroc_camera_load(const char *data_dir, int is_calib_data);
void euroc_camera_free(euroc_camera_t *data);
void euroc_camera_print(const euroc_camera_t *data);

//////////////////////////
// euroc_ground_truth_t //
//////////////////////////

/**
 * EuRoC ground truth
 */
typedef struct euroc_ground_truth_t {
  // Data
  int num_timestamps;
  timestamp_t *timestamps;
  double **p_RS_R;
  double **q_RS;
  double **v_RS_R;
  double **b_w_RS_S;
  double **b_a_RS_S;

} euroc_ground_truth_t;

euroc_ground_truth_t *euroc_ground_truth_load(const char *data_dir);
void euroc_ground_truth_free(euroc_ground_truth_t *data);
void euroc_ground_truth_print(const euroc_ground_truth_t *data);

//////////////////////
// euroc_timeline_t //
//////////////////////

typedef struct euroc_event_t {
  int has_imu0;
  int has_cam0;
  int has_cam1;

  timestamp_t ts;

  size_t imu0_idx;
  double *acc;
  double *gyr;

  size_t cam0_idx;
  char *cam0_image;

  size_t cam1_idx;
  char *cam1_image;
} euroc_event_t;

typedef struct euroc_timeline_t {
  int num_timestamps;
  timestamp_t *timestamps;
  euroc_event_t *events;

} euroc_timeline_t;

euroc_timeline_t *euroc_timeline_create(const euroc_imu_t *imu0_data,
                                        const euroc_camera_t *cam0_data,
                                        const euroc_camera_t *cam1_data);
void euroc_timeline_free(euroc_timeline_t *timeline);

//////////////////
// euroc_data_t //
//////////////////

/**
 * EuRoC data
 */
typedef struct euroc_data_t {
  euroc_imu_t *imu0_data;
  euroc_camera_t *cam0_data;
  euroc_camera_t *cam1_data;
  euroc_ground_truth_t *ground_truth;
  euroc_timeline_t *timeline;
} euroc_data_t;

euroc_data_t *euroc_data_load(const char *data_path);
void euroc_data_free(euroc_data_t *data);

//////////////////////////
// euroc_calib_target_t //
//////////////////////////

/**
 * EuRoC calibration target
 */
typedef struct euroc_calib_target_t {
  char type[100];
  int tag_rows;
  int tag_cols;
  double tag_size;
  double tag_spacing;
} euroc_calib_target_t;

euroc_calib_target_t *euroc_calib_target_load(const char *conf);
void euroc_calib_target_free(euroc_calib_target_t *target);
void euroc_calib_target_print(const euroc_calib_target_t *target);

///////////////////
// euroc_calib_t //
///////////////////

/**
 * EuRoC calibration data
 */
typedef struct euroc_calib_t {
  euroc_imu_t *imu0_data;
  euroc_camera_t *cam0_data;
  euroc_camera_t *cam1_data;
  euroc_calib_target_t *calib_target;
  euroc_timeline_t *timeline;
} euroc_calib_t;

euroc_calib_t *euroc_calib_load(const char *data_path);
void euroc_calib_free(euroc_calib_t *data);

/******************************************************************************
 * KITTI
 ******************************************************************************/
////////////////////
// kitti_camera_t //
////////////////////

typedef struct kitti_camera_t {
  int camera_index;
  int num_timestamps;
  timestamp_t *timestamps;
  char **image_paths;
} kitti_camera_t;

kitti_camera_t *kitti_camera_load(const char *data_dir);
void kitti_camera_free(kitti_camera_t *data);

//////////////////
// kitti_oxts_t //
//////////////////

typedef struct kitti_oxts_t {
  // Timestamps
  int num_timestamps;
  timestamp_t *timestamps;

  // GPS
  double *lat; // Latitude [deg]
  double *lon; // Longitude [deg]
  double *alt; // Altitude [m]

  // Attitude
  double *roll;  // Roll [rad]
  double *pitch; // Pitch [rad]
  double *yaw;   // Yaw [rad]

  // Velocity
  double *vn; // Velocity towards north [m/s]
  double *ve; // Velocity towards east [m/s]
  double *vf; // Forward velocity [m/s]
  double *vl; // Leftward velocity [m/s]
  double *vu; // Upward velocity [m/s]

  // Acceleration
  double *ax; // Acceleration in x [m/s^2]
  double *ay; // Acceleration in y [m/s^2]
  double *az; // Acceleration in z [m/s^2]
  double *af; // Forward acceleration [m/s^2]
  double *al; // Leftward acceleration [m/s^2]
  double *au; // Upward acceleration [m/s^2]

  // Angular velocity
  double *wx; // Angular rate around x [rad/s]
  double *wy; // Angular rate around y [rad/s]
  double *wz; // Angular rate around z [rad/s]
  double *wf; // Angular rate around foward axis [rad/s]
  double *wl; // Angular rate around left axis [rad/s]
  double *wu; // Angular rate around up axis [rad/s]

  // Satellite tracking data
  double *pos_accuracy; // Position accuracy [north / east in m]
  double *vel_accuracy; // Velocity accuracy [north / east in m/s]
  int *navstat;         // Navigation status
  int *numsats;         // Number of satelllites tracked by GPS
  int *posmode;         // Position mode
  int *velmode;         // Velocity mode
  int *orimode;         // Orientation mode
} kitti_oxts_t;

kitti_oxts_t *kitti_oxts_load(const char *data_dir);
void kitti_oxts_free(kitti_oxts_t *data);

//////////////////////
// kitti_velodyne_t //
//////////////////////

typedef struct kitti_velodyne_t {
  int num_timestamps;
  timestamp_t *timestamps;
  timestamp_t *timestamps_start;
  timestamp_t *timestamps_end;
  char **pcd_paths;
} kitti_velodyne_t;

float *kitti_load_points(const char *pcd_path, size_t *num_points);
kitti_velodyne_t *kitti_velodyne_load(const char *data_dir);
void kitti_velodyne_free(kitti_velodyne_t *data);

float *kitti_lidar_xyz(const char *pcd_path,
                       const float voxel_size,
                       size_t *nout);

///////////////////
// kitti_calib_t //
///////////////////

typedef struct kitti_calib_t {
  char calib_time_cam_to_cam[100];
  char calib_time_imu_to_velo[100];
  char calib_time_velo_to_cam[100];
  double corner_dist;

  double S_00[2];       // Image size [pixels]
  double K_00[9];       // Camera 0 intrinsics
  double D_00[5];       // Camera 0 distortion coefficients
  double R_00[9];       // Rotation from camera 0 to camera 0
  double T_00[3];       // Translation from camera 0 to camera 0
  double S_rect_00[2];  // Image size after rectifcation [pixels]
  double R_rect_00[9];  // Rotation after rectification
  double P_rect_00[12]; // Projection matrix after rectification

  double S_01[2];       // Image size [pixels]
  double K_01[9];       // Camera 1 intrinsics
  double D_01[5];       // Camera 1 distortion coefficients
  double R_01[9];       // Rotation from camera 0 to camera 1
  double T_01[3];       // Translation from camera 0 to camera 1
  double S_rect_01[2];  // Image size after rectifcation [pixels]
  double R_rect_01[9];  // Rotation after rectification
  double P_rect_01[12]; // Projection matrix after rectification

  double S_02[2];       // Image size [pixels]
  double K_02[9];       // Camera 2 intrinsics
  double D_02[5];       // Camera 2 distortion coefficients
  double R_02[9];       // Rotation from camera 0 to camera 2
  double T_02[3];       // Translation from camera 0 to camera 2
  double S_rect_02[2];  // Image size after rectifcation [pixels]
  double R_rect_02[9];  // Rotation after rectification
  double P_rect_02[12]; // Projection matrix after rectification

  double S_03[2];       // Image size [pixels]
  double K_03[9];       // Camera 3 intrinsics
  double D_03[5];       // Camera 3 distortion coefficients
  double R_03[9];       // Rotation from camera 0 to camera 3
  double T_03[3];       // Translation from camera 0 to camera 3
  double S_rect_03[2];  // Image size after rectifcation [pixels]
  double R_rect_03[9];  // Rotation after rectification
  double P_rect_03[12]; // Projection matrix after rectification

  double R_velo_imu[9]; // Rotation from imu to velodyne
  double T_velo_imu[3]; // Translation from imu to velodyne

  double R_cam_velo[9]; // Rotation from velodyne to camera
  double T_cam_velo[3]; // Translation from velodyne to camera
  double delta_f[2];
  double delta_c[2];
} kitti_calib_t;

kitti_calib_t *kitti_calib_load(const char *data_dir);
void kitti_calib_free(kitti_calib_t *data);
void kitti_calib_print(const kitti_calib_t *data);

/////////////////
// kitti_raw_t //
/////////////////

typedef struct kitti_raw_t {
   char seq_name[1024];
  kitti_camera_t *image_00;
  kitti_camera_t *image_01;
  kitti_camera_t *image_02;
  kitti_camera_t *image_03;
  kitti_oxts_t *oxts;
  kitti_velodyne_t *velodyne;
  kitti_calib_t *calib;
} kitti_raw_t;

kitti_raw_t *kitti_raw_load(const char *data_dir, const char *seq_name);
void kitti_raw_free(kitti_raw_t *data);


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
