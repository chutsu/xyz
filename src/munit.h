#ifndef MUNIT_H
#define MUNIT_H

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>

/* GLOBAL VARIABLES */
static int num_tests = 0;
static int num_passed = 0;
static int num_failed = 0;
static char *test_target_name = NULL;

/* MUNIT SETTINGS */
#ifndef MU_REDIRECT_STREAMS
#define MU_REDIRECT_STREAMS 0
#endif

#ifndef MU_LOG_DIR
#define MU_LOG_DIR "/tmp"
#endif

#ifndef MU_KEEP_LOGS
#define MU_KEEP_LOGS 1
#endif

#ifndef MU_ENABLE_TERM_COLORS
#define MU_ENABLE_TERM_COLORS 1
#endif

#ifndef MU_ENABLE_PRINT
#define MU_ENABLE_PRINT 1
#endif

/* TERMINAL COLORS */
#if MU_ENABLE_TERM_COLORS == 1
#define MU_RED "\x1B[1;31m"
#define MU_GRN "\x1B[1;32m"
#define MU_WHT "\x1B[1;37m"
#define MU_NRM "\x1B[1;0m"
#else
#define MU_RED
#define MU_GRN
#define MU_WHT
#define MU_NRM
#endif

#define MU_PRINT(...)                                                          \
  do {                                                                         \
    if (MU_ENABLE_PRINT) {                                                     \
      fprintf(stderr, __VA_ARGS__);                                            \
    }                                                                          \
  } while (0)

#define MU_PRINT_MATRIX(PREFIX, A, M, N)                                       \
  {                                                                            \
    if (MU_ENABLE_PRINT) {                                                     \
      size_t idx = 0;                                                          \
      printf("%s:\n", PREFIX);                                                 \
      for (size_t i = 0; i < M; i++) {                                         \
        for (size_t j = 0; j < N; j++) {                                       \
          printf("%.4f  ", A[idx]);                                            \
          idx++;                                                               \
        }                                                                      \
        printf("\n");                                                          \
      }                                                                        \
      printf("\n");                                                            \
    }                                                                          \
  }

#define MU_PRINT_VECTOR(PREFIX, V, N)                                          \
  {                                                                            \
    if (MU_ENABLE_PRINT) {                                                     \
      printf("%s: ", PREFIX);                                                  \
      for (size_t i = 0; i < N; i++) {                                         \
        printf("%f ", V[i]);                                                   \
      }                                                                        \
      printf("\n");                                                            \
    }                                                                          \
  }

/**
 * Redirect stdout and stderr to file.
 * @param[in] output_path Output file path
 * @param[out] stdout_fd Standard out file descriptor
 * @param[out] stderr_fd Standard error file descriptor
 * @param[out] output_fd Output file descriptor
 */
int streams_redirect(const char *output_path,
                     int *stdout_fd,
                     int *stderr_fd,
                     int *output_fd) {
  // Obtain stdout and stderr file descriptors
  *stdout_fd = dup(STDOUT_FILENO);
  *stderr_fd = dup(STDERR_FILENO);

  // Open stdout log file
  *output_fd = open(output_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
  if (*output_fd == -1) {
    perror("opening output.log");
    return -1;
  }

  // Redirect stdout
  if (dup2(*output_fd, STDOUT_FILENO) == -1) {
    perror("cannot redirect stdout");
    return -1;
  }

  // Redirect stderr
  if (dup2(*output_fd, STDERR_FILENO) == -1) {
    perror("cannot redirect stderr");
    return -1;
  }

  return 0;
}

/**
 * Restore stdout and stderr
 * @param[in] stdout_fd Standard output file descriptor
 * @param[in] stderr_fd Standard error file descriptor
 * @param[in] output_fd Output file descriptor
 */
void streams_restore(const int stdout_fd,
                     const int stderr_fd,
                     const int output_fd) {
  // Flush stdout, stderr and close output file
  fflush(stdout);
  fflush(stderr);
  close(output_fd);

  // Restore stdout and stderr
  dup2(stdout_fd, STDOUT_FILENO);
  dup2(stderr_fd, STDERR_FILENO);
}

/**
 * Print test log
 * @param[in] log_path Path to test log
 */
void mu_print_log(const char *log_path) {
  // Open log file
  FILE *log_file = fopen(log_path, "rb");
  if (log_file == NULL) {
    return;
  }

  // Get log file length
  fseek(log_file, 0, SEEK_END);
  size_t log_length = ftell(log_file);
  fseek(log_file, 0, SEEK_SET);

  // Print log and close log file
  char buf[9046] = {0};
  const size_t read = fread(buf, 1, log_length, log_file);
  if (read != log_length) {
    printf("Failed to read log file [%s]\n", log_path);
    exit(-1);
  }
  printf("%s\n", buf);
  fflush(stdout);
  fclose(log_file);
}

/**
 * Print test stats
 */
void mu_print_stats(void) {
  printf("\n");
  printf(MU_WHT "Ran %d tests" MU_NRM " ", num_tests);
  printf("[");
  printf(MU_GRN "%d passed" MU_NRM ", ", num_passed);
  printf(MU_RED "%d failed" MU_NRM, num_failed);
  printf("]\n");
}

/**
 * Run unittests
 * @param[in] test_name Test name
 * @param[in] test_ptr Pointer to unittest
 * @param[in] redirect Redirect stdout and stderr to file
 * @param[in] keep_logs Flag to keep log or not
 */
void mu_run_test(const char *test_name,
                 int (*test_ptr)(void),
                 const int redirect,
                 const int keep_logs) {
  // Check if test target is set and current test is test target
  if (test_target_name != NULL && strcmp(test_target_name, test_name) != 0) {
    return;
  }

  // Redirect stdout and stderr to file
  char log_path[1024] = {0};
  int stdout_fd = 0;
  int stderr_fd = 0;
  int log_fd = 0;
  if (redirect) {
    sprintf(log_path, "%s/mu_%s.log", MU_LOG_DIR, test_name);
    if (streams_redirect(log_path, &stdout_fd, &stderr_fd, &log_fd) == -1) {
      printf("Failed to redirect streams!\n");
      exit(-1);
    }
  }

  // Run test
  if (redirect == 0) {
    printf("-> [%s] ", test_name);
    fflush(stdout);
  }
  int test_retval = (*test_ptr)();

  // Restore stdout and stderr
  if (redirect) {
    streams_restore(stdout_fd, stderr_fd, log_fd);
  }

  // Keep track of test results
  if (test_retval == 0) {
    if (redirect) {
      printf(".");
    } else {
      printf(MU_GRN "OK!\n" MU_NRM);
    }
    fflush(stdout);
    num_passed++;

    if (redirect && keep_logs == 0) {
      remove(log_path);
    }

  } else {
    printf(MU_RED " FAILED!\n" MU_NRM);
    fflush(stdout);
    mu_print_log(log_path);
    num_failed++;
  }

  num_tests++;
}

/**
 * Run python script
 * @param[in] script_path Path to python3 script
 */
int mu_run_python(const char *script_path) {
  char cmd[1024] = {0};
  sprintf(cmd, "python3 %s", script_path);
  if (system(cmd) != 0) {
    printf("Python3 script [%s] failed !", script_path);
    return -1;
  }
  return 0;
}

/**
 * Unit-test assert
 * @param[in] TEST Test condition
 */
#define MU_ASSERT(TEST)                                                        \
  do {                                                                         \
    if ((TEST) == 0) {                                                         \
      printf(MU_RED "ERROR!" MU_NRM " [%s:%d] %s FAILED!\n",                   \
             __func__,                                                         \
             __LINE__,                                                         \
             #TEST);                                                           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

/**
 * Compare reals.
 * @returns
 * - 0 if x == y
 * - 1 if x > y
 * - -1 if x < y
 */
#define MU_ASSERT_FLOAT(X, Y)                                                  \
  do {                                                                         \
    if (fabs(X - Y) > 1e-12) {                                                 \
      return -1;                                                               \
    }                                                                          \
  } while (0)

/**
 * Add unittest
 * @param[in] TEST Test function
 */
#define MU_ADD_TEST(TEST)                                                      \
  mu_run_test(#TEST, TEST, MU_REDIRECT_STREAMS, MU_KEEP_LOGS);

/**
 * Run all unit-tests
 * @param[in] TEST_SUITE Test suite
 */
#define MU_RUN_TESTS(TEST_SUITE)                                               \
  int main(int argc, char *argv[]) {                                           \
    if (argc == 3 && strcmp(argv[1], "--target") == 0) {                       \
      test_target_name = argv[2];                                              \
      printf("TEST TARGET [%s]\n", test_target_name);                          \
    }                                                                          \
                                                                               \
    TEST_SUITE();                                                              \
    mu_print_stats();                                                          \
    return (num_failed) ? -1 : 0;                                              \
  }

#endif // MUNIT_H
