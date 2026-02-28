#include "xyz.h"

/*******************************************************************************
 * SYSTEM
 ******************************************************************************/

void print_stacktrace(void) {
  void *buffer[9046] = {0};
  int nptrs = backtrace(buffer, 100);
  char **strings = backtrace_symbols(buffer, nptrs);
  if (strings == NULL) {
    perror("backtrace_symbols");
    exit(EXIT_FAILURE);
  }

  printf("Stack trace:\n");
  for (int i = 0; i < nptrs; i++) {
    printf("%s\n", strings[i]);
  }

  free(strings);
}

/*******************************************************************************
 * DATA
 ******************************************************************************/

char wait_key(int delay) {
  // Enable raw mode
  struct termios term;
  tcgetattr(STDIN_FILENO, &term);
  term.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &term);

  // Wait
  struct pollfd fd = {STDIN_FILENO, POLLIN, 0};
  int ret = poll(&fd, 1, delay);
  char key = -1;
  if (ret > 0) {
    if (read(STDIN_FILENO, &key, 1) == -1) {
      return -1;
    }
  }

  // Disable raw mode
  tcgetattr(STDIN_FILENO, &term);
  term.c_lflag |= (ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &term);

  return key;
}

/**
 * String copy from `src` to `dst`.
 */
size_t string_copy(char *dst, const char *src) {
  dst[0] = '\0';
  memcpy(dst, src, strlen(src));
  dst[strlen(src)] = '\0'; // Null terminate
  return strlen(dst);
}

/**
 * Copy a substring from `src` to `dst` where `s` and `n` are the start index
 * and length.
 */
void string_subcopy(char *dst, const char *src, const int s, const int n) {
  assert(s >= 0);
  assert(n > 0);

  dst[0] = '\0';
  for (int i = 0; i < n; i++) {
    dst[i] = src[s + i];
  }
}

/**
 * Concatenate string from `src` to `dst`.
 */
void string_cat(char *dst, const char *src) {
  size_t dst_len = strlen(dst);
  strcat(dst + dst_len, src);
  dst[dst_len + strlen(src)] = '\0'; // strncat does not null terminate
}

/**
 * Allocate heap memory for string `s`.
 */
char *string_malloc(const char *s) {
  assert(s != NULL);
  char *retval = malloc(sizeof(char) * strlen(s) + 1);
  memcpy(retval, s, strlen(s));
  retval[strlen(s)] = '\0'; // Null terminate
  return retval;
}

/**
 * Strip whitespace from string `s`.
 */
char *string_strip(char *s) {
  char *end;

  // Trim leading space
  while (*s == ' ') {
    s++;
  }

  if (*s == 0) { // All spaces?
    return s;
  }

  // Trim trailing space
  end = s + strlen(s) - 1;
  while (end > s && (*end == ' ' || *end == '\n')) {
    end--;
  }

  // Write new null terminator character
  end[1] = '\0';

  return s;
}

/**
 * Strip specific character `c` from string `s`.
 */
char *string_strip_char(char *s, const char c) {
  char *end;

  // Trim leading space
  while (*s == c) {
    s++;
  }

  if (*s == 0) { // All spaces?
    return s;
  }

  // Trim trailing space
  end = s + strlen(s) - 1;
  while (end > s && *end == c) {
    end--;
  }

  // Write new null terminator character
  end[1] = '\0';

  return s;
}

/**
 * Split string `s` by delimiter `d`
 */
char **string_split(char *a_str, const char a_delim, size_t *n) {
  char **result = 0;
  char *tmp = a_str;
  char *last_comma = 0;
  char delim[2];
  delim[0] = a_delim;
  delim[1] = 0;

  /* Count how many elements will be extracted. */
  while (*tmp) {
    if (a_delim == *tmp) {
      (*n)++;
      last_comma = tmp;
    }
    tmp++;
  }

  /* Add space for trailing token. */
  *n += last_comma < (a_str + strlen(a_str) - 1);

  /* Add space for terminating null string so caller
     knows where the list of returned strings ends. */
  (*n)++;

  result = malloc(sizeof(char *) * *n);

  if (result) {
    size_t idx = 0;
    char *token = strtok(a_str, delim);

    while (token) {
      assert(idx < *n);
      *(result + idx++) = strdup(token);
      token = strtok(0, delim);
    }
    assert(idx == *n - 1);
    *(result + idx) = 0;
  }

  // Return results
  (*n)--;

  return result;
}

/**
 * Parse integer array line.
 * @returns
 * - 1D vector of integers
 * - NULL for failure
 */
static int *parse_iarray_line(char *line) {
  assert(line != NULL);
  char entry[MAX_LINE_LENGTH] = {0};
  int index = 0;
  int *data = NULL;

  for (size_t i = 0; i < strlen(line); i++) {
    char c = line[i];
    if (c == ' ') {
      continue;
    }

    if (c == ',' || c == '\n') {
      if (data == NULL) {
        size_t array_size = strtod(entry, NULL);
        data = calloc(array_size + 1, sizeof(int));
      }
      data[index] = strtod(entry, NULL);
      index++;
      memset(entry, '\0', sizeof(char) * 100);
    } else {
      entry[strlen(entry)] = c;
    }
  }

  return data;
}

/**
 * Parse 2D integer arrays from csv file.
 * @returns
 * - List of 1D vector of integers
 * - NULL for failure
 */
int **load_iarrays(const char *csv_path, int *num_arrays) {
  assert(csv_path != NULL);
  FILE *csv_file = fopen(csv_path, "r");
  *num_arrays = dsv_rows(csv_path);
  int **array = calloc(*num_arrays, sizeof(int *));

  char line[MAX_LINE_LENGTH] = {0};
  int frame_idx = 0;
  while (fgets(line, MAX_LINE_LENGTH, csv_file) != NULL) {
    if (line[0] == '#') {
      continue;
    }

    array[frame_idx] = parse_iarray_line(line);
    frame_idx++;
  }
  fclose(csv_file);

  return array;
}

/**
 * Parse real array line.
 * @returns
 * - 1D vector of real
 * - NULL for failure
 */
static double *parse_darray_line(char *line) {
  assert(line != NULL);
  char entry[MAX_LINE_LENGTH] = {0};
  int index = 0;
  double *data = NULL;

  for (size_t i = 0; i < strlen(line); i++) {
    char c = line[i];
    if (c == ' ') {
      continue;
    }

    if (c == ',' || c == '\n') {
      if (data == NULL) {
        size_t array_size = strtod(entry, NULL);
        data = calloc(array_size, sizeof(double));
      }
      data[index] = strtod(entry, NULL);
      index++;
      memset(entry, '\0', sizeof(char) * 100);
    } else {
      entry[strlen(entry)] = c;
    }
  }

  return data;
}

/**
 * Parse 2D real arrays from csv file at `csv_path`, on success `num_arrays`
 * will return number of arrays.
 * @returns
 * - List of 1D vector of reals
 * - NULL for failure
 */
double **load_darrays(const char *csv_path, int *num_arrays) {
  assert(csv_path != NULL);
  assert(num_arrays != NULL);
  FILE *csv_file = fopen(csv_path, "r");
  *num_arrays = dsv_rows(csv_path);
  double **array = calloc(*num_arrays, sizeof(double *));

  char line[MAX_LINE_LENGTH] = {0};
  int frame_idx = 0;
  while (fgets(line, MAX_LINE_LENGTH, csv_file) != NULL) {
    if (line[0] == '#') {
      continue;
    }

    array[frame_idx] = parse_darray_line(line);
    frame_idx++;
  }
  fclose(csv_file);

  return array;
}

/**
 * Allocate heap memory for integer `val`.
 */
int *int_malloc(const int val) {
  int *i = malloc(sizeof(int));
  *i = val;
  return i;
}

/**
 * Allocate heap memory for float `val`.
 */
float *float_malloc(const float val) {
  float *f = malloc(sizeof(float));
  *f = val;
  return f;
}

/**
 * Allocate heap memory for double `val`.
 */
double *double_malloc(const double val) {
  double *d = malloc(sizeof(double));
  *d = val;
  return d;
}

/**
 * Allocate heap memory for vector `vec` with length `N`.
 */
double *vector_malloc(const double *vec, const double N) {
  double *retval = malloc(sizeof(double) * N);
  for (int i = 0; i < N; i++) {
    retval[i] = vec[i];
  }
  return retval;
}

/**
 * Get number of rows in a delimited file at `fp`.
 * @returns
 * - Number of rows
 * - -1 for failure
 */
int dsv_rows(const char *fp) {
  assert(fp != NULL);

  // Load file
  FILE *infile = fopen(fp, "r");
  if (infile == NULL) {
    return -1;
  }

  // Loop through lines
  int num_rows = 0;
  char line[MAX_LINE_LENGTH] = {0};
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    if (line[0] != '#') {
      num_rows++;
    }
  }

  // Cleanup
  fclose(infile);

  return num_rows;
}

/**
 * Get number of columns in a delimited file at `fp`.
 * @returns
 * - Number of columns
 * - -1 for failure
 */
int dsv_cols(const char *fp, const char delim) {
  assert(fp != NULL);

  // Load file
  FILE *infile = fopen(fp, "r");
  if (infile == NULL) {
    return -1;
  }

  // Get line that isn't the header
  char line[MAX_LINE_LENGTH] = {0};
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    if (line[0] != '#') {
      break;
    }
  }

  // Parse line to obtain number of elements
  int num_elements = 1;
  int found_separator = 0;
  for (size_t i = 0; i < MAX_LINE_LENGTH; i++) {
    if (line[i] == delim) {
      found_separator = 1;
      num_elements++;
    }
  }

  // Cleanup
  fclose(infile);

  return (found_separator) ? num_elements : -1;
}

/**
 * Get the fields of the delimited file at `fp`, where `delim` is the value
 * separated symbol and `num_fields` returns the length of the fields returned.
 * @returns
 * - List of field strings
 * - NULL for failure
 */
char **dsv_fields(const char *fp, const char delim, int *num_fields) {
  assert(fp != NULL);

  // Load file
  FILE *infile = fopen(fp, "r");
  if (infile == NULL) {
    return NULL;
  }

  // Get last header line
  char field_line[MAX_LINE_LENGTH] = {0};
  char line[MAX_LINE_LENGTH] = {0};
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    if (line[0] != '#') {
      break;
    } else {
      memcpy(field_line, line, strlen(line));
    }
  }

  // Parse fields
  *num_fields = dsv_cols(fp, delim);
  char **fields = malloc(sizeof(char *) * *num_fields);
  int field_idx = 0;
  char field_name[100] = {0};

  for (size_t i = 0; i < strlen(field_line); i++) {
    char c = field_line[i];

    // Ignore # and ' '
    if (c == '#' || c == ' ') {
      continue;
    }

    if (c == ',' || c == '\n') {
      // Add field name to fields
      fields[field_idx] = string_malloc(field_name);
      memset(field_name, '\0', 100);
      field_idx++;
    } else {
      // Append field name
      field_name[strlen(field_name)] = c;
    }
  }

  // Cleanup
  fclose(infile);

  return fields;
}

/**
 * Load delimited separated value data as a matrix.
 * @returns
 * - Matrix of DSV data
 * - NULL for failure
 */
double **
dsv_data(const char *fp, const char delim, int *num_rows, int *num_cols) {
  assert(fp != NULL);

  // Obtain number of rows and columns in dsv data
  *num_rows = dsv_rows(fp);
  *num_cols = dsv_cols(fp, delim);
  if (*num_rows == -1 || *num_cols == -1) {
    return NULL;
  }

  // Load file
  FILE *infile = fopen(fp, "r");
  if (infile == NULL) {
    return NULL;
  }

  // Loop through data
  char line[MAX_LINE_LENGTH] = {0};
  int row_idx = 0;
  int col_idx = 0;

  // Loop through data line by line
  double **data = malloc(sizeof(double *) * *num_rows);
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    // Ignore if comment line
    if (line[0] == '#') {
      continue;
    }

    // Iterate through values in line separated by commas
    data[row_idx] = malloc(sizeof(double) * *num_cols);
    char entry[100] = {0};
    for (size_t i = 0; i < strlen(line); i++) {
      char c = line[i];
      if (c == ' ') {
        continue;
      }

      if (c == ',' || c == '\n') {
        data[row_idx][col_idx] = strtod(entry, NULL);
        memset(entry, '\0', sizeof(char) * 100);
        col_idx++;
      } else {
        entry[strlen(entry)] = c;
      }
    }

    col_idx = 0;
    row_idx++;
  }

  // Clean up
  fclose(infile);

  return data;
}

/**
 * Free DSV data.
 */
void dsv_free(double **data, const int num_rows) {
  assert(data != NULL);
  for (int i = 0; i < num_rows; i++) {
    free(data[i]);
  }
  free(data);
}

/**
 * Load comma separated data as a matrix, where `fp` is the csv file path, on
 * success `num_rows` and `num_cols` will be filled.
 * @returns
 * - Matrix of CSV data
 * - NULL for failure
 */
double **csv_data(const char *fp, int *num_rows, int *num_cols) {
  assert(fp != NULL);
  return dsv_data(fp, ',', num_rows, num_cols);
}

/**
 * Free CSV data.
 */
void csv_free(double **data, const int num_rows) {
  for (int i = 0; i < num_rows; i++) {
    free(data[i]);
  }
  free(data);
}

/**
 * Extract filename from `path` to `fname`.
 */
void path_file_name(const char *path, char *fname) {
  assert(path != NULL);
  assert(fname != NULL);

  char path_copy[9046] = {0};
  memcpy(path_copy, path, strlen(path));

  char *base = strrchr(path_copy, '/');
  base = base ? base + 1 : path_copy;

  memcpy(fname, base, strlen(base));
}

/**
 * Extract file extension from `path` to `fext`.
 */
void path_file_ext(const char *path, char *fext) {
  assert(path != NULL);
  assert(fext != NULL);

  char path_copy[9046] = {0};
  memcpy(path_copy, path, strlen(path));

  char *base = strrchr(path_copy, '.');
  if (base) {
    base = base ? base + 1 : path_copy;
    memcpy(fext, base, strlen(base));
  } else {
    fext[0] = '\0';
  }
}

/**
 * Extract dir name from `path` to `dirname`.
 */
void path_dir_name(const char *path, char *dir_name) {
  assert(path != NULL);
  assert(dir_name != NULL);

  char path_copy[9046] = {0};
  memcpy(path_copy, path, strlen(path));

  char *base = strrchr(path_copy, '/');
  memcpy(dir_name, path_copy, base - path_copy);
}

/**
 * Join two paths `x` and `y`
 */
char *path_join(const char *x, const char *y) {
  assert(x != NULL && y != NULL);

  char *retval = NULL;
  if (x[strlen(x) - 1] == '/') {
    retval = malloc(sizeof(char) * (strlen(x) + strlen(y)) + 1);
    string_copy(retval, x);
    string_copy(retval + strlen(retval), (y[0] == '/') ? y + 1 : y);
  } else {
    retval = malloc(sizeof(char) * (strlen(x) + strlen(y)) + 2);
    string_copy(retval, x);
    string_cat(retval + strlen(retval), "/");
    string_copy(retval + strlen(retval), (y[0] == '/') ? y + 1 : y);
  }

  return retval;
}

/**
 * List files in `path`.
 * @returns List of files in directory and number of files `n`.
 */
char **list_files(const char *path, int *n) {
  assert(path != NULL);
  assert(n != NULL);

  struct dirent **namelist;
  int num_files = scandir(path, &namelist, 0, alphasort);
  if (num_files < 0) {
    return NULL;
  }

  // The first two are '.' and '..'
  free(namelist[0]);
  free(namelist[1]);

  // Allocate memory for list of files
  char **files = malloc(sizeof(char *) * num_files - 2);
  *n = 0;

  // Create list of files
  for (int i = 2; i < num_files; i++) {
    char fp[9046] = {0};
    const char *c = (path[strlen(path) - 1] == '/') ? "" : "/";
    string_cat(fp, path);
    string_cat(fp, c);
    string_cat(fp, namelist[i]->d_name);

    files[*n] = malloc(sizeof(char) * strlen(fp) + 1);
    memcpy(files[*n], fp, strlen(fp));
    files[*n][strlen(fp)] = '\0'; // strncpy does not null terminate
    (*n)++;

    free(namelist[i]);
  }
  free(namelist);

  return files;
}

/**
 * Free list of `files` of length `n`.
 */
void list_files_free(char **data, const int n) {
  assert(data != NULL);
  for (int i = 0; i < n; i++) {
    free(data[i]);
  }
  free(data);
}

/**
 * Count number of lines in file
 * @returns Number of lines or `-1` for failure
 */
size_t file_lines(const char *fp) {
  FILE *f = fopen(fp, "r");
  size_t lines = 0;

  if (f == NULL) {
    return -1;
  }

  int ch;
  while ((ch = getc(f)) != EOF) {
    if (ch == '\n') {
      ++lines;
    }
  }

  return lines;
}

/**
 * Create directory, including parent folders.
 * Returns 0 for success or -1 for failure.
 */
int mkdir_p(const char *path, const mode_t mode) {
  char *tmp = strdup(path);
  char *p = tmp;
  int status = 0;

  // Skip leading slashes
  while (*p == '/')
    p++;

  for (; *p; p++) {
    if (*p == '/') {
      *p = '\0';

      if (mkdir(tmp, mode) != 0) {
        if (errno != EEXIST) {
          status = -1;
          break;
        }
      }

      *p = '/';
    }
  }

  // Create the final directory
  if (status == 0 && mkdir(tmp, mode) != 0) {
    if (errno != EEXIST) {
      status = -1;
    }
  }

  free(tmp);
  return status;
}

int _unlink_cb(const char *fpath,
               const struct stat *sb,
               int typeflag,
               struct FTW *ftwbuf) {
  int rv = remove(fpath);
  if (rv) {
    perror(fpath);
  }
  return rv;
}

/**
 * Delete directrory.
 * Returns 0 for success or -1 for failure.
 */
int rmdir(const char *path) {
  return nftw(path, _unlink_cb, 64, FTW_DEPTH | FTW_PHYS);
}

/**
 * Read file contents in file path `fp`.
 * @returns
 * - Success: File contents
 * - Failure: NULL
 */
char *file_read(const char *fp) {
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

/**
 * Skip line in file.
 */
void skip_line(FILE *fp) {
  assert(fp != NULL);

  char header[BUFSIZ];
  char *retval = fgets(header, BUFSIZ, fp);
  if (retval == NULL) {
    FATAL("Failed to skip line!");
  }
}

/**
 * Check if file exists.
 * @returns
 * - 1 File exists
 * - 0 File does not exist
 */
int file_exists(const char *fp) { return (access(fp, F_OK) == 0) ? 1 : 0; }

/**
 * Get number of rows in file `fp`.
 * @returns
 * - Number of rows in file
 * - -1 for failure.
 */
int file_rows(const char *fp) {
  assert(fp != NULL);

  FILE *file = fopen(fp, "rb");
  if (file == NULL) {
    return -1;
  }

  // Obtain number of lines
  int num_rows = 0;
  char *line = NULL;
  size_t len = 0;
  while (getline(&line, &len, file) != -1) {
    num_rows++;
  }
  free(line);

  // Clean up
  fclose(file);

  return num_rows;
}

/**
 * Copy file from path `src` to path `dst`.
 * @returns
 * - 0 for success
 * - -1 if src file could not be opend
 * - -2 if dst file could not be opened
 */
int file_copy(const char *src, const char *dst) {
  assert(src != NULL);
  assert(dst != NULL);

  FILE *src_file = fopen(src, "rb");
  if (src_file == NULL) {
    return -1;
  }

  FILE *dst_file = fopen(dst, "wb");
  if (dst_file == NULL) {
    fclose(src_file);
    return -2;
  }

  char *line = NULL;
  size_t len = 0;
  ssize_t read = 0;
  while ((read = getline(&line, &len, src_file)) != -1) {
    fwrite(line, sizeof(char), read, dst_file);
  }
  if (line) {
    free(line);
  }

  // Clean up
  fclose(src_file);
  fclose(dst_file);

  return 0;
}

/*******************************************************************************
 * TIME
 ******************************************************************************/

struct timespec __tic_time; // global tic start time

timestamp_t *timestamp_malloc(timestamp_t ts) {
  timestamp_t *ts_ptr = malloc(sizeof(timestamp_t));
  *ts_ptr = ts;
  return ts_ptr;
}

void timestamp_free(timestamp_t *ts_ptr) { free(ts_ptr); }

/**
 * Tic, start timer.
 * @returns A timespec encapsulating the time instance when tic() is called
 */
void tic(void) { clock_gettime(CLOCK_MONOTONIC, &__tic_time); }

/**
 * Toc, stop timer.
 * @returns Time elapsed in seconds
 */
double toc(void) {
  struct timespec toc;
  double time_elasped;

  clock_gettime(CLOCK_MONOTONIC, &toc);
  time_elasped = (toc.tv_sec - __tic_time.tv_sec);
  time_elasped += (toc.tv_nsec - __tic_time.tv_nsec) / 1e9;

  return time_elasped;
}

/**
 * Toc, stop timer.
 * @returns Time elapsed in milli-seconds
 */
double mtoc(void) { return toc() * 1e3; }

/**
 * Get time now since epoch.
 * @return Time now in nano-seconds since epoch
 */
timestamp_t time_now(void) {
  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);

  const time_t sec = spec.tv_sec;
  const long int ns = spec.tv_nsec;
  const uint64_t BILLION = 1000000000L;

  return (uint64_t) sec * BILLION + (uint64_t) ns;
}

/**
 * Convert string to timestamp
 */
timestamp_t str2ts(const char *ts_str) { return strtoll(ts_str, NULL, 10); }

/**
 * Convert timestamp to seconds
 */
double ts2sec(const timestamp_t ts) { return ts * 1e-9; }

/**
 * Convert seconds to timestamp
 */
timestamp_t sec2ts(const double time_s) { return time_s * 1e9; }

/**
 * Extract timestamp from file path.
 */
timestamp_t path2ts(const char *file_path) {
  char fname[128] = {0};
  char fext[128] = {0};
  path_file_name(file_path, fname);
  path_file_ext(file_path, fext);

  char ts_str[128] = {0};
  memcpy(ts_str, fname, strlen(fname) - strlen(fext) - 1);

  char *ptr;
  return strtol(ts_str, &ptr, 10);
}

/*******************************************************************************
 * ARRAY
 ******************************************************************************/

arr_t *arr_malloc(const size_t capacity) {
  arr_t *arr = malloc(sizeof(arr_t));
  arr->data = malloc(sizeof(void **) * capacity);
  arr->size = 0;
  arr->capacity = capacity;
  return arr;
}

void arr_free(arr_t *keys) {
  free(keys->data);
  free(keys);
}

void arr_push_back(arr_t *arr, void *data) {
  if ((arr->size * 2.0) >= arr->capacity) {
    size_t new_capacity = arr->capacity * 2.0;
    arr->data = realloc(arr->data, sizeof(void *) * new_capacity);
    arr->capacity = new_capacity;
  }
  arr->data[arr->size++] = data;
}

/*******************************************************************************
 * DARRAY
 ******************************************************************************/

darray_t *darray_new(size_t element_size, size_t initial_max) {
  assert(element_size > 0);
  assert(initial_max > 0);

  darray_t *array = malloc(sizeof(darray_t) * 1);
  if (array == NULL) {
    return NULL;
  }

  array->end = 0;
  array->max = (int) initial_max;
  array->element_size = element_size;
  array->expand_rate = DEFAULT_EXPAND_RATE;
  array->contents = calloc(initial_max, sizeof(void *));
  if (array->contents == NULL) {
    free(array);
    return NULL;
  }

  return array;
}

void darray_clear(darray_t *array) {
  assert(array != NULL);
  for (int i = 0; i < array->max; i++) {
    if (array->contents[i]) {
      free(array->contents[i]);
    }
  }
}

void darray_destroy(darray_t *array) {
  if (array) {
    if (array->contents) {
      free(array->contents);
    }
    free(array);
  }
}

void darray_clear_destroy(darray_t *array) {
  if (array) {
    darray_clear(array);
    darray_destroy(array);
  }
}

int darray_push(darray_t *array, void *el) {
  assert(array != NULL);

  // Push
  array->contents[array->end] = el;
  array->end++;

  // Expand darray if necessary
  if (array->end >= array->max) {
    return darray_expand(array);
  }

  return 0;
}

void *darray_pop(darray_t *array) {
  assert(array != NULL);

  // pop
  void *el = darray_remove(array, array->end - 1);
  array->end--;

  // contract
  int expanded = array->end > (int) array->expand_rate;
  int trailing_memory = array->end % (int) array->expand_rate;
  if (expanded && trailing_memory) {
    darray_contract(array);
  }

  return el;
}

int darray_contains(darray_t *array,
                    void *el,
                    int (*cmp)(const void *, const void *)) {
  assert(array != NULL);
  assert(el != NULL);
  assert(cmp != NULL);

  // Check first element
  void *element = darray_get(array, 0);
  if (element != NULL && cmp(element, el) == 0) {
    return 1;
  }

  // Rest of element
  for (int i = 0; i < array->end; i++) {
    element = darray_get(array, i);
    if (element != NULL && cmp(element, el) == 0) {
      return 1;
    }
  }

  return 0;
}

darray_t *darray_copy(darray_t *array) {
  assert(array != NULL);

  // Copy first element
  darray_t *array_copy = darray_new(array->element_size, (size_t) array->max);
  void *el = darray_get(array, 0);
  void *el_copy = NULL;

  if (el != NULL) {
    el_copy = darray_new_element(array_copy);
    memcpy(el_copy, el, array->element_size);
    darray_set(array_copy, 0, el_copy);
  }

  // Copy the rest of the elements
  for (int i = 1; i < array->end; i++) {
    el = darray_get(array, i);
    // el_copy = NULL;

    if (el != NULL) {
      memcpy(el_copy, el, array->element_size);
      darray_set(array_copy, i, el);
    }
  }

  return array_copy;
}

void *darray_new_element(darray_t *array) {
  assert(array != NULL);
  assert(array->element_size > 0);
  return calloc(1, array->element_size);
}

void *darray_first(darray_t *array) {
  assert(array != NULL);
  return array->contents[0];
}

void *darray_last(darray_t *array) {
  assert(array != NULL);
  return array->contents[array->end - 1];
}

void darray_set(darray_t *array, int i, void *el) {
  assert(array != NULL);
  assert(i < array->max);

  // Set
  array->contents[i] = el;

  // Update end
  if (i > array->end) {
    array->end = i;
  }
}

void *darray_get(darray_t *array, int i) {
  assert(array != NULL);
  assert(i < array->max);
  return array->contents[i];
}

void *darray_update(darray_t *array, int i, void *el) {
  assert(array != NULL);
  assert(i < array->max);
  void *old_el;

  // Update
  old_el = darray_get(array, i);
  darray_set(array, i, el);

  return old_el;
}

void *darray_remove(darray_t *array, int i) {
  assert(array != NULL);
  void *el = array->contents[i];
  array->contents[i] = NULL;
  return el;
}

static inline int darray_resize(darray_t *array, size_t new_max) {
  assert(array != NULL);

  // Calculate new max and size
  int old_max = (int) array->max;
  array->max = (int) new_max;

  // Reallocate new memory
  void *contents = realloc(array->contents, new_max * sizeof(void *));
  if (contents == NULL) {
    return -1;
  }
  array->contents = contents;

  // Initialize new memory to NULL
  for (int i = old_max; i < (int) new_max; i++) {
    array->contents[i] = NULL;
  }

  return 0;
}

int darray_expand(darray_t *array) {
  assert(array != NULL);
  assert(array->max > 0);

  size_t old_max = (size_t) array->max;
  size_t new_max = (size_t) array->max + array->expand_rate;
  int res = darray_resize(array, new_max);
  if (res != 0) {
    return -1;
  }
  memset(array->contents + old_max, 0, array->expand_rate + 1);

  return 0;
}

int darray_contract(darray_t *array) {
  assert(array != NULL);
  assert(array->max > 0);

  // Contract
  int new_size = 0;
  if (array->end < (int) array->expand_rate) {
    new_size = (int) array->expand_rate;
  } else {
    new_size = array->end;
  }

  return darray_resize(array, (size_t) new_size + 1);
}

/*******************************************************************************
 * LIST
 ******************************************************************************/

list_t *list_malloc(void) {
  list_t *list = calloc(1, sizeof(list_t));
  list->length = 0;
  list->first = NULL;
  list->last = NULL;
  return list;
}

void list_free(list_t *list) {
  assert(list != NULL);

  list_node_t *node;
  list_node_t *next_node;

  // Destroy
  node = list->first;
  while (node != NULL) {
    next_node = node->next;
    if (node) {
      free(node);
    }
    node = next_node;
  }

  free(list);
}

void list_clear(list_t *list) {
  assert(list != NULL);

  list_node_t *node;
  list_node_t *next_node;

  node = list->first;
  while (node != NULL) {
    next_node = node->next;
    free(node->value);
    node = next_node;
  }
}

void list_clear_free(list_t *list) {
  assert(list != NULL);

  list_node_t *node = list->first;
  while (node != NULL) {
    list_node_t *next_node = node->next;
    free(node->value);
    free(node);
    node = next_node;
  }
  free(list);
}

void list_push(list_t *list, void *value) {
  assert(list != NULL);
  assert(value != NULL);

  // Initialize node
  list_node_t *node = calloc(1, sizeof(list_node_t));
  if (node == NULL) {
    return;
  }
  node->value = value;

  // Push node
  if (list->last == NULL) {
    list->first = node;
    list->last = node;
  } else {
    list->last->next = node;
    node->prev = list->last;
    list->last = node;
  }

  list->length++;
}

void *list_pop(list_t *list) {
  assert(list != NULL);

  // Get last
  list_node_t *last = list->last;
  if (last == NULL) {
    return NULL;
  }
  void *value = last->value;
  list_node_t *before_last = last->prev;
  free(last);

  // Pop
  if (before_last == NULL && list->length == 1) {
    list->last = NULL;
    list->first = NULL;
  } else {
    list->last = before_last;
  }
  list->length--;

  return value;
}

void *list_pop_front(list_t *list) {
  assert(list != NULL);
  assert(list->first != NULL);

  // Pop front
  list_node_t *first_node = list->first;
  void *data = first_node->value;
  list_node_t *next_node = first_node->next;

  if (next_node != NULL) {
    list->first = next_node;
  } else {
    list->first = NULL;
  }
  list->length--;

  // Clean up
  free(first_node);

  return data;
}

void *list_shift(list_t *list) {
  assert(list != NULL);

  list_node_t *first = list->first;
  void *value = first->value;
  list_node_t *second = list->first->next;

  list->first = second;
  list->length--;
  free(first);

  return value;
}

void list_unshift(list_t *list, void *value) {
  assert(list != NULL);

  list_node_t *node = calloc(1, sizeof(list_node_t));
  if (node == NULL) {
    return;
  }
  node->value = value;

  if (list->first == NULL) {
    list->first = node;
    list->last = node;
  } else {
    node->next = list->first;
    list->first->prev = node;
    list->first = node;
  }

  list->length++;
}

void *list_remove(list_t *list,
                  void *value,
                  int (*cmp)(const void *, const void *)) {
  assert(list != NULL);
  assert(value != NULL);
  assert(cmp != NULL);

  // Iterate list
  list_node_t *node = list->first;
  while (node != NULL) {

    // Compare target with node value
    if (cmp(node->value, value) == 0) {
      value = node->value;

      if (list->length == 1) {
        // Last node in list
        list->first = NULL;
        list->last = NULL;

      } else if (node == list->first) {
        // First node in list
        list->first = node->next;
        node->next->prev = NULL;

      } else if (node == list->last) {
        // In the case of removing last node in list
        list->last = node->prev;
        node->prev->next = NULL;

      } else {
        // Remove others
        node->prev->next = node->next;
        node->next->prev = node->prev;
      }
      list->length--;
      free(node);

      return value;
    }

    node = node->next;
  }

  return NULL;
}

int list_remove_destroy(list_t *list,
                        void *value,
                        int (*cmp)(const void *, const void *),
                        void (*free_func)(void *)) {
  assert(list != NULL);
  void *result = list_remove(list, value, cmp);
  free_func(result);
  return 0;
}

/*******************************************************************************
 * RED-BLACK-TREE
 ******************************************************************************/

int default_cmp(const void *x, const void *y) {
  if (x < y) {
    return -1;
  } else if (x > y) {
    return 1;
  }
  return 0;
}

int int_cmp(const void *x, const void *y) {
  if (*(int *) x < *(int *) y) {
    return -1;
  } else if (*(int *) x > *(int *) y) {
    return 1;
  }
  return 0;
}

int float_cmp(const void *x, const void *y) {
  if (*(float *) x < *(float *) y) {
    return -1;
  } else if (*(float *) x > *(float *) y) {
    return 1;
  }
  return 0;
}

int double_cmp(const void *x, const void *y) {
  if (*(double *) x < *(double *) y) {
    return -1;
  } else if (*(double *) x > *(double *) y) {
    return 1;
  }
  return 0;
}

int string_cmp(const void *x, const void *y) {
  return strcmp((char *) x, (char *) y);
}

/**
 * The following Red-Black tree implementation is based on Robert Sedgewick's
 * Left Leaninng Red-Black tree (LLRBT), where we have implemented the 2-3
 * variant instead of the 2-3-4 variant for simplicity. Compared to other
 * implementations Sedgewick's by far the simmplest, and requirest the
 * fewest lines of code.
 *
 * Source:
 *
 *   Robert Sedgewick, Kevin Wayne
 *   Algorithms, 4th Edition. Addison-Wesley 2011
 *   Chapter 3.3: Balanced Search Trees, Page 424
 *
 *   Left-leaning Red-Black Trees by Robert Sedgewick
 *   https://sedgewick.io/wp-content/themes/sedgewick/papers/2008LLRB.pdf
 */

rbt_node_t *rbt_node_malloc(const int color, void *key, void *value) {
  rbt_node_t *node = malloc(sizeof(rbt_node_t));
  node->key = key;
  node->value = value;
  node->color = color;
  node->child[0] = NULL;
  node->child[1] = NULL;
  node->size = 1;
  return node;
}

void rbt_node_free(rbt_node_t *n) {
  if (n == NULL) {
    return;
  }
  rbt_node_free(n->child[0]);
  rbt_node_free(n->child[1]);
  free(n);
}

bool rbt_node_is_red(const rbt_node_t *n) {
  if (n == NULL) {
    return false;
  }
  return n->color == RB_RED;
}

rbt_node_t *rbt_node_min(rbt_node_t *n) {
  if (n->child[0] == NULL) {
    return n;
  }
  return rbt_node_min(n->child[0]);
}

rbt_node_t *rbt_node_max(rbt_node_t *n) {
  if (n->child[1] == NULL) {
    return n;
  }
  return rbt_node_max(n->child[1]);
}

size_t rbt_node_height(const rbt_node_t *n) {
  if (n == NULL) {
    return -1;
  }
  return 1 + MAX(rbt_node_height(n->child[0]), rbt_node_height(n->child[1]));
}

size_t rbt_node_size(const rbt_node_t *n) {
  if (n == NULL) {
    return 0;
  }
  return n->size;
}

void rbt_node_keys(const rbt_node_t *n,
                   const void *lo,
                   const void *hi,
                   arr_t *keys,
                   cmp_t cmp) {
  if (n == NULL) {
    return;
  }

  const int cmplo = cmp(lo, n->key);
  const int cmphi = cmp(hi, n->key);
  if (cmplo < 0) {
    rbt_node_keys(n->child[0], lo, hi, keys, cmp);
  }
  if (cmplo <= 0 && cmphi >= 0) {
    arr_push_back(keys, n->key);
  }
  if (cmphi > 0) {
    rbt_node_keys(n->child[1], lo, hi, keys, cmp);
  }
}

void rbt_node_keys_values(const rbt_node_t *n,
                          const void *lo,
                          const void *hi,
                          arr_t *keys,
                          arr_t *values,
                          cmp_t cmp) {
  if (n == NULL) {
    return;
  }

  const int cmplo = cmp(lo, n->key);
  const int cmphi = cmp(hi, n->key);
  if (cmplo < 0) {
    rbt_node_keys_values(n->child[0], lo, hi, keys, values, cmp);
  }
  if (cmplo <= 0 && cmphi >= 0) {
    arr_push_back(keys, n->key);
    arr_push_back(values, n->value);
  }
  if (cmphi > 0) {
    rbt_node_keys_values(n->child[1], lo, hi, keys, values, cmp);
  }
}

int rbt_node_rank(const rbt_node_t *n, const void *key, cmp_t cmp) {
  if (n == NULL) {
    return 0;
  }

  int cmp_val = cmp(key, n->key);
  if (cmp_val < 0) {
    return rbt_node_rank(n->child[0], key, cmp);
  } else if (cmp_val > 0) {
    return 1 + rbt_node_size(n->child[0]) +
           rbt_node_rank(n->child[1], key, cmp);
  }

  return rbt_node_size(n->child[0]);
}

void *rbt_node_select(const rbt_node_t *n, const int rank) {
  if (n == NULL) {
    return NULL;
  }

  const int left_size = rbt_node_size(n->child[0]);
  if (left_size > rank) {
    return rbt_node_select(n->child[0], rank);
  } else if (left_size < rank) {
    return rbt_node_select(n->child[1], rank - left_size - 1);
  }
  return n->key;
}

void rbt_node_flip_colors(rbt_node_t *n) {
  assert(n);
  n->color = !n->color;
  if (n->child[0]) {
    n->child[0]->color = !n->child[0]->color;
  }
  if (n->child[1]) {
    n->child[1]->color = !n->child[1]->color;
  }
}

rbt_node_t *rbt_node_rotate(rbt_node_t *n, const bool dir) {
  assert(n);
  rbt_node_t *tmp = n->child[!dir];
  n->child[!dir] = tmp->child[dir];
  tmp->child[dir] = n;
  tmp->color = n->color;
  n->color = RB_RED;
  n->size = rbt_node_size(n->child[0]) + rbt_node_size(n->child[1]) + 1;
  return tmp;
}

rbt_node_t *rbt_node_move_red_left(rbt_node_t *n) {
  assert(n);
  rbt_node_flip_colors(n);
  if (n && rbt_node_is_red(n->child[1]->child[0])) {
    n->child[1] = rbt_node_rotate(n->child[1], 1);
    n = rbt_node_rotate(n, 0);
    rbt_node_flip_colors(n);
  }
  return n;
}

rbt_node_t *rbt_node_move_red_right(rbt_node_t *n) {
  assert(n);
  rbt_node_flip_colors(n);
  if (n && rbt_node_is_red(n->child[0]->child[0])) {
    n = rbt_node_rotate(n, 1);
    rbt_node_flip_colors(n);
  }
  return n;
}

rbt_node_t *rbt_node_balance(rbt_node_t *n) {
  if (rbt_node_is_red(n->child[1]) && !rbt_node_is_red(n->child[0])) {
    n = rbt_node_rotate(n, 0);
  }
  if (rbt_node_is_red(n->child[0]) && rbt_node_is_red(n->child[0]->child[0])) {
    n = rbt_node_rotate(n, 1);
  }
  if (rbt_node_is_red(n->child[0]) && rbt_node_is_red(n->child[1])) {
    rbt_node_flip_colors(n);
  }
  n->size = 1;
  n->size += rbt_node_size(n->child[0]);
  n->size += rbt_node_size(n->child[1]);

  return n;
}

bool rbt_node_bst_check(const rbt_node_t *n, void *min, void *max, cmp_t cmp) {
  if (n == NULL) {
    return true;
  }
  if (min != NULL && cmp(n->key, min) <= 0) {
    return false;
  }
  if (max != NULL && cmp(n->key, max) >= 0) {
    return false;
  }

  return rbt_node_bst_check(n->child[0], min, n->key, cmp) &&
         rbt_node_bst_check(n->child[1], n->key, max, cmp);
}

bool rbt_node_size_check(const rbt_node_t *n) {
  if (n == NULL) {
    return true;
  }

  const int left_size = rbt_node_size(n->child[0]);
  const int right_size = rbt_node_size(n->child[1]);
  if (n->size != (left_size + right_size + 1)) {
    return false;
  }

  return rbt_node_size_check(n->child[0]) && rbt_node_size_check(n->child[1]);
}

bool rbt_node_23_check(const rbt_node_t *root, rbt_node_t *n) {
  if (n == NULL) {
    return true;
  }
  if (rbt_node_is_red(n->child[1])) {
    return false;
  }
  if (n != root && rbt_node_is_red(n) && rbt_node_is_red(n->child[0])) {
    return false;
  }
  return rbt_node_23_check(root, n->child[0]) &&
         rbt_node_23_check(root, n->child[1]);
}

static bool __rbt_node_balance_check(const rbt_node_t *n, int black) {
  if (n == NULL) {
    return black == 0;
  }
  if (!rbt_node_is_red(n)) {
    black--;
  }
  return __rbt_node_balance_check(n->child[0], black) &&
         __rbt_node_balance_check(n->child[1], black);
}

bool rbt_node_balance_check(rbt_node_t *root) {
  int black = 0;
  rbt_node_t *n = root;
  while (n != NULL) {
    black += (!rbt_node_is_red(n)) ? 1 : 0;
    n = n->child[0];
  }
  return __rbt_node_balance_check(root, black);
}

bool rbt_node_check(rbt_node_t *root, cmp_t cmp) {
  if (!rbt_node_bst_check(root, NULL, NULL, cmp)) {
    printf("Not BST!\n");
    return false;
  }
  if (!rbt_node_size_check(root)) {
    printf("Not size consistent!\n");
    return false;
  }
  if (!rbt_node_23_check(root, root)) {
    printf("Not 2-3 tree!\n");
    return false;
  }
  if (!rbt_node_balance_check(root)) {
    printf("Not balanced!\n");
    return false;
  }

  return true;
}

rbt_node_t *rbt_node_insert(rbt_node_t *n, void *k, void *v, cmp_t cmp) {
  if (n == NULL) {
    return rbt_node_malloc(RB_RED, k, v);
  }

  const int c = cmp(k, n->key);
  if (c < 0) {
    n->child[0] = rbt_node_insert(n->child[0], k, v, cmp);
  } else if (c > 0) {
    n->child[1] = rbt_node_insert(n->child[1], k, v, cmp);
  } else {
    n->value = v;
  }

  return rbt_node_balance(n);
}

rbt_node_t *rbt_node_delete_min(rbt_node_t *n) {
  if (n == NULL) {
    return NULL;
  }

  if (n->child[0] == NULL) {
    return NULL;
  }

  if (!rbt_node_is_red(n->child[0]) &&
      !rbt_node_is_red(n->child[0]->child[0])) {
    n = rbt_node_move_red_left(n);
  }
  n->child[0] = rbt_node_delete_min(n->child[0]);
  return rbt_node_balance(n);
}

rbt_node_t *rbt_node_delete_max(rbt_node_t *n) {
  if (n->child[0] == NULL) {
    n = rbt_node_rotate(n, 1);
  }
  if (n->child[1] == NULL) {
    return NULL;
  }
  if (!rbt_node_is_red(n->child[1]) &&
      !rbt_node_is_red(n->child[1]->child[0])) {
    n = rbt_node_move_red_right(n);
  }

  n->child[1] = rbt_node_delete_max(n->child[1]);
  return rbt_node_balance(n);
}

rbt_node_t *rbt_node_delete(rbt_node_t *n,
                            void *key,
                            cmp_t cmp_func,
                            free_func_t kfree_func) {
  if (n == NULL) {
    return NULL;
  }

  if (cmp_func(key, n->key) < 0) {
    // TRAVERSE LEFT-SUBTREE
    // Prepare n's left child to be red if it's a 2-node
    if (n->child[0] != NULL) {
      if (!rbt_node_is_red(n->child[0]) &&
          !rbt_node_is_red(n->child[0]->child[0])) {
        n = rbt_node_move_red_left(n);
      }
      n->child[0] = rbt_node_delete(n->child[0], key, cmp_func, kfree_func);
    }

  } else {
    // TRAVERSE RIGHT-SUBTREE
    // If n->left is red, rotate right to maintain left-leaning property
    // before potentially moving right. This is crucial for Sedgewick's
    // delete.
    if (rbt_node_is_red(n->child[0])) {
      n = rbt_node_rotate(n, 1);
    }

    // Case 1: Found the node and it's a leaf (or only has a left child
    // already handled)
    if (cmp_func(key, n->key) == 0 && (n->child[1] == NULL)) {
      if (kfree_func) {
        kfree_func(n->key);
      }
      free(n);
      return NULL;
    }

    // Prepare n's right child to be red if it's a 2-node (descending right)
    if (n->child[1] != NULL) {
      if (!rbt_node_is_red(n->child[1]) &&
          !rbt_node_is_red(n->child[1]->child[0])) {
        n = rbt_node_move_red_right(n);
      }

      if (cmp_func(key, n->key) == 0) {
        // Case 2: Found the node and it has a right child
        rbt_node_t *tmp = rbt_node_min(n->child[1]);
        if (kfree_func) {
          free(n->key);
        }
        n->key = tmp->key;
        n->value = tmp->value;
        n->child[1] = rbt_node_delete_min(n->child[1]);
        free(tmp);

      } else {
        // Case 3: Still traversing down the right subtree
        n->child[1] = rbt_node_delete(n->child[1], key, cmp_func, kfree_func);
      }
    }
  }

  return rbt_node_balance(n);
}

void *rbt_node_search(rbt_node_t *n, const void *key, cmp_t cmp_func) {
  while (n != NULL) {
    const int cmp = cmp_func(key, n->key);
    if (cmp < 0) {
      n = n->child[0];
    } else if (cmp > 0) {
      n = n->child[1];
    } else {
      return n->value;
    }
  }

  return NULL;
}

bool rbt_node_contains(const rbt_node_t *n, const void *key, cmp_t cmp_func) {
  while (n != NULL) {
    const int cmp = cmp_func(key, n->key);
    if (cmp < 0) {
      n = n->child[0];
    } else if (cmp > 0) {
      n = n->child[1];
    } else {
      return true;
    }
  }

  return false;
}

rbt_t *rbt_malloc(cmp_t cmp) {
  rbt_t *rbt = malloc(sizeof(rbt_t));
  rbt->root = NULL;
  rbt->cmp = cmp;
  rbt->kcopy = NULL;
  rbt->kfree = NULL;
  rbt->size = 0;
  return rbt;
}

void rbt_free(rbt_t *rbt) {
  rbt_node_free(rbt->root);
  free(rbt);
}

void rbt_insert(rbt_t *rbt, void *key, void *value) {
  assert(rbt);
  void *k = (rbt->kcopy) ? rbt->kcopy(key) : key;
  rbt->root = rbt_node_insert(rbt->root, k, value, rbt->cmp);
  rbt->root->color = RB_BLACK;
  rbt->size++;
}

void rbt_delete(rbt_t *rbt, void *key) {
  assert(rbt);
  if (rbt->size == 0) {
    return;
  }

  rbt_node_t *n = rbt_node_delete(rbt->root, key, rbt->cmp, rbt->kfree);
  rbt->size--;
  if (n) {
    rbt->root = n;
    rbt->root->color = RB_BLACK;
  }
  if (rbt->size == 0) {
    rbt->root = NULL;
  }
}

void *rbt_search(rbt_t *rbt, const void *key) {
  assert(rbt);
  return rbt_node_search(rbt->root, key, rbt->cmp);
}

bool rbt_contains(const rbt_t *rbt, const void *key) {
  assert(rbt);
  return rbt_node_contains(rbt->root, key, rbt->cmp);
}

rbt_node_t *rbt_min(const rbt_t *rbt) {
  assert(rbt);
  return rbt_node_min(rbt->root);
}

rbt_node_t *rbt_max(const rbt_t *rbt) {
  assert(rbt);
  return rbt_node_max(rbt->root);
}

size_t rbt_height(const rbt_t *rbt) {
  assert(rbt);
  return rbt_node_height(rbt->root);
}

size_t rbt_size(const rbt_t *rbt) {
  assert(rbt);
  return rbt_node_size(rbt->root);
}

void rbt_keys(const rbt_t *rbt, arr_t *keys) {
  assert(rbt);
  const rbt_node_t *lo = rbt_node_min(rbt->root);
  const rbt_node_t *hi = rbt_node_max(rbt->root);
  rbt_node_keys(rbt->root, lo->key, hi->key, keys, rbt->cmp);
}

void rbt_keys_values(const rbt_t *rbt, arr_t *keys, arr_t *values) {
  assert(rbt);
  const rbt_node_t *lo = rbt_node_min(rbt->root);
  const rbt_node_t *hi = rbt_node_max(rbt->root);
  rbt_node_keys_values(rbt->root, lo->key, hi->key, keys, values, rbt->cmp);
}

int rbt_rank(const rbt_t *rbt, const void *key) {
  assert(rbt);
  return rbt_node_rank(rbt->root, key, rbt->cmp);
}

void *rbt_select(const rbt_t *rbt, const int rank) {
  assert(rbt);
  return rbt_node_select(rbt->root, rank);
}

/*******************************************************************************
 * HASHMAP
 ******************************************************************************/

// FNV-1a constants (64-bit version is recommended for wider distribution)
#define FNV_PRIME_64 1099511628211ULL
#define FNV_OFFSET_BASIS_64 14695981039346656037ULL

size_t hm_default_hash(const void *key, const size_t key_size) {
  assert(key && key_size > 0);
  size_t hash = FNV_OFFSET_BASIS_64;
  const uint8_t *p = (const uint8_t *) key;
  for (size_t i = 0; i < key_size; i++) {
    hash ^= (size_t) p[i];
    hash *= FNV_PRIME_64;
  }
  return hash;
}

size_t hm_int_hash(const void *key) {
  assert(key);
  return hm_default_hash(key, sizeof(int));
}

size_t hm_float_hash(const void *key) {
  assert(key);
  return hm_default_hash(key, sizeof(float));
}

size_t hm_double_hash(const void *key) {
  assert(key);
  return hm_default_hash(key, sizeof(float));
}

size_t hm_string_hash(const void *key) {
  assert(key);
  return hm_default_hash(key, strlen((char *) key));
}

hm_t *hm_malloc(const size_t capacity,
                size_t (*hash)(const void *),
                int (*cmp)(const void *, const void *)) {
  assert(hash);
  assert(cmp);

  hm_t *hm = malloc(sizeof(hm_t));
  hm->entries = calloc(capacity, sizeof(hm_entry_t));
  hm->length = 0;
  hm->capacity = capacity;
  hm->hash = hash;
  hm->cmp = cmp;

  return hm;
}

void hm_free(hm_t *hm, void (*free_key)(void *), void (*free_value)(void *)) {
  assert(hm);

  for (size_t i = 0; i < hm->capacity; ++i) {
    hm_entry_t *entry = &hm->entries[i];
    if (entry->key == NULL) {
      continue;
    }
    if (free_key) {
      free_key(entry->key);
    }
    if (free_value) {
      free_value(entry->value);
    }
  }
  free(hm->entries);
  free(hm);
}

void *hm_get(const hm_t *hm, const void *key) {
  assert(hm);
  assert(hm->hash);
  assert(key);

  // Hash key
  const size_t hash = hm->hash(key);
  size_t index = hash & (hm->capacity - 1);

  // Linear probing
  while (hm->entries[index].key != NULL) {
    if (hm->cmp(key, hm->entries[index].key) == 0) {
      return hm->entries[index].value;
    }
    index = ((index + 1) >= hm->capacity) ? 0 : index + 1;
  }

  return NULL;
}

int hm_expand(hm_t *hm) {
  assert(hm);
  assert(hm->hash);
  printf("expand!\n");

  // Allocate new hashmap array.
  const size_t new_capacity = hm->capacity * 2;
  hm_entry_t *new_entries = calloc(new_capacity, sizeof(hm_entry_t));
  if (new_entries == NULL) {
    return -1;
  }

  // Iterate and move
  for (size_t i = 0; i < hm->capacity; ++i) {
    hm_entry_t *entry = &hm->entries[i];
    if (entry->key == NULL) {
      continue;
    }

    // Hash key
    const size_t hash = hm->hash(entry->key);
    size_t index = hash & (new_capacity - 1);

    // Linear probing
    while (new_entries[index].key != NULL) {
      if (hm->cmp(entry->key, new_entries[index].key) == 0) {
        break;
      }
      index = ((index + 1) >= new_capacity) ? 0 : index + 1;
    }

    // Copy
    new_entries[index].key = entry->key;
    new_entries[index].value = entry->value;
  }

  // Update
  free(hm->entries);
  hm->entries = new_entries;
  hm->capacity = new_capacity;

  return 0;
}

int hm_set(hm_t *hm, void *key, void *value) {
  assert(hm);
  assert(key);
  assert(value);

  // Expand?
  if (hm->length >= hm->capacity / 2) {
    if (hm_expand(hm) == -1) {
      return -1;
    }
  }

  // Hash key
  const size_t hash = hm->hash(key);
  size_t index = hash & (hm->capacity - 1);

  // Linear probing
  int is_new = 1;
  while (hm->entries[index].key != NULL) {
    if (hm->cmp(key, hm->entries[index].key) == 0) {
      is_new = 0;
      break;
    }
    index = ((index + 1) >= hm->capacity) ? 0 : index + 1;
  }

  // Add value
  hm_entry_t *entry = &hm->entries[index];
  entry->key = key;
  entry->value = value;
  hm->length += is_new;

  return 0;
}

hm_iter_t hm_iterator(hm_t *hm) {
  assert(hm);
  hm_iter_t it;
  it.key = NULL;
  it.value = NULL;
  it._hm = hm;
  it._index = 0;
  return it;
}

int hm_next(hm_iter_t *it) {
  assert(it);

  hm_t *hm = it->_hm;
  while (it->_index < hm->capacity) {
    size_t i = it->_index;
    it->_index++;

    if (hm->entries[i].key) {
      it->key = hm->entries[i].key;
      it->value = hm->entries[i].value;
      return 1;
    }
  }

  return 0;
}

/*******************************************************************************
 * NETWORK
 ******************************************************************************/

/**
 * Return IP and Port info from socket file descriptor `sockfd` to `ip` and
 * `port`. Returns `0` for success and `-1` for failure.
 * @returns
 * - 0 for success
 * - -1 for failure
 */
status_t ip_port_info(const int sockfd, char *ip, int *port) {
  assert(ip != NULL);
  assert(port != NULL);

  struct sockaddr_storage addr;
  socklen_t len = sizeof addr;
  if (getpeername(sockfd, (struct sockaddr *) &addr, &len) != 0) {
    return -1;
  }

  // Deal with both IPv4 and IPv6:
  char ipstr[INET6_ADDRSTRLEN];

  if (addr.ss_family == AF_INET) {
    // IPV4
    struct sockaddr_in *s = (struct sockaddr_in *) &addr;
    *port = ntohs(s->sin_port);
    inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof(ipstr));
  } else {
    // IPV6
    struct sockaddr_in6 *s = (struct sockaddr_in6 *) &addr;
    *port = ntohs(s->sin6_port);
    inet_ntop(AF_INET6, &s->sin6_addr, ipstr, sizeof(ipstr));
  }
  memcpy(ip, ipstr, strlen(ipstr));
  ip[strlen(ip)] = '\0';

  return 0;
}

/**
 * Configure TCP server
 */
status_t tcp_server_setup(tcp_server_t *server, const int port) {
  assert(server != NULL);

  // Setup server struct
  server->port = port;
  server->sockfd = -1;
  server->conn = -1;

  // Create socket
  server->sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (server->sockfd == -1) {
    LOG_ERROR("Socket creation failed...");
    return -1;
  }

  // Socket options
  const int en = 1;
  const size_t int_sz = sizeof(int);
  if (setsockopt(server->sockfd, SOL_SOCKET, SO_REUSEADDR, &en, int_sz) < 0) {
    LOG_ERROR("setsockopt(SO_REUSEADDR) failed");
  }
  // if (setsockopt(server->sockfd, SOL_SOCKET, SO_REUSEPORT, &en, int_sz) < 0)
  // {
  //   LOG_ERROR("setsockopt(SO_REUSEPORT) failed");
  // }

  // Assign IP, PORT
  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(server->port);

  // Bind newly created socket to given IP
  int retval = bind(server->sockfd, (struct sockaddr *) &addr, sizeof(addr));
  if (retval != 0) {
    LOG_ERROR("Socket bind failed: %s", strerror(errno));
    return -1;
  }

  return 0;
}

/**
 * Loop TCP server
 * @returns `0` for success, `-1` for failure
 */
status_t tcp_server_loop(tcp_server_t *server) {
  assert(server != NULL);

  // Server is ready to listen
  if ((listen(server->sockfd, 5)) != 0) {
    LOG_ERROR("Listen failed...");
    return -1;
  }

  // Accept the data packet from client and verification
  DEBUG("Server ready!");
  while (1) {
    // Accept incomming connections
    struct sockaddr_in sockaddr;
    socklen_t len = sizeof(sockaddr);
    int connfd = accept(server->sockfd, (struct sockaddr *) &sockaddr, &len);
    if (connfd < 0) {
      LOG_ERROR("Server acccept failed!");
      return -1;
    } else {
      server->conn = connfd;
      server->conn_handler(&server);
    }
  }
  DEBUG("Server shutting down ...");

  return 0;
}

/**
 * Configure TCP client
 */
status_t tcp_client_setup(tcp_client_t *client,
                          const char *server_ip,
                          const int server_port) {
  assert(client != NULL);
  assert(server_ip != NULL);

  // Setup client struct
  string_copy(client->server_ip, server_ip);
  client->server_port = server_port;
  client->sockfd = -1;

  // Create socket
  client->sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (client->sockfd == -1) {
    LOG_ERROR("Socket creation failed!");
    return -1;
  }

  // Assign IP, PORT
  struct sockaddr_in server = {.sin_family = AF_INET,
                               .sin_addr.s_addr = inet_addr(client->server_ip),
                               .sin_port = htons(client->server_port)};

  // Connect to server
  if (connect(client->sockfd, (struct sockaddr *) &server, sizeof(server)) !=
      0) {
    LOG_ERROR("Failed to connect to server!");
    return -1;
  }
  DEBUG("Connected to the server!");

  return 0;
}

/**
 * Loop TCP client
 */
status_t tcp_client_loop(tcp_client_t *client) {
  while (1) {
    if (client->loop_cb) {
      int retval = client->loop_cb(client);
      switch (retval) {
        case -1:
          return -1;
        case 1:
          break;
      }
    }
  }

  return 0;
}

/*******************************************************************************
 * MATH
 ******************************************************************************/

/**
 * Generate random number between a and b from a uniform distribution.
 * @returns Random number
 */
float randf(const float a, const float b) {
  float random = ((float) rand()) / (float) RAND_MAX;
  float diff = b - a;
  float r = random * diff;
  return a + r;
}

/**
 * Generate random vector of size n where each element is between a and b.
 */
void randvec(const real_t a, const real_t b, const size_t n, real_t *v) {
  for (size_t i = 0; i < n; i++) {
    v[i] = randf(a, b);
  }
}

/**
 * Degrees to radians.
 * @returns Radians
 */
real_t deg2rad(const real_t d) { return d * (M_PI / 180.0); }

/**
 * Radians to degrees.
 * @returns Degrees
 */
real_t rad2deg(const real_t r) { return r * (180.0 / M_PI); }

/**
 * Wrap angle `d` in degrees to +- 180 degrees.
 */
real_t wrap_180(const real_t d) {
  real_t x = fmod(d + 180, 360);
  if (x < 0) {
    x += 360;
  }

  return x - 180;
}

/**
 * Wrap angle `d` in degrees to 0 to 360 degrees.
 */
real_t wrap_360(const real_t d) {
  real_t x = fmod(d, 360);
  if (x < 0) {
    x += 360;
  }
  return x;
}

/**
 * Wrap angle `r` in radians to +- pi radians.
 */
real_t wrap_pi(const real_t r) { return deg2rad(wrap_180(rad2deg(r))); }

/**
 * Wrap angle `r` in radians to 0 to 2pi radians.
 */
real_t wrap_2pi(const real_t r) { return deg2rad(wrap_360(rad2deg(r))); }

/**
 * Compare ints.
 * @returns
 * - 0 if v1 == v2
 * - 1 if v1 > v2
 * - -1 if v1 < v2
 */
int intcmp(const int x, int y) {
  if (x > y) {
    return 1;
  } else if (x < y) {
    return -1;
  }
  return 0;
}

/**
 Compare ints.
 * @returns
 * - 0 if v1 == v2
 * - 1 if v1 > v2
 * - -1 if v1 < v2
 */
int intcmp2(const void *x, const void *y) {
  return intcmp(*(int *) x, *(int *) y);
}

/**
 * Compare reals.
 * @returns
 * - 0 if x == y
 * - 1 if x > y
 * - -1 if x < y
 */
int fltcmp(const real_t x, const real_t y) {
  if (fabs(x - y) < CMP_TOL) {
    return 0;
  } else if (x > y) {
    return 1;
  }

  return -1;
}

/**
 * Compare reals.
 * @returns
 * - 0 if x == y
 * - 1 if x > y
 * - -1 if x < y
 */
int fltcmp2(const void *x, const void *y) {
  assert(x != NULL);
  assert(y != NULL);
  return fltcmp(*(real_t *) x, *(real_t *) y);
}

/**
 * Compare strings.
 */
int strcmp2(const void *x, const void *y) {
  return strcmp((char *) x, (char *) y);
}

/**
 * Check if reals are equal.
 * @returns 1 if x == y, 0 if x != y.
 */
int flteqs(const real_t x, const real_t y) {
  return (fltcmp(x, y) == 0) ? 1 : 0;
}

/**
 * Check if strings are equal.
 * @returns 1 if x == y, 0 if x != y.
 */
int streqs(const char *x, const char *y) { return (strcmp(x, y) == 0) ? 1 : 0; }

/**
 * Cumulative Sum.
 */
void cumsum(const real_t *x, const size_t n, real_t *s) {
  s[0] = x[0];
  for (size_t i = 1; i < n; i++) {
    s[i] = x[i];
    s[i] = s[i] + s[i - 1];
  }
}

/**
 * Logspace. Generates `n` points between decades `10^a` and `10^b`.
 */
void logspace(const real_t a, const real_t b, const size_t n, real_t *x) {
  const real_t h = (b - a) / (n - 1);

  real_t c = a;
  for (size_t i = 0; i < n; i++) {
    x[i] = pow(10, c);
    c += h;
  }
}

/**
 * Pythagoras
 *
 *   c = sqrt(a^2 + b^2)
 *
 * @returns Hypotenuse of a and b
 */
real_t pythag(const real_t a, const real_t b) {
  real_t at = fabs(a);
  real_t bt = fabs(b);
  real_t ct = 0.0;
  real_t result = 0.0;

  if (at > bt) {
    ct = bt / at;
    result = at * sqrt(1.0 + ct * ct);
  } else if (bt > 0.0) {
    ct = at / bt;
    result = bt * sqrt(1.0 + ct * ct);
  } else {
    result = 0.0;
  }

  return result;
}

/**
 * Clip value `x` to be between `val_min` and `val_max`.
 */
real_t clip_value(const real_t x, const real_t vmin, const real_t vmax) {
  real_t x_tmp = x;
  x_tmp = (x_tmp > vmax) ? vmax : x_tmp;
  x_tmp = (x_tmp < vmin) ? vmin : x_tmp;
  return x_tmp;
}

/**
 * Clip vector `x` to be between `val_min` and `val_max`.
 */
void clip(real_t *x, const size_t n, const real_t vmin, const real_t vmax) {
  for (size_t i = 0; i < n; i++) {
    x[i] = (x[i] > vmax) ? vmax : x[i];
    x[i] = (x[i] < vmin) ? vmin : x[i];
  }
}

/**
 * Perform 1D Linear interpolation between `a` and `b` with `t` as the
 * interpolation hyper-parameter.
 * @returns Linear interpolated value between a and b
 */
real_t lerp(const real_t a, const real_t b, const real_t t) {
  return a * (1.0 - t) + b * t;
}

/**
 * Perform 3D Linear interpolation between `a` and `b` with `t` as the
 * interpolation hyper-parameter.
 */
void lerp3(const real_t a[3], const real_t b[3], const real_t t, real_t x[3]) {
  assert(a != NULL);
  assert(b != NULL);
  assert(x != NULL);

  x[0] = lerp(a[0], b[0], t);
  x[1] = lerp(a[1], b[1], t);
  x[2] = lerp(a[2], b[2], t);
}

/**
 * Sinc.
 * @return Result of sinc
 */
real_t sinc(const real_t x) {
  if (fabs(x) > 1e-6) {
    return sin(x) / x;
  } else {
    const real_t c2 = 1.0 / 6.0;
    const real_t c4 = 1.0 / 120.0;
    const real_t c6 = 1.0 / 5040.0;
    const real_t x2 = x * x;
    const real_t x4 = x2 * x2;
    const real_t x6 = x2 * x2 * x2;
    return 1.0 - c2 * x2 + c4 * x4 - c6 * x6;
  }
}

/**
 * Calculate mean from vector `x` of length `n`.
 * @returns Mean of x
 */
real_t mean(const real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  real_t sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += x[i];
  }
  return sum / n;
}

/**
 * Calculate median from vector `x` of length `n`.
 * @returns Median of x
 */
real_t median(const real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  // Make a copy of the original input vector x
  real_t *vals = malloc(sizeof(real_t) * n);
  for (size_t i = 0; i < n; i++) {
    vals[i] = x[i];
  }

  // Sort the values
  qsort(vals, n, sizeof(real_t), fltcmp2);

  // Get median value
  real_t median_value = 0.0;
  if ((n % 2) == 0) {
    const int bwd_idx = (int) (n - 1) / 2.0;
    const int fwd_idx = (int) (n + 1) / 2.0;
    median_value = (vals[bwd_idx] + vals[fwd_idx]) / 2.0;
  } else {
    const int midpoint_idx = n / 2.0;
    median_value = vals[midpoint_idx];
  }

  // Clean up
  free(vals);

  return median_value;
}

/**
 * Calculate variance from vector `x` of length `n`.
 * @returns Variance of x
 */
real_t var(const real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  const real_t mu = mean(x, n);
  real_t sse = 0.0;
  for (size_t i = 0; i < n; i++) {
    sse += (x[i] - mu) * (x[i] - mu);
  }

  return sse / (n - 1);
}

/**
 * Calculate standard deviation from vector `x` of length `n`.
 * @returns Standard deviation of x
 */
real_t stddev(const real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);
  return sqrt(var(x, n));
}

/*******************************************************************************
 * LINEAR ALGEBRA
 ******************************************************************************/

/**
 * Print matrix `A` of size `m x n`.
 */
void print_matrix(const char *prefix,
                  const real_t *A,
                  const size_t m,
                  const size_t n) {
  assert(prefix != NULL);
  assert(A != NULL);
  assert(m != 0);
  assert(n != 0);

  size_t idx = 0;
  printf("%s:\n", prefix);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      // printf("%.4f  ", A[idx]);
      printf("%e  ", A[idx]);
      idx++;
    }
    printf("\n");
  }
  printf("\n");
}

/**
 * Print vector `v` of length `n`.
 */
void print_vector(const char *prefix, const real_t *v, const size_t n) {
  assert(prefix != NULL);
  assert(v != NULL);
  assert(n != 0);

  printf("%s: ", prefix);
  for (size_t i = 0; i < n; i++) {
    printf("%e ", v[i]);
    // printf("%f ", v[i]);
    // printf("%.4f ", v[i]);
    // printf("%.10f ", v[i]);
  }
  printf("\n");
}

/**
 * Print float array.
 */
void print_float_array(const char *prefix, const float *arr, const size_t n) {
  assert(prefix != NULL);
  assert(arr != NULL);
  assert(n != 0);

  printf("%s: ", prefix);
  for (size_t i = 0; i < n; i++) {
    printf("%.4f ", arr[i]);
  }
  printf("\n");
}

/**
 * Print double array.
 */
void print_double_array(const char *prefix, const double *arr, const size_t n) {
  assert(prefix != NULL);
  assert(arr != NULL);
  assert(n != 0);

  printf("%s: ", prefix);
  for (size_t i = 0; i < n; i++) {
    printf("%.4f ", arr[i]);
  }
  printf("\n");
}

/**
 * Convert vector string
 */
void vec2str(const real_t *v, const int n, char *s) {
  s[0] = '[';
  for (int i = 0; i < n; i++) {
    sprintf(s + strlen(s), "%f", v[i]);
    if (i < (n - 1)) {
      strcat(s + strlen(s), ", ");
    }
  }
  strcat(s + strlen(s), "]");
}

/**
 * Convert vector string
 */
void vec2csv(const real_t *v, const int n, char *s) {
  for (int i = 0; i < n; i++) {
    sprintf(s + strlen(s), "%f", v[i]);
    if (i < (n - 1)) {
      strcat(s + strlen(s), ", ");
    }
  }
}

/**
 * Form identity matrix `A` of size `m x n`.
 */
void eye(real_t *A, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(m != 0);
  assert(n != 0);

  size_t idx = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A[idx] = (i == j) ? 1.0 : 0.0;
      idx++;
    }
  }
}

/**
 * Form ones matrix `A` of size `m x n`.
 */
void ones(real_t *A, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(m != 0);
  assert(n != 0);

  size_t idx = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A[idx] = 1.0;
      idx++;
    }
  }
}

/**
 * Form zeros matrix `A` of size `m x n`.
 */
void zeros(real_t *A, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(m != 0);
  assert(n != 0);

  size_t idx = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A[idx] = 0.0;
      idx++;
    }
  }
}

/**
 * Create skew-symmetric matrix `A` from a 3x1 vector `x`.
 */
void hat(const real_t x[3], real_t A[3 * 3]) {
  assert(x != NULL);
  assert(A != NULL);

  // First row
  A[0] = 0.0;
  A[1] = -x[2];
  A[2] = x[1];

  // Second row
  A[3] = x[2];
  A[4] = 0.0;
  A[5] = -x[0];

  // Third row
  A[6] = -x[1];
  A[7] = x[0];
  A[8] = 0.0;
}

/**
 * Opposite of the skew-symmetric matrix
 */
void vee(const real_t A[3 * 3], real_t x[3]) {
  assert(A != NULL);
  assert(x != NULL);

  const real_t A02 = A[2];
  const real_t A10 = A[3];
  const real_t A21 = A[7];

  x[0] = A21;
  x[1] = A02;
  x[2] = A10;
}

/**
 * Perform forward substitution with a lower triangular matrix `L`, column
 * vector `b` and solve for vector `y` of size `n`.
 */
void fwdsubs(const real_t *L, const real_t *b, real_t *y, const size_t n) {
  assert(L != NULL);
  assert(b != NULL);
  assert(y != NULL);
  assert(n > 0);

  for (size_t i = 0; i < n; i++) {
    real_t alpha = b[i];
    for (size_t j = 0; j < i; j++) {
      alpha -= L[i * n + j] * y[j];
    }
    y[i] = alpha / L[i * n + i];
  }
}

/**
 * Perform backward substitution with a upper triangular matrix `U`, column
 * vector `y` and solve for vector `x` of size `n`.
 */
void bwdsubs(const real_t *U, const real_t *y, real_t *x, const size_t n) {
  assert(U != NULL);
  assert(y != NULL);
  assert(x != NULL);
  assert(n > 0);

  for (int i = n - 1; i >= 0; i--) {
    real_t alpha = y[i];
    for (int j = i; j < (int) n; j++) {
      alpha -= U[i * n + j] * x[j];
    }
    x[i] = alpha / U[i * n + i];
  }
}

/**
 * Enforce semi-positive definite. This function assumes the matrix `A` is
 * square where number of rows `m` and columns `n` is equal, and symmetric.
 */
void enforce_spd(real_t *A, const int m, const int n) {
  assert(A != NULL);
  assert(m == n);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      const real_t a = A[(i * n) + j];
      const real_t b = A[(j * n) + i];
      A[(i * n) + j] = (a + b) / 2.0;
    }
  }
}

/**
 * Form identity matrix `A` of size `m x n`.
 */
void eyef(float *A, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(m != 0);
  assert(n != 0);

  size_t idx = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A[idx] = (i == j) ? 1.0 : 0.0;
      idx++;
    }
  }
}

/**
 * Form ones matrix `A` of size `m x n`.
 */
void onesf(float *A, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(m != 0);
  assert(n != 0);

  size_t idx = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A[idx] = 1.0;
      idx++;
    }
  }
}

/**
 * Form zeros matrix `A` of size `m x n`.
 */
void zerosf(float *A, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(m != 0);
  assert(n != 0);

  size_t idx = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A[idx] = 0.0;
      idx++;
    }
  }
}

/**
 * Create skew-symmetric matrix `A` from a 3x1 vector `x`.
 */
void hatf(const float x[3], float A[3 * 3]) {
  assert(x != NULL);
  assert(A != NULL);

  // First row
  A[0] = 0.0;
  A[1] = -x[2];
  A[2] = x[1];

  // Second row
  A[3] = x[2];
  A[4] = 0.0;
  A[5] = -x[0];

  // Third row
  A[6] = -x[1];
  A[7] = x[0];
  A[8] = 0.0;
}

/**
 * Opposite of the skew-symmetric matrix
 */
void veef(const float A[3 * 3], float x[3]) {
  assert(A != NULL);
  assert(x != NULL);

  const float A02 = A[2];
  const float A10 = A[3];
  const float A21 = A[7];

  x[0] = A21;
  x[1] = A02;
  x[2] = A10;
}

/**
 * Perform forward substitution with a lower triangular matrix `L`, column
 * vector `b` and solve for vector `y` of size `n`.
 */
void fwdsubsf(const float *L, const float *b, float *y, const size_t n) {
  assert(L != NULL);
  assert(b != NULL);
  assert(y != NULL);
  assert(n > 0);

  for (size_t i = 0; i < n; i++) {
    float alpha = b[i];
    for (size_t j = 0; j < i; j++) {
      alpha -= L[i * n + j] * y[j];
    }
    y[i] = alpha / L[i * n + i];
  }
}

/**
 * Perform backward substitution with a upper triangular matrix `U`, column
 * vector `y` and solve for vector `x` of size `n`.
 */
void bwdsubsf(const float *U, const float *y, float *x, const size_t n) {
  assert(U != NULL);
  assert(y != NULL);
  assert(x != NULL);
  assert(n > 0);

  for (int i = n - 1; i >= 0; i--) {
    float alpha = y[i];
    for (int j = i; j < (int) n; j++) {
      alpha -= U[i * n + j] * x[j];
    }
    x[i] = alpha / U[i * n + i];
  }
}

/**
 * Enforce semi-positive definite. This function assumes the matrix `A` is
 * square where number of rows `m` and columns `n` is equal, and symmetric.
 */
void enforce_spdf(float *A, const int m, const int n) {
  assert(A != NULL);
  assert(m == n);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      const float a = A[(i * n) + j];
      const float b = A[(j * n) + i];
      A[(i * n) + j] = (a + b) / 2.0;
    }
  }
}

/**
 * Malloc matrix of size `m x n`.
 */
real_t *mat_malloc(const size_t m, const size_t n) {
  assert(m > 0);
  assert(n > 0);
  return calloc(m * n, sizeof(real_t));
}

/**
 * Compare two matrices `A` and `B` of size `m x n`.
 *
 * @returns
 * - 0 if A == B
 * - 1 if A > B
 * - -1 if A < B
 */
int mat_cmp(const real_t *A, const real_t *B, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(B != NULL);
  assert(m > 0);
  assert(n > 0);

  size_t index = 0;

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      int retval = fltcmp(A[index], B[index]);
      if (retval != 0) {
        printf("Failed at index[%zu]\n", index);
        return retval;
      }
      index++;
    }
  }

  return 0;
}

/**
 * Check to see if two matrices `A` and `B` of size `m x n` are equal to a
 * tolerance.
 * @returns 1 if A == B or 0 if A != B
 */
int mat_equals(const real_t *A,
               const real_t *B,
               const size_t m,
               const size_t n,
               const real_t tol) {
  assert(A != NULL);
  assert(B != NULL);
  assert(m > 0);
  assert(n > 0);
  assert(tol > 0);

  size_t index = 0;

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      if (fabs(A[index] - B[index]) > tol) {
        printf("Failed at index[%zu]\n", index);
        return 0;
      }
      index++;
    }
  }

  return 1;
}

// /**
//  * Save matrix `A` of size `m x n` to `save_path`.
//  * @returns `0` for success, `-1` for failure
//  */
// int mat_save(const char *save_path, const real_t *A, const int m, const int
// n) {
//   assert(save_path != NULL);
//   assert(A != NULL);
//   assert(m > 0);
//   assert(n > 0);
//
//   FILE *csv_file = fopen(save_path, "w");
//   if (csv_file == NULL) {
//     return -1;
//   }
//
//   int idx = 0;
//   for (int i = 0; i < m; i++) {
//     for (int j = 0; j < n; j++) {
//       fprintf(csv_file, "%.18e", A[idx]);
//       idx++;
//       if ((j + 1) != n) {
//         fprintf(csv_file, ",");
//       }
//     }
//     fprintf(csv_file, "\n");
//   }
//   fclose(csv_file);
//
//   return 0;
// }
//
// /**
//  * Load matrix from file in `mat_path`, on success `num_rows` and `num_cols`
//  * will be set respectively.
//  */
// real_t *mat_load(const char *mat_path, int *num_rows, int *num_cols) {
//   assert(mat_path != NULL);
//   assert(num_rows != NULL);
//   assert(num_cols != NULL);
//
//   // Obtain number of rows and columns in csv data
//   *num_rows = dsv_rows(mat_path);
//   *num_cols = dsv_cols(mat_path, ',');
//   if (*num_rows == -1 || *num_cols == -1) {
//     return NULL;
//   }
//
//   // Initialize memory for csv data
//   real_t *A = malloc(sizeof(real_t) * *num_rows * *num_cols);
//
//   // Load file
//   FILE *infile = fopen(mat_path, "r");
//   if (infile == NULL) {
//     free(A);
//     return NULL;
//   }
//
//   // Loop through data
//   char line[MAX_LINE_LENGTH] = {0};
//   int row_idx = 0;
//   int col_idx = 0;
//   int idx = 0;
//
//   // Loop through data line by line
//   while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
//     // Ignore if comment line
//     if (line[0] == '#') {
//       continue;
//     }
//
//     // Iterate through values in line separated by commas
//     char entry[100] = {0};
//     for (size_t i = 0; i < strlen(line); i++) {
//       char c = line[i];
//       if (c == ' ') {
//         continue;
//       }
//
//       if (c == ',' || c == '\n') {
//         A[idx] = strtod(entry, NULL);
//         idx++;
//
//         memset(entry, '\0', sizeof(char) * 100);
//         col_idx++;
//       } else {
//         entry[strlen(entry)] = c;
//       }
//     }
//
//     col_idx = 0;
//     row_idx++;
//   }
//
//   // Clean up
//   fclose(infile);
//
//   return A;
// }

/**
 * Set matrix `A` with value `val` at `(i, j)`.
 */
void mat_set(real_t *A,
             const size_t stride,
             const size_t i,
             const size_t j,
             const real_t val) {
  assert(A != NULL);
  assert(stride != 0);
  A[(i * stride) + j] = val;
}

/**
 * Get value from matrix `A` with `stride` at `(i, j)`.
 */
real_t
mat_val(const real_t *A, const size_t stride, const size_t i, const size_t j) {
  assert(A != NULL);
  assert(stride != 0);
  return A[(i * stride) + j];
}

/**
 * Copy matrix `src` of size `m x n` to `dest`.
 */
void mat_copy(const real_t *src, const int m, const int n, real_t *dest) {
  assert(src != NULL);
  assert(m > 0);
  assert(n > 0);
  assert(dest != NULL);

  for (int i = 0; i < (m * n); i++) {
    dest[i] = src[i];
  }
}

/**
 * Set matrix row.
 */
void mat_row_set(real_t *A,
                 const size_t stride,
                 const int row_idx,
                 const real_t *x) {
  int vec_idx = 0;
  for (size_t i = 0; i < stride; i++) {
    A[(stride * row_idx) + i] = x[vec_idx++];
  }
}

/**
 * Set matrix column.
 */
void mat_col_set(real_t *A,
                 const size_t stride,
                 const int num_rows,
                 const int col_idx,
                 const real_t *x) {
  int vec_idx = 0;
  for (int i = 0; i < num_rows; i++) {
    A[i * stride + col_idx] = x[vec_idx++];
  }
}

/**
 * Get matrix column.
 */
void mat_col_get(const real_t *A,
                 const int m,
                 const int n,
                 const int col_idx,
                 real_t *x) {
  int vec_idx = 0;
  for (int i = 0; i < m; i++) {
    x[vec_idx++] = A[i * n + col_idx];
  }
}

/**
 * Get matrix sub-block from `A` with `stride` from row and column start `rs`
 * and `cs`, to row and column end `re` and `ce`. The sub-block is written to
 * `block`.
 */
void mat_block_get(const real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   real_t *block) {
  assert(A != NULL);
  assert(block != NULL);
  assert(A != block);
  assert(stride != 0);

  size_t idx = 0;
  for (size_t i = rs; i <= re; i++) {
    for (size_t j = cs; j <= ce; j++) {
      // block[idx] = mat_val(A, stride, i, j);
      block[idx] = A[(i * stride) + j];
      idx++;
    }
  }
}

/**
 * Set matrix sub-block `block` to `A` with `stride` from row and column start
 * `rs` and `cs`, to row and column end `re` and `ce`.
 */
void mat_block_set(real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   const real_t *block) {
  assert(A != NULL);
  assert(block != NULL);
  assert(A != block);
  assert(stride != 0);

  size_t idx = 0;
  for (size_t i = rs; i <= re; i++) {
    for (size_t j = cs; j <= ce; j++) {
      A[(i * stride) + j] = block[idx];
      idx++;
    }
  }
}

/**
 * Add to matrix sub-block in `A` with `block` from row and column start `rs`
 * and `cs`, to row and column end `re` and `ce`.
 */
void mat_block_add(real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   const real_t *block) {
  assert(A != NULL);
  assert(block != NULL);
  assert(A != block);
  assert(stride != 0);

  size_t idx = 0;
  for (size_t i = rs; i <= re; i++) {
    for (size_t j = cs; j <= ce; j++) {
      A[(i * stride) + j] += block[idx];
      idx++;
    }
  }
}

/**
 * Subtract matrix sub-block in `A` with `block` from row and column start `rs`
 * and `cs`, to row and column end `re` and `ce`.
 */
void mat_block_sub(real_t *A,
                   const size_t stride,
                   const size_t rs,
                   const size_t re,
                   const size_t cs,
                   const size_t ce,
                   const real_t *block) {
  assert(A != NULL);
  assert(block != NULL);
  assert(A != block);
  assert(stride != 0);

  size_t idx = 0;
  for (size_t i = rs; i <= re; i++) {
    for (size_t j = cs; j <= ce; j++) {
      A[(i * stride) + j] -= block[idx];
      idx++;
    }
  }
}

/**
 * Get diagonal vector `d` from matrix `A` of size `m x n`.
 */
void mat_diag_get(const real_t *A, const int m, const int n, real_t *d) {
  int mat_index = 0;
  int vec_index = 0;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        d[vec_index] = A[mat_index];
        vec_index++;
      }
      mat_index++;
    }
  }
}

/**
 * Set the diagonal of matrix `A` of size `m x n` with vector `d`.
 */
void mat_diag_set(real_t *A, const int m, const int n, const real_t *d) {
  assert(A != NULL);
  assert(m > 0);
  assert(n > 0);
  assert(d != NULL);

  int mat_index = 0;
  int vec_index = 0;

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        A[mat_index] = d[vec_index];
        vec_index++;
      } else {
        A[mat_index] = 0.0;
      }
      mat_index++;
    }
  }
}

/**
 * Get upper triangular square matrix of `A` of size `m x m`, results are
 * outputted to `U`.
 */
void mat_triu(const real_t *A, const size_t m, real_t *U) {
  assert(A != NULL);
  assert(m > 0);
  assert(U != NULL);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < m; j++) {
      U[i * m + j] = (j >= i) ? A[i * m + j] : 0.0;
    }
  }
}

/**
 * Get lower triangular square matrix of `A` of size `m x m`, results are
 * outputted to `L`.
 */
void mat_tril(const real_t *A, const size_t m, real_t *L) {
  assert(A != NULL);
  assert(m > 0);
  assert(L != NULL);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < m; j++) {
      L[i * m + j] = (j <= i) ? A[i * m + j] : 0.0;
    }
  }
}

/**
 * Get the trace matrix of `A` of size `m x n`.
 */
real_t mat_trace(const real_t *A, const size_t m, const size_t n) {
  assert(A != NULL);
  assert(m > 0);
  assert(n > 0);

  real_t tr = 0.0;
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      tr += (i == j) ? A[i * n + j] : 0.0;
    }
  }
  return tr;
}

/**
 * Transpose of matrix `A` of size `m x n`, results are outputted to `A_t`.
 */
void mat_transpose(const real_t *A, size_t m, size_t n, real_t *A_t) {
  assert(A != NULL && A != A_t);
  assert(m > 0 && n > 0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      A_t[(j * m) + i] = A[(i * n) + j];
    }
  }
}

/**
 * Add two matrices `A` and `B` of size `m x n`, results are outputted to `C`.
 */
void mat_add(const real_t *A, const real_t *B, real_t *C, size_t m, size_t n) {
  assert(A != NULL && B != NULL && C != NULL);
  assert(m > 0 && n > 0);

  for (size_t i = 0; i < (m * n); i++) {
    C[i] = A[i] + B[i];
  }
}

/**
 * Subtract two matrices `A` and `B` of size `m x n`, results are outputted to
 * matrix `C`.
 */
void mat_sub(const real_t *A, const real_t *B, real_t *C, size_t m, size_t n) {
  assert(A != NULL && B != NULL && C != NULL && B != C && A != C);
  assert(m > 0 && n > 0);

  for (size_t i = 0; i < (m * n); i++) {
    C[i] = A[i] - B[i];
  }
}

/**
 * Scale matrix `A` of size `m x n` inplace with `scale`.
 */
void mat_scale(real_t *A, const size_t m, const size_t n, const real_t scale) {
  assert(A != NULL);
  assert(m > 0 && n > 0);

  for (size_t i = 0; i < (m * n); i++) {
    A[i] = A[i] * scale;
  }
}

/**
 * Copy 3x3 matrix from `src` to `dst`.
 */
void mat3_copy(const real_t src[3 * 3], real_t dst[3 * 3]) {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];

  dst[3] = src[3];
  dst[4] = src[4];
  dst[5] = src[5];

  dst[6] = src[6];
  dst[7] = src[7];
  dst[8] = src[8];
}

/**
 * Add 3x3 matrix `A + B = C`.
 */
void mat3_add(const real_t A[3 * 3], const real_t B[3 * 3], real_t C[3 * 3]) {
  C[0] = A[0] + B[0];
  C[1] = A[1] + B[1];
  C[2] = A[2] + B[2];

  C[3] = A[3] + B[3];
  C[4] = A[4] + B[4];
  C[5] = A[5] + B[5];

  C[6] = A[6] + B[6];
  C[7] = A[7] + B[7];
  C[8] = A[8] + B[8];
}

/**
 * Subtract 3x3 matrix `A - B = C`.
 */
void mat3_sub(const real_t A[3 * 3], const real_t B[3 * 3], real_t C[3 * 3]) {
  C[0] = A[0] - B[0];
  C[1] = A[1] - B[1];
  C[2] = A[2] - B[2];

  C[3] = A[3] - B[3];
  C[4] = A[4] - B[4];
  C[5] = A[5] - B[5];

  C[6] = A[6] - B[6];
  C[7] = A[7] - B[7];
  C[8] = A[8] - B[8];
}

/**
 * Create new vector of length `n` in heap memory.
 * @returns Heap allocated vector
 */
real_t *vec_malloc(const real_t *x, const size_t n) {
  assert(n > 0);
  real_t *vec = calloc(n, sizeof(real_t));
  for (size_t i = 0; i < n; i++) {
    vec[i] = x[i];
  }

  return vec;
}

/**
 * Copy vector `src` of length `n` to `dest`.
 */
void vec_copy(const real_t *src, const size_t n, real_t *dest) {
  assert(src != NULL);
  assert(n > 0);
  assert(dest != NULL);

  for (size_t i = 0; i < n; i++) {
    dest[i] = src[i];
  }
}

/**
 * Check if vectors `x` and `y` of length `n` are equal.
 * @returns
 * - 1 for x == y
 * - 0 for x != y
 */
int vec_equals(const real_t *x, const real_t *y, const size_t n) {
  assert(x != NULL);
  assert(y != NULL);
  assert(n > 0);

  for (size_t i = 0; i < n; i++) {
    if (fltcmp(x[i], y[i]) != 0) {
      return 0;
    }
  }

  return 1;
}

/**
 * Get minimal value in vector `x` of length `n`.
 */
real_t vec_min(const real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  real_t y = x[0];
  for (size_t i = 1; i < n; i++) {
    y = (x[i] < y) ? x[i] : y;
  }

  return y;
}

/**
 * Get minimal value in vector `x` of length `n`.
 */
real_t vec_max(const real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  real_t y = x[0];
  for (size_t i = 1; i < n; i++) {
    y = (x[i] > y) ? x[i] : y;
  }

  return y;
}

/**
 * Get minimal, maximum value in vector `x` of length `n` as `vmin`, `vmax` as
 * well as the range `r`.
 */
void vec_range(const real_t *x,
               const size_t n,
               real_t *vmin,
               real_t *vmax,
               real_t *r) {
  assert(x != NULL);
  assert(n > 0);
  assert(vmin != NULL);
  assert(vmax != NULL);
  assert(r != NULL);

  *vmin = x[0];
  *vmax = x[0];
  for (size_t i = 1; i < n; i++) {
    *vmin = (x[i] < *vmin) ? x[i] : *vmin;
    *vmax = (x[i] > *vmax) ? x[i] : *vmax;
  }
  *r = vmax - vmin;
}

// /**
//  * Load vector.
//  *
//  * @param vec_path Path to csv containing vector values
//  * @param m Number of rows
//  * @param n Number of cols
//  *
//  * @returns Vector or `NULL` for failure
//  */
// real_t *vec_load(const char *vec_path, int *m, int *n) {
//   assert(vec_path != NULL);
//   assert(m != NULL);
//   assert(n != NULL);
//
//   // Obtain number of rows and columns in csv data
//   *m = dsv_rows(vec_path);
//   *n = dsv_cols(vec_path, ',');
//   if (*m > 0 && *n == -1) {
//     // Load file
//     FILE *infile = fopen(vec_path, "r");
//     if (infile == NULL) {
//       return NULL;
//     }
//
//     // Loop through data line by line
//     char line[MAX_LINE_LENGTH] = {0};
//     while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
//       // Ignore if comment line
//       if (line[0] == '#') {
//         continue;
//       }
//
//       if (strlen(line) == 0) {
//         fclose(infile);
//         return NULL;
//       }
//     }
//
//     *n = 1;
//   } else if (*m == -1 || *n == -1) {
//     return NULL;
//   }
//
//   // Initialize memory for csv data
//   real_t *x = malloc(sizeof(real_t) * *m * *n);
//
//   // Load file
//   FILE *infile = fopen(vec_path, "r");
//   if (infile == NULL) {
//     free(x);
//     return NULL;
//   }
//
//   // Loop through data
//   char line[MAX_LINE_LENGTH] = {0};
//   int row_idx = 0;
//   int col_idx = 0;
//   int idx = 0;
//
//   // Loop through data line by line
//   while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
//     // Ignore if comment line
//     if (line[0] == '#') {
//       continue;
//     }
//
//     // Iterate through values in line separated by commas
//     char entry[100] = {0};
//     for (size_t i = 0; i < strlen(line); i++) {
//       char c = line[i];
//       if (c == ' ') {
//         continue;
//       }
//
//       if (c == ',' || c == '\n') {
//         x[idx] = strtod(entry, NULL);
//         idx++;
//
//         memset(entry, '\0', sizeof(char) * 100);
//         col_idx++;
//       } else {
//         entry[strlen(entry)] = c;
//       }
//     }
//
//     col_idx = 0;
//     row_idx++;
//   }
//
//   // Clean up
//   fclose(infile);
//
//   return x;
// }

/**
 * Sum two vectors `x` and `y` of length `n` to `z`.
 */
void vec_add(const real_t *x, const real_t *y, real_t *z, size_t n) {
  assert(x != NULL && y != NULL && z != NULL && x != y && x != z);
  assert(n > 0);

  for (size_t i = 0; i < n; i++) {
    z[i] = x[i] + y[i];
  }
}

/**
 * Subtract two vectors `x` and `y` of length `n` to `z`.
 */
void vec_sub(const real_t *x, const real_t *y, real_t *z, size_t n) {
  assert(x != NULL && y != NULL && z != NULL && x != y && x != z);
  assert(n > 0);

  for (size_t i = 0; i < n; i++) {
    z[i] = x[i] - y[i];
  }
}

/**
 * Scale a vector `x` of length `n` with `scale` in-place.
 */
void vec_scale(real_t *x, const size_t n, const real_t scale) {
  assert(x != NULL);
  assert(n > 0);

  for (size_t i = 0; i < n; i++) {
    x[i] = x[i] * scale;
  }
}

/**
 * Calculate vector norm of `x` of length `n`.
 * @returns Norm of vector x
 */
real_t vec_norm(const real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  real_t sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sqrt(sum);
}

/**
 * Normalize vector `x` of length `n` in place.
 */
void vec_normalize(real_t *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  const real_t norm = vec_norm(x, n);
  for (size_t i = 0; i < n; i++) {
    x[i] = x[i] / norm;
  }
}

/**
 * Copy vector of size 3 from `src` to `dst`.
 */
void vec3_copy(const real_t src[3], real_t dst[3]) {
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
}

/**
 * Add vector of size 3 `x + y = z`.
 */
void vec3_add(const real_t x[3], const real_t y[3], real_t z[3]) {
  z[0] = x[0] + y[0];
  z[1] = x[1] + y[1];
  z[2] = x[2] + y[2];
}

/**
 * Subtract vector of size 3 `x - y = z`.
 */
void vec3_sub(const real_t x[3], const real_t y[3], real_t z[3]) {
  z[0] = x[0] - y[0];
  z[1] = x[1] - y[1];
  z[2] = x[2] - y[2];
}

/**
 * Scale vector of size 3.
 */
void vec3_scale(const real_t a[3], const real_t s, real_t b[3]) {
  b[0] = a[0] * s;
  b[1] = a[1] * s;
  b[2] = a[2] * s;
}

/**
 * Dot product vector of size 3 `a * b`.
 */
real_t vec3_dot(const real_t a[3], const real_t b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Cross product between vector `a` and `b`, output is written to `c`.
 */
void vec3_cross(const real_t a[3], const real_t b[3], real_t c[3]) {
  assert(a != b);
  assert(a != c);

  // cx = ay * bz - az * by
  // cy = az * bx - ax * bz
  // cz = ax * by - ay * bx
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * Calculate the norm of vector `x` of size 3.
 */
real_t vec3_norm(const real_t x[3]) {
  return sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
}

/**
 * Normalize vector `x` of size 3.
 */
void vec3_normalize(real_t x[3]) {
  const real_t n = vec3_norm(x);
  x[0] = x[0] / n;
  x[1] = x[1] / n;
  x[2] = x[2] / n;
}

/**
 * Dot product of two matrices or vectors `A` and `B` of size `A_m x A_n` and
 * `B_m x B_n`. Results are written to `C`.
 */
void dot(const real_t *A,
         const size_t A_m,
         const size_t A_n,
         const real_t *B,
         const size_t B_m,
         const size_t B_n,
         real_t *C) {
  assert(A != NULL && B != NULL && A != C && B != C);
  assert(A_m > 0 && A_n > 0 && B_m > 0 && B_n > 0);
  assert(A_n == B_m);

#ifdef USE_CBLAS
#if PRECISION == 1
  cblas_sgemm(CblasRowMajor, // Matrix data arrangement
              CblasNoTrans,  // Transpose A
              CblasNoTrans,  // Transpose B
              A_m,           // Number of rows in A and C
              B_n,           // Number of cols in B and C
              A_n,           // Number of cols in A
              1.0,           // Scaling factor for the product of A and B
              A,             // Matrix A
              A_n,           // First dimension of A
              B,             // Matrix B
              B_n,           // First dimension of B
              0.0,           // Scale factor for C
              C,             // Output
              B_n);          // First dimension of C
#elif PRECISION == 2
  cblas_dgemm(CblasRowMajor, // Matrix data arrangement
              CblasNoTrans,  // Transpose A
              CblasNoTrans,  // Transpose B
              A_m,           // Number of rows in A and C
              B_n,           // Number of cols in B and C
              A_n,           // Number of cols in A
              1.0,           // Scaling factor for the product of A and B
              A,             // Matrix A
              A_n,           // First dimension of A
              B,             // Matrix B
              B_n,           // First dimension of B
              0.0,           // Scale factor for C
              C,             // Output
              B_n);          // First dimension of C
#endif
#else
  size_t m = A_m;
  size_t n = B_n;

  memset(C, 0, sizeof(real_t) * A_m * B_n);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < A_n; k++) {
        C[(i * n) + j] += A[(i * A_n) + k] * B[(k * B_n) + j];
      }
    }
  }

#endif
}

void dotf(const float *A,
          const size_t A_m,
          const size_t A_n,
          const float *B,
          const size_t B_m,
          const size_t B_n,
          float *C) {
#ifdef USE_CBLAS
  assert(A != NULL && B != NULL && A != C && B != C);
  assert(A_m > 0 && A_n > 0 && B_m > 0 && B_n > 0);
  assert(A_n == B_m);
  cblas_sgemm(CblasRowMajor, // Matrix data arrangement
              CblasNoTrans,  // Transpose A
              CblasNoTrans,  // Transpose B
              A_m,           // Number of rows in A and C
              B_n,           // Number of cols in B and C
              A_n,           // Number of cols in A
              1.0,           // Scaling factor for the product of A and B
              A,             // Matrix A
              A_n,           // First dimension of A
              B,             // Matrix B
              B_n,           // First dimension of B
              0.0,           // Scale factor for C
              C,             // Output
              B_n);          // First dimension of C
#else
  size_t m = A_m;
  size_t n = B_n;
  memset(C, 0, sizeof(float) * A_m * B_n);
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      for (size_t k = 0; k < A_n; k++) {
        C[(i * n) + j] += A[(i * A_n) + k] * B[(k * B_n) + j];
      }
    }
  }
#endif
}

/**
 * Dot product of two matrices or vectors `A` and `B` of size `A_m x A_n` and
 * `B_m x B_n`. Results are written to `C`.
 */
void dot3(const real_t *A,
          const size_t A_m,
          const size_t A_n,
          const real_t *B,
          const size_t B_m,
          const size_t B_n,
          const real_t *C,
          const size_t C_m,
          const size_t C_n,
          real_t *D) {
  real_t *AB = malloc(sizeof(real_t) * A_m * B_n);
  dot(A, A_m, A_n, B, B_m, B_n, AB);
  dot(AB, A_m, B_m, C, C_m, C_n, D);
  free(AB);
}

/**
 * Mulitply Y = X' * A * X
 */
void dot_XtAX(const real_t *X,
              const size_t X_m,
              const size_t X_n,
              const real_t *A,
              const size_t A_m,
              const size_t A_n,
              real_t *Y) {
  assert(X != NULL);
  assert(A != NULL);
  assert(Y != NULL);
  assert(X_m == A_m);

  real_t *XtA = malloc(sizeof(real_t) * (X_m * A_m));
  real_t *Xt = malloc(sizeof(real_t) * (X_m * X_n));

  mat_transpose(X, X_m, X_n, Xt);
  dot(Xt, X_n, X_m, A, A_m, A_n, XtA);
  dot(XtA, X_m, A_m, Xt, X_n, X_m, Y);

  free(Xt);
  free(XtA);
}

/**
 * Mulitply Y = X * A * X'
 */
void dot_XAXt(const real_t *X,
              const size_t X_m,
              const size_t X_n,
              const real_t *A,
              const size_t A_m,
              const size_t A_n,
              real_t *Y) {
  assert(X != NULL);
  assert(A != NULL);
  assert(Y != NULL);
  assert(X_n == A_m);

  real_t *Xt = malloc(sizeof(real_t) * (X_m * X_n));
  real_t *XA = malloc(sizeof(real_t) * (X_m * A_n));

  mat_transpose(X, X_m, X_n, Xt);
  dot(X, X_m, X_n, A, A_m, A_n, XA);
  dot(XA, X_m, A_n, Xt, X_n, X_m, Y);

  free(XA);
  free(Xt);
}

/**
 * Invert a block diagonal matrix.
 */
void bdiag_inv(const real_t *A, const int m, const int bs, real_t *A_inv) {
  real_t *A_sub = malloc(sizeof(real_t) * bs * bs);
  real_t *A_sub_inv = malloc(sizeof(real_t) * bs * bs);
  zeros(A_inv, m, m);

  for (int idx = 0; idx < m; idx += bs) {
    const int rs = idx;
    const int re = idx + bs - 1;
    const int cs = idx;
    const int ce = idx + bs - 1;
    mat_block_get(A, m, rs, re, cs, ce, A_sub);

    // Invert using SVD
    // pinv(A_sub, bs, bs, A_sub_inv);

    // Inverse using Eigen-decomp
    eig_inv(A_sub, bs, bs, 0, A_sub_inv);

    mat_block_set(A_inv, m, rs, re, cs, ce, A_sub_inv);
  }

  free(A_sub);
  free(A_sub_inv);
}

/**
 * Invert a sub block diagonal matrix.
 */
void bdiag_inv_sub(const real_t *A,
                   const int stride,
                   const int m,
                   const int bs,
                   real_t *A_inv) {
  real_t *A_sub = malloc(sizeof(real_t) * bs * bs);
  real_t *A_sub_inv = malloc(sizeof(real_t) * bs * bs);
  zeros(A_inv, m, m);

  for (int idx = 0; idx < m; idx += bs) {
    const int rs = idx;
    const int re = idx + bs - 1;
    const int cs = idx;
    const int ce = idx + bs - 1;
    mat_block_get(A, stride, rs, re, cs, ce, A_sub);

    // Invert using SVD
    // pinv(A_sub, bs, bs, A_sub_inv);

    // Inverse using Eigen-decomp
    eig_inv(A_sub, bs, bs, 0, A_sub_inv);

    mat_block_set(A_inv, m, rs, re, cs, ce, A_sub_inv);
  }

  free(A_sub);
  free(A_sub_inv);
}

/**
 * Dot product of A * x = b, where A is a block diagonal matrix, x is a vector
 * and b is the result.
 */
void bdiag_dot(const real_t *A,
               const int m,
               const int n,
               const int bs,
               const real_t *x,
               real_t *b) {
  real_t *A_sub = malloc(sizeof(real_t) * bs * bs);
  real_t *x_sub = malloc(sizeof(real_t) * bs);

  for (int idx = 0; idx < m; idx += bs) {
    const int rs = idx;
    const int re = idx + bs - 1;
    const int cs = idx;
    const int ce = idx + bs - 1;
    mat_block_get(A, m, rs, re, cs, ce, A_sub);
    vec_copy(x + rs, bs, x_sub);
    dot(A_sub, bs, bs, x_sub, bs, 1, b + rs);
  }

  free(A_sub);
  free(x_sub);
}

/**
 * Check inverted matrix A by multiplying by its inverse.
 * @returns `0` for succces, `-1` for failure.
 */
int check_inv(const real_t *A, const real_t *A_inv, const int m) {
  const real_t tol = 1e-2;
  real_t *inv_check = calloc(m * m, sizeof(real_t));
  dot(A, m, m, A_inv, m, m, inv_check);
  // print_matrix("inv_check", inv_check, m, m);
  // gnuplot_matshow(inv_check, m, m);
  // exit(0);

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      const real_t target = (i == j) ? 1.0 : 0.0;
      const real_t val = inv_check[i * m + j];
      const real_t diff = fabs(val - target);

      if ((diff > tol) != 0) {
        printf("[%d, %d] got %e, diff: %e\n", i, j, val, diff);
        free(inv_check);
        return -1;
      }
    }
  }

  free(inv_check);
  return 0;
}

/**
 * Return the linear least squares norm.
 */
real_t check_Axb(const real_t *A,
                 const real_t *x,
                 const real_t *b,
                 const int m,
                 const int n) {
  real_t *b_est = malloc(sizeof(real_t) * m);
  real_t *diff = malloc(sizeof(real_t) * m);
  real_t r_sq = 0.0;

  dot(A, m, n, x, n, 1, b_est);
  vec_sub(b_est, b, diff, m);
  dot(diff, 1, m, diff, m, 1, &r_sq);

  free(b_est);
  free(diff);

  return sqrt(r_sq);
}

/**
 * Check analytical jacobian `jac` with name `jac_name` and compare it with a
 * numerically differentiated jacobian `fdiff` (finite diff). Where both
 * matrices are of size `m x n`, `tol` denotes the tolerance to consider both
 * `jac` and `fdiff` to be close enough. For debugging purposes use `verbose`
 * to show the matrices and differences.
 *
 * @returns
 * - 0 for success
 * - -1 for failure
 */
int check_jacobian(const char *jac_name,
                   const real_t *fdiff,
                   const real_t *jac,
                   const size_t m,
                   const size_t n,
                   const real_t tol,
                   const int verbose) {
  assert(jac_name != NULL);
  assert(fdiff != NULL);
  assert(jac != NULL);
  assert(m > 0);
  assert(n > 0);
  assert(tol > 0);

  int retval = 0;
  int ok = 1;
  real_t *delta = mat_malloc(m, n);
  mat_sub(fdiff, jac, delta, m, n);

  // Check if any of the values are beyond the tol
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      if (fabs(mat_val(delta, n, i, j)) >= tol) {
        ok = 0;
      }
    }
  }

  // Print result
  if (ok == 0) {
    if (verbose) {
      LOG_ERROR("Bad jacobian [%s]!\n", jac_name);
      print_matrix("analytical jac", jac, m, n);
      print_matrix("num diff jac", fdiff, m, n);
      print_matrix("difference matrix", delta, m, n);
    }
    retval = -1;
  } else {
    if (verbose) {
      printf("Check [%s] ok!\n", jac_name);
      print_matrix("analytical jac", jac, m, n);
      print_matrix("num diff jac", fdiff, m, n);
      print_matrix("difference matrix", delta, m, n);
    }
    retval = 0;
  }

  // Clean up
  free(delta);

  return retval;
}

/////////
// SVD //
/////////

#ifdef USE_LAPACK

// LAPACK fortran prototypes
extern void sgesdd_(char *jobz,
                    int *m,
                    int *n,
                    float *a,
                    int *lda,
                    float *s,
                    float *u,
                    int *ldu,
                    float *vt,
                    int *ldvt,
                    float *work,
                    int *lwork,
                    int *iwork,
                    int *info);
extern void dgesdd_(char *jobz,
                    int *m,
                    int *n,
                    double *a,
                    int *lda,
                    double *s,
                    double *u,
                    int *ldu,
                    double *vt,
                    int *ldvt,
                    double *work,
                    int *lwork,
                    int *iwork,
                    int *info);

/**
 * Decompose matrix A with SVD
 */
int __lapack_svd(real_t *A, int m, int n, real_t *s, real_t *U, real_t *Vt) {
  // Transpose matrix A because LAPACK is column major
  real_t *At = malloc(sizeof(real_t) * m * n);
  mat_transpose(A, m, n, At);

  // Query and allocate optimal workspace
  int lda = m;
  int lwork = -1;
  int info = 0;
  real_t work_size;
  real_t *work = &work_size;
  int num_sv = (m < n) ? m : n;
  int *iwork = malloc(sizeof(int) * 8 * num_sv);

#if PRECISION == 1
  sgesdd_("A", &m, &n, At, &lda, s, U, &m, Vt, &n, work, &lwork, iwork, &info);
#else
  dgesdd_("A", &m, &n, At, &lda, s, U, &m, Vt, &n, work, &lwork, iwork, &info);
#endif
  lwork = work_size;
  work = malloc(sizeof(real_t) * lwork);

  // Compute SVD
#if PRECISION == 1
  sgesdd_("A", &m, &n, At, &lda, s, U, &m, Vt, &n, work, &lwork, iwork, &info);
#else
  dgesdd_("A", &m, &n, At, &lda, s, U, &m, Vt, &n, work, &lwork, iwork, &info);
#endif
  if (info > 0) {
    LOG_ERROR("Failed to compute svd!\n");
  }

  // Clean up
  free(At);
  free(iwork);
  free(work);

  return (info == 0) ? 0 : -1;
}

#endif // USE_LAPACK

/**
 * Singular Value Decomposition
 *
 * Given a matrix A of size m x n, compute the singular value decomposition of
 * A = U * W * Vt, where the input A is replaced by U, the diagonal matrix W is
 * output as a vector w of size n, and the matrix V (not V transpose) is of
 * size n x n.
 *
 * Source (Singular-Value-Decomposition: page 59-70):
 *
 *   Press, William H., et al. "Numerical recipes in C++." The art of
 *   scientific computing 2 (2007): 1002.
 *
 * @returns 0 for success, -1 for failure
 */
int __svd(real_t *A, const int m, const int n, real_t *w, real_t *V) {
  int flag, i, its, j, jj, k, l, nm;
  double anorm, c, f, g, h, s, scale, x, y, z, *rv1;
  l = 0;

  rv1 = malloc(sizeof(double) * n);
  if (rv1 == NULL) {
    printf("svd(): Unable to allocate vector\n");
    return (-1);
  }

  g = scale = anorm = 0.0;
  for (i = 0; i < n; i++) {
    l = i + 1;
    rv1[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) {
        scale += fabs(A[k * n + i]);
      }
      if (scale) {
        for (k = i; k < m; k++) {
          A[k * n + i] /= scale;
          s += A[k * n + i] * A[k * n + i];
        }
        f = A[i * n + i];
        g = -SIGN2(sqrt(s), f);
        h = f * g - s;
        A[i * n + i] = f - g;
        for (j = l; j < n; j++) {
          for (s = 0.0, k = i; k < m; k++) {
            s += A[k * n + i] * A[k * n + j];
          }
          f = s / h;
          for (k = i; k < m; k++) {
            A[k * n + j] += f * A[k * n + i];
          }
        }
        for (k = i; k < m; k++) {
          A[k * n + i] *= scale;
        }
      }
    }
    w[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m && i != n - 1) {
      for (k = l; k < n; k++) {
        scale += fabs(A[i * n + k]);
      }
      if (scale) {
        for (k = l; k < n; k++) {
          A[i * n + k] /= scale;
          s += A[i * n + k] * A[i * n + k];
        }
        f = A[i * n + l];
        g = -SIGN2(sqrt(s), f);
        h = f * g - s;
        A[i * n + l] = f - g;
        for (k = l; k < n; k++) {
          rv1[k] = A[i * n + k] / h;
        }
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < n; k++) {
            s += A[j * n + k] * A[i * n + k];
          }
          for (k = l; k < n; k++) {
            A[j * n + k] += s * rv1[k];
          }
        }
        for (k = l; k < n; k++) {
          A[i * n + k] *= scale;
        }
      }
    }
    anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  for (i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g) {
        for (j = l; j < n; j++) {
          V[j * n + i] = (A[i * n + j] / A[i * n + l]) / g;
        }
        for (j = l; j < n; j++) {
          for (s = 0.0, k = l; k < n; k++) {
            s += A[i * n + k] * V[k * n + j];
          }
          for (k = l; k < n; k++) {
            V[k * n + j] += s * V[k * n + i];
          }
        }
      }
      for (j = l; j < n; j++) {
        V[i * n + j] = V[j * n + i] = 0.0;
      }
    }
    V[i * n + i] = 1.0;
    g = rv1[i];
    l = i;
  }

  for (i = MIN(m, n) - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    for (j = l; j < n; j++) {
      A[i * n + j] = 0.0;
    }
    if (g) {
      g = 1.0 / g;
      for (j = l; j < n; j++) {
        for (s = 0.0, k = l; k < m; k++) {
          s += A[k * n + i] * A[k * n + j];
        }
        f = (s / A[i * n + i]) * g;
        for (k = i; k < m; k++) {
          A[k * n + j] += f * A[k * n + i];
        }
      }
      for (j = i; j < m; j++) {
        A[j * n + i] *= g;
      }
    } else
      for (j = i; j < m; j++) {
        A[j * n + i] = 0.0;
      }
    ++A[i * n + i];
  }

  for (k = n - 1; k >= 0; k--) {
    for (its = 0; its < 30; its++) {
      flag = 1;
      for (l = k; l >= 0; l--) {
        nm = l - 1;
        if ((fabs(rv1[l]) + anorm) == anorm) {
          flag = 0;
          break;
        }
        if ((fabs(w[nm]) + anorm) == anorm) {
          break;
        }
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          rv1[i] = c * rv1[i];
          if ((fabs(f) + anorm) == anorm) {
            break;
          }
          g = w[i];
          h = pythag(f, g);
          w[i] = h;
          h = 1.0 / h;
          c = g * h;
          s = -f * h;
          for (j = 0; j < m; j++) {
            y = A[j * n + nm];
            z = A[j * n + i];
            A[j * n + nm] = y * c + z * s;
            A[j * n + i] = z * c - y * s;
          }
        }
      }
      z = w[k];
      if (l == k) {
        if (z < 0.0) {
          w[k] = -z;
          for (j = 0; j < n; j++) {
            V[j * n + k] = -V[j * n + k];
          }
        }
        break;
      }
      if (its == 29) {
        printf("no convergence in 30 svd iterations\n");
      }
      x = w[l];
      nm = k - 1;
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
      g = pythag(f, 1.0);
      f = ((x - z) * (x + z) + h * ((y / (f + SIGN2(g, f))) - h)) / x;
      c = s = 1.0;
      for (j = l; j <= nm; j++) {
        i = j + 1;
        g = rv1[i];
        y = w[i];
        h = s * g;
        g = c * g;
        z = pythag(f, h);
        rv1[j] = z;
        c = f / z;
        s = h / z;
        f = x * c + g * s;
        g = g * c - x * s;
        h = y * s;
        y *= c;
        for (jj = 0; jj < n; jj++) {
          x = V[jj * n + j];
          z = V[jj * n + i];
          V[jj * n + j] = x * c + z * s;
          V[jj * n + i] = z * c - x * s;
        }
        z = pythag(f, h);
        w[j] = z;
        if (z) {
          z = 1.0 / z;
          c = f * z;
          s = h * z;
        }
        f = c * g + s * y;
        x = c * y - s * g;
        for (jj = 0; jj < m; jj++) {
          y = A[jj * n + j];
          z = A[jj * n + i];
          A[jj * n + j] = y * c + z * s;
          A[jj * n + i] = z * c - y * s;
        }
      }
      rv1[l] = 0.0;
      rv1[k] = f;
      w[k] = x;
    }
  }
  free(rv1);

  return 0;
}

/**
 * Decompose matrix A with SVD
 */
int svd(const real_t *A,
        const int m,
        const int n,
        real_t *U,
        real_t *s,
        real_t *V) {
#ifdef USE_LAPACK
  real_t *A_copy = malloc(sizeof(real_t) * m * n);
  real_t *U_ = malloc(sizeof(real_t) * m * m);
  real_t *Ut_ = malloc(sizeof(real_t) * m * m);
  real_t *Vt = malloc(sizeof(real_t) * n * n);

  mat_copy(A, m, n, A_copy);
  const int retval = __lapack_svd(A_copy, m, n, s, U_, Vt);

  mat_transpose(U_, m, m, Ut_);
  mat_block_get(Ut_, m, 0, m - 1, 0, n - 1, U);
  mat_copy(Vt, n, n, V);

  free(U_);
  free(Ut_);
  free(Vt);
  free(A_copy);

  return retval;
#else
  mat_copy(A, m, n, U);
  return __svd(U, m, n, s, V);
#endif // USE_LAPACK
}

/**
 * Pseudo inverse of matrix A with SVD
 */
void pinv(const real_t *A, const int m, const int n, real_t *A_inv) {
  assert(m == n);

  // Decompose A = U * S * Vt
  const int diag_size = (m < n) ? m : n;
  real_t *s = calloc(diag_size, sizeof(real_t));
  real_t *U = calloc(m * n, sizeof(real_t));
  real_t *V = calloc(n * n, sizeof(real_t));
  svd(A, m, n, U, s, V);

  // Form Sinv diagonal matrix
  real_t *Si = calloc(m * n, sizeof(real_t));
  zeros(Si, n, m);
  for (int idx = 0; idx < m; idx++) {
    const int diag_idx = idx * n + idx;

    if (s[idx] > 1e-24) {
      Si[diag_idx] = 1.0 / s[idx];
    } else {
      Si[diag_idx] = 0.0;
    }
  }

  // A_inv = Vt * Si * U
  // real_t *V_Si = malloc(sizeof(real_t) * m * m);
  // zeros(A_inv, m, n);
  // dot(V, m, n, Si, n, m, V_Si);
  // dot(V_Si, m, m, Ut, m, n, A_inv);

  // A_inv = U * Si * Ut
  real_t *Ut = calloc(m * n, sizeof(real_t));
  real_t *Si_Ut = calloc(diag_size * n, sizeof(real_t));
  mat_transpose(U, m, n, Ut);
  dot(Si, diag_size, diag_size, Ut, m, n, Si_Ut);
  dot(V, m, n, Si_Ut, diag_size, n, A_inv);

  // Clean up
  free(s);
  free(U);
  free(V);

  free(Si);

  free(Ut);
  free(Si_Ut);
}

/**
 * Use SVD to find the determinant of matrix `A` of size `m` x `n`.
 *
 * WARNING: This method assumes the matrix `A` is invertible, additionally the
 * returned determinant is the **absolute** value, the sign is not returned.
 */
int svd_det(const real_t *A, const int m, const int n, real_t *det) {
  assert(m == n);

  // Decompose matrix A with SVD
  const int k = (m < n) ? m : n;
  real_t *U = malloc(sizeof(real_t) * m * n);
  real_t *s = malloc(sizeof(real_t) * k);
  real_t *V = malloc(sizeof(real_t) * k * k);
  int retval = svd(A, m, n, U, s, V);

  // Calculate determinant by via product of the diagonal singular values
  *det = s[0];
  for (int i = 1; i < k; i++) {
    *det *= s[i];
  }

  // Clean up
  free(U);
  free(s);
  free(V);

  return retval;
}

/**
 * Calculate matrix rank of `A` of size `m` x `n`.
 */
int svd_rank(const real_t *A, const int m, const int n, real_t tol) {
  // Decompose matrix A with SVD
  const int k = (m < n) ? m : n;
  real_t *U = malloc(sizeof(real_t) * m * n);
  real_t *s = malloc(sizeof(real_t) * k);
  real_t *V = malloc(sizeof(real_t) * k * k);
  int retval = svd(A, m, n, U, s, V);
  if (retval != 0) {
    free(U);
    free(s);
    free(V);
    return -1;
  }

  // Calculate determinant by via product of the diagonal singular values
  int rank = 0;
  for (int i = 0; i < k; i++) {
    if (s[i] >= tol) {
      rank++;
    }
  }

  // Clean up
  free(U);
  free(s);
  free(V);

  return rank;
}

//////////
// CHOL //
//////////

#ifdef USE_LAPACK

// LAPACK fortran prototypes
extern int spotrf_(char *uplo, int *n, float *A, int *lda, int *info);
extern int spotrs_(char *uplo,
                   int *n,
                   int *nrhs,
                   float *A,
                   int *lda,
                   float *B,
                   int *ldb,
                   int *info);
extern int dpotrf_(char *uplo, int *n, double *A, int *lda, int *info);
extern int dpotrs_(char *uplo,
                   int *n,
                   int *nrhs,
                   double *A,
                   int *lda,
                   double *B,
                   int *ldb,
                   int *info);

/**
 * Decompose matrix A to lower triangular matrix L
 */
void __lapack_chol(const real_t *A, const size_t m, real_t *L) {
  assert(A != NULL);
  assert(m > 0);
  assert(L != NULL);

  // Cholesky Decomposition
  int lda = m;
  int n = m;
  char uplo = 'L';
  int info = 0;
  mat_copy(A, m, m, L);

#if PRECISION == 1
  spotrf_(&uplo, &n, L, &lda, &info);
#elif PRECISION == 2
  dpotrf_(&uplo, &n, L, &lda, &info);
#endif
  if (info != 0) {
    fprintf(stderr, "Failed to decompose A using Cholesky Decomposition!\n");
  }

  // Transpose and zero upper triangular result
  for (size_t i = 0; i < m; i++) {
    for (size_t j = i; j < m; j++) {
      if (i != j) {
        L[(j * m) + i] = L[(i * m) + j];
        L[(i * m) + j] = 0;
      }
    }
  }
}

/**
 * Solve Ax = b using LAPACK's implementation of Cholesky decomposition, where
 * `A` is a square matrix, `b` is a vector and `x` is the solution vector of
 * size `n`.
 */
void __lapack_chol_solve(const real_t *A,
                         const real_t *b,
                         real_t *x,
                         const size_t m) {
  assert(A != NULL);
  assert(b != NULL);
  assert(x != NULL);
  assert(m > 0);

  // Cholesky Decomposition
  int info = 0;
  int lda = m;
  int n = m;
  char uplo = 'U';
  real_t *a = mat_malloc(m, m);
  mat_copy(A, m, m, a);
#if PRECISION == 1
  spotrf_(&uplo, &n, a, &lda, &info);
#elif PRECISION == 2
  dpotrf_(&uplo, &n, a, &lda, &info);
#endif
  if (info != 0) {
    fprintf(stderr, "Failed to decompose A using Cholesky Decomposition!\n");
  }

  // Solve Ax = b using Cholesky decomposed A from above
  vec_copy(b, m, x);
  int nhrs = 1;
  int ldb = m;
#if PRECISION == 1
  spotrs_(&uplo, &n, &nhrs, a, &lda, x, &ldb, &info);
#elif PRECISION == 2
  dpotrs_(&uplo, &n, &nhrs, a, &lda, x, &ldb, &info);
#endif
  if (info != 0) {
    fprintf(stderr, "Failed to solve Ax = b!\n");
  }

  free(a);
}
#endif // USE_LAPACK

/**
 * Cholesky decomposition. Takes a `m x m` matrix `A` and decomposes it into a
 * lower and upper triangular matrix `L` and `U` with Cholesky decomposition.
 * This function only returns the `L` triangular matrix.
 */
void __chol(const real_t *A, const size_t m, real_t *L) {
  assert(A != NULL);
  assert(m > 0);
  assert(L != NULL);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < (i + 1); j++) {

      if (i == j) {
        real_t s = 0.0;
        for (size_t k = 0; k < j; k++) {
          s += L[j * m + k] * L[j * m + k];
        }
        L[i * m + j] = sqrt(A[i * m + i] - s);

      } else {
        real_t s = 0.0;
        for (size_t k = 0; k < j; k++) {
          s += L[i * m + k] * L[j * m + k];
        }
        L[i * m + j] = (1.0 / L[j * m + j] * (A[i * m + j] - s));
      }
    }
  }
}

/**
 * Solve `Ax = b` using Cholesky decomposition, where `A` is a square matrix,
 * `b` is a vector and `x` is the solution vector of size `n`.
 */
void __chol_solve(const real_t *A, const real_t *b, real_t *x, const size_t n) {
  assert(A != NULL);
  assert(b != NULL);
  assert(x != NULL);
  assert(n > 0);

  // Allocate memory
  real_t *L = calloc(n * n, sizeof(real_t));
  real_t *Lt = calloc(n * n, sizeof(real_t));
  real_t *y = calloc(n, sizeof(real_t));

  // Cholesky decomposition
  chol(A, n, L);
  mat_transpose(L, n, n, Lt);

  // Forward substitution
  // Ax = b -> LLt x = b.
  // Let y = Lt x, L y = b (Solve for y)
  for (int i = 0; i < (int) n; i++) {
    real_t alpha = b[i];

    if (fltcmp(L[i * n + i], 0.0) == 0) {
      y[i] = 0.0;

    } else {
      for (int j = 0; j < i; j++) {
        alpha -= L[i * n + j] * y[j];
      }
      y[i] = alpha / L[i * n + i];
    }
  }

  // Backward substitution
  // Now we have y, we can go back to (Lt x = y) and solve for x
  for (int i = n - 1; i >= 0; i--) {
    real_t alpha = y[i];

    if (fltcmp(Lt[i * n + i], 0.0) == 0) {
      x[i] = 0.0;

    } else {
      for (int j = i; j < (int) n; j++) {
        alpha -= Lt[i * n + j] * x[j];
      }
      x[i] = alpha / Lt[i * n + i];
    }
  }

  // Clean up
  free(y);
  free(L);
  free(Lt);
}

/**
 * Cholesky decomposition. Takes a `m x m` matrix `A` and decomposes it into a
 * lower and upper triangular matrix `L` and `U` with Cholesky decomposition.
 * This function only returns the `L` triangular matrix.
 */
void chol(const real_t *A, const size_t m, real_t *L) {
#ifdef USE_LAPACK
  __lapack_chol(A, m, L);
#else
  __chol(A, m, L);
#endif // USE_LAPACK
}

/**
 * Solve `Ax = b` using Cholesky decomposition, where `A` is a square matrix,
 * `b` is a vector and `x` is the solution vector of size `n`.
 */
void chol_solve(const real_t *A, const real_t *b, real_t *x, const size_t n) {
#ifdef USE_LAPACK
  __lapack_chol_solve(A, b, x, n);
#else
  __chol_solve(A, b, x, n);
#endif // USE_LAPACK
}

////////
// QR //
////////

#ifdef USE_LAPACK

// LAPACK fortran prototypes
void sgeqrf_(const int *M,
             const int *N,
             float *A,
             const int *lda,
             float *TAU,
             float *work,
             const int *lwork,
             int *info);
void dgeqrf_(const int *M,
             const int *N,
             double *A,
             const int *lda,
             double *TAU,
             double *work,
             const int *lwork,
             int *info);

void __lapack_qr(real_t *A, int m, int n, real_t *R) {
  // Transpose matrix A because LAPACK is column major
  real_t *At = malloc(sizeof(real_t) * m * n);
  mat_transpose(A, m, n, At);

  // Query and allocate optimal workspace
  int lda = m;
  int lwork = -1;
  int info = 0;
  real_t work_size;
  real_t *work = &work_size;
  real_t *tau = malloc(sizeof(real_t) * ((m < n) ? m : n));

#if PRECISION == 1
  sgeqrf_(&m, &n, At, &lda, tau, work, &lwork, &info);
#else
  dgeqrf_(&m, &n, At, &lda, tau, work, &lwork, &info);
#endif
  lwork = work_size;
  work = malloc(sizeof(real_t) * lwork);

  // Compute QR
#if PRECISION == 1
  sgeqrf_(&m, &n, At, &lda, tau, work, &lwork, &info);
#else
  dgeqrf_(&m, &n, At, &lda, tau, work, &lwork, &info);
#endif
  // mat_transpose(At, m, n, R);

  // Transpose result and zero lower triangular
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < m; j++) {
      if (i <= j) {
        R[(i * m) + j] = At[(j * m) + i];
      } else {
        R[(i * m) + j] = 0;
      }
    }
  }

  // Recover matrix Q
  // From the LAPACK documentation:
  //
  // The matrix Q is represented as a product of elementary reflectors
  //
  //   Q = H(1) H(2) . . . H(k), where k = min(m, n).
  //
  // Each H(i) has the form
  //
  //   H(i) = I - tau * v * v**T
  //
  // where tau is a real scalar, and v is a real vector with v(1:i-1) = 0 and
  // v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i), and tau in (i).
  //
  // Q = eye(6); % Initial
  // v = [ 0 0 0 0 0 0];
  // m = 6;
  // for i = 1:4
  //   v(1:i-1) = 0;
  //   v(i) = 1;
  //   v(i+1:m) = A(i+1:m,i);
  //   A(i+1:m,i)
  //   Q = Q*(eye(6) - tau(i)*v'*v);
  // end
  // real_t *Q = calloc(m * m, sizeof(real_t));
  // real_t *v = calloc(m, sizeof(real_t));
  // for (int i = 0; i < n; i++) {
  //   for (int ii = 0; ii < i; i++) {
  //     v[ii] = 0.0;
  //   }
  //   v[i] = 1.0;
  //   for (int ii = i+1; ii < m; ii++) {
  //     v[ii] = At[(i + 1) * m + i];
  //   }

  // }
  // free(Q);
  // free(v);

  // print_matrix("R", R, m, n);
  // print_vector("tau", tau, m);

  // Clean up
  free(At);
  free(work);
  free(tau);
}
#endif // USE_LAPACK

void qr(real_t *A, const int m, const int n, real_t *R) {
#ifdef USE_LAPACK
  __lapack_qr(A, m, n, R);
#else
  FATAL("NOT IMPLEMENTED!");
#endif // USE_LAPACK
}

/////////
// EIG //
/////////

#ifdef USE_LAPACK

// LAPACK fortran prototypes
void ssyev_(char *jobz,
            char *uplo,
            int *n,
            float *a,
            int *lda,
            float *w,
            float *work,
            int *lwork,
            int *info);
void dsyev_(char *jobz,
            char *uplo,
            int *n,
            double *a,
            int *lda,
            double *w,
            double *work,
            int *lwork,
            int *info);

int __lapack_eig(const real_t *A,
                 const int m,
                 const int n,
                 real_t *V,
                 real_t *w) {
  assert(A != NULL);
  assert(m > 0 && n > 0);
  assert(m == n);
  int n_ = n;
  int lda = n;

  // Copy matrix A to output matrix V
  mat_triu(A, n, V);

  // Query and allocate the optimal workspace
  int lwork = -1;
  int info = 0;
  real_t wkopt;
#if PRECISION == 1
  ssyev_("Vectors", "Lower", &n_, V, &lda, w, &wkopt, &lwork, &info);
#elif PRECISION == 2
  dsyev_("Vectors", "Lower", &n_, V, &lda, w, &wkopt, &lwork, &info);
#endif // Precision
  lwork = (int) wkopt;
  real_t *work = malloc(sizeof(double) * lwork);

  // Solve eigenproblem
#if PRECISION == 1
  ssyev_("Vectors", "Lower", &n_, V, &lda, w, work, &lwork, &info);
#elif PRECISION == 2
  dsyev_("Vectors", "Lower", &n_, V, &lda, w, work, &lwork, &info);
#endif // Precision
  if (info > 0) {
    LOG_ERROR("The algorithm failed to compute eigenvalues.\n");
    free(work);
    return -1;
  }

  // Clean up
  real_t *Vt = malloc(sizeof(real_t) * n * n);
  mat_transpose(V, n, n, Vt);
  mat_copy(Vt, n, n, V);
  free(Vt);
  free(work);

  return 0;
}
#endif // USE_LAPACK

/**
 * Perform Eigen-Decomposition of a symmetric matrix `A` of size `m` x `n`.
 */
int eig_sym(const real_t *A, const int m, const int n, real_t *V, real_t *w) {
  assert(A != NULL);
  assert(m > 0 && n > 0);
  assert(m == n);
  assert(V != NULL && w != NULL);
#ifdef USE_LAPACK
  return __lapack_eig(A, m, n, V, w);
#else
  FATAL("Not implemented!");
  return 0;
#endif // USE_LAPACK
}

/**
 * Invert matrix `A` of size `m` x `n` with Eigen-decomposition.
 */
int eig_inv(real_t *A, const int m, const int n, const int c, real_t *A_inv) {
  assert(A != NULL);
  assert(m == n);
  assert(A_inv != NULL);

  // Enforce Symmetric Positive Definite
  if (c) {
    enforce_spd(A, m, m);
  }

  // Invert matrix via Eigen-decomposition
  real_t *V = malloc(sizeof(real_t) * m * m);
  real_t *Vt = malloc(sizeof(real_t) * m * m);
  real_t *Lambda_inv = malloc(sizeof(real_t) * m * m);
  real_t *w = malloc(sizeof(real_t) * m);

  eig_sym(A, m, m, V, w);
  for (int i = 0; i < m; i++) {
    w[i] = 1.0 / w[i];
  }
  mat_diag_set(Lambda_inv, m, m, w);
  mat_transpose(V, m, m, Vt);
  dot3(V, m, m, Lambda_inv, m, m, Vt, m, m, A_inv);

  // Clean up
  free(V);
  free(Vt);
  free(Lambda_inv);
  free(w);

  return 0;
}

/**
 * Calculate matrix rank of `A` of size `m` x `n`.
 */
int eig_rank(const real_t *A, const int m, const int n, const real_t tol) {
  assert(A != NULL);
  assert(m > 0 && n > 0);
  assert(m == n);

  real_t *V = malloc(sizeof(real_t) * m * n);
  real_t *w = malloc(sizeof(real_t) * m);
  int retval = eig_sym(A, m, n, V, w);
  if (retval != 0) {
    free(V);
    free(w);
    return -1;
  }

  int rank = 0;
  for (int i = 0; i < m; i++) {
    if (w[i] >= tol) {
      rank++;
    }
  }

  // Clean up
  free(V);
  free(w);

  return rank;
}

int schur_complement(const real_t *H,
                     const real_t *b,
                     const int H_size,
                     const int m,
                     const int r,
                     real_t *H_marg,
                     real_t *b_marg) {
  assert(H != NULL);
  assert(b);
  assert(H_size > 0);
  assert((m + r) == H_size);
  assert(H_marg != NULL && b_marg != NULL);

  // Extract sub-blocks of matrix H
  // H = [Hmm, Hmr,
  //      Hrm, Hrr]
  real_t *Hmm = malloc(sizeof(real_t) * m * m);
  real_t *Hmr = malloc(sizeof(real_t) * m * r);
  real_t *Hrm = malloc(sizeof(real_t) * m * r);
  real_t *Hrr = malloc(sizeof(real_t) * r * r);
  real_t *Hmm_inv = malloc(sizeof(real_t) * m * m);

  mat_block_get(H, H_size, 0, m - 1, 0, m - 1, Hmm);
  mat_block_get(H, H_size, 0, m - 1, m, H_size - 1, Hmr);
  mat_block_get(H, H_size, m, H_size - 1, 0, m - 1, Hrm);
  mat_block_get(H, H_size, m, H_size - 1, m, H_size - 1, Hrr);

  // Extract sub-blocks of vector b
  // b = [b_mm, b_rr]
  real_t *bmm = malloc(sizeof(real_t) * m);
  real_t *brr = malloc(sizeof(real_t) * r);
  vec_copy(b, m, bmm);
  vec_copy(b + m, r, brr);

  // Invert Hmm
  int status = 0;
  if (eig_inv(Hmm, m, m, 1, Hmm_inv) != 0) {
    status = -1;
  }
  // pinv(Hmm, m, m, Hmm_inv);
  if (check_inv(Hmm, Hmm_inv, m) == -1) {
    status = -1;
    printf("Inverse Hmm failed!\n");
  }

  // Shur-Complement
  // H_marg = H_rr - H_rm * H_mm_inv * H_mr
  // b_marg = b_rr - H_rm * H_mm_inv * b_mm
  if (status == 0) {
    dot3(Hrm, r, m, Hmm_inv, m, m, Hmr, m, r, H_marg);
    dot3(Hrm, r, m, Hmm_inv, m, m, bmm, m, 1, b_marg);
    for (int i = 0; i < (r * r); i++) {
      H_marg[i] = Hrr[i] - H_marg[i];
    }
    for (int i = 0; i < r; i++) {
      b_marg[i] = brr[i] - b_marg[i];
    }
  }

  // Clean-up
  free(Hmm);
  free(Hmr);
  free(Hrm);
  free(Hrr);
  free(Hmm_inv);

  free(bmm);
  free(brr);

  return status;
}

/**
 * Calculate the Shannon Entropy
 */
int shannon_entropy(const real_t *covar, const int m, real_t *entropy) {
  assert(covar != NULL);
  assert(m > 0);
  assert(entropy != NULL);

  real_t covar_det = 0.0f;
  if (svd_det(covar, m, m, &covar_det) != 0) {
    return -1;
  }

  const real_t k = pow(2 * M_PI * exp(1), m);
  *entropy = 0.5 * log(k * covar_det);

  return 0;
}

/*******************************************************************************
 * SUITE-SPARSE
 ******************************************************************************/

#define CHOLMOD_NZERO_EPS 1e-12

/**
 * Allocate memory and form a sparse matrix
 *
 * @param c Cholmod workspace
 * @param A Matrix A
 * @param m Number of rows in A
 * @param n Number of cols in A
 * @param stype
 *
 *   0:  matrix is "unsymmetric": use both upper and lower triangular parts
 *       (the matrix may actually be symmetric in pattern and value, but
 *       both parts are explicitly stored and used).  May be square or
 *       rectangular.
 *   >0: matrix is square and symmetric, use upper triangular part.
 *       Entries in the lower triangular part are ignored.
 *   <0: matrix is square and symmetric, use lower triangular part.
 *       Entries in the upper triangular part are ignored.
 *
 *   Note that stype>0 and stype<0 are different for cholmod_sparse and
 *   cholmod_triplet.  See the cholmod_triplet data structure for more
 *   details.
 *
 * @returns A suite-sparse sparse matrix
 */
cholmod_sparse *cholmod_sparse_malloc(cholmod_common *c,
                                      const real_t *A,
                                      const int m,
                                      const int n,
                                      const int stype) {
  assert(c != NULL);
  assert(A != NULL);
  assert(m > 0 && n > 0);

  // Count number of non-zeros
  size_t nzmax = 0;
  for (long int idx = 0; idx < (m * n); idx++) {
    if (fabs(A[idx]) > CHOLMOD_NZERO_EPS) {
      nzmax++;
    }
  }

  // Allocate memory for sparse matrix
  cholmod_sparse *A_cholmod =
      cholmod_allocate_sparse(m, n, nzmax, 1, 1, stype, CHOLMOD_REAL, c);
  assert(A_cholmod);

  // Fill sparse matrix
  int *row_ind = A_cholmod->i;
  int *col_ptr = A_cholmod->p;
  real_t *values = A_cholmod->x;
  size_t row_it = 0;
  size_t col_it = 1;
  for (long int j = 0; j < n; ++j) {
    for (long int i = 0; i < m; ++i) {
      if (fabs(A[(i * n) + j]) > CHOLMOD_NZERO_EPS) {
        values[row_it] = A[(i * n) + j];
        row_ind[row_it] = i;
        row_it++;
      }
    }
    col_ptr[col_it] = row_it;
    col_it++;
  }

  return A_cholmod;
}

/**
 * Allocate memory and form a dense vector
 *
 * @param c Cholmod workspace
 * @param x Vector x
 * @param n Length of vector x
 *
 * @returns A dense suite-sparse vector
 */
cholmod_dense *cholmod_dense_malloc(cholmod_common *c,
                                    const real_t *x,
                                    const int n) {
  assert(c != NULL);
  assert(x != NULL);

  cholmod_dense *out = cholmod_allocate_dense(n, 1, n, CHOLMOD_REAL, c);
  assert(out != NULL);

  double *out_x = out->x;
  for (size_t i = 0; i < n; i++) {
    out_x[i] = x[i];
  }

  return out;
}

/**
 * Extract dense suite-sparse vector of length `n` from `src` to `dst`.
 */
void cholmod_dense_raw(const cholmod_dense *src, real_t *dst, const int n) {
  assert(src != NULL);
  assert(dst != NULL);
  assert(n > 0);

  double *data = src->x;
  for (int i = 0; i < n; i++) {
    dst[i] = data[i];
  }
}

/**
 * Solve Ax = b with Suite-Sparse's CHOLMOD package
 *
 * @param c Cholmod workspace
 * @param A Matrix A
 * @param A_m Number of rows in A
 * @param A_n Number of cols in A
 * @param b Vector b
 * @param b_m Number of rows in A
 * @param x Vector x
 *
 * @returns the residual norm of (Ax - b)
 */
real_t suitesparse_chol_solve(cholmod_common *c,
                              const real_t *A,
                              const int A_m,
                              const int A_n,
                              const real_t *b,
                              const int b_m,
                              real_t *x) {
  assert(c != NULL);
  assert(A != NULL && A_m > 0 && A_n > 0);
  assert(b != NULL && b_m > 0);
  assert(A_n == b_m);
  assert(x != NULL);

  // Setup
  cholmod_sparse *A_sparse = cholmod_sparse_malloc(c, A, A_m, A_n, 1);
  cholmod_dense *b_dense = cholmod_dense_malloc(c, b, b_m);
  assert(A_sparse);
  assert(b_dense);
  assert(cholmod_check_sparse(A_sparse, c) != -1);
  assert(cholmod_check_dense(b_dense, c) != -1);

  // Analyze and factorize
  cholmod_factor *L_factor = cholmod_analyze(A_sparse, c);
  cholmod_factorize(A_sparse, L_factor, c);
  assert(cholmod_check_factor(L_factor, c) != -1);

  // Solve A * x = b
  cholmod_dense *x_dense = cholmod_solve(CHOLMOD_A, L_factor, b_dense, c);
  cholmod_dense_raw(x_dense, x, b_m);

  // r = r - A * x
  double m1[2] = {-1, 0};
  double one[2] = {1, 0};
  cholmod_dense *r_dense = cholmod_copy_dense(b_dense, c);
  cholmod_sdmult(A_sparse, 0, m1, one, x_dense, r_dense, c);
  const real_t norm = cholmod_norm_dense(r_dense, 0, c);

  // Clean up
  cholmod_free_sparse(&A_sparse, c);
  cholmod_free_dense(&b_dense, c);
  cholmod_free_factor(&L_factor, c);
  cholmod_free_dense(&x_dense, c);
  cholmod_free_dense(&r_dense, c);

  return norm;
}

/*******************************************************************************
 * TRANSFORMS
 ******************************************************************************/

/** Form rotation matrix around x axis **/
void rotx(const real_t theta, real_t C[3 * 3]) {
  C[0] = 1.0;
  C[1] = 0.0;
  C[2] = 0.0;

  C[3] = 0.0;
  C[4] = cos(theta);
  C[5] = -sin(theta);

  C[6] = 0.0;
  C[7] = sin(theta);
  C[8] = cos(theta);
}

/** Form rotation matrix around y axis */
void roty(const real_t theta, real_t C[3 * 3]) {
  C[0] = cos(theta);
  C[1] = 0.0;
  C[2] = sin(theta);

  C[3] = 0.0;
  C[4] = 1.0;
  C[5] = 0.0;

  C[6] = -sin(theta);
  C[7] = 0.0;
  C[8] = cos(theta);
}

/** Form rotation matrix around z axis */
void rotz(const real_t theta, real_t C[3 * 3]) {
  C[0] = cos(theta);
  C[1] = -sin(theta);
  C[2] = 0.0;

  C[3] = sin(theta);
  C[4] = cos(theta);
  C[5] = 0.0;

  C[6] = 0.0;
  C[7] = 0.0;
  C[8] = 1.0;
}

/** Compare two rotation matrices **/
real_t rot_diff(const real_t R0[3 * 3], const real_t R1[3 * 3]) {
  real_t R0t[3 * 3] = {0};
  real_t dR[3 * 3] = {0};
  mat_transpose(R0, 3, 3, R0t);
  dot(R0t, 3, 3, R1, 3, 3, dR);

  const real_t tr = mat_trace(dR, 3, 3);
  if (fabs(tr - 3.0) < 1e-5) {
    return 0.0;
  }

  return acos((tr - 1.0) / 2.0);
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a 7x1 pose vector
 * `params`.
 *
 *    pose = (translation, rotation)
 *    pose = (rx, ry, rz, qw, qx, qy, qz)
 */
void tf(const real_t params[7], real_t T[4 * 4]) {
  assert(params != NULL);
  assert(T != NULL);

  const real_t r[3] = {params[0], params[1], params[2]};
  const real_t q[4] = {params[3], params[4], params[5], params[6]};

  real_t C[3 * 3] = {0};
  quat2rot(q, C);

  T[0] = C[0];
  T[1] = C[1];
  T[2] = C[2];
  T[3] = r[0];

  T[4] = C[3];
  T[5] = C[4];
  T[6] = C[5];
  T[7] = r[1];

  T[8] = C[6];
  T[9] = C[7];
  T[10] = C[8];
  T[11] = r[2];

  T[12] = 0.0;
  T[13] = 0.0;
  T[14] = 0.0;
  T[15] = 1.0;
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a rotation matrix `C`
 * and translation vector `r`.
 */
void tf_cr(const real_t C[3 * 3], const real_t r[3], real_t T[4 * 4]) {
  T[0] = C[0];
  T[1] = C[1];
  T[2] = C[2];
  T[3] = r[0];

  T[4] = C[3];
  T[5] = C[4];
  T[6] = C[5];
  T[7] = r[1];

  T[8] = C[6];
  T[9] = C[7];
  T[10] = C[8];
  T[11] = r[2];

  T[12] = 0.0;
  T[13] = 0.0;
  T[14] = 0.0;
  T[15] = 1.0;
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a quaternion `q` and
 * translation vector `r`.
 */
void tf_qr(const real_t q[4], const real_t r[3], real_t T[4 * 4]) {
  real_t C[3 * 3] = {0};
  quat2rot(q, C);
  tf_cr(C, r, T);
}

/**
 * Form 4x4 homogeneous transformation matrix `T` from a euler-angles `ypr`
 * (yaw-pitch-roll) and translation vector `r`.
 */
void tf_er(const real_t ypr[3], const real_t r[3], real_t T[4 * 4]) {
  real_t C[3 * 3] = {0};
  euler321(ypr, C);
  tf_cr(C, r, T);
}

/**
 * Form 7x1 pose parameter vector `params` from 4x4 homogeneous transformation
 * matrix `T`.
 */
void tf_vector(const real_t T[4 * 4], real_t params[7]) {
  assert(T != NULL);
  assert(params != NULL);

  real_t C[3 * 3] = {0};
  tf_rot_get(T, C);

  real_t r[3] = {0};
  tf_trans_get(T, r);

  real_t q[4] = {0};
  rot2quat(C, q);

  params[0] = r[0];
  params[1] = r[1];
  params[2] = r[2];

  params[3] = q[0];
  params[4] = q[1];
  params[5] = q[2];
  params[6] = q[3];
}

/**
 * Decompose transform `T` into the rotation `C` and translation `r`
 * components.
 */
void tf_decompose(const real_t T[4 * 4], real_t C[3 * 3], real_t r[3]) {
  assert(T != NULL);
  assert(C != NULL);
  assert(r != NULL);

  C[0] = T[0];
  C[1] = T[1];
  C[2] = T[2];
  r[0] = T[3];

  C[3] = T[4];
  C[4] = T[5];
  C[5] = T[6];
  r[1] = T[7];

  C[6] = T[8];
  C[7] = T[9];
  C[8] = T[10];
  r[1] = T[11];
}

/**
 * Set the rotational component in the 4x4 transformation matrix `T` using a
 * 3x3 rotation matrix `C`.
 */
void tf_rot_set(real_t T[4 * 4], const real_t C[3 * 3]) {
  assert(T != NULL);
  assert(C != NULL);
  assert(T != C);

  T[0] = C[0];
  T[1] = C[1];
  T[2] = C[2];

  T[4] = C[3];
  T[5] = C[4];
  T[6] = C[5];

  T[8] = C[6];
  T[9] = C[7];
  T[10] = C[8];
}

/**
 * Get the rotation matrix `C` from the 4x4 transformation matrix `T`.
 */
void tf_rot_get(const real_t T[4 * 4], real_t C[3 * 3]) {
  assert(T != NULL);
  assert(C != NULL);
  assert(T != C);

  C[0] = T[0];
  C[1] = T[1];
  C[2] = T[2];

  C[3] = T[4];
  C[4] = T[5];
  C[5] = T[6];

  C[6] = T[8];
  C[7] = T[9];
  C[8] = T[10];
}

/**
 * Set the rotation component in the 4x4 transformation matrix `T` using a 4x1
 * quaternion `q`.
 */
void tf_quat_set(real_t T[4 * 4], const real_t q[4]) {
  assert(T != NULL);
  assert(q != NULL);
  assert(T != q);

  real_t C[3 * 3] = {0};
  quat2rot(q, C);
  tf_rot_set(T, C);
}

/**
 * Get the quaternion `q` from the 4x4 transformation matrix `T`.
 */
void tf_quat_get(const real_t T[4 * 4], real_t q[4]) {
  assert(T != NULL);
  assert(q != NULL);
  assert(T != q);

  real_t C[3 * 3] = {0};
  tf_rot_get(T, C);
  rot2quat(C, q);
}

/**
 * Set the rotational component in the 4x4 transformation matrix `T` using a
 * 3x1 euler angle vector `euler`.
 */
void tf_euler_set(real_t T[4 * 4], const real_t ypr[3]) {
  assert(T != NULL);
  assert(ypr != NULL);
  assert(T != ypr);

  real_t C[3 * 3] = {0};
  euler321(ypr, C);
  tf_rot_set(T, C);
}

/**
 * Get the rotational component in the 4x4 transformation matrix `T` in the
 * form of a 3x1 euler angle vector `ypr`.
 */
void tf_euler_get(const real_t T[4 * 4], real_t ypr[3]) {
  assert(T != NULL);
  assert(ypr != NULL);
  assert(T != ypr);

  real_t C[3 * 3] = {0};
  tf_rot_get(T, C);
  rot2euler(C, ypr);
}

/**
 * Set the translational component in the 4x4 transformation matrix `T` using
 * a 3x1 translation vector `r`.
 */
void tf_trans_set(real_t T[4 * 4], const real_t r[3]) {
  assert(T != NULL);
  assert(r != NULL);
  assert(T != r);

  T[3] = r[0];
  T[7] = r[1];
  T[11] = r[2];
}

/**
 * Get the translational vector `r` from the 4x4 transformation matrix `T`.
 */
void tf_trans_get(const real_t T[4 * 4], real_t r[3]) {
  assert(T != NULL);
  assert(r != NULL);
  assert(T != r);

  r[0] = T[3];
  r[1] = T[7];
  r[2] = T[11];
}

/**
 * Invert the 4x4 homogeneous transformation matrix `T` where results are
 * written to `T_inv`.
 */
void tf_inv(const real_t T[4 * 4], real_t T_inv[4 * 4]) {
  assert(T != NULL);
  assert(T_inv != NULL);
  assert(T != T_inv);

  /**
   * Transformation T comprises of rotation C and translation r:
   *
   *   T = [C0, C1, C2, r0]
   *       [C3, C4, C5, r1]
   *       [C6, C7, C8, r2]
   *       [0, 0, 0, 1]
   *
   * The inverse is therefore:
   *
   *   C_inv = C^T
   *   r_inv = -C^T * r
   *
   *   T_inv = [C0, C3, C6, -C0*r0 - C3*r1 - C6*r2]
   *           [C1, C4, C7, -C1*r0 - C4*r1 - C7*r2]
   *           [C2, C5, C8, -C2*r0 - C5*r1 - C8*r2]
   *           [0, 0, 0, 1]
   */

  // Get rotation and translation components
  real_t C[3 * 3] = {0};
  real_t r[3] = {0};
  tf_rot_get(T, C);
  tf_trans_get(T, r);

  // Invert translation
  real_t r_out[3] = {0};
  r_out[0] = -C[0] * r[0] - C[3] * r[1] - C[6] * r[2];
  r_out[1] = -C[1] * r[0] - C[4] * r[1] - C[7] * r[2];
  r_out[2] = -C[2] * r[0] - C[5] * r[1] - C[8] * r[2];

  // First row
  T_inv[0] = C[0];
  T_inv[1] = C[3];
  T_inv[2] = C[6];
  T_inv[3] = r_out[0];

  // Second row
  T_inv[4] = C[1];
  T_inv[5] = C[4];
  T_inv[6] = C[7];
  T_inv[7] = r_out[1];

  // Third row
  T_inv[8] = C[2];
  T_inv[9] = C[5];
  T_inv[10] = C[8];
  T_inv[11] = r_out[2];

  // Fourth row
  T_inv[12] = 0.0;
  T_inv[13] = 0.0;
  T_inv[14] = 0.0;
  T_inv[15] = 1.0;
}

/**
 * Transform 3x1 point `p` using 4x4 homogeneous transformation matrix `T` and
 * output to 3x1 `retval`.
 */
void tf_point(const real_t T[4 * 4], const real_t p[3], real_t retval[3]) {
  assert(T != NULL);
  assert(p != NULL);
  assert(retval != NULL);
  assert(p != retval);

  const real_t hp_a[4] = {p[0], p[1], p[2], 1.0};
  real_t hp_b[4] = {0.0, 0.0, 0.0, 0.0};
  dot(T, 4, 4, hp_a, 4, 1, hp_b);

  retval[0] = hp_b[0];
  retval[1] = hp_b[1];
  retval[2] = hp_b[2];
}

/**
 * Transform 4x1 homogeneous point `hp` using 4x4 homogeneous transformation
 * matrix `T` and output to 4x1 `retval`.
 */
void tf_hpoint(const real_t T[4 * 4], const real_t hp[4], real_t retval[4]) {
  assert(T != NULL);
  assert(hp != retval);
  assert(retval != NULL);
  dot(T, 4, 4, hp, 4, 1, retval);
}

/**
 * Perturb the `i`-th rotational component of a 4x4 homogeneous transformation
 * matrix `T` with `step_size`.
 */
void tf_perturb_rot(real_t T[4 * 4], const real_t step_size, const int i) {
  assert(T != NULL);
  assert(i >= 0 && i <= 2);

  // Build perturb drvec
  real_t drvec[3] = {0};
  drvec[i] = step_size;

  // Decompose transform to rotation and translation
  real_t C[3 * 3] = {0};
  tf_rot_get(T, C);

  // Perturb rotation
  real_t C_rvec[3 * 3] = {0};
  real_t C_diff[3 * 3] = {0};
  rvec2rot(drvec, 1e-8, C_rvec);
  dot(C, 3, 3, C_rvec, 3, 3, C_diff);
  tf_rot_set(T, C_diff);
}

/**
 * Perturb the `i`-th translation component of a 4x4 homogeneous
 * transformation matrix with `step_size`.
 */
void tf_perturb_trans(real_t T[4 * 4], const real_t step_size, const int i) {
  assert(T != NULL);
  assert(i >= 0 && i <= 2);

  // Build perturb dr
  real_t dr[3] = {0};
  dr[i] = step_size;

  // Decompose transform get translation
  real_t r[3] = {0};
  tf_trans_get(T, r);

  // Perturb translation
  const real_t r_diff[3] = {r[0] + dr[0], r[1] + dr[1], r[2] + dr[2]};
  tf_trans_set(T, r_diff);
}

/**
 * Chain `N` homogeneous transformations `tfs`.
 */
void tf_chain(const real_t **tfs, const int N, real_t T_out[4 * 4]) {
  assert(tfs != NULL);
  assert(T_out != NULL);

  // Initialize T_out with the first transform in tfs
  mat_copy(tfs[0], 4, 4, T_out);

  // Chain transforms
  for (int i = 1; i < N; i++) {
    real_t T_from[4 * 4] = {0};
    mat_copy(T_out, 4, 4, T_from);
    dot(T_from, 4, 4, tfs[i], 4, 4, T_out);
  }
}

/**
 * Chain `N` homogeneous transformations `tfs`.
 */
void tf_chain2(const int n, ...) {
  va_list args;
  real_t T_out[4 * 4] = {0};
  eye(T_out, 4, 4);

  va_start(args, n);
  for (int i = 0; i < n; i++) {
    real_t T_from[4 * 4] = {0};
    mat_copy(T_out, 4, 4, T_from);
    dot(T_from, 4, 4, va_arg(args, real_t *), 4, 4, T_out);
  }
  mat_copy(T_out, 4, 4, va_arg(args, real_t *));
  va_end(args);
}

/**
 * Pose difference between `pose0` and `pose`, returns difference in
 * translation and rotation `diff`.
 */
void tf_diff(const real_t Ti[4 * 4], const real_t Tj[4 * 4], real_t diff[6]) {
  TF_VECTOR(Ti, pose_i);
  TF_VECTOR(Tj, pose_j);
  pose_diff(pose_i, pose_j, diff);
}

/**
 * Find the difference between two transforms, returns difference in
 * translation `dr` and rotation angle `dtheta` in radians.
 */
void tf_diff2(const real_t Ti[4 * 4],
              const real_t Tj[4 * 4],
              real_t dr[3],
              real_t *dtheta) {
  TF_VECTOR(Ti, pose_i);
  TF_VECTOR(Tj, pose_j);
  pose_diff2(pose_i, pose_j, dr, dtheta);
}

/**
 * Return translation vector `r` from pose vector `p`.
 */
void pose_get_trans(const real_t p[7], real_t r[3]) {
  r[0] = p[0];
  r[1] = p[1];
  r[2] = p[2];
}

/**
 * Return Quaternion `q` from pose vector `p`.
 */
void pose_get_quat(const real_t p[7], real_t q[4]) {
  q[0] = p[3];
  q[1] = p[4];
  q[2] = p[5];
  q[3] = p[6];
}

/**
 * Return rotation matrix `C` from pose vector `p`.
 */
void pose_get_rot(const real_t p[7], real_t C[3 * 3]) {
  const real_t q[4] = {p[3], p[4], p[5], p[6]};
  quat2rot(q, C);
}

/**
 * Pose difference between `pose0` and `pose1`, returns difference in
 * translation and rotation `diff`.
 */
void pose_diff(const real_t pose0[7], const real_t pose1[7], real_t diff[6]) {
  assert(pose0 != NULL);
  assert(pose1 != NULL);
  assert(diff != NULL);

  // dr
  diff[0] = pose0[0] - pose1[0];
  diff[1] = pose0[1] - pose1[1];
  diff[2] = pose0[2] - pose1[2];

  // dq = quat_mul(quat_inv(q_meas), q_est);
  const real_t *q0 = pose0 + 3;
  const real_t *q1 = pose1 + 3;
  real_t q0_inv[4] = {0};
  real_t dq[4] = {0};
  quat_inv(q0, q0_inv);
  quat_mul(q0_inv, q1, dq);

  // dtheta = 2 * dq;
  const real_t dtheta[3] = {2.0 * dq[1], 2.0 * dq[2], 2.0 * dq[3]};
  diff[3] = dtheta[0];
  diff[4] = dtheta[1];
  diff[5] = dtheta[2];
}

/**
 * Pose difference between `pose0` and `pose1`, returns difference in
 * translation `dr` and rotation angle `dtheta` in radians.
 */
void pose_diff2(const real_t pose0[7],
                const real_t pose1[7],
                real_t dr[3],
                real_t *dtheta) {
  assert(pose0 != NULL);
  assert(pose1 != NULL);
  assert(dr != NULL);
  assert(dtheta != NULL);

  // dr
  dr[0] = pose0[0] - pose1[0];
  dr[1] = pose0[1] - pose1[1];
  dr[2] = pose0[2] - pose1[2];

  // dC = C0.T * C1
  // dtheta = acos((tr(dC) - 1.0) / 2.0)
  const real_t *q0 = pose0 + 3;
  const real_t *q1 = pose1 + 3;
  real_t C0[3 * 3] = {0};
  real_t C0t[3 * 3] = {0};
  real_t C1[3 * 3] = {0};
  real_t dC[3 * 3] = {0};
  quat2rot(q0, C0);
  quat2rot(q1, C1);
  mat_transpose(C0, 3, 3, C0t);
  dot(C0t, 3, 3, C1, 3, 3, dC);
  const real_t tr = mat_trace(dC, 3, 3);
  if (fabs(tr - 3.0) < 1e-5) {
    *dtheta = 0.0;
  } else {
    *dtheta = acos((tr - 1.0) / 2.0);
  }
}

/**
 * Update pose vector `pose` with update vector `dx`
 */
void pose_update(real_t pose[7], const real_t dx[6]) {
  // Update translation
  pose[0] += dx[0];
  pose[1] += dx[1];
  pose[2] += dx[2];

  // Update rotation
  real_t dq[4] = {0};
  real_t q_new[4] = {0};
  quat_delta(dx + 3, dq);
  quat_mul(pose + 3, dq, q_new);
  pose[3] = q_new[0];
  pose[4] = q_new[1];
  pose[5] = q_new[2];
  pose[6] = q_new[3];
}

/**
 * Perturb pose vector `pose` randomly.
 */
void pose_random_perturb(real_t pose[7],
                         const real_t dtrans,
                         const real_t drot) {
  // Perturb pose position
  pose[0] += randf(-dtrans, dtrans);
  pose[1] += randf(-dtrans, dtrans);
  pose[2] += randf(-dtrans, dtrans);

  // Pertrub pose rotation
  real_t dalpha[3] = {0};
  randvec(-drot, drot, 3, dalpha);
  quat_update(pose + 3, dalpha);
}

/**
 * Print pose vector
 */
void print_pose(const char *prefix, const real_t pose[7]) {
  if (prefix) {
    printf("%s: ", prefix);
  }

  for (int i = 0; i < 7; i++) {
    printf("%.2f", pose[i]);
    if ((i + 1) < 7) {
      printf(", ");
    }
  }
  printf("\n");
}

void vecs2rot(const real_t acc[3], const real_t gravity[3], real_t *C) {
  // Normalize vectors
  real_t a[3] = {acc[0], acc[1], acc[2]};
  real_t g[3] = {gravity[0], gravity[1], gravity[2]};
  vec3_normalize(a);
  vec3_normalize(g);

  // Create Quaternion from two vectors
  const real_t cos_theta = a[0] * g[0] + a[1] * g[1] + a[2] * g[2];
  const real_t half_cos = sqrt(0.5 * (1.0 + cos_theta));
  const real_t half_sin = sqrt(0.5 * (1.0 - cos_theta));
  real_t w[3] = {0};
  w[0] = a[1] * g[2] - a[2] * g[1];
  w[1] = -a[0] * g[2] + a[2] * g[0];
  w[2] = a[0] * g[1] - a[1] * g[0];
  vec3_normalize(w);

  const real_t qw = half_cos;
  const real_t qx = half_sin * w[0];
  const real_t qy = half_sin * w[1];
  const real_t qz = half_sin * w[2];

  // Convert Quaternion to rotation matrix
  const real_t qx2 = qx * qx;
  const real_t qy2 = qy * qy;
  const real_t qz2 = qz * qz;
  const real_t qw2 = qw * qw;

  C[0] = qw2 + qx2 - qy2 - qz2;
  C[1] = 2 * (qx * qy - qw * qz);
  C[2] = 2 * (qx * qz + qw * qy);

  C[3] = 2 * (qx * qy + qw * qz);
  C[4] = qw2 - qx2 + qy2 - qz2;
  C[5] = 2 * (qy * qz - qw * qx);

  C[6] = 2 * (qx * qz - qw * qy);
  C[7] = 2 * (qy * qz + qw * qx);
  C[8] = qw2 - qx2 - qy2 + qz2;
}

/**
 * Convert rotation vector `rvec` to 3x3 rotation matrix `R`, where `eps` is
 * the tolerance to determine if the rotation is too small.
 */
void rvec2rot(const real_t *rvec, const real_t eps, real_t *R) {
  assert(rvec != NULL);
  assert(eps > 0);
  assert(R != NULL);

  // Magnitude of rvec
  const real_t theta = sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1]);
  // ^ basically norm(rvec), but faster

  // Check if rotation is too small
  if (theta < eps) {
    R[0] = 1.0;
    R[1] = -rvec[2];
    R[2] = rvec[1];

    R[3] = rvec[2];
    R[4] = 1.0;
    R[5] = -rvec[0];

    R[6] = -rvec[1];
    R[7] = rvec[0], R[8] = 1.0;
    return;
  }

  // Convert rvec to rotation matrix
  real_t rvec_normed[3] = {rvec[0], rvec[1], rvec[2]};
  vec_scale(rvec_normed, 3, 1 / theta);
  const real_t x = rvec_normed[0];
  const real_t y = rvec_normed[1];
  const real_t z = rvec_normed[2];

  const real_t c = cos(theta);
  const real_t s = sin(theta);
  const real_t C = 1 - c;

  const real_t xs = x * s;
  const real_t ys = y * s;
  const real_t zs = z * s;

  const real_t xC = x * C;
  const real_t yC = y * C;
  const real_t zC = z * C;

  const real_t xyC = x * yC;
  const real_t yzC = y * zC;
  const real_t zxC = z * xC;

  R[0] = x * xC + c;
  R[1] = xyC - zs;
  R[2] = zxC + ys;

  R[3] = xyC + zs;
  R[4] = y * yC + c;
  R[5] = yzC - xs;

  R[6] = zxC - ys;
  R[7] = yzC + xs;
  R[8] = z * zC + c;
}

/**
 * Convert Euler angles `ypr` (yaw, pitch, roll) in degrees to a 3x3 rotation
 * matrix `C`.
 */
void euler321(const real_t ypr[3], real_t C[3 * 3]) {
  assert(ypr != NULL);
  assert(C != NULL);

  const real_t psi = ypr[0];
  const real_t theta = ypr[1];
  const real_t phi = ypr[2];

  const real_t cpsi = cos(psi);
  const real_t spsi = sin(psi);

  const real_t ctheta = cos(theta);
  const real_t stheta = sin(theta);

  const real_t cphi = cos(phi);
  const real_t sphi = sin(phi);

  // 1st row
  C[0] = cpsi * ctheta;
  C[1] = cpsi * stheta * sphi - spsi * cphi;
  C[2] = cpsi * stheta * cphi + spsi * sphi;

  // 2nd row
  C[3] = spsi * ctheta;
  C[4] = spsi * stheta * sphi + cpsi * cphi;
  C[5] = spsi * stheta * cphi - cpsi * sphi;

  // 3rd row
  C[6] = -stheta;
  C[7] = ctheta * sphi;
  C[8] = ctheta * cphi;
}

/**
 * Convert Euler angles `ypr` in radians to a Hamiltonian Quaternion.
 */
void euler2quat(const real_t ypr[3], real_t q[4]) {
  const real_t psi = ypr[0];
  const real_t theta = ypr[1];
  const real_t phi = ypr[2];

  const real_t cphi = cos(phi / 2.0);
  const real_t ctheta = cos(theta / 2.0);
  const real_t cpsi = cos(psi / 2.0);
  const real_t sphi = sin(phi / 2.0);
  const real_t stheta = sin(theta / 2.0);
  const real_t spsi = sin(psi / 2.0);

  const real_t qx = sphi * ctheta * cpsi - cphi * stheta * spsi;
  const real_t qy = cphi * stheta * cpsi + sphi * ctheta * spsi;
  const real_t qz = cphi * ctheta * spsi - sphi * stheta * cpsi;
  const real_t qw = cphi * ctheta * cpsi + sphi * stheta * spsi;

  const real_t mag = sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  q[0] = qw / mag;
  q[1] = qx / mag;
  q[2] = qy / mag;
  q[3] = qz / mag;
}

/**
 * Convert 3x3 rotation matrix `C` to Quaternion `q`.
 */
void rot2quat(const real_t C[3 * 3], real_t q[4]) {
  assert(C != NULL);
  assert(q != NULL);

  const real_t C00 = C[0];
  const real_t C01 = C[1];
  const real_t C02 = C[2];
  const real_t C10 = C[3];
  const real_t C11 = C[4];
  const real_t C12 = C[5];
  const real_t C20 = C[6];
  const real_t C21 = C[7];
  const real_t C22 = C[8];

  const real_t tr = C00 + C11 + C22;
  real_t S = 0.0f;
  real_t qw = 0.0f;
  real_t qx = 0.0f;
  real_t qy = 0.0f;
  real_t qz = 0.0f;

  if (tr > 0) {
    S = sqrt(tr + 1.0) * 2; // S=4*qw
    qw = 0.25 * S;
    qx = (C21 - C12) / S;
    qy = (C02 - C20) / S;
    qz = (C10 - C01) / S;
  } else if ((C00 > C11) && (C[0] > C22)) {
    S = sqrt(1.0 + C[0] - C11 - C22) * 2; // S=4*qx
    qw = (C21 - C12) / S;
    qx = 0.25 * S;
    qy = (C01 + C10) / S;
    qz = (C02 + C20) / S;
  } else if (C11 > C22) {
    S = sqrt(1.0 + C11 - C[0] - C22) * 2; // S=4*qy
    qw = (C02 - C20) / S;
    qx = (C01 + C10) / S;
    qy = 0.25 * S;
    qz = (C12 + C21) / S;
  } else {
    S = sqrt(1.0 + C22 - C[0] - C11) * 2; // S=4*qz
    qw = (C10 - C01) / S;
    qx = (C02 + C20) / S;
    qy = (C12 + C21) / S;
    qz = 0.25 * S;
  }

  q[0] = qw;
  q[1] = qx;
  q[2] = qy;
  q[3] = qz;
}

/**
 * Convert 3 x 3 rotation matrix `C` to euler angles `euler`.
 */
void rot2euler(const real_t C[3 * 3], real_t ypr[3]) {
  assert(C != NULL);
  assert(ypr != NULL);

  real_t q[4] = {0};
  rot2quat(C, q);
  quat2euler(q, ypr);
}

/**
 * Convert Quaternion `q` to Euler angles 3x1 vector `euler`.
 */
void quat2euler(const real_t q[4], real_t ypr[3]) {
  assert(q != NULL);
  assert(ypr != NULL);

  const real_t qw = q[0];
  const real_t qx = q[1];
  const real_t qy = q[2];
  const real_t qz = q[3];

  const real_t qw2 = qw * qw;
  const real_t qx2 = qx * qx;
  const real_t qy2 = qy * qy;
  const real_t qz2 = qz * qz;

  const real_t t1 = atan2(2 * (qx * qw + qz * qy), (qw2 - qx2 - qy2 + qz2));
  const real_t t2 = asin(2 * (qy * qw - qx * qz));
  const real_t t3 = atan2(2 * (qx * qy + qz * qw), (qw2 + qx2 - qy2 - qz2));

  ypr[0] = t3;
  ypr[1] = t2;
  ypr[2] = t1;
}

/**
 * Print Quaternion
 */
void print_quat(const char *prefix, const real_t q[4]) {
  printf("%s: [w: %.10f, x: %.10f, y: %.10f, z: %.10f]\n",
         prefix,
         q[0],
         q[1],
         q[2],
         q[3]);
}

/**
 * Return Quaternion norm
 */
real_t quat_norm(const real_t q[4]) {
  return sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

/**
 * Setup Quaternion
 */
void quat_setup(real_t q[4]) {
  q[0] = 1.0;
  q[1] = 0.0;
  q[2] = 0.0;
  q[3] = 0.0;
}

/**
 * Normalize Quaternion
 */
void quat_normalize(real_t q[4]) {
  const real_t n = quat_norm(q);
  q[0] = q[0] / n;
  q[1] = q[1] / n;
  q[2] = q[2] / n;
  q[3] = q[3] / n;
}

/**
 * Normalize Quaternion
 */
void quat_normalize_copy(const real_t q[4], real_t q_normalized[4]) {
  const real_t n = quat_norm(q);
  q_normalized[0] = q[0] / n;
  q_normalized[1] = q[1] / n;
  q_normalized[2] = q[2] / n;
  q_normalized[3] = q[3] / n;
}

/**
 * Convert Quaternion `q` to 3x3 rotation matrix `C`.
 */
void quat2rot(const real_t q[4], real_t C[3 * 3]) {
  assert(q != NULL);
  assert(C != NULL);

  const real_t qw = q[0];
  const real_t qx = q[1];
  const real_t qy = q[2];
  const real_t qz = q[3];

  const real_t qx2 = qx * qx;
  const real_t qy2 = qy * qy;
  const real_t qz2 = qz * qz;
  const real_t qw2 = qw * qw;

  // Homogeneous form
  // -- 1st row
  C[0] = qw2 + qx2 - qy2 - qz2;
  C[1] = 2 * (qx * qy - qw * qz);
  C[2] = 2 * (qx * qz + qw * qy);
  // -- 2nd row
  C[3] = 2 * (qx * qy + qw * qz);
  C[4] = qw2 - qx2 + qy2 - qz2;
  C[5] = 2 * (qy * qz - qw * qx);
  // -- 3rd row
  C[6] = 2 * (qx * qz - qw * qy);
  C[7] = 2 * (qy * qz + qw * qx);
  C[8] = qw2 - qx2 - qy2 + qz2;
}

/**
 * Inverse Quaternion `q`.
 */
void quat_inv(const real_t q[4], real_t q_inv[4]) {
  q_inv[0] = q[0];
  q_inv[1] = -q[1];
  q_inv[2] = -q[2];
  q_inv[3] = -q[3];
}

/**
 * Form Quaternion left multiplication matrix.
 */
void quat_left(const real_t q[4], real_t left[4 * 4]) {
  const real_t qw = q[0];
  const real_t qx = q[1];
  const real_t qy = q[2];
  const real_t qz = q[3];

  // clang-format off
  left[0]  = qw; left[1]  = -qx; left[2]  = -qy; left[3]  = -qz;
  left[4]  = qx; left[5]  =  qw; left[6]  = -qz; left[7]  =  qy;
  left[8]  = qy; left[9]  =  qz; left[10] =  qw; left[11] = -qx;
  left[12] = qz; left[13] = -qy; left[14] =  qx; left[15] =  qw;
  // clang-format on
}

/**
 * Form Quaternion left multiplication matrix.
 */
void quat_left_xyz(const real_t q[4], real_t left_xyz[3 * 3]) {
  const real_t qw = q[0];
  const real_t qx = q[1];
  const real_t qy = q[2];
  const real_t qz = q[3];

  // clang-format off
  left_xyz[0] =  qw; left_xyz[1] = -qz; left_xyz[2]  =  qy;
  left_xyz[3] =  qz; left_xyz[4] =  qw;  left_xyz[5] = -qx;
  left_xyz[6] = -qy; left_xyz[7] =  qx;  left_xyz[8] =  qw;
  // clang-format on
}

/**
 * Form Quaternion right multiplication matrix.
 */
void quat_right(const real_t q[4], real_t right[4 * 4]) {
  const real_t qw = q[0];
  const real_t qx = q[1];
  const real_t qy = q[2];
  const real_t qz = q[3];

  // clang-format off
  right[0]  = qw; right[1]  = -qx; right[2]  = -qy; right[3]  = -qz;
  right[4]  = qx; right[5]  =  qw; right[6]  =  qz; right[7]  = -qy;
  right[8]  = qy; right[9]  = -qz; right[10] =  qw; right[11] =  qx;
  right[12] = qz; right[13] =  qy; right[14] = -qx; right[15] =  qw;
  // clang-format on
}

/**
 * Quaternion left-multiply `p` with `q`, results are outputted to `r`.
 */
void quat_lmul(const real_t p[4], const real_t q[4], real_t r[4]) {
  assert(p != NULL);
  assert(q != NULL);
  assert(r != NULL);

  const real_t pw = p[0];
  const real_t px = p[1];
  const real_t py = p[2];
  const real_t pz = p[3];

  r[0] = pw * q[0] - px * q[1] - py * q[2] - pz * q[3];
  r[1] = px * q[0] + pw * q[1] - pz * q[2] + py * q[3];
  r[2] = py * q[0] + pz * q[1] + pw * q[2] - px * q[3];
  r[3] = pz * q[0] - py * q[1] + px * q[2] + pw * q[3];
}

/**
 * Quaternion right-multiply `p` with `q`, results are outputted to `r`.
 */
void quat_rmul(const real_t p[4], const real_t q[4], real_t r[4]) {
  assert(p != NULL);
  assert(q != NULL);
  assert(r != NULL);

  const real_t qw = q[0];
  const real_t qx = q[1];
  const real_t qy = q[2];
  const real_t qz = q[3];

  r[0] = qw * q[0] - qx * q[1] - qy * q[2] - qz * q[3];
  r[1] = qx * q[0] + qw * q[1] + qz * q[2] - qy * q[3];
  r[2] = qy * q[0] - qz * q[1] + qw * q[2] + qx * q[3];
  r[3] = qz * q[0] + qy * q[1] - qx * q[2] + qw * q[3];
}

/**
 * Quaternion multiply `p` with `q`, results are outputted to `r`.
 */
void quat_mul(const real_t p[4], const real_t q[4], real_t r[4]) {
  assert(p != NULL);
  assert(q != NULL);
  assert(r != NULL);
  quat_lmul(p, q, r);
}

/**
 * Form delta quaternion `dq` from a small rotation vector `dalpha`.
 */
void quat_delta(const real_t dalpha[3], real_t dq[4]) {
  assert(dalpha != NULL);
  assert(dq != NULL);

  const real_t half_norm = 0.5 * vec_norm(dalpha, 3);
  const real_t k = sinc(half_norm) * 0.5;
  const real_t vector[3] = {k * dalpha[0], k * dalpha[1], k * dalpha[2]};
  real_t scalar = cos(half_norm);

  dq[0] = scalar;
  dq[1] = vector[0];
  dq[2] = vector[1];
  dq[3] = vector[2];
  quat_normalize(dq);
}

/**
 * Update quaternion with small update dalpha.
 */
void quat_update(real_t q[4], const real_t dalpha[3]) {
  const real_t dq[4] = {1.0, 0.5 * dalpha[0], 0.5 * dalpha[1], 0.5 * dalpha[2]};
  real_t q_new[4] = {0};
  quat_mul(q, dq, q_new);
  quat_normalize(q_new);
  q[0] = q_new[0];
  q[1] = q_new[1];
  q[2] = q_new[2];
  q[3] = q_new[3];
}

/**
 * Update quaternion with angular velocity and dt.
 */
void quat_update_dt(real_t q[4], const real_t w[3], const real_t dt) {
  real_t dalpha[3] = {w[0] * dt, w[1] * dt, w[2] * dt};
  quat_update(q, dalpha);
}

/**
 * Perturb quaternion
 */
void quat_perturb(real_t q[4], const int i, const real_t h) {
  assert(i >= 0 && i <= 2);

  // Form small pertubation quaternion dq
  real_t dalpha[3] = {0};
  real_t dq[4] = {0};
  dalpha[i] = h;
  quat_delta(dalpha, dq);

  // Perturb quaternion
  real_t q_[4] = {q[0], q[1], q[2], q[3]};
  quat_mul(q_, dq, q);
  quat_normalize(q);
}

/**
 * Rotate vector `x` with quaternion `q`.
 */
void quat_transform(const real_t q[4], const real_t x[3], real_t y[3]) {
  // y = q * p * q_conj
  const real_t q_conj[4] = {q[0], -q[1], -q[2], -q[3]};
  const real_t p[4] = {0.0, x[0], x[1], x[2]};

  real_t qp[4] = {0};
  real_t p_new[4] = {0};
  quat_mul(q, p, qp);
  quat_mul(qp, q_conj, p_new);

  y[0] = p_new[1];
  y[1] = p_new[2];
  y[2] = p_new[3];
}

/*******************************************************************************
 * LIE
 ******************************************************************************/

/**
 * Exponential Map
 */
void lie_Exp(const real_t phi[3], real_t C[3 * 3]) {
  assert(phi != NULL);
  assert(C != NULL);

  real_t phi_norm = vec_norm(phi, 3);
  real_t phi_skew[3 * 3] = {0};
  real_t phi_skew_sq[3 * 3] = {0};

  hat(phi, phi_skew);
  dot(phi_skew, 3, 3, phi_skew, 3, 3, phi_skew_sq);

  if (phi_norm < 1e-3) {
    // C = eye(3) + hat(phi);
    eye(C, 3, 3);
    mat_add(C, phi_skew, C, 3, 3);
  } else {
    // C = eye(3);
    // C += (sin(phi_norm) / phi_norm) * phi_skew;
    // C += ((1 - cos(phi_norm)) / phi_norm ^ 2) * phi_skew_sq;
    real_t A[3 * 3] = {0};
    mat_copy(phi_skew, 3, 3, A);
    mat_scale(A, 3, 3, (sin(phi_norm) / phi_norm));

    real_t B[3 * 3] = {0};
    mat_copy(phi_skew_sq, 3, 3, B);
    mat_scale(B, 3, 3, (1.0 - cos(phi_norm)) / (phi_norm * phi_norm));

    eye(C, 3, 3);
    mat_add(C, A, C, 3, 3);
    mat_add(C, B, C, 3, 3);
  }
}

/**
 * Logarithmic Map
 */
void lie_Log(const real_t C[3 * 3], real_t rvec[3]) {
  assert(C != NULL);
  assert(rvec != NULL);

  /**
   * phi = acos((trace(C) - 1) / 2);
   * vec = vee(C - C') / (2 * sin(phi));
   * rvec = phi * vec;
   */
  const real_t tr = C[0] + C[4] + C[8];
  const real_t phi = acos((tr - 1.0) / 2.0);

  real_t C_t[3 * 3] = {0};
  real_t dC[3 * 3] = {0};
  mat_transpose(C, 3, 3, C_t);
  mat_sub(C, C_t, dC, 3, 3);
  real_t u[3] = {0};
  vee(dC, u);
  const real_t s = 1.0 / (2 * sin(phi));
  const real_t vec[3] = {s * u[0], s * u[1], s * u[2]};

  rvec[0] = phi * vec[0];
  rvec[1] = phi * vec[1];
  rvec[2] = phi * vec[2];
}

/**
 * Box-plus operator:
 *
 *   C_new = C [+] alpha
 *
 */
void box_plus(const real_t C[3 * 3],
              const real_t alpha[3],
              real_t C_new[3 * 3]) {
  real_t dC[3 * 3] = {0};
  lie_Exp(alpha, dC);
  dot(C, 3, 3, dC, 3, 3, C_new);
}

/**
 * Box-minus operator:
 *
 *   alpha = C_a [-] C_b
 *
 */
void box_minus(const real_t Ca[3 * 3],
               const real_t Cb[3 * 3],
               real_t alpha[3]) {
  real_t Cbt[3 * 3] = {0};
  real_t dC[3 * 3] = {0};
  mat_transpose(Cb, 3, 3, Cbt);
  dot(Cbt, 3, 3, Ca, 3, 3, dC);
  lie_Log(dC, alpha);
}

/*******************************************************************************
 * GNUPLOT
 ******************************************************************************/

FILE *gnuplot_init(void) { return popen("gnuplot -persistent", "w"); }

void gnuplot_close(FILE *pipe) { fclose(pipe); }

void gnuplot_multiplot(FILE *pipe, const int num_rows, const int num_cols) {
  fprintf(pipe, "set multiplot layout %d, %d\n", num_rows, num_cols);
}

void gnuplot_send(FILE *pipe, const char *cmd) { fprintf(pipe, "%s\n", cmd); }

void gnuplot_xrange(FILE *pipe, const double xmin, const double xmax) {
  fprintf(pipe, "set xrange [%f:%f]\n", xmin, xmax);
}

void gnuplot_yrange(FILE *pipe, const double ymin, const double ymax) {
  fprintf(pipe, "set yrange [%f:%f]\n", ymin, ymax);
}

void gnuplot_send_xy(FILE *pipe,
                     const char *data_name,
                     const double *xvals,
                     const double *yvals,
                     const int n) {
  fprintf(pipe, "%s << EOD \n", data_name);
  for (int i = 0; i < n; i++) {
    fprintf(pipe, "%lf %lf\n", xvals[i], yvals[i]);
  }
  fprintf(pipe, "EOD\n");
}

void gnuplot_send_matrix(FILE *pipe,
                         const char *data_name,
                         const double *A,
                         const int m,
                         const int n) {
  // Start data
  fprintf(pipe, "%s << EOD \n", data_name);

  // Print first row with column indices
  fprintf(pipe, "%d ", n);
  for (int j = 0; j < n; j++) {
    fprintf(pipe, "%d ", j);
  }
  fprintf(pipe, "\n");

  // Print rows here first number is row index
  for (int i = 0; i < m; i++) {
    fprintf(pipe, "%d ", i);
    for (int j = 0; j < n; j++) {
      fprintf(pipe, "%lf ", A[(i * n) + j]);
    }
    fprintf(pipe, "\n");
  }

  // End data
  fprintf(pipe, "EOD\n");
}

void gnuplot_matshow(const double *A, const int m, const int n) {
  // Open gnuplot
  FILE *gnuplot = gnuplot_init();

  // Set color scheme
  gnuplot_send(gnuplot, "set palette gray");

  // Set x and y tic labels
  gnuplot_send(gnuplot, "set size square");
  gnuplot_send(gnuplot, "set xtics format ''");
  gnuplot_send(gnuplot, "set x2tics");
  gnuplot_send(gnuplot, "set yrange [* : *] reverse");
  gnuplot_send(gnuplot, "set autoscale x2fix");
  gnuplot_send(gnuplot, "set autoscale yfix");
  gnuplot_send(gnuplot, "set autoscale cbfix");

  // Plot
  gnuplot_send_matrix(gnuplot, "$A", A, m, n);
  gnuplot_send(gnuplot, "plot $A matrix with image notitle axes x2y1");
  gnuplot_send(gnuplot, "pause mouse close");

  // Close gnuplot
  gnuplot_close(gnuplot);
}

/******************************************************************************
 * CONTROL
 *****************************************************************************/

void pid_ctrl_setup(pid_ctrl_t *pid,
                    const real_t kp,
                    const real_t ki,
                    const real_t kd) {
  pid->error_prev = 0.0;
  pid->error_sum = 0.0;

  pid->error_p = 0.0;
  pid->error_i = 0.0;
  pid->error_d = 0.0;

  pid->k_p = kp;
  pid->k_i = ki;
  pid->k_d = kd;
}

real_t pid_ctrl_update(pid_ctrl_t *pid,
                       const real_t setpoint,
                       const real_t input,
                       const real_t dt) {
  // Calculate errors
  real_t error = setpoint - input;
  pid->error_sum += error * dt;

  // Calculate output
  pid->error_p = pid->k_p * error;
  pid->error_i = pid->k_i * pid->error_sum;
  pid->error_d = pid->k_d * (error - pid->error_prev) / dt;
  real_t output = pid->error_p + pid->error_i + pid->error_d;

  // Update error
  pid->error_prev = error;

  return output;
}

void pid_ctrl_reset(pid_ctrl_t *pid) {
  pid->error_prev = 0;
  pid->error_sum = 0;

  pid->error_p = 0;
  pid->error_i = 0;
  pid->error_d = 0;
}

/******************************************************************************
 * MAV
 *****************************************************************************/

void mav_model_setup(mav_model_t *mav,
                     const real_t x[12],
                     const real_t inertia[3],
                     const real_t kr,
                     const real_t kt,
                     const real_t l,
                     const real_t d,
                     const real_t m,
                     const real_t g) {
  vec_copy(x, 12, mav->x);            // State
  vec_copy(inertia, 3, mav->inertia); // Moment of inertia
  mav->kr = kr;                       // Rotation drag constant
  mav->kt = kt;                       // Translation drag constant
  mav->l = l;                         // Arm length
  mav->d = d;                         // Drag co-efficient
  mav->m = m;                         // Mass
  mav->g = g;                         // Gravitational constant
}

void mav_model_print_state(const mav_model_t *mav, const real_t time) {
  printf("time: %f, ", time);
  printf("pos: [%f, %f, %f], ", mav->x[6], mav->x[7], mav->x[8]);
  printf("att: [%f, %f, %f], ", mav->x[0], mav->x[1], mav->x[2]);
  printf("vel: [%f, %f, %f], ", mav->x[9], mav->x[10], mav->x[11]);
  printf("\n");
}

void mav_model_update(mav_model_t *mav, const real_t u[4], const real_t dt) {
  // Map out previous state
  // -- Attitude
  const real_t ph = mav->x[0];
  const real_t th = mav->x[1];
  const real_t ps = mav->x[2];
  // -- Angular velocity
  const real_t p = mav->x[3];
  const real_t q = mav->x[4];
  const real_t r = mav->x[5];
  // -- Velocity
  const real_t vx = mav->x[9];
  const real_t vy = mav->x[10];
  const real_t vz = mav->x[11];

  // Map out constants
  const real_t Ix = mav->inertia[0];
  const real_t Iy = mav->inertia[1];
  const real_t Iz = mav->inertia[2];
  const real_t kr = mav->kr;
  const real_t kt = mav->kt;
  const real_t m = mav->m;
  const real_t mr = 1.0 / m;
  const real_t g = mav->g;

  // Convert motor inputs to angular p, q, r and total thrust
  // clang-format off
  real_t A[4 * 4] = {
    1.0,          1.0,     1.0,   1.0,
    0.0,      -mav->l,     0.0,   mav->l,
    -mav->l,      0.0,  mav->l,   0.0,
    -mav->d,   mav->d,  -mav->d,  mav->d
  };
  // clang-format on

  // tau = A * u
  const real_t mt = 5.0; // Max-thrust
  const real_t s[4] = {mt * u[0], mt * u[1], mt * u[2], mt * u[3]};
  const real_t tauf = A[0] * s[0] + A[1] * s[1] + A[2] * s[2] + A[3] * s[3];
  const real_t taup = A[4] * s[0] + A[5] * s[1] + A[6] * s[2] + A[7] * s[3];
  const real_t tauq = A[8] * s[0] + A[9] * s[1] + A[10] * s[2] + A[11] * s[3];
  const real_t taur = A[12] * s[0] + A[13] * s[1] + A[14] * s[2] + A[15] * s[3];

  // Update state
  const real_t cph = cos(ph);
  const real_t sph = sin(ph);
  const real_t cth = cos(th);
  const real_t sth = sin(th);
  const real_t tth = tan(th);
  const real_t cps = cos(ps);
  const real_t sps = sin(ps);

  real_t *x = mav->x;
  // -- Attitude
  x[0] += (p + q * sph * tth + r * cos(ph) * tth) * dt;
  x[1] += (q * cph - r * sph) * dt;
  x[2] += ((1 / cth) * (q * sph + r * cph)) * dt;
  // s[2] = wrapToPi(s[2]);
  // -- Angular velocity
  x[3] += (-((Iz - Iy) / Ix) * q * r - (kr * p / Ix) + (1 / Ix) * taup) * dt;
  x[4] += (-((Ix - Iz) / Iy) * p * r - (kr * q / Iy) + (1 / Iy) * tauq) * dt;
  x[5] += (-((Iy - Ix) / Iz) * p * q - (kr * r / Iz) + (1 / Iz) * taur) * dt;
  // -- Position
  x[6] += vx * dt;
  x[7] += vy * dt;
  x[8] += vz * dt;
  // -- Linear velocity
  x[9] += ((-kt * vx / m) + mr * (cph * sth * cps + sph * sps) * tauf) * dt;
  x[10] += ((-kt * vy / m) + mr * (cph * sth * sps - sph * cps) * tauf) * dt;
  x[11] += (-(kt * vz / m) + mr * (cph * cth) * tauf - g) * dt;
}

void mav_model_attitude(const mav_model_t *mav, real_t rpy[3]) {
  rpy[0] = mav->x[0];
  rpy[1] = mav->x[1];
  rpy[2] = mav->x[2];
}

void mav_model_angular_velocity(const mav_model_t *mav, real_t pqr[3]) {
  pqr[0] = mav->x[3];
  pqr[1] = mav->x[4];
  pqr[2] = mav->x[5];
}

void mav_model_position(const mav_model_t *mav, real_t pos[3]) {
  pos[0] = mav->x[6];
  pos[1] = mav->x[7];
  pos[2] = mav->x[8];
}

void mav_model_velocity(const mav_model_t *mav, real_t vel[3]) {
  vel[0] = mav->x[9];
  vel[1] = mav->x[10];
  vel[2] = mav->x[11];
}

mav_model_telem_t *mav_model_telem_malloc(void) {
  mav_model_telem_t *telem = malloc(sizeof(mav_model_telem_t) * 1);

  telem->num_events = 0;
  telem->time = NULL;
  telem->roll = NULL;
  telem->pitch = NULL;
  telem->yaw = NULL;
  telem->wx = NULL;
  telem->wy = NULL;
  telem->wz = NULL;
  telem->x = NULL;
  telem->y = NULL;
  telem->z = NULL;
  telem->vx = NULL;
  telem->vy = NULL;
  telem->vz = NULL;

  return telem;
}

void mav_model_telem_free(mav_model_telem_t *telem) {
  free(telem->time);
  free(telem->roll);
  free(telem->pitch);
  free(telem->yaw);
  free(telem->wx);
  free(telem->wy);
  free(telem->wz);
  free(telem->x);
  free(telem->y);
  free(telem->z);
  free(telem->vx);
  free(telem->vy);
  free(telem->vz);
  free(telem);
}

void mav_model_telem_update(mav_model_telem_t *telem,
                            const mav_model_t *mav,
                            const real_t time) {
  const int idx = telem->num_events;
  const int ns = idx + 1;

  telem->time = realloc(telem->time, sizeof(real_t) * ns);
  telem->roll = realloc(telem->roll, sizeof(real_t) * ns);
  telem->pitch = realloc(telem->pitch, sizeof(real_t) * ns);
  telem->yaw = realloc(telem->yaw, sizeof(real_t) * ns);
  telem->wx = realloc(telem->wx, sizeof(real_t) * ns);
  telem->wy = realloc(telem->wy, sizeof(real_t) * ns);
  telem->wz = realloc(telem->wz, sizeof(real_t) * ns);
  telem->x = realloc(telem->x, sizeof(real_t) * ns);
  telem->y = realloc(telem->y, sizeof(real_t) * ns);
  telem->z = realloc(telem->z, sizeof(real_t) * ns);
  telem->vx = realloc(telem->vx, sizeof(real_t) * ns);
  telem->vy = realloc(telem->vy, sizeof(real_t) * ns);
  telem->vz = realloc(telem->vz, sizeof(real_t) * ns);

  telem->num_events = ns;
  telem->time[idx] = time;
  telem->roll[idx] = rad2deg(mav->x[0]);
  telem->pitch[idx] = rad2deg(mav->x[1]);
  telem->yaw[idx] = rad2deg(mav->x[2]);
  telem->wx[idx] = mav->x[3];
  telem->wy[idx] = mav->x[4];
  telem->wz[idx] = mav->x[5];
  telem->x[idx] = mav->x[6];
  telem->y[idx] = mav->x[7];
  telem->z[idx] = mav->x[8];
  telem->vx[idx] = mav->x[9];
  telem->vy[idx] = mav->x[10];
  telem->vz[idx] = mav->x[11];
}

void mav_model_telem_plot(const mav_model_telem_t *telem) {
  // Plot
  FILE *g = gnuplot_init();

  // -- Plot settings
  gnuplot_send(g, "set multiplot layout 3,1");
  gnuplot_send(g, "set colorsequence classic");
  gnuplot_send(g, "set style line 1 lt 1 pt -1 lw 1");
  gnuplot_send(g, "set style line 2 lt 2 pt -1 lw 1");
  gnuplot_send(g, "set style line 3 lt 3 pt -1 lw 1");

  // -- Attitude
  gnuplot_send(g, "set title 'Attitude'");
  gnuplot_send(g, "set xlabel 'Time [s]'");
  gnuplot_send(g, "set ylabel 'Attitude [deg]'");
  gnuplot_send_xy(g, "$roll", telem->time, telem->roll, telem->num_events);
  gnuplot_send_xy(g, "$pitch", telem->time, telem->pitch, telem->num_events);
  gnuplot_send_xy(g, "$yaw", telem->time, telem->yaw, telem->num_events);
  gnuplot_send(g, "plot $roll with lines, $pitch with lines, $yaw with lines");

  // -- Displacement
  gnuplot_send(g, "set title 'Displacement'");
  gnuplot_send(g, "set xlabel 'Time [s]'");
  gnuplot_send(g, "set ylabel 'Displacement [m]'");
  gnuplot_send_xy(g, "$x", telem->time, telem->x, telem->num_events);
  gnuplot_send_xy(g, "$y", telem->time, telem->y, telem->num_events);
  gnuplot_send_xy(g, "$z", telem->time, telem->z, telem->num_events);
  gnuplot_send(g, "plot $x with lines, $y with lines, $z with lines");

  // -- Velocity
  gnuplot_send(g, "set title 'Velocity'");
  gnuplot_send(g, "set xlabel 'Time [s]'");
  gnuplot_send(g, "set ylabel 'Velocity [m/s]'");
  gnuplot_send_xy(g, "$vx", telem->time, telem->vx, telem->num_events);
  gnuplot_send_xy(g, "$vy", telem->time, telem->vy, telem->num_events);
  gnuplot_send_xy(g, "$vz", telem->time, telem->vz, telem->num_events);
  gnuplot_send(g, "plot $vx with lines, $vy with lines, $vz with lines");

  // Clean up
  gnuplot_close(g);
}

void mav_model_telem_plot_xy(const mav_model_telem_t *telem) {
  FILE *g = gnuplot_init();

  real_t x_min = vec_min(telem->x, telem->num_events);
  real_t x_max = vec_max(telem->x, telem->num_events);
  real_t y_min = vec_min(telem->y, telem->num_events);
  real_t y_max = vec_max(telem->y, telem->num_events);
  real_t x_pad = (x_max - x_min) * 0.1;
  real_t y_pad = (x_max - x_min) * 0.1;

  gnuplot_send(g, "set colorsequence classic");
  gnuplot_send_xy(g, "$DATA", telem->x, telem->y, telem->num_events);
  gnuplot_xrange(g, x_min - x_pad, x_max + x_pad);
  gnuplot_yrange(g, y_min - y_pad, y_max + y_pad);
  gnuplot_send(g, "set xlabel 'X [m]'");
  gnuplot_send(g, "set ylabel 'Y [m]'");
  gnuplot_send(g, "plot $DATA with lines lt 1 lw 2");

  gnuplot_close(g);
}

void mav_att_ctrl_setup(mav_att_ctrl_t *ctrl) {
  ctrl->dt = 0;
  pid_ctrl_setup(&ctrl->roll, 100.0, 0.0, 5.0);
  pid_ctrl_setup(&ctrl->pitch, 100.0, 0.0, 5.0);
  pid_ctrl_setup(&ctrl->yaw, 10.0, 0.0, 1.0);
  zeros(ctrl->u, 4, 1);
}

void mav_att_ctrl_update(mav_att_ctrl_t *ctrl,
                         const real_t sp[4],
                         const real_t pv[3],
                         const real_t dt,
                         real_t u[4]) {
  // Check rate
  ctrl->dt += dt;
  if (ctrl->dt < 0.001) {
    // Return previous command
    u[0] = ctrl->u[0];
    u[1] = ctrl->u[1];
    u[2] = ctrl->u[2];
    u[3] = ctrl->u[3];
    return;
  }

  // Roll, pitch, yaw and thrust
  const real_t error_yaw = wrap_pi(sp[2] - pv[2]);
  const real_t r = pid_ctrl_update(&ctrl->roll, sp[0], pv[0], ctrl->dt);
  const real_t p = pid_ctrl_update(&ctrl->pitch, sp[1], pv[1], ctrl->dt);
  const real_t y = pid_ctrl_update(&ctrl->yaw, error_yaw, 0.0, ctrl->dt);
  const real_t t = clip_value(sp[3], 0.0, 1.0);

  // Map roll, pitch, yaw and thrust to motor outputs
  u[0] = clip_value(-p - y + t, 0.0, 1.0);
  u[1] = clip_value(-r + y + t, 0.0, 1.0);
  u[2] = clip_value(p - y + t, 0.0, 1.0);
  u[3] = clip_value(r + y + t, 0.0, 1.0);

  // Keep track of control action
  ctrl->u[0] = u[0];
  ctrl->u[1] = u[1];
  ctrl->u[2] = u[2];
  ctrl->u[3] = u[3];
  ctrl->dt = 0.0; // Reset dt
}

void mav_vel_ctrl_setup(mav_vel_ctrl_t *ctrl) {
  ctrl->dt = 0;
  pid_ctrl_setup(&ctrl->vx, 1.0, 0.0, 0.05);
  pid_ctrl_setup(&ctrl->vy, 1.0, 0.0, 0.05);
  pid_ctrl_setup(&ctrl->vz, 10.0, 0.0, 0.0);
  zeros(ctrl->u, 4, 1);
}

void mav_vel_ctrl_update(mav_vel_ctrl_t *ctrl,
                         const real_t sp[4],
                         const real_t pv[4],
                         const real_t dt,
                         real_t u[4]) {
  // Check rate
  ctrl->dt += dt;
  if (ctrl->dt < 0.001) {
    // Return previous command
    u[0] = ctrl->u[0];
    u[1] = ctrl->u[1];
    u[2] = ctrl->u[2];
    u[3] = ctrl->u[3];
    return;
  }

  // Calculate RPY errors relative to quadrotor by incorporating yaw
  const real_t errors_W[3] = {sp[0] - pv[0], sp[1] - pv[1], sp[2] - pv[2]};
  const real_t ypr[3] = {pv[3], 0.0, 0.0};
  real_t C_WS[3 * 3] = {0};
  real_t C_SW[3 * 3] = {0};
  real_t errors[3] = {0};
  euler321(ypr, C_WS);
  mat_transpose(C_WS, 3, 3, C_SW);
  dot(C_SW, 3, 3, errors_W, 3, 1, errors);

  // Roll, pitch, yaw and thrust
  const real_t r = -pid_ctrl_update(&ctrl->vy, errors[1], 0.0, dt);
  const real_t p = pid_ctrl_update(&ctrl->vx, errors[0], 0.0, dt);
  const real_t y = sp[3];
  const real_t t = 0.5 + pid_ctrl_update(&ctrl->vz, errors[2], 0.0, dt);

  u[0] = clip_value(r, deg2rad(-20.0), deg2rad(20.0));
  u[1] = clip_value(p, deg2rad(-20.0), deg2rad(20.0));
  u[2] = y;
  u[3] = clip_value(t, 0.0, 1.0);

  // // Yaw first if threshold reached
  // if (fabs(sp[3] - pv[3]) > deg2rad(2)) {
  //   outputs[0] = 0.0;
  //   outputs[1] = 0.0;
  // }

  // Keep track of control action
  ctrl->u[0] = u[0];
  ctrl->u[1] = u[1];
  ctrl->u[2] = u[2];
  ctrl->u[3] = u[3];
  ctrl->dt = 0.0; // Reset dt
}

void mav_pos_ctrl_setup(mav_pos_ctrl_t *ctrl) {
  ctrl->dt = 0;
  pid_ctrl_setup(&ctrl->x, 0.5, 0.0, 0.05);
  pid_ctrl_setup(&ctrl->y, 0.5, 0.0, 0.05);
  pid_ctrl_setup(&ctrl->z, 1.0, 0.0, 0.1);
  zeros(ctrl->u, 4, 1);
}

void mav_pos_ctrl_update(mav_pos_ctrl_t *ctrl,
                         const real_t sp[4],
                         const real_t pv[4],
                         const real_t dt,
                         real_t u[4]) {
  // Check rate
  ctrl->dt += dt;
  if (ctrl->dt < 0.01) {
    // Return previous command
    u[0] = ctrl->u[0];
    u[1] = ctrl->u[1];
    u[2] = ctrl->u[2];
    u[3] = ctrl->u[3];
    return;
  }

  // Calculate RPY errors relative to quadrotor by incorporating yaw
  const real_t errors_W[3] = {sp[0] - pv[0], sp[1] - pv[1], sp[2] - pv[2]};
  const real_t ypr[3] = {pv[3], 0.0, 0.0};
  real_t C_WS[3 * 3] = {0};
  real_t C_SW[3 * 3] = {0};
  real_t errors[3] = {0};
  euler321(ypr, C_WS);
  mat_transpose(C_WS, 3, 3, C_SW);
  dot(C_SW, 3, 3, errors_W, 3, 1, errors);

  // Velocity commands
  const real_t vx = pid_ctrl_update(&ctrl->x, errors[0], 0.0, ctrl->dt);
  const real_t vy = pid_ctrl_update(&ctrl->y, errors[1], 0.0, ctrl->dt);
  const real_t vz = pid_ctrl_update(&ctrl->z, errors[2], 0.0, ctrl->dt);
  const real_t yaw = sp[3];

  u[0] = clip_value(vx, -2.5, 2.5);
  u[1] = clip_value(vy, -2.5, 2.5);
  u[2] = clip_value(vz, -5.0, 5.0);
  u[3] = yaw;

  // Keep track of control action
  ctrl->u[0] = u[0];
  ctrl->u[1] = u[1];
  ctrl->u[2] = u[2];
  ctrl->u[3] = u[3];
  ctrl->dt = 0.0;
}

mav_waypoints_t *mav_waypoints_malloc(void) {
  mav_waypoints_t *wps = malloc(sizeof(mav_waypoints_t) * 1);

  wps->num_waypoints = 0;
  wps->waypoints = NULL;
  wps->index = 0;

  wps->wait_mode = 0;
  wps->wait_time = 0.0;

  wps->threshold_dist = 0.1;
  wps->threshold_yaw = 0.1;
  wps->threshold_wait = 2.0;

  return wps;
}

void mav_waypoints_free(mav_waypoints_t *wps) {
  free(wps->waypoints);
  free(wps);
}

void mav_waypoints_print(const mav_waypoints_t *wps) {
  printf("num_waypoints: %d\n", wps->num_waypoints);
  for (int k = 0; k < wps->num_waypoints; k++) {
    const real_t x = wps->waypoints[k * 4 + 0];
    const real_t y = wps->waypoints[k * 4 + 1];
    const real_t z = wps->waypoints[k * 4 + 2];
    const real_t yaw = wps->waypoints[k * 4 + 3];
    printf("[%d]: (%.2f, %.2f, %.2f, %.2f)\n", k, x, y, z, yaw);
  }
}

int mav_waypoints_done(const mav_waypoints_t *wps) {
  return wps->index == wps->num_waypoints;
}

void mav_waypoints_add(mav_waypoints_t *wps, real_t wp[4]) {
  const int n = wps->num_waypoints;
  wps->waypoints = realloc(wps->waypoints, sizeof(real_t) * (n + 1) * 4);
  wps->waypoints[n * 4 + 0] = wp[0];
  wps->waypoints[n * 4 + 1] = wp[1];
  wps->waypoints[n * 4 + 2] = wp[2];
  wps->waypoints[n * 4 + 3] = wp[3];
  wps->num_waypoints++;
}

void mav_waypoints_target(const mav_waypoints_t *wps, real_t wp[4]) {
  if (mav_waypoints_done(wps) == 1) {
    wp[0] = wps->waypoints[(wps->index - 1) * 4 + 0];
    wp[1] = wps->waypoints[(wps->index - 1) * 4 + 1];
    wp[2] = wps->waypoints[(wps->index - 1) * 4 + 2];
    wp[3] = wps->waypoints[(wps->index - 1) * 4 + 3];
    return;
  }

  wp[0] = wps->waypoints[wps->index * 4 + 0];
  wp[1] = wps->waypoints[wps->index * 4 + 1];
  wp[2] = wps->waypoints[wps->index * 4 + 2];
  wp[3] = wps->waypoints[wps->index * 4 + 3];
}

int mav_waypoints_update(mav_waypoints_t *wps,
                         const real_t state[4],
                         const real_t dt,
                         real_t wp[4]) {
  assert(wps->index >= 0 && wps->index <= wps->num_waypoints);

  // Check if waypoints completed - return last waypoint
  if (mav_waypoints_done(wps) == 1) {
    mav_waypoints_target(wps, wp);
    return -1;
  }

  // Check if in wait mode - gap between waypoints
  if (wps->wait_mode == 1 && (wps->wait_time >= wps->threshold_wait)) {
    // Go to next waypoint
    wps->index++;
    mav_waypoints_target(wps, wp);

    // Reset wait mode
    wps->wait_mode = 0;
    wps->wait_time = 0.0;
    return 0;
  }

  // Check if we're close to current waypoint
  mav_waypoints_target(wps, wp);
  const real_t dx = state[0] - wp[0];
  const real_t dy = state[1] - wp[1];
  const real_t dz = state[2] - wp[2];
  const real_t diff_dist = sqrt(dx * dx + dy * dy + dz * dz);
  const real_t diff_yaw = fabs(state[3] - wp[3]);
  if (diff_dist < wps->threshold_dist && diff_yaw < wps->threshold_yaw) {
    // Transition to wait mode
    wps->wait_mode = 1;
    wps->wait_time += dt;
  } else {
    // Reset wait mode
    wps->wait_mode = 0;
    wps->wait_time = 0.0;
  }

  return 0;
}

/******************************************************************************
 * COMPUTER-VISION
 *****************************************************************************/

///////////
// IMAGE //
///////////

/**
 * Setup image `img` with `width`, `height` and `data`.
 */
void image_setup(image_t *img,
                 const int width,
                 const int height,
                 uint8_t *data) {
  assert(img != NULL);
  img->width = width;
  img->height = height;
  img->data = data;
}

/**
 * Load image at `file_path`.
 * @returns Heap allocated image
 */
image_t *image_load(const char *file_path) {
  assert(file_path != NULL);

#ifdef USE_STB
  int img_w = 0;
  int img_h = 0;
  int img_c = 0;
  stbi_set_flip_vertically_on_load(1);
  uint8_t *data = stbi_load(file_path, &img_w, &img_h, &img_c, 0);
  if (!data) {
    FATAL("Failed to load image file: [%s]", file_path);
  }

  image_t *img = malloc(sizeof(image_t) * 1);
  img->width = img_w;
  img->height = img_h;
  img->channels = img_c;
  img->data = data;
  return img;
#else
  FATAL("Not Implemented!");
#endif
}

/**
 * Print image properties.
 */
void image_print_properties(const image_t *img) {
  assert(img != NULL);
  printf("img.width: %d\n", img->width);
  printf("img.height: %d\n", img->height);
  printf("img.channels: %d\n", img->channels);
}

/**
 * Free image.
 */
void image_free(image_t *img) {
  assert(img != NULL);
  free(img->data);
  free(img);
}

////////////
// RADTAN //
////////////

/**
 * Distort 2x1 point `p` using Radial-Tangential distortion, where the
 * distortion params are stored in `params` (k1, k2, p1, p2), results are
 * written to 2x1 vector `p_d`.
 */
void radtan4_distort(const real_t params[4],
                     const real_t p_in[2],
                     real_t p_out[2]) {
  assert(params != NULL);
  assert(p_in != NULL);
  assert(p_out != NULL);

  // Distortion parameters
  const real_t k1 = params[0];
  const real_t k2 = params[1];
  const real_t p1 = params[2];
  const real_t p2 = params[3];

  // Point
  const real_t x = p_in[0];
  const real_t y = p_in[1];

  // Apply radial distortion
  const real_t x2 = x * x;
  const real_t y2 = y * y;
  const real_t r2 = x2 + y2;
  const real_t r4 = r2 * r2;
  const real_t radial_factor = 1.0 + (k1 * r2) + (k2 * r4);
  const real_t x_dash = x * radial_factor;
  const real_t y_dash = y * radial_factor;

  // Apply tangential distortion
  const real_t xy = x * y;
  const real_t x_ddash = x_dash + (2.0 * p1 * xy + p2 * (r2 + 2.0 * x2));
  const real_t y_ddash = y_dash + (2.0 * p2 * xy + p1 * (r2 + 2.0 * y2));

  // Distorted point
  p_out[0] = x_ddash;
  p_out[1] = y_ddash;
}

/**
 * Radial-Tangential undistort
 */
void radtan4_undistort(const real_t params[4],
                       const real_t p_in[2],
                       real_t p_out[2]) {
  const int max_iter = 5;
  real_t p[2] = {p_in[0], p_in[1]};

  for (int i = 0; i < max_iter; i++) {
    // Error
    real_t p_d[2] = {0};
    radtan4_distort(params, p, p_d);
    const real_t e[2] = {p_in[0] - p_d[0], p_in[1] - p_d[1]};

    // Jacobian
    real_t J[2 * 2] = {0};
    radtan4_point_jacobian(params, p, J);

    // Calculate update
    // dp = inv(J' * J) * J' * e;
    real_t Jt[2 * 2] = {0};
    real_t JtJ[2 * 2] = {0};
    real_t JtJ_inv[2 * 2] = {0};
    real_t dp[2] = {0};

    mat_transpose(J, 2, 2, Jt);
    dot(Jt, 2, 2, J, 2, 2, JtJ);
    pinv(JtJ, 2, 2, JtJ_inv);
    dot3(JtJ_inv, 2, 2, Jt, 2, 2, e, 2, 1, dp);

    // Update
    p[0] += dp[0];
    p[1] += dp[1];

    // Calculate cost
    // cost = e' * e
    const real_t cost = e[0] * e[0] + e[1] * e[1];
    if (cost < 1.0e-15) {
      break;
    }
  }

  // Return result
  p_out[0] = p[0];
  p_out[1] = p[1];
}

/**
 * Form Radial-Tangential point jacobian, using distortion `params` (k1, k2,
 * p1, p2), 2x1 image point `p`, the jacobian is written to 2x2 `J_point`.
 */
void radtan4_point_jacobian(const real_t params[4],
                            const real_t p[2],
                            real_t J_point[2 * 2]) {
  assert(params != NULL);
  assert(p != NULL);
  assert(J_point != NULL);

  // Distortion parameters
  const real_t k1 = params[0];
  const real_t k2 = params[1];
  const real_t p1 = params[2];
  const real_t p2 = params[3];

  // Point
  const real_t x = p[0];
  const real_t y = p[1];

  // Apply radial distortion
  const real_t x2 = x * x;
  const real_t y2 = y * y;
  const real_t r2 = x2 + y2;
  const real_t r4 = r2 * r2;

  // Point Jacobian is 2x2
  J_point[0] = k1 * r2 + k2 * r4 + 2 * p1 * y + 6 * p2 * x;
  J_point[0] += x * (2 * k1 * x + 4 * k2 * x * r2) + 1;
  J_point[1] = 2 * p1 * x + 2 * p2 * y + y * (2 * k1 * x + 4 * k2 * x * r2);
  J_point[2] = J_point[1];
  J_point[3] = k1 * r2 + k2 * r4 + 6 * p1 * y + 2 * p2 * x;
  J_point[3] += y * (2 * k1 * y + 4 * k2 * y * r2) + 1;
}

/**
 * Form Radial-Tangential parameter jacobian, using distortion `params` (k1,
 * k2, p1, p2), 2x1 image point `p`, the jacobian is written to 2x4 `J_param`.
 */
void radtan4_params_jacobian(const real_t params[4],
                             const real_t p[2],
                             real_t J_param[2 * 4]) {
  assert(params != NULL);
  assert(p != NULL);
  assert(J_param != NULL);
  UNUSED(params);

  // Point
  const real_t x = p[0];
  const real_t y = p[1];

  // Setup
  const real_t x2 = x * x;
  const real_t y2 = y * y;
  const real_t xy = x * y;
  const real_t r2 = x2 + y2;
  const real_t r4 = r2 * r2;

  // Param Jacobian is 2x4
  J_param[0] = x * r2;
  J_param[1] = x * r4;
  J_param[2] = 2 * xy;
  J_param[3] = 3 * x2 + y2;

  J_param[4] = y * r2;
  J_param[5] = y * r4;
  J_param[6] = x2 + 3 * y2;
  J_param[7] = 2 * xy;
}

//////////
// EQUI //
//////////

/**
 * Distort 2x1 point `p` using Equi-Distant distortion, where the
 * distortion params are stored in `params` (k1, k2, k3, k4), results are
 * written to 2x1 vector `p_d`.
 */
void equi4_distort(const real_t params[4],
                   const real_t p_in[2],
                   real_t p_out[2]) {
  assert(params != NULL);
  assert(p_in != NULL);
  assert(p_out != NULL);

  const real_t k1 = params[0];
  const real_t k2 = params[1];
  const real_t k3 = params[2];
  const real_t k4 = params[3];

  const real_t x = p_in[0];
  const real_t y = p_in[1];
  const real_t r = sqrt(x * x + y * y);

  const real_t th = atan(r);
  const real_t th2 = th * th;
  const real_t th4 = th2 * th2;
  const real_t th6 = th4 * th2;
  const real_t th8 = th4 * th4;
  const real_t thd = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8);
  const real_t s = thd / r;

  p_out[0] = s * x;
  p_out[1] = s * y;
}

/**
 * Equi-distant un-distort
 */
void equi4_undistort(const real_t dist_params[4],
                     const real_t p_in[2],
                     real_t p_out[2]) {
  const real_t k1 = dist_params[0];
  const real_t k2 = dist_params[1];
  const real_t k3 = dist_params[2];
  const real_t k4 = dist_params[3];

  const real_t thd = sqrt(p_in[0] * p_in[0] + p_in[1] * p_in[1]);
  real_t th = thd; // Initial guess
  for (int i = 20; i > 0; i--) {
    const real_t th2 = th * th;
    const real_t th4 = th2 * th2;
    const real_t th6 = th4 * th2;
    const real_t th8 = th4 * th4;
    th = thd / (1 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8);
  }

  const real_t scaling = tan(th) / thd;
  p_out[0] = p_in[0] * scaling;
  p_out[1] = p_in[1] * scaling;
}

/**
 * Form Equi-Distant point jacobian, using distortion `params` (k1, k2, k3,
 * k4), 2x1 image point `p`, the jacobian is written to 2x2 `J_point`.
 */
void equi4_point_jacobian(const real_t params[4],
                          const real_t p[2],
                          real_t J_point[2 * 2]) {
  assert(params != NULL);
  assert(p != NULL);
  assert(J_point != NULL);

  const real_t k1 = params[0];
  const real_t k2 = params[1];
  const real_t k3 = params[2];
  const real_t k4 = params[3];

  const real_t x = p[0];
  const real_t y = p[1];
  const real_t r = sqrt(x * x + y * y);

  const real_t th = atan(r);
  const real_t th2 = th * th;
  const real_t th4 = th2 * th2;
  const real_t th6 = th4 * th2;
  const real_t th8 = th4 * th4;
  const real_t thd = th * (1.0 + k1 * th2 + k2 * th4 + k3 * th6 + k4 * th8);

  const real_t th_r = 1.0 / (r * r + 1.0);
  const real_t thd_th =
      1.0 + 3.0 * k1 * th2 + 5.0 * k2 * th4 + 7.0 * k3 * th6 + 9.0 * k4 * th8;
  const real_t s = thd / r;
  const real_t s_r = thd_th * th_r / r - thd / (r * r);
  const real_t r_x = 1.0 / r * x;
  const real_t r_y = 1.0 / r * y;

  // Point Jacobian is 2x2
  J_point[0] = s + x * s_r * r_x;
  J_point[1] = x * s_r * r_y;
  J_point[2] = y * s_r * r_x;
  J_point[3] = s + y * s_r * r_y;
}

/**
 * Form Equi-Distant parameter jacobian, using distortion `params` (k1, k2,
 * k3, k4), 2x1 image point `p`, the jacobian is written to 2x4 `J_param`.
 */
void equi4_params_jacobian(const real_t params[4],
                           const real_t p[2],
                           real_t J_param[2 * 4]) {
  assert(params != NULL);
  assert(p != NULL);
  assert(J_param != NULL);
  UNUSED(params);

  const real_t x = p[0];
  const real_t y = p[1];
  const real_t r = sqrt(x * x + y * y);

  const real_t th = atan(r);
  const real_t th2 = th * th;
  const real_t th3 = th2 * th;
  const real_t th5 = th3 * th2;
  const real_t th7 = th5 * th2;
  const real_t th9 = th7 * th2;

  // Param Jacobian is 2x4
  J_param[0] = x * th3 / r;
  J_param[1] = x * th5 / r;
  J_param[2] = x * th7 / r;
  J_param[3] = x * th9 / r;

  J_param[4] = y * th3 / r;
  J_param[5] = y * th5 / r;
  J_param[6] = y * th7 / r;
  J_param[7] = y * th9 / r;
}

/////////////
// PINHOLE //
/////////////

/**
 * Estimate pinhole focal length. The focal length is estimated using
 * `image_width` [pixels], and `fov` (Field of view of the camera) [rad].
 */
real_t pinhole_focal(const int image_width, const real_t fov) {
  return ((image_width / 2.0) / tan(deg2rad(fov) / 2.0));
}

/**
 * From 3x3 camera matrix K using pinhole camera parameters.
 *
 *   K = [fx,  0,  cx,
 *         0  fy,  cy,
 *         0   0,   1];
 *
 * where `params` is assumed to contain the fx, fy, cx, cy in that order.
 */
void pinhole_K(const real_t params[4], real_t K[3 * 3]) {
  K[0] = params[0];
  K[1] = 0.0;
  K[2] = params[2];

  K[3] = 0.0;
  K[4] = params[1];
  K[5] = params[3];

  K[6] = 0.0;
  K[7] = 0.0;
  K[8] = 1.0;
}

/**
 * Form 3x4 pinhole projection matrix `P`:
 *
 *   P = K * [-C | -C * r];
 *
 * Where K is the pinhole camera matrix formed using the camera parameters
 * `params` (fx, fy, cx, cy), C and r is the rotation and translation
 * component of the camera pose represented as a 4x4 homogenous transform `T`.
 */
void pinhole_projection_matrix(const real_t params[4],
                               const real_t T[4 * 4],
                               real_t P[3 * 4]) {
  assert(params != NULL);
  assert(T != NULL);
  assert(P != NULL);

  // Form K matrix
  real_t K[3 * 3] = {0};
  pinhole_K(params, K);

  // Invert camera pose
  real_t T_inv[4 * 4] = {0};
  tf_inv(T, T_inv);

  // Extract rotation and translation component
  real_t C[3 * 3] = {0};
  real_t r[3] = {0};
  tf_rot_get(T_inv, C);
  tf_trans_get(T_inv, r);

  // Form [C | r] matrix
  real_t Cr[3 * 4] = {0};
  Cr[0] = C[0];
  Cr[1] = C[1];
  Cr[2] = C[2];
  Cr[3] = r[0];

  Cr[4] = C[3];
  Cr[5] = C[4];
  Cr[6] = C[5];
  Cr[7] = r[1];

  Cr[8] = C[6];
  Cr[9] = C[7];
  Cr[10] = C[8];
  Cr[11] = r[2];

  // Form projection matrix P = K * [C | r]
  dot(K, 3, 3, Cr, 3, 4, P);
}

/**
 * Project 3D point `p_C` observed from the camera to the image plane `z`
 * using pinhole parameters `params` (fx, fy, cx, cy).
 */
void pinhole_project(const real_t params[4], const real_t p_C[3], real_t z[2]) {
  assert(params != NULL);
  assert(p_C != NULL);
  assert(z != NULL);

  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];

  const real_t px = p_C[0] / p_C[2];
  const real_t py = p_C[1] / p_C[2];

  z[0] = px * fx + cx;
  z[1] = py * fy + cy;
}

/**
 * Form Pinhole point jacobian `J` using pinhole parameters `params`.
 */
void pinhole_point_jacobian(const real_t params[4], real_t J[2 * 2]) {
  assert(params != NULL);
  assert(J != NULL);

  J[0] = params[0];
  J[1] = 0.0;
  J[2] = 0.0;
  J[3] = params[1];
}

/**
 * Form Pinhole parameter jacobian `J` using pinhole parameters `params` and
 * 2x1 image point `x`.
 */
void pinhole_params_jacobian(const real_t params[4],
                             const real_t x[2],
                             real_t J[2 * 4]) {
  assert(params != NULL);
  assert(x != NULL);
  assert(J != NULL);
  UNUSED(params);

  J[0] = x[0];
  J[1] = 0.0;
  J[2] = 1.0;
  J[3] = 0.0;

  J[4] = 0.0;
  J[5] = x[1];
  J[6] = 0.0;
  J[7] = 1.0;
}

/////////////////////
// PINHOLE-RADTAN4 //
/////////////////////

/**
 * Projection of 3D point to image plane using Pinhole + Radial-Tangential.
 */
void pinhole_radtan4_project(const real_t params[8],
                             const real_t p_C[3],
                             real_t z[2]) {
  assert(params != NULL);
  assert(p_C != NULL);
  assert(z != NULL);

  // Project
  const real_t p[2] = {p_C[0] / p_C[2], p_C[1] / p_C[2]};

  // Distort
  const real_t d[4] = {params[4], params[5], params[6], params[7]};
  real_t p_d[2] = {0};
  radtan4_distort(d, p, p_d);

  // Scale and center
  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];

  z[0] = p_d[0] * fx + cx;
  z[1] = p_d[1] * fy + cy;
}

/**
 * Pinhole Radial-Tangential Undistort
 */
void pinhole_radtan4_undistort(const real_t params[8],
                               const real_t z_in[2],
                               real_t z_out[2]) {
  assert(params != NULL);
  assert(z_in != NULL);
  assert(z_out != NULL);

  // Back project and undistort
  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];
  const real_t p[2] = {(z_in[0] - cx) / fx, (z_in[1] - cy) / fy};
  real_t p_undist[2] = {0};
  radtan4_undistort(params + 4, p, p_undist);

  // Project undistorted point to image plane
  z_out[0] = p_undist[0] * fx + cx;
  z_out[1] = p_undist[1] * fy + cy;
}

/**
 * Pinhole Radial-Tangential back project
 */
void pinhole_radtan4_back_project(const real_t params[8],
                                  const real_t z[2],
                                  real_t ray[3]) {
  assert(params != NULL);
  assert(z != NULL);
  assert(ray != NULL);

  // Back project and undistort
  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];
  const real_t p[2] = {(z[0] - cx) / fx, (z[1] - cy) / fy};
  real_t p_undist[2] = {0};
  radtan4_undistort(params + 4, p, p_undist);

  ray[0] = p_undist[0];
  ray[1] = p_undist[1];
  ray[2] = 1.0;
}

/**
 * Projection Jacobian of Pinhole + Radial-Tangential.
 */
void pinhole_radtan4_project_jacobian(const real_t params[8],
                                      const real_t p_C[3],
                                      real_t J[2 * 3]) {
  assert(params != NULL);
  assert(p_C != NULL);
  assert(J != NULL);

  // Project
  const real_t x = p_C[0];
  const real_t y = p_C[1];
  const real_t z = p_C[2];
  const real_t p[2] = {x / z, y / z};

  // Projection Jacobian
  real_t J_p[2 * 3] = {0};
  J_p[0] = 1.0 / z;
  J_p[1] = 0.0;
  J_p[2] = -x / (z * z);
  J_p[3] = 0.0;
  J_p[4] = 1.0 / z;
  J_p[5] = -y / (z * z);

  // Distortion Point Jacobian
  const real_t k1 = params[4];
  const real_t k2 = params[5];
  const real_t p1 = params[6];
  const real_t p2 = params[7];
  const real_t d[4] = {k1, k2, p1, p2};
  real_t J_d[2 * 2] = {0};
  radtan4_point_jacobian(d, p, J_d);

  // Project Point Jacobian
  real_t J_k[2 * 3] = {0};
  pinhole_point_jacobian(params, J_k);

  // J = J_k * J_d * J_p;
  real_t J_dp[2 * 3] = {0};
  dot(J_d, 2, 2, J_p, 2, 3, J_dp);
  dot(J_k, 2, 2, J_dp, 2, 3, J);
}

/**
 * Parameter Jacobian of Pinhole + Radial-Tangential.
 */
void pinhole_radtan4_params_jacobian(const real_t params[8],
                                     const real_t p_C[3],
                                     real_t J[2 * 8]) {
  assert(params != NULL);
  assert(p_C != NULL);
  assert(J != NULL);

  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];
  const real_t k[4] = {fx, fy, cx, cy};

  const real_t k1 = params[4];
  const real_t k2 = params[5];
  const real_t p1 = params[6];
  const real_t p2 = params[7];
  const real_t d[4] = {k1, k2, p1, p2};

  // Project
  const real_t x = p_C[0];
  const real_t y = p_C[1];
  const real_t z = p_C[2];
  const real_t p[2] = {x / z, y / z};

  // Distort
  real_t p_d[2] = {0};
  radtan4_distort(d, p, p_d);

  // Project params Jacobian: J_proj_params
  real_t J_proj_params[2 * 4] = {0};
  pinhole_params_jacobian(k, p_d, J_proj_params);

  // Project point Jacobian: J_proj_point
  real_t J_proj_point[2 * 2] = {0};
  pinhole_point_jacobian(k, J_proj_point);

  // Distortion point Jacobian: J_dist_params
  real_t J_dist_params[2 * 4] = {0};
  radtan4_params_jacobian(d, p, J_dist_params);

  // Radtan4 params Jacobian: J_radtan4
  real_t J_radtan4[2 * 4] = {0};
  dot(J_proj_point, 2, 2, J_dist_params, 2, 4, J_radtan4);

  // J = [J_proj_params, J_proj_point * J_dist_params]
  J[0] = J_proj_params[0];
  J[1] = J_proj_params[1];
  J[2] = J_proj_params[2];
  J[3] = J_proj_params[3];

  J[8] = J_proj_params[4];
  J[9] = J_proj_params[5];
  J[10] = J_proj_params[6];
  J[11] = J_proj_params[7];

  J[4] = J_radtan4[0];
  J[5] = J_radtan4[1];
  J[6] = J_radtan4[2];
  J[7] = J_radtan4[3];

  J[12] = J_radtan4[4];
  J[13] = J_radtan4[5];
  J[14] = J_radtan4[6];
  J[15] = J_radtan4[7];
}

///////////////////
// PINHOLE-EQUI4 //
///////////////////

/**
 * Projection of 3D point to image plane using Pinhole + Equi-Distant.
 */
void pinhole_equi4_project(const real_t params[8],
                           const real_t p_C[3],
                           real_t z[2]) {
  assert(params != NULL);
  assert(p_C != NULL);
  assert(z != NULL);

  // Project
  const real_t p[2] = {p_C[0] / p_C[2], p_C[1] / p_C[2]};

  // Distort
  const real_t d[4] = {params[4], params[5], params[6], params[7]};
  real_t p_d[2] = {0};
  equi4_distort(d, p, p_d);

  // Scale and center
  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];

  z[0] = p_d[0] * fx + cx;
  z[1] = p_d[1] * fy + cy;
}

/**
 * Pinhole Equi-distant Undistort
 */
void pinhole_equi4_undistort(const real_t params[8],
                             const real_t z_in[2],
                             real_t z_out[2]) {
  assert(params != NULL);
  assert(z_in != NULL);
  assert(z_out != NULL);

  // Back project and undistort
  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];
  const real_t p[2] = {(z_in[0] - cx) / fx, (z_in[1] - cy) / fy};
  real_t p_undist[2] = {0};
  equi4_undistort(params + 4, p, p_undist);

  // Project undistorted point to image plane
  z_out[0] = p_undist[0] * fx + cx;
  z_out[1] = p_undist[1] * fy + cy;
}

/**
 * Pinhole Equi-distant back project
 */
void pinhole_equi4_back_project(const real_t params[8],
                                const real_t z[2],
                                real_t ray[3]) {
  assert(params != NULL);
  assert(z != NULL);
  assert(ray != NULL);

  // Back project and undistort
  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];
  const real_t p[2] = {(z[0] - cx) / fx, (z[1] - cy) / fy};
  real_t p_undist[2] = {0};
  equi4_undistort(params + 4, p, p_undist);

  ray[0] = p_undist[0];
  ray[1] = p_undist[1];
  ray[2] = 1.0;
}

/**
 * Projection Jacobian of Pinhole + Equi-Distant.
 */
void pinhole_equi4_project_jacobian(const real_t params[8],
                                    const real_t p_C[3],
                                    real_t J[2 * 3]) {
  assert(params != NULL);
  assert(p_C != NULL);
  assert(J != NULL);

  // Project
  const real_t x = p_C[0];
  const real_t y = p_C[1];
  const real_t z = p_C[2];
  const real_t p[2] = {x / z, y / z};

  // Projection Jacobian
  real_t J_p[2 * 3] = {0};
  J_p[0] = 1.0 / z;
  J_p[1] = 0.0;
  J_p[2] = -x / (z * z);
  J_p[3] = 0.0;
  J_p[4] = 1.0 / z;
  J_p[5] = -y / (z * z);

  // Distortion Point Jacobian
  const real_t k1 = params[4];
  const real_t k2 = params[5];
  const real_t k3 = params[6];
  const real_t k4 = params[7];
  const real_t d[4] = {k1, k2, k3, k4};
  real_t J_d[2 * 2] = {0};
  equi4_point_jacobian(d, p, J_d);

  // Project Point Jacobian
  real_t J_k[2 * 3] = {0};
  pinhole_point_jacobian(params, J_k);

  // J = J_k * J_d * J_p;
  real_t J_dp[2 * 3] = {0};
  dot(J_d, 2, 2, J_p, 2, 3, J_dp);
  dot(J_k, 2, 2, J_dp, 2, 3, J);
}

void pinhole_equi4_params_jacobian(const real_t params[8],
                                   const real_t p_C[3],
                                   real_t J[2 * 8]) {
  assert(params != NULL);
  assert(p_C != NULL);
  assert(J != NULL);

  const real_t fx = params[0];
  const real_t fy = params[1];
  const real_t cx = params[2];
  const real_t cy = params[3];
  const real_t k[4] = {fx, fy, cx, cy};

  const real_t k1 = params[4];
  const real_t k2 = params[5];
  const real_t p1 = params[6];
  const real_t p2 = params[7];
  const real_t d[4] = {k1, k2, p1, p2};

  // Project
  const real_t x = p_C[0];
  const real_t y = p_C[1];
  const real_t z = p_C[2];
  const real_t p[2] = {x / z, y / z};

  // Distort
  real_t p_d[2] = {0};
  equi4_distort(d, p, p_d);

  // Project params Jacobian: J_proj_params
  real_t J_proj_params[2 * 4] = {0};
  pinhole_params_jacobian(k, p_d, J_proj_params);

  // Project point Jacobian: J_proj_point
  real_t J_proj_point[2 * 2] = {0};
  pinhole_point_jacobian(k, J_proj_point);

  // Distortion point Jacobian: J_dist_params
  real_t J_dist_params[2 * 4] = {0};
  equi4_params_jacobian(d, p, J_dist_params);

  // Radtan4 params Jacobian: J_equi4
  real_t J_equi4[2 * 4] = {0};
  dot(J_proj_point, 2, 2, J_dist_params, 2, 4, J_equi4);

  // J = [J_proj_params, J_proj_point * J_dist_params]
  J[0] = J_proj_params[0];
  J[1] = J_proj_params[1];
  J[2] = J_proj_params[2];
  J[3] = J_proj_params[3];

  J[8] = J_proj_params[4];
  J[9] = J_proj_params[5];
  J[10] = J_proj_params[6];
  J[11] = J_proj_params[7];

  J[4] = J_equi4[0];
  J[5] = J_equi4[1];
  J[6] = J_equi4[2];
  J[7] = J_equi4[3];

  J[12] = J_equi4[4];
  J[13] = J_equi4[5];
  J[14] = J_equi4[6];
  J[15] = J_equi4[7];
}

//////////////
// GEOMETRY //
//////////////

/**
 * Triangulate a single 3D point `p` observed by two different camera frames
 * represented by two 3x4 camera projection matrices `P_i` and `P_j`, and the
 * 2D image point correspondance `z_i` and `z_j`.
 */
void linear_triangulation(const real_t P_i[3 * 4],
                          const real_t P_j[3 * 4],
                          const real_t z_i[2],
                          const real_t z_j[2],
                          real_t p[3]) {
  assert(P_i != NULL);
  assert(P_j != NULL);
  assert(z_i != NULL);
  assert(z_j != NULL);
  assert(p != NULL);

  // Form A matrix
  real_t A[4 * 4] = {0};
  // -- ROW 1
  A[0] = -P_i[4] + P_i[8] * z_i[1];
  A[1] = -P_i[5] + P_i[9] * z_i[1];
  A[2] = P_i[10] * z_i[1] - P_i[6];
  A[3] = P_i[11] * z_i[1] - P_i[7];
  // -- ROW 2
  A[4] = -P_i[0] + P_i[8] * z_i[0];
  A[5] = -P_i[1] + P_i[9] * z_i[0];
  A[6] = P_i[10] * z_i[0] - P_i[2];
  A[7] = P_i[11] * z_i[0] - P_i[3];
  // -- ROW 3
  A[8] = -P_j[4] + P_j[8] * z_j[1];
  A[9] = -P_j[5] + P_j[9] * z_j[1];
  A[10] = P_j[10] * z_j[1] - P_j[6];
  A[11] = P_j[11] * z_j[1] - P_j[7];
  // -- ROW 4
  A[12] = -P_j[0] + P_j[8] * z_j[0];
  A[13] = -P_j[1] + P_j[9] * z_j[0];
  A[14] = P_j[10] * z_j[0] - P_j[2];
  A[15] = P_j[11] * z_j[0] - P_j[3];

  // Form A_t
  real_t A_t[4 * 4] = {0};
  mat_transpose(A, 4, 4, A_t);

  // SVD
  real_t A2[4 * 4] = {0};
  real_t s[4] = {0};
  real_t U[4 * 4] = {0};
  real_t V[4 * 4] = {0};
  dot(A_t, 4, 4, A, 4, 4, A2);
  svd(A2, 4, 4, U, s, V);

  // Get best row of V_t
  real_t min_s = s[0];
  real_t x = V[0];
  real_t y = V[4];
  real_t z = V[8];
  real_t w = V[12];
  for (int i = 1; i < 4; i++) {
    if (s[i] < min_s) {
      min_s = s[i];
      x = V[i + 0];
      y = V[i + 4];
      z = V[i + 8];
      w = V[i + 12];
    }
  }

  // Normalize the scale to obtain the 3D point
  p[0] = x / w;
  p[1] = y / w;
  p[2] = z / w;
}

/**
 * Find Homography.
 *
 * A Homography is a transformation (a 3x3 matrix) that maps the normalized
 * image points from one image to the corresponding normalized image points in
 * the other image. Specifically, let x and y be the n-th homogeneous points
 * of pts_i and pts_j:
 *
 *   x = [u_i, v_i, 1.0]
 *   y = [u_j, v_j, 1.0]
 *
 * The Homography is a 3x3 matrix that transforms x to y:
 *
 *   y = H * x
 *
 * **IMPORTANT**: The normalized image points `pts_i` and `pts_j` must
 * correspond to points in 3D that on a plane.
 */
int homography_find(const real_t *pts_i,
                    const real_t *pts_j,
                    const int num_points,
                    real_t H[3 * 3]) {

  const int Am = 2 * num_points;
  const int An = 9;
  real_t *A = malloc(sizeof(real_t) * Am * An);

  for (int n = 0; n < num_points; n++) {
    const real_t x_i = pts_i[n * 2 + 0];
    const real_t y_i = pts_i[n * 2 + 1];
    const real_t x_j = pts_j[n * 2 + 0];
    const real_t y_j = pts_j[n * 2 + 1];

    const int rs = n * 18;
    const int re = n * 18 + 9;
    A[rs + 0] = -x_i;
    A[rs + 1] = -y_i;
    A[rs + 2] = -1.0;
    A[rs + 3] = 0.0;
    A[rs + 4] = 0.0;
    A[rs + 5] = 0.0;
    A[rs + 6] = x_i * x_j;
    A[rs + 7] = y_i * x_j;
    A[rs + 8] = x_j;

    A[re + 0] = 0.0;
    A[re + 1] = 0.0;
    A[re + 2] = 0.0;
    A[re + 3] = -x_i;
    A[re + 4] = -y_i;
    A[re + 5] = -1.0;
    A[re + 6] = x_i * y_j;
    A[re + 7] = y_i * y_j;
    A[re + 8] = y_j;
  }

  real_t *U = malloc(sizeof(real_t) * Am * Am);
  real_t *s = malloc(sizeof(real_t) * Am);
  real_t *V = malloc(sizeof(real_t) * An * An);
  if (svd(A, Am, An, U, s, V) != 0) {
    return -1;
  }

  // Form the Homography matrix using the last column of V and normalize
  H[0] = V[8] / V[80];
  H[1] = V[17] / V[80];
  H[2] = V[26] / V[80];

  H[3] = V[35] / V[80];
  H[4] = V[44] / V[80];
  H[5] = V[53] / V[80];

  H[6] = V[62] / V[80];
  H[7] = V[71] / V[80];
  H[8] = V[80] / V[80];

  // Clean up
  free(A);
  free(U);
  free(s);
  free(V);

  return 0;
}

/**
 * Compute relative pose between camera and planar object `T_CF` using `N` 3D
 * object points `obj_pts`, 2D image points in pixels, as well as the pinhole
 * focal lengths `fx`, `fy` and principal centers `cx` and `cy`.
 *
 * Source:
 *
 *   Section 4.1.3: From homography to pose computation
 *
 *   Marchand, Eric, Hideaki Uchiyama, and Fabien Spindler. "Pose estimation
 *   for augmented reality: a hands-on survey." IEEE transactions on
 *   visualization and computer graphics 22.12 (2015): 2633-2651.
 *
 *   https://github.com/lagadic/camera_localization
 *
 * Returns:
 *
 *   `0` for success and `-1` for failure.
 *
 */
int homography_pose(const real_t *proj_params,
                    const real_t *img_pts,
                    const real_t *obj_pts,
                    const int N,
                    real_t T_CF[4 * 4]) {
  // Form A to compute ||Ah|| = 0 using SVD, where A is an (N * 2) x 9 matrix
  // and h is the vectorized Homography matrix h, N is the number of points.
  // if N == 4, the matrix has more columns than rows. The solution is to add
  // an extra line with zeros.
  const int num_rows = 2 * N + ((N == 4) ? 1 : 0);
  const int num_cols = 9;
  const real_t fx = proj_params[0];
  const real_t fy = proj_params[1];
  const real_t cx = proj_params[2];
  const real_t cy = proj_params[3];
  real_t *A = malloc(sizeof(real_t) * num_rows * num_cols);

  for (int i = 0; i < N; i++) {
    const real_t kp[2] = {img_pts[i * 2 + 0], img_pts[i * 2 + 1]};
    const real_t x0[2] = {obj_pts[i * 3 + 0], obj_pts[i * 3 + 1]};
    const real_t x1[2] = {(kp[0] - cx) / fx, (kp[1] - cy) / fy};

    const int rs = i * 18;
    const int re = i * 18 + 9;
    A[rs + 0] = 0.0;
    A[rs + 1] = 0.0;
    A[rs + 2] = 0.0;
    A[rs + 3] = -x0[0];
    A[rs + 4] = -x0[1];
    A[rs + 5] = -1.0;
    A[rs + 6] = x1[1] * x0[0];
    A[rs + 7] = x1[1] * x0[1];
    A[rs + 8] = x1[1];

    A[re + 0] = x0[0];
    A[re + 1] = x0[1];
    A[re + 2] = 1.0;
    A[re + 3] = 0.0;
    A[re + 4] = 0.0;
    A[re + 5] = 0.0;
    A[re + 6] = -x1[0] * x0[0];
    A[re + 7] = -x1[0] * x0[1];
    A[re + 8] = -x1[0];
  }

  const int Am = num_rows;
  const int An = num_cols;
  real_t *U = malloc(sizeof(real_t) * Am * Am);
  real_t *s = malloc(sizeof(real_t) * Am);
  real_t *V = malloc(sizeof(real_t) * An * An);
  if (svd(A, Am, An, U, s, V) != 0) {
    free(A);
    free(U);
    free(s);
    free(V);
    return -1;
  }

  // Form the Homography matrix using the last column of V
  real_t H[3 * 3] = {0};
  H[0] = V[8];
  H[1] = V[17];
  H[2] = V[26];

  H[3] = V[35];
  H[4] = V[44];
  H[5] = V[53];

  H[6] = V[62];
  H[7] = V[71];
  H[8] = V[80];

  if (H[8] < 0) {
    for (int i = 0; i < 9; i++) {
      H[i] *= -1.0;
    }
  }

  // Normalize H to ensure that || c1 || = 1
  const real_t H_norm = sqrt(H[0] * H[0] + H[3] * H[3] + H[6] * H[6]);
  for (int i = 0; i < 9; i++) {
    H[i] /= H_norm;
  }

  // Form translation vector
  const real_t r[3] = {H[2], H[5], H[8]};

  // Form Rotation matrix
  const real_t c1[3] = {H[0], H[3], H[6]};
  const real_t c2[3] = {H[1], H[4], H[7]};
  real_t c3[3] = {0};
  vec3_cross(c1, c2, c3);

  real_t C[3 * 3] = {0};
  for (int i = 0; i < 3; i++) {
    C[(i * 3) + 0] = c1[i];
    C[(i * 3) + 1] = c2[i];
    C[(i * 3) + 2] = c3[i];
  }

  // Set T_CF
  T_CF[0] = C[0];
  T_CF[1] = C[1];
  T_CF[2] = C[2];
  T_CF[3] = r[0];

  T_CF[4] = C[3];
  T_CF[5] = C[4];
  T_CF[6] = C[5];
  T_CF[7] = r[1];

  T_CF[8] = C[6];
  T_CF[9] = C[7];
  T_CF[10] = C[8];
  T_CF[11] = r[2];

  T_CF[12] = 0.0;
  T_CF[13] = 0.0;
  T_CF[14] = 0.0;
  T_CF[15] = 1.0;

  // Clean up
  free(A);
  free(U);
  free(s);
  free(V);

  return 0;
}

static real_t *_solvepnp_residuals(const real_t *proj_params,
                                   const real_t *img_pts,
                                   const real_t *obj_pts,
                                   const int N,
                                   real_t *param) {
  POSE2TF(param, T_FC_est);
  TF_INV(T_FC_est, T_CF_est);
  real_t *r = malloc(sizeof(real_t) * 2 * N);

  for (int n = 0; n < N; n++) {
    // Calculate residual
    real_t z[2] = {img_pts[n * 2 + 0], img_pts[n * 2 + 1]};
    real_t p_F[3] = {obj_pts[n * 3 + 0],
                     obj_pts[n * 3 + 1],
                     obj_pts[n * 3 + 2]};
    TF_POINT(T_CF_est, p_F, p_C);
    real_t zhat[2] = {0};
    pinhole_project(proj_params, p_C, zhat);
    real_t res[2] = {z[0] - zhat[0], z[1] - zhat[1]};

    // Form R.H.S.Gauss Newton g
    r[n * 2 + 0] = res[0];
    r[n * 2 + 1] = res[1];
  }

  return r;
}

static real_t _solvepnp_cost(const real_t *proj_params,
                             const real_t *img_pts,
                             const real_t *obj_pts,
                             const int N,
                             real_t *param) {
  real_t *r = _solvepnp_residuals(proj_params, img_pts, obj_pts, N, param);
  real_t cost = 0;
  dot(r, 1, 2 * N, r, 2 * N, 1, &cost);
  free(r);

  return 0.5 * cost;
}

static void _solvepnp_linearize(const real_t *proj_params,
                                const real_t *img_pts,
                                const real_t *obj_pts,
                                const int N,
                                const real_t *param,
                                real_t *H,
                                real_t *g) {
  // Form Gauss-Newton system
  POSE2TF(param, T_FC_est);
  TF_INV(T_FC_est, T_CF_est);
  zeros(H, 6, 6);
  zeros(g, 6, 1);

  for (int i = 0; i < N; i++) {
    // Calculate residual
    const real_t z[2] = {img_pts[i * 2 + 0], img_pts[i * 2 + 1]};
    const real_t p_F[3] = {obj_pts[i * 3 + 0],
                           obj_pts[i * 3 + 1],
                           obj_pts[i * 3 + 2]};
    TF_POINT(T_CF_est, p_F, p_C);
    real_t zhat[2] = {0};
    pinhole_project(proj_params, p_C, zhat);
    const real_t r[2] = {z[0] - zhat[0], z[1] - zhat[1]};

    // Calculate Jacobian
    TF_DECOMPOSE(T_FC_est, C_FC, r_FC);
    TF_DECOMPOSE(T_CF_est, C_CF, r_CF);
    // -- Jacobian w.r.t 3D point p_C
    // clang-format off
    const real_t Jp[2 * 3] = {
      1.0 / p_C[2], 0.0, -p_C[0] / (p_C[2] * p_C[2]),
      0.0, 1.0 / p_C[2], -p_C[1] / (p_C[2] * p_C[2])
    };
    // clang-format on
    // -- Jacobian w.r.t 2D point x
    const real_t Jk[2 * 2] = {proj_params[0], 0, 0, proj_params[1]};
    // -- Pinhole projection Jacobian
    // Jh = -1 * Jk @ Jp
    real_t Jh[2 * 3] = {0};
    dot(Jk, 2, 2, Jp, 2, 3, Jh);
    for (int i = 0; i < 6; i++) {
      Jh[i] *= -1.0;
    }
    // -- Jacobian of reprojection w.r.t. pose T_FC
    real_t nC_CF[3 * 3] = {0};
    real_t nC_FC[3 * 3] = {0};
    mat3_copy(C_CF, nC_CF);
    mat3_copy(C_FC, nC_FC);
    mat_scale(nC_CF, 3, 3, -1.0);
    mat_scale(nC_FC, 3, 3, -1.0);
    // -- J_pos = Jh * -C_CF
    real_t J_pos[2 * 3] = {0};
    dot(Jh, 2, 3, nC_CF, 3, 3, J_pos);
    // -- J_rot = Jh * -C_CF * hat(p_F - r_FC) * -C_FC
    real_t J_rot[2 * 3] = {0};
    real_t A[3 * 3] = {0};
    real_t dp[3] = {0};
    real_t dp_hat[3 * 3] = {0};
    dp[0] = p_F[0] - r_FC[0];
    dp[1] = p_F[1] - r_FC[1];
    dp[2] = p_F[2] - r_FC[2];
    hat(dp, dp_hat);
    dot(dp_hat, 3, 3, nC_FC, 3, 3, A);
    dot(J_pos, 2, 3, A, 3, 3, J_rot);
    // -- J = [J_pos | J_rot]
    real_t J[2 * 6] = {0};
    real_t Jt[6 * 2] = {0};
    J[0] = J_pos[0];
    J[1] = J_pos[1];
    J[2] = J_pos[2];
    J[6] = J_pos[3];
    J[7] = J_pos[4];
    J[8] = J_pos[5];

    J[3] = J_rot[0];
    J[4] = J_rot[1];
    J[5] = J_rot[2];
    J[9] = J_rot[3];
    J[10] = J_rot[4];
    J[11] = J_rot[5];
    mat_transpose(J, 2, 6, Jt);

    // Form Hessian
    // H += J.T * J
    real_t Hi[6 * 6] = {0};
    dot(Jt, 6, 2, J, 2, 6, Hi);
    for (int i = 0; i < 36; i++) {
      H[i] += Hi[i];
    }

    // Form R.H.S. Gauss Newton g
    // g += -J.T @ r
    real_t gi[6] = {0};
    mat_scale(Jt, 6, 2, -1.0);
    dot(Jt, 6, 2, r, 2, 1, gi);
    g[0] += gi[0];
    g[1] += gi[1];
    g[2] += gi[2];
    g[3] += gi[3];
    g[4] += gi[4];
    g[5] += gi[5];
  }
}

static void _solvepnp_solve(real_t lambda_k, real_t *H, real_t *g, real_t *dx) {
  // Damp Hessian: H = H + lambda * I
  for (int i = 0; i < 6; i++) {
    H[(i * 6) + i] += lambda_k;
  }

  // Solve: H * dx = g
  chol_solve(H, g, dx, 6);
}

static void _solvepnp_update(const real_t *param_k,
                             const real_t *dx,
                             real_t *param_kp1) {
  param_kp1[0] = param_k[0];
  param_kp1[1] = param_k[1];
  param_kp1[2] = param_k[2];
  param_kp1[3] = param_k[3];
  param_kp1[4] = param_k[4];
  param_kp1[5] = param_k[5];
  param_kp1[6] = param_k[6];
  pose_update(param_kp1, dx);
}

/**
 * Solve the Perspective-N-Points problem.
 *
 * **IMPORTANT**: This function assumes that object points lie on the plane
 * because the initialization step uses DLT to estimate the homography between
 * camera and planar object, then the relative pose between them is recovered.
 */
int solvepnp(const real_t proj_params[4],
             const real_t *img_pts,
             const real_t *obj_pts,
             const int N,
             real_t T_CO[4 * 4]) {
  assert(proj_params != NULL);
  assert(img_pts != NULL);
  assert(obj_pts != NULL);
  assert(N > 0);
  assert(T_CO != NULL);

  const int verbose = 0;
  const int max_iter = 10;
  const real_t lambda_init = 1e4;
  const real_t lambda_factor = 10.0;
  const real_t dx_threshold = 1e-5;
  const real_t J_threshold = 1e-5;

  // Initialize pose with DLT
  if (homography_pose(proj_params, img_pts, obj_pts, N, T_CO) != 0) {
    return -1;
  }

  TF_INV(T_CO, T_OC);
  TF_VECTOR(T_OC, param_k);
  real_t lambda_k = lambda_init;
  real_t J_k = _solvepnp_cost(proj_params, img_pts, obj_pts, N, param_k);

  real_t H[6 * 6] = {0};
  real_t g[6 * 1] = {0};
  real_t dx[6 * 1] = {0};
  real_t dJ = 0;
  real_t dx_norm = 0;
  real_t J_kp1 = 0;
  real_t param_kp1[7] = {0};

  for (int iter = 0; iter < max_iter; iter++) {
    // Solve
    _solvepnp_linearize(proj_params, img_pts, obj_pts, N, param_k, H, g);
    _solvepnp_solve(lambda_k, H, g, dx);
    _solvepnp_update(param_k, dx, param_kp1);
    J_kp1 = _solvepnp_cost(proj_params, img_pts, obj_pts, N, param_kp1);

    // Accept or reject update
    dJ = J_kp1 - J_k;
    dx_norm = vec_norm(dx, 6);
    if (J_kp1 < J_k) {
      // Accept update
      J_k = J_kp1;
      vec_copy(param_kp1, 7, param_k);
      lambda_k /= lambda_factor;
    } else {
      // Reject update
      lambda_k *= lambda_factor;
    }

    // Display
    if (verbose) {
      printf("iter: %d, ", iter);
      printf("lambda_k: %.2e, ", lambda_k);
      printf("norm(dx): %.2e, ", dx_norm);
      printf("dcost: %.2e, ", dJ);
      printf("cost:  %.2e\n", J_k);
    }

    // Terminate?
    if (J_k < J_threshold) {
      break;
    } else if (dx_threshold > dx_norm) {
      break;
    }
  }

  // // Calculate reprojection errors
  // real_t *r = _solvepnp_residuals(proj_params, img_pts, obj_pts, N,
  // param_kp1); real_t *errors = malloc(sizeof(real_t) * N); for (int i = 0; i
  // < N; i++) {
  //   const real_t x = r[i * 2 + 0];
  //   const real_t y = r[i * 2 + 1];
  //   errors[i] = sqrt(x * x + y * y);
  // }

  // // Calculate RMSE
  // real_t sum = 0.0;
  // real_t sse = 0.0;
  // for (int i = 0; i < N; i++) {
  //   sum += errors[i];
  //   sse += errors[i] * errors[i];
  // }
  // const real_t reproj_rmse = sqrt(sse / N);
  // const real_t reproj_mean = sum / N;
  // const real_t reproj_median = median(errors, N);
  // printf("rmse: %f, mean: %f, median: %f\n", reproj_rmse, reproj_mean,
  // reproj_median);

  // free(r);
  // free(errors);

  return 0;
}

/*******************************************************************************
 * APRILGRID
 ******************************************************************************/

// APRILGRID /////////////////////////////////////////////////////////////////

/**
 * Malloc AprilGrid.
 *
 * @param ts Timestamp
 * @param grid AprilGrid
 */
aprilgrid_t *aprilgrid_malloc(const int num_rows,
                              const int num_cols,
                              const real_t tag_size,
                              const real_t tag_spacing) {
  aprilgrid_t *grid = malloc(sizeof(aprilgrid_t));
  grid->num_rows = num_rows;
  grid->num_cols = num_cols;
  grid->tag_size = tag_size;
  grid->tag_spacing = tag_spacing;

  // Grid data
  grid->timestamp = 0;
  const int max_corners = (num_rows * num_cols * 4);
  grid->corners_detected = 0;
  grid->data = calloc(max_corners * 6, sizeof(real_t));

  return grid;
}

void aprilgrid_free(aprilgrid_t *grid) {
  if (grid) {
    free(grid->data);
    free(grid);
  }
}

/**
 * Clear Aprilgrid
 */
void aprilgrid_clear(aprilgrid_t *grid) {
  assert(grid != NULL);

  grid->timestamp = 0;
  grid->corners_detected = 0;
  const int max_corners = (grid->num_rows * grid->num_cols * 4);
  for (int i = 0; i < (max_corners * 6); i++) {
    grid->data[i] = 0;
  }
}

/**
 * Reset Aprilgrid
 */
void aprilgrid_reset(aprilgrid_t *grid) {
  assert(grid != NULL);

  grid->timestamp = 0;
  grid->corners_detected = 0;
  const int max_corners = (grid->num_rows * grid->num_cols * 4);
  for (int i = 0; i < (max_corners * 6); i++) {
    grid->data[i] = 0;
  }

  grid->num_rows = 0;
  grid->num_cols = 0;
  grid->tag_size = 0;
  grid->tag_spacing = 0;
}

/**
 * Copy AprilGrid
 */
void aprilgrid_copy(const aprilgrid_t *src, aprilgrid_t *dst) {
  dst->timestamp = src->timestamp;
  dst->num_rows = src->num_rows;
  dst->num_cols = src->num_cols;
  dst->tag_size = src->tag_size;
  dst->tag_spacing = src->tag_spacing;

  dst->corners_detected = src->corners_detected;
  for (size_t i = 0; i < (dst->num_rows * dst->num_cols * 4); i++) {
    for (size_t j = 0; j < 6; j++) {
      dst->data[i * 6 + j] = src->data[i * 6 + j];
    }
  }
}

/**
 * Check AprilGrids are equal
 */
int aprilgrid_equals(const aprilgrid_t *grid0, const aprilgrid_t *grid1) {
  APRILGRID_CHECK(grid0->timestamp == grid1->timestamp);
  APRILGRID_CHECK(grid0->num_rows == grid1->num_rows);
  APRILGRID_CHECK(grid0->num_cols == grid1->num_cols);
  APRILGRID_CHECK(fabs(grid0->tag_size - grid1->tag_size) < 1e-8);
  APRILGRID_CHECK(fabs(grid0->tag_spacing - grid1->tag_spacing) < 1e-8);
  APRILGRID_CHECK(grid0->corners_detected == grid1->corners_detected);

  for (size_t i = 0; i < (grid0->num_rows * grid0->num_cols * 4); i++) {
    for (size_t j = 0; j < 6; j++) {
      APRILGRID_CHECK(fabs(grid0->data[i * 6 + j] - grid1->data[i * 6 + j]) <
                      1e-8);
    }
  }

  return 1;
error:
  return 0;
}

/**
 * Print Aprilgrid
 */
void aprilgrid_print(const aprilgrid_t *grid) {
  assert(grid != NULL);

  printf("timestamp: %ld\n", grid->timestamp);
  printf("num_rows: %d\n", grid->num_rows);
  printf("num_cols: %d\n", grid->num_cols);
  printf("tag_size: %f\n", grid->tag_size);
  printf("tag_spacing: %f\n", grid->tag_spacing);
  printf("\n");
  printf("corners_detected: %d\n", grid->corners_detected);
  printf("#tag_id, corner_idx, kp_x, kp_y, p_x, p_y, p_z\n");
  const int max_corners = (grid->num_rows * grid->num_cols * 4);
  for (int i = 0; i < max_corners; i++) {
    if (grid->data[i * 6 + 0] <= 0) {
      continue;
    }

    const int tag_id = i / 4;
    const int corner_idx = i % 4;
    printf("%d, ", tag_id);
    printf("%d, ", corner_idx);
    printf("%.2f, ", grid->data[i * 6 + 1]);
    printf("%.2f, ", grid->data[i * 6 + 2]);
    printf("%.2f, ", grid->data[i * 6 + 3]);
    printf("%.2f, ", grid->data[i * 6 + 4]);
    printf("%.2f", grid->data[i * 6 + 5]);
    printf("\n");
  }
}

/**
 * Return center of AprilGrid
 */
void aprilgrid_center(const aprilgrid_t *grid, real_t *cx, real_t *cy) {
  assert(grid != NULL);
  assert(cx != NULL);
  assert(cy != NULL);

  *cx = ((grid->num_cols / 2.0) * grid->tag_size);
  *cx += (((grid->num_cols / 2.0) - 1) * grid->tag_spacing * grid->tag_size);
  *cx += (0.5 * grid->tag_spacing * grid->tag_size);

  *cy = ((grid->num_rows / 2.0) * grid->tag_size);
  *cy += (((grid->num_rows / 2.0) - 1) * grid->tag_spacing * grid->tag_size);
  *cy += (0.5 * grid->tag_spacing * grid->tag_size);
}

/**
 * Return AprilTag grid index within the AprilGrid based on tag id
 */
void aprilgrid_grid_index(const aprilgrid_t *grid,
                          const int tag_id,
                          int *i,
                          int *j) {
  assert(grid != NULL);
  assert(tag_id >= 0 && tag_id <= (grid->num_rows * grid->num_cols - 1));
  assert(i != NULL);
  assert(j != NULL);

  if (tag_id > (grid->num_rows * grid->num_cols)) {
    APRILGRID_FATAL("tag_id > (num_rows * num_cols)!\n");
  } else if (tag_id < 0) {
    APRILGRID_FATAL("tag_id < 0!\n");
  }

  *i = (int) (tag_id / grid->num_cols);
  *j = (int) (tag_id % grid->num_cols);
}

/**
 * Return AprilGrid object point from tag id and corner index
 */
void aprilgrid_object_point(const aprilgrid_t *grid,
                            const int tag_id,
                            const int corner_idx,
                            real_t object_point[3]) {
  assert(grid != NULL);
  assert(tag_id >= 0 && tag_id <= (grid->num_rows * grid->num_cols - 1));
  assert(corner_idx >= 0 && corner_idx <= 3);
  assert(object_point != NULL);

  // Calculate the AprilGrid index using tag id
  int i = 0;
  int j = 0;
  aprilgrid_grid_index(grid, tag_id, &i, &j);

  // Caculate the x and y of the tag origin (bottom left corner of tag)
  // relative to grid origin (bottom left corner of entire grid)
  const real_t x = j * (grid->tag_size + grid->tag_size * grid->tag_spacing);
  const real_t y = i * (grid->tag_size + grid->tag_size * grid->tag_spacing);

  // Calculate the x and y of each corner
  switch (corner_idx) {
    case 0: // Bottom left
      object_point[0] = x;
      object_point[1] = y;
      object_point[2] = 0;
      break;
    case 1: // Bottom right
      object_point[0] = x + grid->tag_size;
      object_point[1] = y;
      object_point[2] = 0;
      break;
    case 2: // Top right
      object_point[0] = x + grid->tag_size;
      object_point[1] = y + grid->tag_size;
      object_point[2] = 0;
      break;
    case 3: // Top left
      object_point[0] = x;
      object_point[1] = y + grid->tag_size;
      object_point[2] = 0;
      break;
    default:
      APRILGRID_FATAL("Incorrect corner id [%d]!\n", corner_idx);
      break;
  }
}

/**
 * Add AprilGrid corner measurement
 */
void aprilgrid_add_corner(aprilgrid_t *grid,
                          const int tag_id,
                          const int corner_idx,
                          const real_t kp[2]) {
  assert(grid != NULL);
  assert(tag_id >= 0 && tag_id <= (grid->num_rows * grid->num_cols - 1));
  assert(corner_idx >= 0 && corner_idx <= 3);

  // Set AprilGrid as detected
  grid->corners_detected++;

  // Push tag_id and keypoints
  const int data_row = (tag_id * 4) + corner_idx;
  real_t p[3] = {0};
  aprilgrid_object_point(grid, tag_id, corner_idx, p);

  grid->data[data_row * 6 + 0] = 1;
  grid->data[data_row * 6 + 1] = kp[0];
  grid->data[data_row * 6 + 2] = kp[1];
  grid->data[data_row * 6 + 3] = p[0];
  grid->data[data_row * 6 + 4] = p[1];
  grid->data[data_row * 6 + 5] = p[2];
}

/**
 * Remove AprilGrid corner measurement
 */
void aprilgrid_remove_corner(aprilgrid_t *grid,
                             const int tag_id,
                             const int corner_idx) {
  assert(grid != NULL);
  assert(tag_id >= 0 && tag_id <= (grid->num_rows * grid->num_cols - 1));
  assert(corner_idx >= 0 && corner_idx <= 3);

  const int data_row = (tag_id * 4) + corner_idx;
  assert(data_row >= 0);
  grid->data[data_row * 6 + 0] = 0;
  grid->data[data_row * 6 + 1] = 0;
  grid->data[data_row * 6 + 2] = 0;
  grid->data[data_row * 6 + 3] = 0;
  grid->data[data_row * 6 + 4] = 0;
  grid->data[data_row * 6 + 5] = 0;
  grid->corners_detected--;
}

/**
 * Add AprilGrid AprilTag measurement
 */
void aprilgrid_add_tag(aprilgrid_t *grid,
                       const int tag_id,
                       const real_t tag_kps[4][2]) {
  assert(grid != NULL);
  assert(tag_id >= 0 && tag_id <= (grid->num_rows * grid->num_cols - 1));
  assert(tag_kps != NULL);

  for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
    if (tag_kps[corner_idx] == NULL) {
      continue;
    }
    aprilgrid_add_corner(grid, tag_id, corner_idx, tag_kps[corner_idx]);
  }
}

/**
 * Remove AprilGrid AprilTag measurement
 */
void aprilgrid_remove_tag(aprilgrid_t *grid, const int tag_id) {
  assert(grid != NULL);
  assert(tag_id >= 0 && tag_id <= (grid->num_rows * grid->num_cols - 1));

  for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
    aprilgrid_remove_corner(grid, tag_id, corner_idx);
  }
}

/**
 * Return AprilGrid measurements
 */
void aprilgrid_measurements(const aprilgrid_t *grid,
                            int *tag_ids,
                            int *corner_idxs,
                            real_t *tag_kps,
                            real_t *obj_pts) {
  assert(grid != NULL);
  assert(tag_ids != NULL);
  assert(corner_idxs != NULL);
  assert(tag_kps != NULL);
  assert(obj_pts != NULL);

  // Pre-check
  if (grid->corners_detected == 0) {
    return;
  }

  // Get measurements
  int meas_idx = 0;
  for (long i = 0; i < (grid->num_rows * grid->num_cols * 4); i++) {
    if (grid->data[i * 6 + 0] < 1.0) {
      continue;
    }

    const int tag_id = i / 4;
    const int corner_idx = i % 4;
    const real_t kp_x = grid->data[i * 6 + 1];
    const real_t kp_y = grid->data[i * 6 + 2];
    const real_t p_x = grid->data[i * 6 + 3];
    const real_t p_y = grid->data[i * 6 + 4];
    const real_t p_z = grid->data[i * 6 + 5];

    tag_ids[meas_idx] = tag_id;
    corner_idxs[meas_idx] = corner_idx;
    tag_kps[meas_idx * 2] = kp_x;
    tag_kps[meas_idx * 2 + 1] = kp_y;
    obj_pts[meas_idx * 3] = p_x;
    obj_pts[meas_idx * 3 + 1] = p_y;
    obj_pts[meas_idx * 3 + 2] = p_z;
    meas_idx++;
  }
}

/**
 * Save AprilGrid
 */
int aprilgrid_save(const aprilgrid_t *grid, const char *save_path) {
  assert(grid != NULL);
  assert(save_path != NULL);

  // Open file for saving
  FILE *fp = fopen(save_path, "w");
  if (fp == NULL) {
    APRILGRID_LOG("Failed to open [%s] for saving!", save_path);
    return -1;
  }

  // Output header
  // -- Configuration
  fprintf(fp, "timestamp:%ld\n", grid->timestamp);
  fprintf(fp, "num_rows:%d\n", grid->num_rows);
  fprintf(fp, "num_cols:%d\n", grid->num_cols);
  fprintf(fp, "tag_size:%f\n", grid->tag_size);
  fprintf(fp, "tag_spacing:%f\n", grid->tag_spacing);
  fprintf(fp, "\n");
  // -- Data
  fprintf(fp, "corners_detected:%d\n", grid->corners_detected);
  fprintf(fp, "tag_id,corner_idx,kp_x,kp_y,p_x,p_y,p_z\n");

  // Output data
  if (grid->corners_detected) {
    // vec2s_t kps = keypoints();
    for (long i = 0; i < (grid->num_rows * grid->num_cols * 4); i++) {
      const int tag_id = i / 4;
      const int corner_idx = i % 4;

      if (grid->data[i * 6 + 0] > 0) { // Corner detected?
        fprintf(fp, "%d,", tag_id);
        fprintf(fp, "%d,", corner_idx);
        fprintf(fp, "%f,", grid->data[i * 6 + 1]);
        fprintf(fp, "%f,", grid->data[i * 6 + 2]);
        fprintf(fp, "%f,", grid->data[i * 6 + 3]);
        fprintf(fp, "%f,", grid->data[i * 6 + 4]);
        fprintf(fp, "%f", grid->data[i * 6 + 5]);
        fprintf(fp, "\n");
      }
    }
  }

  // Close up
  fclose(fp);

  return 0;
}

static void aprilgrid_parse_line(FILE *fp,
                                 const char *key,
                                 const char *value_type,
                                 void *value) {
  assert(fp != NULL);
  assert(key != NULL);
  assert(value_type != NULL);
  assert(value != NULL);

  // Parse line
  const size_t buf_len = 1024;
  char buf[1024] = {0};
  if (fgets(buf, buf_len, fp) == NULL) {
    APRILGRID_FATAL("Failed to parse [%s]\n", key);
  }

  // Split key-value
  char delim[2] = ":";
  char *key_str = strtok(buf, delim);
  char *value_str = strtok(NULL, delim);

  // Check key matches
  if (strcmp(key_str, key) != 0) {
    APRILGRID_FATAL("Failed to parse [%s]\n", key);
  }

  // Typecase value
  if (value_type == NULL) {
    APRILGRID_FATAL("Value type not set!\n");
  }

  if (strcmp(value_type, "uint64_t") == 0) {
    *(uint64_t *) value = atol(value_str);
  } else if (strcmp(value_type, "int") == 0) {
    *(int *) value = atoi(value_str);
  } else if (strcmp(value_type, "real_t") == 0) {
    *(real_t *) value = atof(value_str);
  } else {
    APRILGRID_FATAL("Invalid value type [%s]\n", value_type);
  }
}

static void aprilgrid_parse_skip_line(FILE *fp) {
  assert(fp != NULL);
  const size_t buf_len = 1024;
  char buf[1024] = {0};
  char *retval = fgets(buf, buf_len, fp);
  APRILGRID_UNUSED(retval);
}

/**
 * Load AprilGrid
 */
aprilgrid_t *aprilgrid_load(const char *data_path) {
  assert(data_path != NULL);

  // Open file for loading
  FILE *fp = fopen(data_path, "r");
  if (fp == NULL) {
    APRILGRID_LOG("Failed to open [%s]!\n", data_path);
    return NULL;
  }

  // Parse configuration
  timestamp_t timestamp;
  int num_rows = 0;
  int num_cols = 0;
  real_t tag_size = 0;
  real_t tag_spacing = 0;
  aprilgrid_parse_line(fp, "timestamp", "uint64_t", &timestamp);
  aprilgrid_parse_line(fp, "num_rows", "int", &num_rows);
  aprilgrid_parse_line(fp, "num_cols", "int", &num_cols);
  aprilgrid_parse_line(fp, "tag_size", "real_t", &tag_size);
  aprilgrid_parse_line(fp, "tag_spacing", "real_t", &tag_spacing);
  aprilgrid_parse_skip_line(fp);
  aprilgrid_t *grid =
      aprilgrid_malloc(num_rows, num_cols, tag_size, tag_spacing);

  // Parse data
  int corners_detected = 0;
  grid->timestamp = timestamp;
  aprilgrid_parse_line(fp, "corners_detected", "int", &corners_detected);
  aprilgrid_parse_skip_line(fp);

#if PRECISION == 1
  const char *fmt = "%d,%d,%f,%f,%f,%f,%f";
#elif PRECISION == 2
  const char *fmt = "%d,%d,%lf,%lf,%lf,%lf,%lf";
#else
#error "Invalid precision!"
#endif
  for (int i = 0; i < corners_detected; i++) {
    // Parse data line
    int tag_id = 0;
    int corner_idx = 0;
    real_t kp[2] = {0};
    real_t p[3] = {0};
    const int retval = fscanf(fp,
                              fmt,
                              &tag_id,
                              &corner_idx,
                              &kp[0],
                              &kp[1],
                              &p[0],
                              &p[1],
                              &p[2]);
    if (retval != 7) {
      APRILGRID_FATAL("Failed to parse data line in [%s]\n", data_path);
    }

    // Add corner
    aprilgrid_add_corner(grid, tag_id, corner_idx, kp);
  }

  // Clean up
  fclose(fp);

  return grid;
}

// APRILGRID DETECTOR ////////////////////////////////////////////////////////

aprilgrid_detector_t *aprilgrid_detector_malloc(int num_rows,
                                                int num_cols,
                                                real_t tag_size,
                                                real_t tag_spacing) {
  aprilgrid_detector_t *det = malloc(sizeof(aprilgrid_detector_t));
  // det->tf = tagStandard41h12_create();
  det->tf = tag36h11_create();
  det->td = apriltag_detector_create();
  apriltag_detector_add_family_bits(det->td, det->tf, 1);

  det->num_rows = num_rows;
  det->num_cols = num_cols;
  det->tag_size = tag_size;
  det->tag_spacing = tag_spacing;

  return det;
}

void aprilgrid_detector_free(aprilgrid_detector_t *det) {
  assert(det != NULL);
  apriltag_detector_destroy(det->td);
  // tagStandard41h12_destroy(det->tf);
  tag36h11_destroy(det->tf);
  free(det);
  det = NULL;
}

aprilgrid_t *aprilgrid_detector_detect(const aprilgrid_detector_t *det,
                                       const timestamp_t ts,
                                       const int32_t image_width,
                                       const int32_t image_height,
                                       const int32_t image_stride,
                                       uint8_t *image_data) {
  assert(det != NULL);
  assert(image_width > 0);
  assert(image_height > 0);
  assert(image_stride > 0);
  assert(image_data != NULL);

  // Form image_u8_t
  image_u8_t im = {.width = image_width,
                   .height = image_height,
                   .stride = image_stride,
                   .buf = image_data};

  // Detect AprilTags
  aprilgrid_t *grid = aprilgrid_malloc(det->num_rows,
                                       det->num_cols,
                                       det->tag_size,
                                       det->tag_spacing);
  grid->timestamp = ts;
  zarray_t *dets = apriltag_detector_detect(det->td, &im);
  // int num_corners = 0;
  for (int i = 0; i < zarray_size(dets); i++) {
    apriltag_detection_t *det;
    zarray_get(dets, i, &det);

    const real_t p0[2] = {det->p[0][0], det->p[0][1]};
    const real_t p1[2] = {det->p[1][0], det->p[1][1]};
    const real_t p2[2] = {det->p[2][0], det->p[2][1]};
    const real_t p3[2] = {det->p[3][0], det->p[3][1]};

    aprilgrid_add_corner(grid, det->id, 0, p0);
    aprilgrid_add_corner(grid, det->id, 1, p1);
    aprilgrid_add_corner(grid, det->id, 2, p2);
    aprilgrid_add_corner(grid, det->id, 3, p3);
    // num_corners += 4;
  }
  apriltag_detections_destroy(dets);

  // // Return NULL if no apriltag detected
  // if (num_corners == 0) {
  //   aprilgrid_free(grid);
  //   return NULL;
  // }

  return grid;
}

/*******************************************************************************
 * MORTON CODES
 ******************************************************************************/

// "Insert" a 0 bit after each of the 16 low bits of x
uint32_t part1by1(uint32_t x) {
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x &= 0x0000ffff;
  x = (x ^ (x << 8)) & 0x00ff00ff;
  x = (x ^ (x << 4)) & 0x0f0f0f0f;
  x = (x ^ (x << 2)) & 0x33333333;
  x = (x ^ (x << 1)) & 0x55555555;
  return x;
}

// "Insert" two 0 bits after each of the 10 low bits of x
uint32_t part1by2(uint32_t x) {
  // x = ---- ---- ---- ---- ---- --98 7654 3210
  // x = ---- --98 ---- ---- ---- ---- 7654 3210
  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  x &= 0x000003ff;
  x = (x ^ (x << 16)) & 0xff0000ff;
  x = (x ^ (x << 8)) & 0x0300f00f;
  x = (x ^ (x << 4)) & 0x030c30c3;
  x = (x ^ (x << 2)) & 0x09249249;
  return x;
}

// Inverse of part1by1 - "delete" all odd-indexed bits
uint32_t compact1by1(uint32_t x) {
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x &= 0x55555555;
  x = (x ^ (x >> 1)) & 0x33333333;
  x = (x ^ (x >> 2)) & 0x0f0f0f0f;
  x = (x ^ (x >> 4)) & 0x00ff00ff;
  x = (x ^ (x >> 8)) & 0x0000ffff;
  return x;
}

// Inverse of part1by2 - "delete" all bits not at positions divisible by 3
uint32_t compact1by2(uint32_t x) {
  // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
  // x = ---- --98 ---- 76-- --54 ---- 32-- --10
  // x = ---- --98 ---- ---- 7654 ---- ---- 3210
  // x = ---- --98 ---- ---- ---- ---- 7654 3210
  // x = ---- ---- ---- ---- ---- --98 7654 3210
  x &= 0x09249249;
  x = (x ^ (x >> 2)) & 0x030c30c3;
  x = (x ^ (x >> 4)) & 0x0300f00f;
  x = (x ^ (x >> 8)) & 0xff0000ff;
  x = (x ^ (x >> 16)) & 0x000003ff;
  return x;
}

uint32_t morton_encode_2d(uint32_t x, uint32_t y) {
  return (part1by1(y) << 1) + part1by1(x);
}

uint32_t morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
  return (part1by2(z) << 2) + (part1by2(y) << 1) + part1by2(x);
}

void morton_decode_2d(uint32_t code, uint32_t *x, uint32_t *y) {
  *x = compact1by1(code >> 0);
  *y = compact1by1(code >> 1);
}

void morton_decode_3d(uint32_t code, uint32_t *x, uint32_t *y, uint32_t *z) {
  *x = compact1by2(code >> 0);
  *y = compact1by2(code >> 1);
  *z = compact1by2(code >> 2);
}

/*******************************************************************************
 * PLANE
 ******************************************************************************/

void plane_setup(plane_t *plane,
                 const real_t normal[3],
                 const real_t p[3],
                 const real_t d) {
  plane->normal[0] = normal[0];
  plane->normal[1] = normal[1];
  plane->normal[2] = normal[2];

  plane->p[0] = p[0];
  plane->p[1] = p[1];
  plane->p[2] = p[2];

  plane->d = d;
}

void plane_vector(const plane_t *plane, real_t v[4]) {
  v[0] = plane->normal[0];
  v[1] = plane->normal[1];
  v[2] = plane->normal[2];
  v[3] = plane->d;
}

void plane_set_transform(plane_t *plane, const real_t T[4 * 4]) {
  real_t T_inv[4 * 4] = {0};
  real_t T_inv_T[4 * 4] = {0};
  real_t v[4] = {0};
  real_t v_[4] = {0};

  // [x, y, z, d] = inv(T)' * v
  tf_inv(T, T_inv);
  mat_transpose(T_inv, 4, 4, T_inv_T);
  plane_vector(plane, v);
  dot(T_inv_T, 4, 4, v, 4, 1, v_);

  // Normalize plane and distance
  const float norm = sqrt(v_[0] * v_[0] + v_[1] * v_[1] + v_[2] * v_[2]);
  plane->normal[0] = v_[0] / norm;
  plane->normal[1] = v_[1] / norm;
  plane->normal[2] = v_[2] / norm;
  plane->d = plane->d / norm;
}

void plane_get_transform(const plane_t *plane,
                         const real_t world_up[3],
                         real_t T[4 * 4]) {
  // Setup
  const real_t *p = plane->p;
  real_t xax[3] = {0};
  real_t yax[3] = {0};
  real_t zax[3] = {0};

  // z-axis
  vec3_copy(plane->normal, zax);
  vec3_normalize(zax);

  // x-axis
  vec3_cross(zax, world_up, xax);
  vec3_normalize(xax);

  // y-axis
  vec3_cross(zax, xax, yax);

  // Plane transform
  // clang-format off
  T[0]  = xax[0]; T[1]  = yax[0]; T[2]  = zax[0]; T[3]  = p[0];
  T[4]  = xax[1]; T[5]  = yax[1]; T[6]  = zax[1]; T[7]  = p[1];
  T[8]  = xax[2]; T[9]  = yax[2]; T[10] = zax[2]; T[11] = p[2];
  T[12] = 0.0;    T[13] = 0.0;    T[14] = 0.0;    T[15] = 1.0;
  // clang-format on
}

inline __attribute__((always_inline)) real_t
plane_point_dist(const plane_t *plane, const real_t p[3]) {
  // dist = a * x + b * y + c * z - d
  return plane->normal[0] * p[0] + plane->normal[1] * p[1] +
         plane->normal[2] * p[2] - plane->d;
}

/*******************************************************************************
 * FRUSTUM
 ******************************************************************************/

void frustum_setup(frustum_t *frustum,
                   const real_t hfov,
                   const real_t aspect,
                   const real_t znear,
                   const real_t zfar) {
  frustum->hfov = hfov;
  frustum->aspect = aspect;
  frustum->znear = znear;
  frustum->zfar = zfar;

  const real_t wnear = 2.0 * tan(hfov / 2.0) * znear;
  const real_t hnear = wnear * (1.0 / aspect);
  const real_t wfar = 2.0 * tan(hfov / 2.0) * zfar;
  const real_t hfar = wfar * (1.0 / aspect);

  // OpenGL
  const real_t front[3] = {0.0, 0.0, -1.0};
  const real_t right[3] = {1.0, 0.0, 0.0};
  const real_t left[3] = {-right[0], -right[1], -right[2]};
  const real_t up[3] = {0.0, 1.0, 0.0};
  const real_t cam_pos[3] = {0.0, 0.0, 0.0};

  // Near and far center
  real_t nc[3] = {0};
  real_t fc[3] = {0};

  nc[0] = cam_pos[0] + front[0] * znear;
  nc[1] = cam_pos[1] + front[1] * znear;
  nc[2] = cam_pos[2] + front[2] * znear;

  fc[0] = cam_pos[0] + front[0] * zfar;
  fc[1] = cam_pos[1] + front[1] * zfar;
  fc[2] = cam_pos[2] + front[2] * zfar;

  // Near frustum corners
  real_t ntl[3] = {0};
  real_t ntr[3] = {0};
  real_t nbl[3] = {0};
  real_t nbr[3] = {0};

  ntl[0] = nc[0] + (up[0] * hnear / 2.0) - (right[0] * wnear / 2.0);
  ntl[1] = nc[1] + (up[1] * hnear / 2.0) - (right[1] * wnear / 2.0);
  ntl[2] = nc[2] + (up[2] * hnear / 2.0) - (right[2] * wnear / 2.0);

  ntr[0] = nc[0] + (up[0] * hnear / 2.0) + (right[0] * wnear / 2.0);
  ntr[1] = nc[1] + (up[1] * hnear / 2.0) + (right[1] * wnear / 2.0);
  ntr[2] = nc[2] + (up[2] * hnear / 2.0) + (right[2] * wnear / 2.0);

  nbl[0] = nc[0] - (up[0] * hnear / 2.0) - (right[0] * wnear / 2.0);
  nbl[1] = nc[1] - (up[1] * hnear / 2.0) - (right[1] * wnear / 2.0);
  nbl[2] = nc[2] - (up[2] * hnear / 2.0) - (right[2] * wnear / 2.0);

  nbr[0] = nc[0] - (up[0] * hnear / 2.0) + (right[0] * wnear / 2.0);
  nbr[1] = nc[1] - (up[1] * hnear / 2.0) + (right[1] * wnear / 2.0);
  nbr[2] = nc[2] - (up[2] * hnear / 2.0) + (right[2] * wnear / 2.0);

  // Far frustum corners
  real_t ftl[3] = {0};
  real_t ftr[3] = {0};
  real_t fbl[3] = {0};
  real_t fbr[3] = {0};

  ftl[0] = fc[0] + (up[0] * hfar / 2.0) - (right[0] * wfar / 2.0);
  ftl[1] = fc[1] + (up[1] * hfar / 2.0) - (right[1] * wfar / 2.0);
  ftl[2] = fc[2] + (up[2] * hfar / 2.0) - (right[2] * wfar / 2.0);

  ftr[0] = fc[0] + (up[0] * hfar / 2.0) + (right[0] * wfar / 2.0);
  ftr[1] = fc[1] + (up[1] * hfar / 2.0) + (right[1] * wfar / 2.0);
  ftr[2] = fc[2] + (up[2] * hfar / 2.0) + (right[2] * wfar / 2.0);

  fbl[0] = fc[0] - (up[0] * hfar / 2.0) - (right[0] * wfar / 2.0);
  fbl[1] = fc[1] - (up[1] * hfar / 2.0) - (right[1] * wfar / 2.0);
  fbl[2] = fc[2] - (up[2] * hfar / 2.0) - (right[2] * wfar / 2.0);

  fbr[0] = fc[0] - (up[0] * hfar / 2.0) + (right[0] * wfar / 2.0);
  fbr[1] = fc[1] - (up[1] * hfar / 2.0) + (right[1] * wfar / 2.0);
  fbr[2] = fc[2] - (up[2] * hfar / 2.0) + (right[2] * wfar / 2.0);

  // Points on the near plane
  real_t pl[3] = {0};
  real_t pr[3] = {0};
  real_t pt[3] = {0};
  real_t pb[3] = {0};

  pl[0] = (nc[0] - right[0] * wnear / 2.0) - cam_pos[0];
  pl[1] = (nc[1] - right[1] * wnear / 2.0) - cam_pos[1];
  pl[2] = (nc[2] - right[2] * wnear / 2.0) - cam_pos[2];

  pr[0] = (nc[0] + right[0] * wnear / 2.0) - cam_pos[0];
  pr[1] = (nc[1] + right[1] * wnear / 2.0) - cam_pos[1];
  pr[2] = (nc[2] + right[2] * wnear / 2.0) - cam_pos[2];

  pt[0] = (nc[0] + right[0] * hnear / 2.0) - cam_pos[0];
  pt[1] = (nc[1] + right[1] * hnear / 2.0) - cam_pos[1];
  pt[2] = (nc[2] + right[2] * hnear / 2.0) - cam_pos[2];

  pb[0] = (nc[0] - right[0] * hnear / 2.0) - cam_pos[0];
  pb[1] = (nc[1] - right[1] * hnear / 2.0) - cam_pos[1];
  pb[2] = (nc[2] - right[2] * hnear / 2.0) - cam_pos[2];

  // Normals on the left, right, top and bottom planes
  real_t normal_near[3] = {0};
  real_t normal_far[3] = {0};
  real_t normal_left[3] = {0};
  real_t normal_right[3] = {0};
  real_t normal_top[3] = {0};
  real_t normal_bottom[3] = {0};

  vec3_normalize(pl);
  vec3_normalize(pr);
  vec3_normalize(pt);
  vec3_normalize(pb);

  vec3_copy(front, normal_near);
  vec3_copy(front, normal_far);
  vec3_scale(normal_far, -1, normal_far);

  vec3_cross(pl, up, normal_left);
  vec3_cross(up, pr, normal_right);
  vec3_cross(left, pt, normal_top);
  vec3_cross(right, pb, normal_bottom);

  // Distance
  const real_t dnear = vec3_dot(nc, normal_near);
  const real_t dfar = vec3_dot(fc, normal_far);
  const real_t dleft = vec3_dot(pl, normal_left);
  const real_t dright = vec3_dot(pr, normal_right);
  const real_t dtop = vec3_dot(pt, normal_top);
  const real_t dbottom = vec3_dot(pb, normal_bottom);

  // Form planes
  plane_setup(&frustum->near, normal_near, nc, dnear);
  plane_setup(&frustum->far, normal_far, fc, dfar);
  plane_setup(&frustum->left, normal_left, pl, dleft);
  plane_setup(&frustum->right, normal_right, pr, dright);
  plane_setup(&frustum->top, normal_top, pt, dtop);
  plane_setup(&frustum->bottom, normal_bottom, pb, dbottom);
}

bool frustum_check_point(const frustum_t *frustum, const real_t p[3]) {
  int status = 0;
  status += plane_point_dist(&frustum->near, p) >= 0;
  status += plane_point_dist(&frustum->far, p) >= 0;
  status += plane_point_dist(&frustum->left, p) >= 0;
  status += plane_point_dist(&frustum->right, p) >= 0;
  status += plane_point_dist(&frustum->top, p) >= 0;
  status += plane_point_dist(&frustum->bottom, p) >= 0;
  return (status == 6) ? true : false;
}

/*******************************************************************************
 * POINT CLOUD
 ******************************************************************************/

/**
 * Estimates scale `c`, rotation matrix `R` and translation vector `t` between
 * two sets of points `X` and `Y` such that:
 *
 *   Y ~= scale * R * X + t
 *
 * Source:
 *
 *   Least-Squares Estimation of Transformation Parameters Between Two Point
 *   Patterns (Shinji Umeyama, 1991)
 *
 * Args:
 *
 *   X: src 3D points
 *   Y: dest 3D points
 *   n: Number of 3D points
 *
 * Returns:
 *
 *   scale: Scale factor
 *   R: Rotation matrix
 *   t: translation vector
 *
 */
void umeyama(const float *X,
             const float *Y,
             const size_t n,
             real_t scale[1],
             real_t R[3 * 3],
             real_t t[3]) {
  // Compute centroid
  real_t mu_x[3] = {0};
  real_t mu_y[3] = {0};
  for (size_t i = 0; i < n; ++i) {
    mu_x[0] += X[i * 3 + 0];
    mu_x[1] += X[i * 3 + 1];
    mu_x[2] += X[i * 3 + 2];

    mu_y[0] += Y[i * 3 + 0];
    mu_y[1] += Y[i * 3 + 1];
    mu_y[2] += Y[i * 3 + 2];
  }
  mu_x[0] /= n;
  mu_x[1] /= n;
  mu_x[2] /= n;
  mu_y[0] /= n;
  mu_y[1] /= n;
  mu_y[2] /= n;

  // Calculate variance of points X relative to its centroid
  // var_x = square(X - mu_x).sum(axis=0).mean()
  real_t var_x = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const real_t dx0 = X[i * 3 + 0] - mu_x[0];
    const real_t dx1 = X[i * 3 + 1] - mu_x[1];
    const real_t dx2 = X[i * 3 + 2] - mu_x[2];
    var_x += dx0 * dx0 + dx1 * dx1 + dx2 * dx2;
  }
  var_x /= n;

  // Form covariance matrix and decompose with SVD
  // covar = ((Y - mu_y) * (X - mu_x).T) / X.shape[1]
  // -- Accumulate covariance sums
  real_t *covar = calloc(3 * 3, sizeof(real_t));
  for (int k = 0; k < n; k++) {
    const real_t dx0 = X[k * 3 + 0] - mu_x[0];
    const real_t dx1 = X[k * 3 + 1] - mu_x[1];
    const real_t dx2 = X[k * 3 + 2] - mu_x[2];

    const real_t dy0 = Y[k * 3 + 0] - mu_y[0];
    const real_t dy1 = Y[k * 3 + 1] - mu_y[1];
    const real_t dy2 = Y[k * 3 + 2] - mu_y[2];

    covar[0] += dy0 * dx0;
    covar[1] += dy0 * dx1;
    covar[2] += dy0 * dx2;

    covar[3] += dy1 * dx0;
    covar[4] += dy1 * dx1;
    covar[5] += dy1 * dx2;

    covar[6] += dy2 * dx0;
    covar[7] += dy2 * dx1;
    covar[8] += dy2 * dx2;
  }
  // -- Normalize by n
  const real_t covar_scale = 1.0 / n;
  covar[0] *= covar_scale;
  covar[1] *= covar_scale;
  covar[2] *= covar_scale;
  covar[3] *= covar_scale;
  covar[4] *= covar_scale;
  covar[5] *= covar_scale;
  covar[6] *= covar_scale;
  covar[7] *= covar_scale;
  covar[8] *= covar_scale;
  // -- U, s, V = svd(covar)
  real_t U[3 * 3] = {0};
  real_t s[3] = {0};
  real_t V[3 * 3] = {0};
  svd(covar, 3, 3, U, s, V);

  // Check to see if rotation matrix det(R) is 1
  real_t U_det = 0;
  real_t V_det = 0;
  svd_det(U, 3, 3, &U_det);
  svd_det(V, 3, 3, &V_det);
  const real_t d = U_det * V_det;
  const real_t D[3 * 3] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, d};

  // Calculate scale, rotation matrix and translation vector
  // -- Scale
  // scale = trace(diag(s) * D) / var_x
  real_t S[3 * 3] = {0};
  real_t SD[3 * 3] = {0};
  mat_diag_set(S, 3, 3, s);
  dot(S, 3, 3, D, 3, 3, SD);
  scale[0] = mat_trace(SD, 3, 3) / var_x;

  // -- Rotation
  // R = U * D * V'
  real_t Vt[3 * 3] = {0};
  mat_transpose(V, 3, 3, Vt);
  dot3(U, 3, 3, D, 3, 3, Vt, 3, 3, R);

  // -- Translation
  // t = mu_y - scale * R * mu_x
  real_t y[3] = {0};
  dot(R, 3, 3, mu_x, 3, 1, y);
  t[0] = mu_y[0] - scale[0] * y[0];
  t[1] = mu_y[1] - scale[0] * y[1];
  t[2] = mu_y[2] - scale[0] * y[2];

  // real_t diff = 0.0;
  // for (int i = 0; i < n; ++i) {
  //   // y_est = R * x + t
  //   const real_t px = X[i * 3 + 0];
  //   const real_t py = X[i * 3 + 1];
  //   const real_t pz = X[i * 3 + 2];
  //   const real_t x[3] = {px, py, pz};
  //
  //   real_t y_est[3] = {0};
  //   dot(R, 3, 3, x, 3, 1, y_est);
  //
  //   y_est[0] = y_est[0] + t[0];
  //   y_est[1] = y_est[1] + t[1];
  //   y_est[2] = y_est[2] + t[2];
  //
  //   diff += (y[0] - y_est[0]) * (y[0] - y_est[0]);
  //   diff += (y[1] - y_est[1]) * (y[1] - y_est[1]);
  //   diff += (y[2] - y_est[2]) * (y[2] - y_est[2]);
  // }

  // Clean up
  free(covar);
}

/*******************************************************************************
 * VOXEL
 ******************************************************************************/

void voxel_setup(voxel_t *voxel, const int32_t key[3]) {
  assert(voxel);
  assert(key);

  voxel->key[0] = key[0];
  voxel->key[1] = key[1];
  voxel->key[2] = key[2];

  for (int i = 0; i < VOXEL_MAX_POINTS; ++i) {
    voxel->points[i * 3 + 0] = 0.0f;
    voxel->points[i * 3 + 1] = 0.0f;
    voxel->points[i * 3 + 2] = 0.0f;
  }

  voxel->length = 0;
}

voxel_t *voxel_malloc(const int32_t key[3]) {
  voxel_t *voxel = malloc(sizeof(voxel_t));
  voxel->points = calloc(VOXEL_MAX_POINTS * 3, sizeof(float));
  voxel_setup(voxel, key);
  return voxel;
}

void voxel_free(voxel_t *voxel) {
  assert(voxel);
  free(voxel->points);
  free(voxel);
}

void voxel_print(voxel_t *voxel) {
  assert(voxel);

  printf("key: [%d, %d, %d]\n", voxel->key[0], voxel->key[1], voxel->key[2]);
  printf("length: %ld\n", voxel->length);
  for (int i = 0; i < voxel->length; ++i) {
    const float x = voxel->points[i * 3 + 0];
    const float y = voxel->points[i * 3 + 1];
    const float z = voxel->points[i * 3 + 2];
    printf("%d: [%.2f, %.2f, %.2f]\n", i, x, y, z);
  }
}

void voxel_reset(voxel_t *voxel) {
  assert(voxel);
  assert(voxel->points != NULL);

  voxel->key[0] = -1;
  voxel->key[1] = -1;
  voxel->key[2] = -1;

  for (int i = 0; i < VOXEL_MAX_POINTS; ++i) {
    voxel->points[i * 3 + 0] = 0.0f;
    voxel->points[i * 3 + 1] = 0.0f;
    voxel->points[i * 3 + 2] = 0.0f;
  }

  voxel->length = 0;
}

void voxel_copy(const voxel_t *src, voxel_t *dst) {
  assert(src && dst);
  assert(src->points != NULL);
  assert(dst->points != NULL);

  dst->key[0] = src->key[0];
  dst->key[1] = src->key[1];
  dst->key[2] = src->key[2];
  for (int i = 0; i < src->length; ++i) {
    dst->points[i] = src->points[i];
  }
  dst->length = src->length;
}

void voxel_add(voxel_t *voxel, const float p[3]) {
  if (voxel->length >= VOXEL_MAX_POINTS) {
    return;
  }
  voxel->points[voxel->length * 3 + 0] = p[0];
  voxel->points[voxel->length * 3 + 1] = p[1];
  voxel->points[voxel->length * 3 + 2] = p[2];
  voxel->length++;
}

typedef struct {
  uint32_t key;
  uint32_t value;
} voxel_kv_t;

void voxel_radix_sort(voxel_kv_t *arr, const int n) {
  if (n <= 1)
    return;

  const int radix_bits = 8;
  const int radix = 1 << radix_bits;
  const int MASK = radix - 1;

  voxel_kv_t *temp = malloc(n * sizeof(voxel_kv_t));
  if (!temp) {
    fprintf(stderr, "Memory allocation failed\n");
    return;
  }

  // Find maximum to determine number of passes needed
  uint32_t max_val = arr[0].key;
  for (int i = 1; i < n; i++) {
    if (arr[i].key > max_val) {
      max_val = arr[i].key;
    }
  }

  voxel_kv_t *current = arr;
  voxel_kv_t *next = temp;

  // Process 8 bits at a time
  for (int shift = 0; shift < 32 && (max_val >> shift) > 0;
       shift += radix_bits) {
    int count[256] = {0};

    // Count occurrences
    for (int i = 0; i < n; i++) {
      int digit = (current[i].key >> shift) & MASK;
      count[digit]++;
    }

    // Convert counts to positions
    for (int i = 1; i < radix; i++) {
      count[i] += count[i - 1];
    }

    // Build output array (from right to left for stability)
    for (int i = n - 1; i >= 0; i--) {
      int digit = (current[i].key >> shift) & MASK;
      next[count[digit] - 1] = current[i];
      count[digit]--;
    }

    // Swap arrays
    voxel_kv_t *swap = current;
    current = next;
    next = swap;
  }

  // Copy back if needed
  if (current != arr) {
    memcpy(arr, current, n * sizeof(voxel_kv_t));
  }

  free(temp);
}

float *voxel_grid_downsample(const float *points,
                             const int num_points,
                             const float voxel_size,
                             size_t *output_count) {
  const int min_points_per_voxel = 1;

  // First pass: Get min x-y-z
  float min_x = FLT_MAX;
  float min_y = FLT_MAX;
  float min_z = FLT_MAX;
  float max_x = -FLT_MAX;
  float max_y = -FLT_MAX;
  float max_z = -FLT_MAX;
  for (int i = 0; i < num_points; ++i) {
    const float x = points[i * 3 + 0];
    const float y = points[i * 3 + 1];
    const float z = points[i * 3 + 2];
    min_x = (x < min_x) ? x : min_x;
    min_y = (y < min_y) ? y : min_y;
    min_z = (z < min_z) ? z : min_z;
    max_x = (x > max_x) ? x : max_x;
    max_y = (y > max_y) ? y : max_y;
    max_z = (z > max_z) ? z : max_z;
  }

  const float inv_voxel_size = 1.0 / voxel_size;
  const uint64_t dx = (uint64_t) ((max_x - min_x) * inv_voxel_size) + 1;
  const uint64_t dy = (uint64_t) ((max_y - min_y) * inv_voxel_size) + 1;
  const uint64_t dz = (uint64_t) ((max_z - min_z) * inv_voxel_size) + 1;
  if ((dx * dy * dz) > UINT32_MAX) {
    LOG_ERROR("Voxel size too small! Index overflow!");
    return NULL;
  }

  // Compute the min and max bounding box values
  const int32_t bb_min_x = (int32_t) (floor(min_x * inv_voxel_size));
  const int32_t bb_max_x = (int32_t) (floor(max_x * inv_voxel_size));
  const int32_t bb_min_y = (int32_t) (floor(min_y * inv_voxel_size));
  const int32_t bb_max_y = (int32_t) (floor(max_y * inv_voxel_size));
  const int32_t bb_min_z = (int32_t) (floor(min_z * inv_voxel_size));
  const int32_t bb_dx = bb_max_x - bb_min_x + 1;
  const int32_t bb_dy = bb_max_y - bb_min_y + 1;
  const int32_t bb_dxdy = bb_dx * bb_dy;

  // Second pass: Convert 3D points to voxel index, and sort by voxel index
  voxel_kv_t *kvs = malloc(sizeof(voxel_kv_t) * num_points);
  for (int i = 0; i < num_points; ++i) {
    const uint32_t vx = floor(points[i * 3 + 0] * inv_voxel_size - bb_min_x);
    const uint32_t vy = floor(points[i * 3 + 1] * inv_voxel_size - bb_min_y);
    const uint32_t vz = floor(points[i * 3 + 2] * inv_voxel_size - bb_min_z);
    const uint32_t vindex = vx + vy * bb_dx + vz * bb_dxdy;
    kvs[i].key = vindex;
    kvs[i].value = i;
  }
  voxel_radix_sort(kvs, num_points);

  // Third pass: Form pairs of first and last index
  size_t total = 0;
  size_t index = 0;
  typedef struct {
    int first;
    int second;
  } pair_t;
  pair_t *pairs = malloc(sizeof(pair_t) * num_points);

  while (index < num_points) {
    size_t i = index + 1;
    while (i < num_points && kvs[i].key == kvs[index].key) {
      ++i;
    }

    if ((i - index) >= min_points_per_voxel) {
      pairs[total].first = index;
      pairs[total].second = i;
      ++total;
    }
    index = i;
  }

  // Fourth pass: compute centroids, insert them into their final position
  float *output = malloc(sizeof(float) * 3 * total);
  for (int i = 0; i < total; ++i) {
    // Calculate centroid
    const int first_index = pairs[i].first;
    const int last_index = pairs[i].second;
    float centroid_x = 0.0f;
    float centroid_y = 0.0f;
    float centroid_z = 0.0f;
    const float count = last_index - first_index;
    for (int j = first_index; j < last_index; ++j) {
      const int point_index = kvs[j].value;
      centroid_x += points[point_index * 3 + 0];
      centroid_y += points[point_index * 3 + 1];
      centroid_z += points[point_index * 3 + 2];
    }
    centroid_x /= count;
    centroid_y /= count;
    centroid_z /= count;

    output[i * 3 + 0] = centroid_x;
    output[i * 3 + 1] = centroid_y;
    output[i * 3 + 2] = centroid_z;
  }
  *output_count = total;

  // Clean up
  free(kvs);
  free(pairs);

  return output;
}

/*******************************************************************************
 * OCTREE
 ******************************************************************************/

/////////////////
// OCTREE NODE //
/////////////////

octree_node_t *octree_node_malloc(const float center[3],
                                  const float size,
                                  const int depth,
                                  const int max_depth,
                                  const int max_points) {
  octree_node_t *node = malloc(sizeof(octree_node_t));

  node->center[0] = center[0];
  node->center[1] = center[1];
  node->center[2] = center[2];
  node->size = size;
  node->depth = depth;
  node->max_depth = max_depth;
  node->max_points = max_points;

  for (int i = 0; i < 8; ++i) {
    node->children[i] = NULL;
  }
  node->points = malloc(sizeof(float) * 3 * max_points);
  node->num_points = 0;
  node->capacity = max_points;

  return node;
}

void octree_node_free(octree_node_t *node) {
  if (node == NULL) {
    return;
  }

  for (int i = 0; i < 8; ++i) {
    octree_node_free(node->children[i]);
  }
  free(node->points);
  free(node);
}

bool octree_node_check_point(const octree_node_t *node, const float point[3]) {
  const float hsize = node->size / 2.0;
  const float xmin = node->center[0] - hsize;
  const float xmax = node->center[0] + hsize;
  const float ymin = node->center[1] - hsize;
  const float ymax = node->center[1] + hsize;
  const float zmin = node->center[2] - hsize;
  const float zmax = node->center[2] + hsize;

  const bool x_ok = xmin <= point[0] && point[0] <= xmax;
  const bool y_ok = ymin <= point[1] && point[1] <= ymax;
  const bool z_ok = zmin <= point[2] && point[2] <= zmax;
  return x_ok && y_ok && z_ok;
}

////////////
// OCTREE //
////////////

octree_t *octree_malloc(const float octree_center[3],
                        const float map_size,
                        const int octree_max_depth,
                        const int voxel_max_points,
                        const float *octree_points,
                        const size_t num_points) {
  assert(octree_center);
  octree_t *octree = malloc(sizeof(octree_t));

  octree->center[0] = octree_center[0];
  octree->center[1] = octree_center[1];
  octree->center[2] = octree_center[2];
  octree->map_size = map_size;
  octree->root = octree_node_malloc(octree->center,
                                    octree->map_size,
                                    0,
                                    octree_max_depth,
                                    voxel_max_points);
  for (size_t i = 0; i < num_points; i++) {
    octree_add_point(octree->root, &octree_points[i * 3]);
  }

  return octree;
}

void octree_free(octree_t *octree) {
  if (octree == NULL) {
    return;
  }
  octree_node_free(octree->root);
  free(octree);
}

void octree_add_point(octree_node_t *node, const float point[3]) {
  assert(node);
  assert(point);

  // Max depth reached? Add the point
  if (node->depth == node->max_depth) {
    assert(octree_node_check_point(node, point));
    if (node->num_points >= node->max_points) {
      return;
    }
    node->points[node->num_points * 3 + 0] = point[0];
    node->points[node->num_points * 3 + 1] = point[1];
    node->points[node->num_points * 3 + 2] = point[2];
    node->num_points++;
    if (node->num_points >= node->capacity) {
      node->capacity = node->capacity * 2;
      node->points = realloc(node->points, sizeof(float) * 3 * node->capacity);
    }
    return;
  }

  // Calculate node index
  int index = 0;
  index |= (point[0] >= node->center[0]) ? 1 : 0;
  index |= (point[1] >= node->center[1]) ? 2 : 0;
  index |= (point[2] >= node->center[2]) ? 4 : 0;

  // Create new child node if it doesn't exist already
  octree_node_t *child = node->children[index];
  if (child == NULL) {
    const float offset = node->size / 4;
    const float *node_center = node->center;
    const float center[3] = {node_center[0] + (index & 1 ? offset : -offset),
                             node_center[1] + (index & 2 ? offset : -offset),
                             node_center[2] + (index & 4 ? offset : -offset)};
    const float size = node->size / 2.0;
    const int depth = node->depth + 1;
    const int max_depth = node->max_depth;
    const int max_points = node->max_points;
    child = octree_node_malloc(center, size, depth, max_depth, max_points);
    node->children[index] = child;
  }

  // Recurse down the octree
  octree_add_point(child, point);
}

static void octree_get_points_recurse(const octree_node_t *node,
                                      octree_data_t *data) {
  assert(node);
  assert(data && data->points && data->capacity > 0);

  if (node->num_points > 0) {
    for (size_t i = 0; i < node->num_points; ++i) {
      data->points[data->num_points * 3 + 0] = node->points[i * 3 + 0];
      data->points[data->num_points * 3 + 1] = node->points[i * 3 + 1];
      data->points[data->num_points * 3 + 2] = node->points[i * 3 + 2];
      data->num_points++;
      if (data->num_points >= data->capacity) {
        data->capacity = data->capacity * 2;
        data->points =
            realloc(data->points, sizeof(float) * 3 * data->capacity);
      }
    }
  }

  for (size_t i = 0; i < 8; ++i) {
    if (node->children[i]) {
      octree_get_points_recurse(node->children[i], data);
    }
  }
}

void octree_get_points(const octree_node_t *node, octree_data_t *data) {
  assert(node);
  assert(data && data->points && data->capacity > 0);

  for (size_t i = 0; i < 8; ++i) {
    if (node->children[i]) {
      octree_get_points_recurse(node->children[i], data);
    }
  }
}

float *octree_downsample(const float *octree_data,
                         const size_t n,
                         const float voxel_size,
                         const size_t voxel_max_points,
                         size_t *n_out) {
  assert(octree_data);
  assert(n > 0);

  // Find center
  float xmin = FLT_MAX;
  float ymin = FLT_MAX;
  float zmin = FLT_MAX;
  float xmax = -FLT_MAX;
  float ymax = -FLT_MAX;
  float zmax = -FLT_MAX;
  for (size_t i = 0; i < n; ++i) {
    const float x = octree_data[i * 3 + 0];
    const float y = octree_data[i * 3 + 1];
    const float z = octree_data[i * 3 + 2];
    xmin = (x <= xmin) ? x : xmin;
    xmax = (x >= xmax) ? x : xmax;
    ymin = (y <= ymin) ? y : ymin;
    ymax = (y >= ymax) ? y : ymax;
    zmin = (z <= zmin) ? z : zmin;
    zmax = (z >= zmax) ? z : zmax;
  }
  const float octree_center[3] = {
      (xmax + xmin) / 2.0,
      (ymax + ymin) / 2.0,
      (zmax + zmin) / 2.0,
  };

  // Find octree size
  float map_size = 0.0f;
  map_size = (fabs(xmin) > map_size) ? fabs(xmin) : map_size;
  map_size = (fabs(ymin) > map_size) ? fabs(ymin) : map_size;
  map_size = (fabs(zmin) > map_size) ? fabs(zmin) : map_size;
  map_size = (xmax > map_size) ? xmax : map_size;
  map_size = (ymax > map_size) ? ymax : map_size;
  map_size = (zmax > map_size) ? zmax : map_size;
  map_size *= 2.0;

  // Create octree
  const int octree_max_depth = ceil(log2(map_size / voxel_size));
  octree_t *octree = octree_malloc(octree_center,
                                   map_size,
                                   octree_max_depth,
                                   voxel_max_points,
                                   octree_data,
                                   n);

  // Get points
  octree_data_t data = {0};
  data.points = malloc(sizeof(float) * 3 * n);
  data.num_points = 0;
  data.capacity = n;
  octree_get_points(octree->root, &data);

  // Clean up
  octree_free(octree);

  // Return
  *n_out = data.num_points;
  data.points = realloc(data.points, sizeof(float) * 3 * *n_out);
  return data.points;
}

/*****************************************************************************
 * KD-TREE
 ****************************************************************************/

//////////////////
// KD-TREE NODE //
//////////////////

kdtree_node_t *kdtree_node_malloc(const float p[3], const int k) {
  assert(p);
  assert(k >= 0 && k <= 2);
  kdtree_node_t *node = malloc(sizeof(kdtree_node_t));

  node->p[0] = p[0];
  node->p[1] = p[1];
  node->p[2] = p[2];
  node->k = k;
  node->left = NULL;
  node->right = NULL;

  return node;
}

void kdtree_node_free(kdtree_node_t *node) {
  if (node == NULL) {
    return;
  }
  kdtree_node_free(node->left);
  kdtree_node_free(node->right);
  free(node);
}

/////////////
// KD-TREE //
/////////////

int point_cmp(const void *a, const void *b, void *k) {
  return (((float *) a)[*(int *) k] < ((float *) b)[*(int *) k]) ? -1 : 1;
}

kdtree_node_t *kdtree_insert(kdtree_node_t *node,
                             const float p[3],
                             const int depth) {
  const int k = depth % KDTREE_KDIM;
  if (node == NULL) {
    return kdtree_node_malloc(p, k);
  }

  if (p[k] < node->p[k]) {
    node->left = kdtree_insert(node->left, p, depth + 1);
  } else {
    node->right = kdtree_insert(node->right, p, depth + 1);
  }

  return node;
}

static kdtree_node_t *
_kdtree_build(float *points, const int start, const int end, const int depth) {
  if (start > end) {
    return NULL;
  }

  int k = depth % KDTREE_KDIM;
  const int mid = (start + end + 1) / 2;
  qsort_r(points + start * 3,
          end - start + 1,
          sizeof(float) * 3,
          point_cmp,
          &k);

  kdtree_node_t *root = kdtree_node_malloc(points + mid * 3, k);
  root->left = _kdtree_build(points, start, mid - 1, depth + 1);
  root->right = _kdtree_build(points, mid + 1, end, depth + 1);

  return root;
}

kdtree_t *kdtree_malloc(float *points, size_t num_points) {
  kdtree_t *kdtree = malloc(sizeof(kdtree_t));
  kdtree->root = _kdtree_build(points, 0, num_points - 1, 0);
  return kdtree;
}

void kdtree_free(kdtree_t *kdtree) {
  kdtree_node_free(kdtree->root);
  free(kdtree);
}

static void _kdtree_points(const kdtree_node_t *node, kdtree_data_t *data) {
  assert(data);
  if (node == NULL) {
    return;
  }

  data->points[data->size * 3 + 0] = node->p[0];
  data->points[data->size * 3 + 1] = node->p[1];
  data->points[data->size * 3 + 2] = node->p[2];
  data->size++;
  if (data->size >= data->capacity) {
    data->capacity = data->capacity * 2;
    data->points = realloc(data->points, sizeof(float) * 3 * data->capacity);
  }

  _kdtree_points(node->left, data);
  _kdtree_points(node->right, data);
}

void kdtree_points(const kdtree_t *kdtree, kdtree_data_t *data) {
  assert(kdtree && kdtree->root);
  assert(data && data->points && data->capacity > 0);
  _kdtree_points(kdtree->root, data);
}

void _kdtree_nn(const kdtree_node_t *node,
                const float target[3],
                float *best_dist,
                float *best_point,
                int depth) {
  // Pre-check
  if (node == NULL) {
    return;
  }

  // Calculate distance and keep track of best
  float sq_dist = 0.0f;
  sq_dist += (node->p[0] - target[0]) * (node->p[0] - target[0]);
  sq_dist += (node->p[1] - target[1]) * (node->p[1] - target[1]);
  sq_dist += (node->p[2] - target[2]) * (node->p[2] - target[2]);
  if (sq_dist <= *best_dist) {
    best_point[0] = node->p[0];
    best_point[1] = node->p[1];
    best_point[2] = node->p[2];
    *best_dist = sq_dist;
  }

  // Determine which side to search first
  const int axis = node->k;
  const float diff = target[axis] - node->p[axis];

  // Search the closer subtree first
  const kdtree_node_t *closer = (diff <= 0) ? node->left : node->right;
  const kdtree_node_t *farther = (diff <= 0) ? node->right : node->left;
  _kdtree_nn(closer, target, best_dist, best_point, depth + 1);

  // Search the farther subtree
  if (fabs(diff) < *best_dist) {
    _kdtree_nn(farther, target, best_dist, best_point, depth + 1);
  }
}

void kdtree_nn(const kdtree_t *kdtree,
               const float target[3],
               float *best_point,
               float *best_dist) {
  *best_dist = INFINITY;
  best_point[0] = target[0];
  best_point[1] = target[1];
  best_point[2] = target[2];
  _kdtree_nn(kdtree->root, target, best_dist, best_point, 0);
}

kdtree_data_t *kdtree_nns(const kdtree_t *kdtree,
                          const float *query_points,
                          const size_t n) {
  kdtree_data_t *data = malloc(sizeof(kdtree_data_t));
  data->points = malloc(sizeof(float) * 3 * n);
  data->size = 0;
  data->capacity = n;

  for (size_t i = 0; i < n; ++i) {
    float best_point[3] = {0};
    float best_dist = 0.0f;
    kdtree_nn(kdtree, &query_points[i * 3], best_point, &best_dist);
    data->points[data->size * 3 + 0] = best_point[0];
    data->points[data->size * 3 + 1] = best_point[1];
    data->points[data->size * 3 + 2] = best_point[2];
    data->size++;
  }

  return data;
}

/*******************************************************************************
 * STATE-ESTIMATION
 ******************************************************************************/

//////////////
// FIDUCIAL //
//////////////

fiducial_info_t *fiducial_info_malloc(const timestamp_t ts, const int cam_idx) {
  fiducial_info_t *finfo = malloc(sizeof(fiducial_info_t));
  finfo->ts = ts;
  finfo->cam_idx = cam_idx;
  finfo->num_corners = 0;
  finfo->capacity = 200;
  finfo->tag_ids = calloc(200, sizeof(int));
  finfo->corner_indices = calloc(200, sizeof(int));
  finfo->pts = calloc(200 * 3, sizeof(real_t));
  finfo->kps = calloc(200 * 2, sizeof(real_t));
  return finfo;
}

void fiducial_info_free(fiducial_info_t *finfo) {
  free(finfo->tag_ids);
  free(finfo->corner_indices);
  free(finfo->pts);
  free(finfo->kps);
  free(finfo);
}

void fiducial_info_print(const fiducial_info_t *finfo) {
  printf("ts: %ld\n", finfo->ts);
  printf("cam_idx: %d\n", finfo->cam_idx);
  printf("num_corners: %d\n", finfo->num_corners);
  printf("\n");
  printf("#tag_id, corner_idx, kp_x, kp_y, p_x, p_y, p_z\n");
  for (int i = 0; i < finfo->num_corners; i++) {
    const int tag_id = finfo->tag_ids[i];
    const int corner_idx = finfo->corner_indices[i];
    printf("%d, ", tag_id);
    printf("%d, ", corner_idx);
    printf("%.2f, ", finfo->kps[i * 2 + 0]);
    printf("%.2f, ", finfo->kps[i * 2 + 1]);
    printf("%.2f, ", finfo->pts[i * 3 + 0]);
    printf("%.2f, ", finfo->pts[i * 3 + 1]);
    printf("%.2f", finfo->pts[i * 3 + 2]);
    printf("\n");
  }
  printf("\n");
}

void fiducial_info_add(fiducial_info_t *finfo,
                       const int tag_id,
                       const int corner_index,
                       const real_t p[3],
                       const real_t z[2]) {
  if (finfo->num_corners >= finfo->capacity) {
    finfo->capacity = finfo->capacity * 2;
    const size_t n = finfo->capacity;

    finfo->tag_ids = realloc(finfo->tag_ids, sizeof(int) * n);
    finfo->corner_indices = realloc(finfo->tag_ids, sizeof(int) * n);
    finfo->pts = realloc(finfo->pts, sizeof(real_t) * 3 * n);
    finfo->kps = realloc(finfo->kps, sizeof(real_t) * 2 * n);
  }

  const size_t i = finfo->num_corners;
  finfo->tag_ids[i] = tag_id;
  finfo->corner_indices[i] = corner_index;
  finfo->pts[i * 3 + 0] = p[0];
  finfo->pts[i * 3 + 1] = p[1];
  finfo->pts[i * 3 + 2] = p[2];
  finfo->kps[i * 2 + 0] = z[0];
  finfo->kps[i * 2 + 1] = z[1];
  finfo->num_corners++;
}

////////////
// CAMERA //
////////////

/**
 * Setup camera parameters
 */
void camera_setup(camera_t *camera,
                  const int cam_idx,
                  const int cam_res[2],
                  const char *proj_model,
                  const char *dist_model,
                  const real_t *data) {
  assert(camera != NULL);
  assert(cam_res != NULL);
  assert(proj_model != NULL);
  assert(dist_model != NULL);
  assert(data != NULL);

  camera->cam_idx = cam_idx;
  camera->resolution[0] = cam_res[0];
  camera->resolution[1] = cam_res[1];

  string_copy(camera->proj_model, proj_model);
  string_copy(camera->dist_model, dist_model);

  camera->data[0] = data[0];
  camera->data[1] = data[1];
  camera->data[2] = data[2];
  camera->data[3] = data[3];
  camera->data[4] = data[4];
  camera->data[5] = data[5];
  camera->data[6] = data[6];
  camera->data[7] = data[7];

  if (streqs(proj_model, "pinhole") && streqs(dist_model, "radtan4")) {
    camera->proj_func = pinhole_radtan4_project;
    camera->back_proj_func = pinhole_radtan4_back_project;
    camera->undistort_func = pinhole_radtan4_undistort;
  } else if (streqs(proj_model, "pinhole") && streqs(dist_model, "equi4")) {
    camera->proj_func = pinhole_equi4_project;
    camera->back_proj_func = pinhole_equi4_back_project;
    camera->undistort_func = pinhole_equi4_undistort;
  } else {
    FATAL("Unknown [%s-%s] camera model!\n", proj_model, dist_model);
  }
}

/**
 * Copy camera parameters.
 */
void camera_copy(const camera_t *src, camera_t *dst) {
  dst->cam_idx = src->cam_idx;
  dst->resolution[0] = src->resolution[0];
  dst->resolution[1] = src->resolution[1];
  strcpy(dst->proj_model, src->proj_model);
  strcpy(dst->dist_model, src->dist_model);
  dst->data[0] = src->data[0];
  dst->data[1] = src->data[1];
  dst->data[2] = src->data[2];
  dst->data[3] = src->data[3];
  dst->data[4] = src->data[4];
  dst->data[5] = src->data[5];
  dst->data[6] = src->data[6];
  dst->data[7] = src->data[7];

  dst->proj_func = src->proj_func;
  dst->back_proj_func = src->back_proj_func;
  dst->undistort_func = src->undistort_func;
}

/**
 * Print camera parameters
 */
void camera_fprint(const camera_t *cam, FILE *f) {
  assert(cam != NULL);

  fprintf(f, "cam%d:\n", cam->cam_idx);
  fprintf(f,
          "  resolution: [%d, %d]\n",
          cam->resolution[0],
          cam->resolution[1]);
  fprintf(f, "  proj_model: %s\n", cam->proj_model);
  fprintf(f, "  dist_model: %s\n", cam->dist_model);
  fprintf(f, "  data: [");
  for (int i = 0; i < 8; i++) {
    if ((i + 1) < 8) {
      fprintf(f, "%.4f, ", cam->data[i]);
    } else {
      fprintf(f, "%.4f", cam->data[i]);
    }
  }
  fprintf(f, "]\n");
}

/**
 * Print camera parameters
 */
void camera_print(const camera_t *cam) {
  assert(cam != NULL);
  camera_fprint(cam, stdout);
}

/**
 * Project 3D point to image point.
 */
void camera_project(const camera_t *camera, const real_t p_C[3], real_t z[2]) {
  assert(camera != NULL);
  assert(camera->proj_func != NULL);
  assert(p_C != NULL);
  assert(z != NULL);
  camera->proj_func(camera->data, p_C, z);
}

/**
 * Back project image point to bearing vector.
 */
void camera_back_project(const camera_t *camera,
                         const real_t z[2],
                         real_t bearing[3]) {
  assert(camera != NULL);
  assert(z != NULL);
  assert(bearing != NULL);
  camera->back_proj_func(camera->data, z, bearing);
}

/**
 * Undistort image points.
 */
void camera_undistort_points(const camera_t *camera,
                             const real_t *kps,
                             const int num_points,
                             real_t *kps_und) {
  assert(camera != NULL);
  assert(kps != NULL);
  assert(kps_und != NULL);

  for (int i = 0; i < num_points; i++) {
    const real_t *z_in = &kps[2 * i];
    real_t *z_out = &kps_und[i * 2];
    camera->undistort_func(camera->data, z_in, z_out);
  }
}

/**
 * Solve the Perspective-N-Points problem.
 *
 * **IMPORTANT**: This function assumes that object points lie on the plane
 * because the initialization step uses DLT to estimate the homography between
 * camera and planar object, then the relative pose between them is recovered.
 */
int solvepnp_camera(const camera_t *cam_params,
                    const real_t *img_pts,
                    const real_t *obj_pts,
                    const int N,
                    real_t T_CO[4 * 4]) {
  assert(cam_params != NULL);
  assert(img_pts != NULL);
  assert(obj_pts != NULL);
  assert(N > 0);
  assert(T_CO != NULL);

  // Undistort keypoints
  real_t *img_pts_ud = malloc(sizeof(real_t) * N * 2);
  camera_undistort_points(cam_params, img_pts, N, img_pts_ud);

  // Estimate relative pose T_CO
  const int status = solvepnp(cam_params->data, img_pts_ud, obj_pts, N, T_CO);
  free(img_pts_ud);

  return status;
}

/**
 * Triangulate features in batch.
 */
void triangulate_batch(const camera_t *cam_i,
                       const camera_t *cam_j,
                       const real_t T_CiCj[4 * 4],
                       const real_t *kps_i,
                       const real_t *kps_j,
                       const int n,
                       real_t *points,
                       int *status) {
  assert(cam_i != NULL);
  assert(cam_j != NULL);
  assert(T_CiCj != NULL);
  assert(kps_i != NULL);
  assert(kps_j != NULL);
  assert(n > 0);
  assert(points != NULL);
  assert(status != NULL);

  // Setup projection matrices
  real_t P_i[3 * 4] = {0};
  real_t P_j[3 * 4] = {0};
  TF_IDENTITY(T_eye);
  pinhole_projection_matrix(cam_i->data, T_eye, P_i);
  pinhole_projection_matrix(cam_j->data, T_CiCj, P_j);

  // Triangulate features
  for (int i = 0; i < n; i++) {
    // Undistort keypoints
    real_t z_i[2] = {0};
    real_t z_j[2] = {0};
    cam_i->undistort_func(cam_i->data, &kps_i[i * 2], z_i);
    cam_j->undistort_func(cam_j->data, &kps_j[i * 2], z_j);

    // Triangulate
    real_t p[3] = {0};
    linear_triangulation(P_i, P_j, z_i, z_j, p);
    points[i * 3 + 0] = p[0];
    points[i * 3 + 1] = p[1];
    points[i * 3 + 2] = p[2];
    status[i] = 0;
  }
}

////////////////
// IMU-BUFFER //
////////////////

/**
 * Setup IMU buffer
 */
void imu_buffer_setup(imu_buffer_t *imu_buf) {
  for (int k = 0; k < IMU_BUFFER_MAX_SIZE; k++) {
    imu_buf->ts[k] = 0.0;

    imu_buf->acc[k][0] = 0.0;
    imu_buf->acc[k][1] = 0.0;
    imu_buf->acc[k][2] = 0.0;

    imu_buf->gyr[k][0] = 0.0;
    imu_buf->gyr[k][1] = 0.0;
    imu_buf->gyr[k][2] = 0.0;
  }

  imu_buf->size = 0;
}

/**
 * Print IMU buffer
 */
void imu_buffer_print(const imu_buffer_t *imu_buf) {
  for (int k = 0; k < imu_buf->size; k++) {
    const real_t *acc = imu_buf->acc[k];
    const real_t *gyr = imu_buf->gyr[k];

    printf("ts: %ld ", imu_buf->ts[k]);
    printf("acc: [%.2f, %.2f, %.2f] ", acc[0], acc[1], acc[2]);
    printf("gyr: [%.2f, %.2f, %.2f] ", gyr[0], gyr[1], gyr[2]);
    printf("\n");
  }
}

/**
 * Add measurement to IMU buffer
 */
void imu_buffer_add(imu_buffer_t *imu_buf,
                    const timestamp_t ts,
                    const real_t acc[3],
                    const real_t gyr[3]) {
  assert(imu_buf->size < IMU_BUFFER_MAX_SIZE);
  const int k = imu_buf->size;
  imu_buf->ts[k] = ts;
  imu_buf->acc[k][0] = acc[0];
  imu_buf->acc[k][1] = acc[1];
  imu_buf->acc[k][2] = acc[2];
  imu_buf->gyr[k][0] = gyr[0];
  imu_buf->gyr[k][1] = gyr[1];
  imu_buf->gyr[k][2] = gyr[2];
  imu_buf->size++;
}

/**
 * Return first timestamp in IMU buffer
 */
timestamp_t imu_buffer_first_ts(const imu_buffer_t *imu_buf) {
  assert(imu_buf != NULL);
  return imu_buf->ts[0];
}

/**
 * Return last timestamp in IMU buffer
 */
timestamp_t imu_buffer_last_ts(const imu_buffer_t *imu_buf) {
  assert(imu_buf != NULL);
  return imu_buf->ts[imu_buf->size - 1];
}

/**
 * Clear IMU buffer
 */
void imu_buffer_clear(imu_buffer_t *imu_buf) {
  for (int k = 0; k < imu_buf->size; k++) {
    timestamp_t *ts = &imu_buf->ts[k];
    real_t *acc = imu_buf->acc[k];
    real_t *gyr = imu_buf->gyr[k];

    *ts = 0;

    acc[0] = 0.0;
    acc[1] = 0.0;
    acc[2] = 0.0;

    gyr[0] = 0.0;
    gyr[1] = 0.0;
    gyr[2] = 0.0;
  }
  imu_buf->size = 0;
}

/**
 * Copy IMU buffer
 */
void imu_buffer_copy(const imu_buffer_t *src, imu_buffer_t *dst) {
  dst->size = 0;
  for (int k = 0; k < src->size; k++) {
    dst->ts[k] = src->ts[k];

    dst->acc[k][0] = src->acc[k][0];
    dst->acc[k][1] = src->acc[k][1];
    dst->acc[k][2] = src->acc[k][2];

    dst->gyr[k][0] = src->gyr[k][0];
    dst->gyr[k][1] = src->gyr[k][1];
    dst->gyr[k][2] = src->gyr[k][2];
  }
  dst->size = src->size;
}

/////////////
// FEATURE //
/////////////

/**
 * Setup feature.
 */
void feature_setup(feature_t *f, const size_t feature_id) {
  assert(f != NULL);

  f->marginalize = 0;
  f->fix = 0;

  f->type = FEATURE_XYZ;
  f->feature_id = feature_id;
  f->status = 0;
  f->data[0] = 0.0;
  f->data[1] = 0.0;
  f->data[2] = 0.0;
}

/**
 * Initialize feature.
 */
void feature_init(feature_t *f, const size_t feature_id, const real_t *data) {
  assert(f != NULL);
  assert(data != NULL);

  f->marginalize = 0;
  f->fix = 0;

  f->type = FEATURE_XYZ;
  f->feature_id = feature_id;
  f->status = 1;
  f->data[0] = data[0];
  f->data[1] = data[1];
  f->data[2] = data[2];
}

/**
 * Print feature.
 */
void feature_print(const feature_t *f) {
  printf("feature_id: %ld\n", f->feature_id);
  printf("status: %d\n", f->status);
  printf("data: (%.2f, %.2f, %.2f)\n", f->data[0], f->data[1], f->data[2]);
  printf("\n");
}

////////////////
// PARAMETERS //
////////////////

/**
 * Return parameter type as a string
 */
void param_type_string(const int param_type, char *s) {
  switch (param_type) {
    case POSITION_PARAM:
      strcpy(s, "POSITION_PARAM");
      break;
    case ROTATION_PARAM:
      strcpy(s, "ROTATION_PARAM");
      break;
    case POSE_PARAM:
      strcpy(s, "POSE_PARAM");
      break;
    case EXTRINSIC_PARAM:
      strcpy(s, "EXTRINSIC_PARAM");
      break;
    case FIDUCIAL_PARAM:
      strcpy(s, "FIDUCIAL_PARAM");
      break;
    case VELOCITY_PARAM:
      strcpy(s, "VELOCITY_PARAM");
      break;
    case IMU_BIASES_PARAM:
      strcpy(s, "IMU_BIASES_PARAM");
      break;
    case FEATURE_PARAM:
      strcpy(s, "FEATURE_PARAM");
      break;
    case JOINT_PARAM:
      strcpy(s, "JOINT_PARAM");
      break;
    case CAMERA_PARAM:
      strcpy(s, "CAMERA_PARAM");
      break;
    case TIME_DELAY_PARAM:
      strcpy(s, "TIME_DELAY_PARAM");
      break;
    default:
      FATAL("Invalid param type [%d]!\n", param_type);
      break;
  }
}

/**
 * Return parameter global size depending on parameter type
 */
size_t param_global_size(const int param_type) {
  size_t param_size = 0;

  switch (param_type) {
    case POSITION_PARAM:
      param_size = 3;
      break;
    case ROTATION_PARAM:
      param_size = 4;
      break;
    case POSE_PARAM:
    case EXTRINSIC_PARAM:
    case FIDUCIAL_PARAM:
      param_size = 7;
      break;
    case VELOCITY_PARAM:
      param_size = 3;
      break;
    case IMU_BIASES_PARAM:
      param_size = 6;
      break;
    case FEATURE_PARAM:
      param_size = 3;
      break;
    case JOINT_PARAM:
      param_size = 1;
      break;
    case CAMERA_PARAM:
      param_size = 8;
      break;
    case TIME_DELAY_PARAM:
      param_size = 1;
      break;
    default:
      FATAL("Invalid param type [%d]!\n", param_type);
      break;
  }

  return param_size;
}

/**
 * Return parameter local size depending on parameter type
 */
size_t param_local_size(const int param_type) {
  size_t param_size = 0;

  switch (param_type) {
    case POSITION_PARAM:
      param_size = 3;
      break;
    case ROTATION_PARAM:
      param_size = 3;
      break;
    case POSE_PARAM:
    case EXTRINSIC_PARAM:
    case FIDUCIAL_PARAM:
      param_size = 6;
      break;
    case VELOCITY_PARAM:
      param_size = 3;
      break;
    case IMU_BIASES_PARAM:
      param_size = 6;
      break;
    case FEATURE_PARAM:
      param_size = 3;
      break;
    case JOINT_PARAM:
      param_size = 1;
      break;
    case CAMERA_PARAM:
      param_size = 8;
      break;
    case TIME_DELAY_PARAM:
      param_size = 1;
      break;
    default:
      FATAL("Invalid param type [%d]!\n", param_type);
      break;
  }

  return param_size;
}

/**
 * Malloc param index.
 */
rbt_t *param_index_malloc(void) { return rbt_malloc(default_cmp); }

/**
 * Free param index.
 */
void param_index_free(rbt_t *param_index) {
  // Free values
  const size_t n = rbt_size(param_index);
  arr_t *keys = arr_malloc(n);
  rbt_keys(param_index, keys);
  for (size_t i = 0; i < n; ++i) {
    free(rbt_search(param_index, keys->data[i]));
  }

  // Free param_index and keys book keeping
  rbt_free(param_index);
  arr_free(keys);
}

/**
 * Print param index.
 */
void param_index_print(const rbt_t *param_index) {
  const size_t n = rbt_size(param_index);
  arr_t *keys = arr_malloc(n);
  arr_t *vals = arr_malloc(n);
  rbt_keys_values(param_index, keys, vals);

  for (size_t i = 0; i < n; ++i) {
    param_info_t *info = vals->data[i];
    if (info->idx != -1) {
      char s[100] = {0};
      param_type_string(info->type, s);
      printf("param[%04ld]: %15s, idx: %6d, addr: %p\n",
             i,
             s,
             info->idx,
             (void *) info->data);
    }
  }

  arr_free(keys);
  arr_free(vals);
}

/**
 * Check if param has already been added.
 */
bool param_index_exists(rbt_t *param_index, real_t *key) {
  return rbt_contains(param_index, key);
}

/**
 * Add parameter to hash
 */
void param_index_add(rbt_t *param_index,
                     const int param_type,
                     const int fix,
                     real_t *data,
                     int *col_idx) {
  if (fix == 0) {
    param_info_t *param_meta = malloc(sizeof(param_info_t));
    param_meta->data = data;
    param_meta->idx = *col_idx;
    param_meta->type = param_type;
    param_meta->fix = fix;
    rbt_insert(param_index, data, param_meta);
    *col_idx += param_local_size(param_type);

  } else {
    param_info_t *param_meta = malloc(sizeof(param_info_t));
    param_meta->data = data;
    param_meta->idx = -1;
    param_meta->type = param_type;
    param_meta->fix = fix;
    rbt_insert(param_index, data, param_meta);
  }
}

////////////
// FACTOR //
////////////

int check_factor_jacobian(const void *factor,
                          FACTOR_EVAL_PTR,
                          real_t **params,
                          real_t **jacobians,
                          const int r_size,
                          const int param_size,
                          const int param_idx,
                          const real_t step_size,
                          const real_t tol,
                          const int verbose) {
  // Form jacobian name
  char J_name[10] = {0};
  if (snprintf(J_name, 10, "J%d", param_idx) <= 0) {
    return -1;
  }

  // Setup
  real_t *r = calloc(r_size, sizeof(real_t));
  real_t *J_numdiff = calloc(r_size * param_size, sizeof(real_t));

  // Evaluate factor
  if (factor_eval(factor, params, r, NULL) != 0) {
    free(r);
    free(J_numdiff);
    return -2;
  }

  // Numerical diff - forward finite difference
  for (int i = 0; i < param_size; i++) {
    real_t *r_fwd = calloc(r_size, sizeof(real_t));
    real_t *r_diff = calloc(r_size, sizeof(real_t));

    params[param_idx][i] += step_size;
    factor_eval(factor, params, r_fwd, NULL);
    params[param_idx][i] -= step_size;

    vec_sub(r_fwd, r, r_diff, r_size);
    vec_scale(r_diff, r_size, 1.0 / step_size);
    mat_col_set(J_numdiff, param_size, r_size, i, r_diff);

    free(r_fwd);
    free(r_diff);
  }

  // Check jacobian
  const int retval = check_jacobian(J_name,
                                    J_numdiff,
                                    jacobians[param_idx],
                                    r_size,
                                    param_size,
                                    tol,
                                    verbose);
  free(r);
  free(J_numdiff);

  return retval;
}

int check_factor_so3_jacobian(const void *factor,
                              FACTOR_EVAL_PTR,
                              real_t **params,
                              real_t **jacobians,
                              const int r_size,
                              const int param_idx,
                              const real_t step_size,
                              const real_t tol,
                              const int verbose) {
  // Form jacobian name
  char J_name[10] = {0};
  if (snprintf(J_name, 10, "J%d", param_idx) <= 0) {
    return -1;
  }

  // Setup
  const int param_size = 3;
  real_t *r = calloc(r_size, sizeof(real_t));
  real_t *J_numdiff = calloc(r_size * param_size, sizeof(real_t));

  // Evaluate factor
  if (factor_eval(factor, params, r, NULL) != 0) {
    free(r);
    free(J_numdiff);
    return -2;
  }

  for (int i = 0; i < param_size; i++) {
    real_t *r_fwd = calloc(r_size, sizeof(real_t));
    real_t *r_diff = calloc(r_size, sizeof(real_t));

    quat_perturb(params[param_idx], i, step_size);
    factor_eval(factor, params, r_fwd, NULL);
    quat_perturb(params[param_idx], i, -step_size);

    vec_sub(r_fwd, r, r_diff, r_size);
    vec_scale(r_diff, r_size, 1.0 / step_size);
    mat_col_set(J_numdiff, param_size, r_size, i, r_diff);

    free(r_fwd);
    free(r_diff);
  }

  // Check Jacobian
  const int retval = check_jacobian(J_name,
                                    J_numdiff,
                                    jacobians[param_idx],
                                    r_size,
                                    param_size,
                                    tol,
                                    verbose);
  free(r);
  free(J_numdiff);

  return retval;
}

/////////////////
// POSE FACTOR //
/////////////////

/**
 * Setup pose factor
 */
void pose_factor_setup(pose_factor_t *factor,
                       real_t *pose,
                       const real_t var[6]) {
  assert(factor != NULL);
  assert(pose != NULL);
  assert(var != NULL);

  // Parameters
  factor->pose_est = pose;

  // Measurement
  factor->pos_meas[0] = pose[0];
  factor->pos_meas[1] = pose[1];
  factor->pos_meas[2] = pose[2];
  factor->quat_meas[0] = pose[3];
  factor->quat_meas[1] = pose[4];
  factor->quat_meas[2] = pose[5];
  factor->quat_meas[3] = pose[6];

  // Measurement covariance matrix
  zeros(factor->covar, 6, 6);
  factor->covar[0] = var[0];
  factor->covar[7] = var[1];
  factor->covar[14] = var[2];
  factor->covar[21] = var[3];
  factor->covar[28] = var[4];
  factor->covar[35] = var[5];

  // Square root information matrix
  zeros(factor->sqrt_info, 6, 6);
  factor->sqrt_info[0] = sqrt(1.0 / factor->covar[0]);
  factor->sqrt_info[7] = sqrt(1.0 / factor->covar[7]);
  factor->sqrt_info[14] = sqrt(1.0 / factor->covar[14]);
  factor->sqrt_info[21] = sqrt(1.0 / factor->covar[21]);
  factor->sqrt_info[28] = sqrt(1.0 / factor->covar[28]);
  factor->sqrt_info[35] = sqrt(1.0 / factor->covar[35]);

  // Factor residuals, parameters and Jacobians
  factor->r_size = 6;
  factor->num_params = 1;
  factor->param_types[0] = POSE_PARAM;
  factor->params[0] = factor->pose_est;
  factor->jacs[0] = factor->J_pose;
}

/**
 * Evaluate pose factor
 * @returns `0` for success, `-1` for failure
 */
int pose_factor_eval(void *factor_ptr) {
  assert(factor_ptr != NULL);
  pose_factor_t *factor = (pose_factor_t *) factor_ptr;

  // Map params
  const real_t r_est[3] = {factor->params[0][0],
                           factor->params[0][1],
                           factor->params[0][2]};
  const real_t q_est[4] = {factor->params[0][3],
                           factor->params[0][4],
                           factor->params[0][5],
                           factor->params[0][6]};
  const real_t *r_meas = factor->pos_meas;
  const real_t *q_meas = factor->quat_meas;

  // Calculate pose error
  // -- Translation error
  // dr = r_meas - r_est;
  real_t dr[3] = {0};
  dr[0] = r_meas[0] - r_est[0];
  dr[1] = r_meas[1] - r_est[1];
  dr[2] = r_meas[2] - r_est[2];

  // -- Rotation error
  // dq = quat_mul(quat_inv(q_meas), q_est);
  real_t dq[4] = {0};
  real_t q_meas_inv[4] = {0};
  quat_inv(q_meas, q_meas_inv);
  quat_mul(q_meas_inv, q_est, dq);

  // dtheta = 2 * dq;
  real_t dtheta[3] = {0};
  dtheta[0] = 2 * dq[1];
  dtheta[1] = 2 * dq[2];
  dtheta[2] = 2 * dq[3];

  // -- Set residuals
  // r = factor.sqrt_info * [dr; dtheta];
  real_t r[6] = {0};
  r[0] = dr[0];
  r[1] = dr[1];
  r[2] = dr[2];
  r[3] = dtheta[0];
  r[4] = dtheta[1];
  r[5] = dtheta[2];
  dot(factor->sqrt_info, 6, 6, r, 6, 1, factor->r);

  // Calculate Jacobians
  const real_t dqw = dq[0];
  const real_t dqx = dq[1];
  const real_t dqy = dq[2];
  const real_t dqz = dq[3];

  real_t J[6 * 6] = {0};

  J[0] = -1.0;
  J[1] = 0.0;
  J[2] = 0.0;
  J[6] = 0.0;
  J[7] = -1.0;
  J[8] = 0.0;
  J[12] = 0.0;
  J[13] = 0.0;
  J[14] = -1.0;

  J[21] = dqw;
  J[22] = -dqz;
  J[23] = dqy;
  J[27] = dqz;
  J[28] = dqw;
  J[29] = -dqx;
  J[33] = -dqy;
  J[34] = dqx;
  J[35] = dqw;

  dot(factor->sqrt_info, 6, 6, J, 6, 6, factor->jacs[0]);

  return 0;
}

///////////////
// BA FACTOR //
///////////////

/**
 * Setup bundle adjustment factor
 */
void ba_factor_setup(ba_factor_t *factor,
                     real_t *pose,
                     real_t *feature,
                     camera_t *camera,
                     const real_t z[2],
                     const real_t var[2]) {
  assert(factor != NULL);
  assert(pose != NULL);
  assert(feature != NULL);
  assert(camera != NULL);
  assert(var != NULL);

  // Parameters
  factor->pose = pose;
  factor->feature = feature;
  factor->camera = camera;
  factor->num_params = 3;

  // Measurement covariance
  factor->covar[0] = var[0];
  factor->covar[1] = 0.0;
  factor->covar[2] = 0.0;
  factor->covar[3] = var[1];

  // Square-root information matrix
  factor->sqrt_info[0] = sqrt(1.0 / factor->covar[0]);
  factor->sqrt_info[1] = 0.0;
  factor->sqrt_info[2] = 0.0;
  factor->sqrt_info[3] = sqrt(1.0 / factor->covar[3]);

  // Measurement
  factor->z[0] = z[0];
  factor->z[1] = z[1];

  // Factor parameters, residuals and Jacobians
  factor->r_size = 2;
  factor->num_params = 3;

  factor->param_types[0] = POSE_PARAM;
  factor->param_types[1] = FEATURE_PARAM;
  factor->param_types[2] = CAMERA_PARAM;

  factor->params[0] = factor->pose;
  factor->params[1] = factor->feature;
  factor->params[2] = factor->camera->data;

  factor->jacs[0] = factor->J_pose;
  factor->jacs[1] = factor->J_feature;
  factor->jacs[2] = factor->J_camera;
}

/**
 * Camera pose jacobian
 */
static void ba_factor_pose_jacobian(const real_t Jh_weighted[2 * 3],
                                    const real_t T_WC[4 * 4],
                                    const real_t p_W[3],
                                    real_t *J) {
  // Pre-check
  if (J == NULL) {
    return;
  }

  // Jh_weighted = -1 * sqrt_info * Jh;
  // J_pos = Jh_weighted * -C_CW;
  // J_rot = Jh_weighted * -C_CW * hat(p_W - r_WC) * -C_WC;
  // J = [J_pos, J_rot]

  // Setup
  real_t C_WC[3 * 3] = {0};
  real_t C_CW[3 * 3] = {0};
  real_t r_WC[3] = {0};
  tf_rot_get(T_WC, C_WC);
  tf_trans_get(T_WC, r_WC);
  mat_transpose(C_WC, 3, 3, C_CW);

  // J_pos = -1 * sqrt_info * Jh * -C_CW;
  real_t J_pos[2 * 3] = {0};
  real_t neg_C_CW[3 * 3] = {0};
  mat_copy(C_CW, 3, 3, neg_C_CW);
  mat_scale(neg_C_CW, 3, 3, -1.0);
  dot(Jh_weighted, 2, 3, neg_C_CW, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  /**
   * Jh_weighted = -1 * sqrt_info * Jh;
   * J_rot = Jh_weighted * -C_CW * hat(p_W - r_WC) * -C_WC;
   * where:
   *
   *   A = -C_CW;
   *   B = hat(p_W - r_WC);
   *   C = -C_WC;
   */
  real_t J_rot[2 * 3] = {0};
  real_t A[3 * 3] = {0};
  mat_copy(neg_C_CW, 3, 3, A);

  real_t B[3 * 3] = {0};
  real_t dp[3] = {0};
  dp[0] = p_W[0] - r_WC[0];
  dp[1] = p_W[1] - r_WC[1];
  dp[2] = p_W[2] - r_WC[2];
  hat(dp, B);

  real_t C[3 * 3] = {0};
  mat_copy(C_WC, 3, 3, C);
  mat_scale(C, 3, 3, -1.0);

  real_t AB[3 * 3] = {0};
  real_t ABC[3 * 3] = {0};
  dot(A, 3, 3, B, 3, 3, AB);
  dot(AB, 3, 3, C, 3, 3, ABC);
  dot(Jh_weighted, 2, 3, ABC, 3, 3, J_rot);

  J[3] = J_rot[0];
  J[4] = J_rot[1];
  J[5] = J_rot[2];

  J[9] = J_rot[3];
  J[10] = J_rot[4];
  J[11] = J_rot[5];
}

/**
 * Feature jacobian
 */
static void ba_factor_feature_jacobian(const real_t Jh_weighted[2 * 3],
                                       const real_t T_WC[4 * 4],
                                       real_t *J) {
  // Pre-check
  if (J == NULL) {
    return;
  }

  // Jh_weighted = -1 * sqrt_info * Jh;
  // J = Jh_weighted * C_CW;
  real_t C_WC[3 * 3] = {0};
  real_t C_CW[3 * 3] = {0};
  tf_rot_get(T_WC, C_WC);
  mat_transpose(C_WC, 3, 3, C_CW);
  dot(Jh_weighted, 2, 3, C_CW, 3, 3, J);
}

/**
 * Camera parameters jacobian
 */
static void ba_factor_camera_jacobian(const real_t neg_sqrt_info[2 * 2],
                                      const real_t J_cam_params[2 * 8],
                                      real_t *J) {
  // Pre-check
  if (J == NULL) {
    return;
  }

  // J = -1 * sqrt_info * J_cam_params;
  dot(neg_sqrt_info, 2, 2, J_cam_params, 2, 8, J);
}

/**
 * Evaluate bundle adjustment factor
 * @returns `0` for success, `-1` for failure
 */
int ba_factor_eval(void *factor_ptr) {
  assert(factor_ptr != NULL);
  ba_factor_t *factor = (ba_factor_t *) factor_ptr;

  // Map params
  real_t T_WCi[4 * 4] = {0};
  tf(factor->params[0], T_WCi);
  const real_t *p_W = factor->params[1];
  const real_t *cam_params = factor->params[2];

  // Calculate residuals
  // -- Project point from world to image plane
  real_t T_CiW[4 * 4] = {0};
  real_t p_Ci[3] = {0};
  real_t z_hat[2];
  tf_inv(T_WCi, T_CiW);
  tf_point(T_CiW, p_W, p_Ci);
  camera_project(factor->camera, p_Ci, z_hat);
  // -- Residual
  real_t r[2] = {0};
  r[0] = factor->z[0] - z_hat[0];
  r[1] = factor->z[1] - z_hat[1];
  // -- Weighted residual
  dot(factor->sqrt_info, 2, 2, r, 2, 1, factor->r);

  // Calculate jacobians
  // -- Form: -1 * sqrt_info
  real_t neg_sqrt_info[2 * 2] = {0};
  mat_copy(factor->sqrt_info, 2, 2, neg_sqrt_info);
  mat_scale(neg_sqrt_info, 2, 2, -1.0);
  // -- Form: Jh_weighted = -1 * sqrt_info * Jh
  real_t Jh[2 * 3] = {0};
  real_t Jh_w[2 * 3] = {0};
  pinhole_radtan4_project_jacobian(cam_params, p_Ci, Jh);
  dot(neg_sqrt_info, 2, 2, Jh, 2, 3, Jh_w);
  // -- Form: J_cam_params
  real_t J_cam_params[2 * 8] = {0};
  pinhole_radtan4_params_jacobian(cam_params, p_Ci, J_cam_params);
  // -- Fill jacobians
  ba_factor_pose_jacobian(Jh_w, T_WCi, p_W, factor->jacs[0]);
  ba_factor_feature_jacobian(Jh_w, T_WCi, factor->jacs[1]);
  ba_factor_camera_jacobian(neg_sqrt_info, J_cam_params, factor->jacs[2]);

  return 0;
}

///////////////////
// CAMERA FACTOR //
///////////////////

/**
 * Setup camera factor
 */
void camera_factor_setup(camera_factor_t *factor,
                         real_t *pose,
                         real_t *extrinsic,
                         real_t *feature,
                         camera_t *camera,
                         const real_t z[2],
                         const real_t var[2]) {
  assert(factor != NULL);
  assert(pose != NULL);
  assert(extrinsic != NULL);
  assert(feature != NULL);
  assert(camera != NULL);
  assert(z != NULL);
  assert(var != NULL);

  // Parameters
  factor->pose = pose;
  factor->extrinsic = extrinsic;
  factor->feature = feature;
  factor->camera = camera;

  // Measurement covariance matrix
  factor->covar[0] = var[0];
  factor->covar[1] = 0.0;
  factor->covar[2] = 0.0;
  factor->covar[3] = var[1];

  // Square-root information matrix
  factor->sqrt_info[0] = sqrt(1.0 / factor->covar[0]);
  factor->sqrt_info[1] = 0.0;
  factor->sqrt_info[2] = 0.0;
  factor->sqrt_info[3] = sqrt(1.0 / factor->covar[3]);

  // Measurement
  factor->z[0] = z[0];
  factor->z[1] = z[1];

  // Parameters, residuals, jacobians
  factor->r_size = 2;
  factor->num_params = 4;

  factor->param_types[0] = POSE_PARAM;
  factor->param_types[1] = EXTRINSIC_PARAM;
  factor->param_types[2] = FEATURE_PARAM;
  factor->param_types[3] = CAMERA_PARAM;

  factor->params[0] = factor->pose;
  factor->params[1] = factor->extrinsic;
  factor->params[2] = factor->feature;
  factor->params[3] = factor->camera->data;

  factor->jacs[0] = factor->J_pose;
  factor->jacs[1] = factor->J_extrinsic;
  factor->jacs[2] = factor->J_feature;
  factor->jacs[3] = factor->J_camera;
}

/**
 * Pose jacobian
 */
static void camera_factor_pose_jacobian(const real_t Jh_w[2 * 3],
                                        const real_t T_WB[4 * 4],
                                        const real_t T_BC[4 * 4],
                                        const real_t p_W[3],
                                        real_t J[2 * 6]) {
  assert(Jh_w != NULL);
  assert(T_BC != NULL);
  assert(T_WB != NULL);
  assert(p_W != NULL);
  assert(J != NULL);

  // Jh_w = -1 * sqrt_info * Jh;
  // J_pos = Jh_w * C_CB * -C_BW;
  // J_rot = Jh_w * C_CB * C_BW * hat(p_W - r_WB) * -C_WB;
  // J = [J_pos, J_rot];

  // Setup
  real_t C_BW[3 * 3] = {0};
  real_t C_CB[3 * 3] = {0};
  real_t C_CW[3 * 3] = {0};

  TF_ROT(T_WB, C_WB);
  TF_ROT(T_BC, C_BC);
  mat_transpose(C_WB, 3, 3, C_BW);
  mat_transpose(C_BC, 3, 3, C_CB);
  dot(C_CB, 3, 3, C_BW, 3, 3, C_CW);

  // Form: -C_BW
  real_t neg_C_BW[3 * 3] = {0};
  mat_copy(C_BW, 3, 3, neg_C_BW);
  mat_scale(neg_C_BW, 3, 3, -1.0);

  // Form: -C_CW
  real_t neg_C_CW[3 * 3] = {0};
  dot(C_CB, 3, 3, neg_C_BW, 3, 3, neg_C_CW);

  // Form: -C_WB
  real_t neg_C_WB[3 * 3] = {0};
  mat_copy(C_WB, 3, 3, neg_C_WB);
  mat_scale(neg_C_WB, 3, 3, -1.0);

  // Form: C_CB * -C_BW * hat(p_W - r_WB) * -C_WB
  real_t p[3] = {0};
  real_t S[3 * 3] = {0};
  TF_TRANS(T_WB, r_WB);
  vec_sub(p_W, r_WB, p, 3);
  hat(p, S);

  real_t A[3 * 3] = {0};
  real_t B[3 * 3] = {0};
  dot(neg_C_CW, 3, 3, S, 3, 3, A);
  dot(A, 3, 3, neg_C_WB, 3, 3, B);

  // Form: J_pos = Jh_w * C_CB * -C_BW;
  real_t J_pos[2 * 3] = {0};
  dot(Jh_w, 2, 3, neg_C_CW, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // Form: J_rot = Jh_w * C_CB * -C_BW * hat(p_W - r_WB) * -C_WB;
  real_t J_rot[2 * 3] = {0};
  dot(Jh_w, 2, 3, B, 3, 3, J_rot);

  J[3] = J_rot[0];
  J[4] = J_rot[1];
  J[5] = J_rot[2];

  J[9] = J_rot[3];
  J[10] = J_rot[4];
  J[11] = J_rot[5];
}

/**
 * Body-camera extrinsic jacobian
 */
static void camera_factor_extrinsic_jacobian(const real_t Jh_w[2 * 3],
                                             const real_t T_BC[4 * 4],
                                             const real_t p_C[3],
                                             real_t J[2 * 6]) {
  assert(Jh_w != NULL);
  assert(T_BC != NULL);
  assert(p_C != NULL);
  assert(J != NULL);

  // Jh_w = -1 * sqrt_info * Jh;
  // J_pos = Jh_w * -C_CB;
  // J_rot = Jh_w * C_CB * hat(C_BC * p_C);

  // Setup
  real_t C_BC[3 * 3] = {0};
  real_t C_CB[3 * 3] = {0};
  real_t C_BW[3 * 3] = {0};
  real_t C_CW[3 * 3] = {0};

  tf_rot_get(T_BC, C_BC);
  mat_transpose(C_BC, 3, 3, C_CB);
  dot(C_CB, 3, 3, C_BW, 3, 3, C_CW);

  // Form: -C_CB
  real_t neg_C_CB[3 * 3] = {0};
  mat_copy(C_CB, 3, 3, neg_C_CB);
  mat_scale(neg_C_CB, 3, 3, -1.0);

  // Form: -C_BC
  real_t neg_C_BC[3 * 3] = {0};
  mat_copy(C_BC, 3, 3, neg_C_BC);
  mat_scale(neg_C_BC, 3, 3, -1.0);

  // Form: -C_CB * hat(C_BC * p_C) * -C_BC
  real_t p[3] = {0};
  real_t S[3 * 3] = {0};
  dot(C_BC, 3, 3, p_C, 3, 1, p);
  hat(p, S);

  real_t A[3 * 3] = {0};
  real_t B[3 * 3] = {0};
  dot(neg_C_CB, 3, 3, S, 3, 3, A);
  dot(A, 3, 3, neg_C_BC, 3, 3, B);

  // Form: J_rot = Jh_w * -C_CB;
  real_t J_pos[2 * 3] = {0};
  dot(Jh_w, 2, 3, neg_C_CB, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // Form: J_rot = Jh_w * -C_CB * hat(C_BC * p_C) * -C_BC;
  real_t J_rot[2 * 3] = {0};
  dot(Jh_w, 2, 3, B, 3, 3, J_rot);

  J[3] = J_rot[0];
  J[4] = J_rot[1];
  J[5] = J_rot[2];

  J[9] = J_rot[3];
  J[10] = J_rot[4];
  J[11] = J_rot[5];
}

/**
 * Camera parameters jacobian
 */
static void camera_factor_camera_jacobian(const real_t neg_sqrt_info[2 * 2],
                                          const real_t J_cam_params[2 * 8],
                                          real_t J[2 * 8]) {
  assert(neg_sqrt_info != NULL);
  assert(J_cam_params != NULL);
  assert(J != NULL);

  // J = -1 * sqrt_info * J_cam_params;
  dot(neg_sqrt_info, 2, 2, J_cam_params, 2, 8, J);
}

/**
 * Feature jacobian
 */
static void camera_factor_feature_jacobian(const real_t Jh_w[2 * 3],
                                           const real_t T_WB[4 * 4],
                                           const real_t T_BC[4 * 4],
                                           real_t J[2 * 3]) {
  if (J == NULL) {
    return;
  }
  assert(Jh_w != NULL);
  assert(T_WB != NULL);
  assert(T_BC != NULL);
  assert(J != NULL);

  // Jh_w = -1 * sqrt_info * Jh;
  // J = Jh_w * C_CW;

  // Setup
  real_t T_WC[4 * 4] = {0};
  real_t C_WC[3 * 3] = {0};
  real_t C_CW[3 * 3] = {0};
  dot(T_WB, 4, 4, T_BC, 4, 4, T_WC);
  tf_rot_get(T_WC, C_WC);
  mat_transpose(C_WC, 3, 3, C_CW);

  // Form: J = -1 * sqrt_info * Jh * C_CW;
  dot(Jh_w, 2, 3, C_CW, 3, 3, J);
}

/**
 * Evaluate vision factor
 * @returns `0` for success, `-1` for failure
 */
int camera_factor_eval(void *factor_ptr) {
  camera_factor_t *factor = (camera_factor_t *) factor_ptr;
  assert(factor != NULL);
  assert(factor->pose);
  assert(factor->extrinsic);
  assert(factor->feature);
  assert(factor->camera);

  // Map params
  real_t T_WB[4 * 4] = {0};
  real_t T_BCi[4 * 4] = {0};
  tf(factor->params[0], T_WB);
  tf(factor->params[1], T_BCi);
  const real_t *p_W = factor->params[2];
  const real_t *cam_params = factor->params[3];

  // Form camera pose
  TF_CHAIN(T_WCi, 2, T_WB, T_BCi);
  TF_INV(T_WCi, T_CiW);

  // Transform feature from world to camera frame
  TF_POINT(T_CiW, p_W, p_Ci);

  // Calculate residuals
  // -- Project point from world to image plane
  real_t z_hat[2];
  camera_project(factor->camera, p_Ci, z_hat);
  // -- Residual
  real_t r[2] = {0};
  r[0] = factor->z[0] - z_hat[0];
  r[1] = factor->z[1] - z_hat[1];
  // -- Weighted residual
  dot(factor->sqrt_info, 2, 2, r, 2, 1, factor->r);

  // Calculate jacobians
  // -- Form: -1 * sqrt_info
  real_t neg_sqrt_info[2 * 2] = {0};
  mat_copy(factor->sqrt_info, 2, 2, neg_sqrt_info);
  mat_scale(neg_sqrt_info, 2, 2, -1.0);
  // -- Form: Jh_ = -1 * sqrt_info * Jh
  real_t Jh[2 * 3] = {0};
  real_t Jh_[2 * 3] = {0};
  pinhole_radtan4_project_jacobian(cam_params, p_Ci, Jh);
  dot(neg_sqrt_info, 2, 2, Jh, 2, 3, Jh_);
  // -- Form: J_cam_params
  real_t J_cam_params[2 * 8] = {0};
  pinhole_radtan4_params_jacobian(cam_params, p_Ci, J_cam_params);
  // -- Fill Jacobians
  camera_factor_pose_jacobian(Jh_, T_WB, T_BCi, p_W, factor->jacs[0]);
  camera_factor_extrinsic_jacobian(Jh_, T_BCi, p_Ci, factor->jacs[1]);
  camera_factor_feature_jacobian(Jh_, T_WB, T_BCi, factor->jacs[2]);
  camera_factor_camera_jacobian(neg_sqrt_info, J_cam_params, factor->jacs[3]);

  return 0;
}

////////////////
// IMU FACTOR //
////////////////

/**
 * Form IMU state vector
 */
void imu_state_vector(const real_t r[3],
                      const real_t q[4],
                      const real_t v[3],
                      const real_t ba[3],
                      const real_t bg[3],
                      real_t x[16]) {
  assert(r != NULL);
  assert(q != NULL);
  assert(v != NULL);
  assert(ba != NULL);
  assert(bg != NULL);
  assert(x != NULL);

  x[0] = r[0];
  x[1] = r[1];
  x[2] = r[2];

  x[3] = q[0];
  x[4] = q[1];
  x[5] = q[2];
  x[6] = q[3];

  x[7] = v[0];
  x[8] = v[1];
  x[9] = v[2];

  x[10] = ba[0];
  x[11] = ba[1];
  x[12] = ba[2];

  x[13] = bg[0];
  x[14] = bg[1];
  x[15] = bg[2];
}

/**
 * Propagate IMU measurement
 */
void imu_propagate(const real_t pose_k[7],
                   const real_t vel_k[3],
                   const imu_buffer_t *imu_buf,
                   real_t pose_kp1[7],
                   real_t vel_kp1[3]) {
  // Initialize state
  real_t r[3] = {pose_k[0], pose_k[1], pose_k[2]};
  real_t v[3] = {vel_k[0], vel_k[1], vel_k[2]};
  real_t q[4] = {pose_k[3], pose_k[4], pose_k[5], pose_k[6]};
  const real_t g[3] = {0.0, 0.0, -9.81};

  real_t dt = 0.0;
  for (int k = 0; k < imu_buf->size; k++) {
    // Calculate dt
    if ((k + 1) < imu_buf->size) {
      timestamp_t ts_k = imu_buf->ts[k];
      timestamp_t ts_kp1 = imu_buf->ts[k + 1];
      dt = ts2sec(ts_kp1) - ts2sec(ts_k);
    }

    // Map out accelerometer and gyroscope measurements
    const real_t *a = imu_buf->acc[k];
    const real_t *w = imu_buf->gyr[k];

    // Precompute:
    // acc = (q * a * q_conj) + g
    //     = (C * a) + g
    real_t acc[3] = {0};
    quat_transform(q, a, acc);
    acc[0] += g[0];
    acc[1] += g[1];
    acc[2] += g[2];

    // Update position:
    // r = r + (v * dt) + (0.5 * ((C * a) + g) * dt_sq);
    r[0] += (v[0] * dt) + (0.5 * acc[0] * dt * dt);
    r[1] += (v[1] * dt) + (0.5 * acc[1] * dt * dt);
    r[2] += (v[2] * dt) + (0.5 * acc[2] * dt * dt);

    // Update velocity
    // v = v + (C * a + g) * dt;
    v[0] += acc[0] * dt;
    v[1] += acc[1] * dt;
    v[2] += acc[2] * dt;

    // Update rotation
    quat_update_dt(q, w, dt);
    quat_normalize(q);
  }

  // Map results
  pose_kp1[0] = r[0];
  pose_kp1[1] = r[1];
  pose_kp1[2] = r[2];
  pose_kp1[3] = q[0];
  pose_kp1[4] = q[1];
  pose_kp1[5] = q[2];
  pose_kp1[6] = q[3];

  vel_kp1[0] = v[0];
  vel_kp1[1] = v[1];
  vel_kp1[2] = v[2];
}

/**
 * Initialize roll and pitch with accelerometer measurements.
 */
void imu_initial_attitude(const imu_buffer_t *imu_buf, real_t q_WS[4]) {
  // Get mean accelerometer measurements
  real_t ax = 0.0;
  real_t ay = 0.0;
  real_t az = 0.0;
  for (size_t k = 0; k < imu_buf->size; k++) {
    ax += imu_buf->acc[k][0];
    ay += imu_buf->acc[k][1];
    az += imu_buf->acc[k][2];
  }
  ax /= imu_buf->size;
  ay /= imu_buf->size;
  az /= imu_buf->size;

  // Initialize orientation
  const real_t ypr[3] = {0.0,
                         atan2(-ax, sqrt(ay * ay + az * az)),
                         atan2(ay, az)};
  euler2quat(ypr, q_WS);

  // const real_t a[3] = {ax, ay, az};
  // const real_t g[3] = {0.0, 0.0, 9.81};
  // real_t C[3 * 3] = {0};
  // real_t q[4] = {0};
  // real_t euler[3] = {0};
  // vecs2rot(a, g, C);
  // rot2quat(C, q);
  // quat2euler(q, euler);
  // print_vector("euler", euler, 3);
  // print_vector("ypr", ypr, 3);
  // exit(0);
}

/**
 * Propagate IMU measurement
 */
void imu_factor_propagate_step(imu_factor_t *factor,
                               const real_t a_i[3],
                               const real_t w_i[3],
                               const real_t a_j[3],
                               const real_t w_j[3],
                               const real_t dt) {
  assert(factor != NULL);
  assert(a_i != NULL);
  assert(w_i != NULL);
  assert(a_j != NULL);
  assert(w_j != NULL);
  assert(dt > 0.0);

  // Setup
  const real_t dt_sq = dt * dt;
  const real_t *r_i = factor->dr;
  const real_t *v_i = factor->dv;
  const real_t *q_i = factor->dq;
  const real_t *ba_i = factor->ba;
  const real_t *bg_i = factor->bg;

  // Gyroscope measurement
  const real_t wx = 0.5 * (w_i[0] + w_j[0]) - bg_i[0];
  const real_t wy = 0.5 * (w_i[1] + w_j[1]) - bg_i[1];
  const real_t wz = 0.5 * (w_i[2] + w_j[2]) - bg_i[2];
  const real_t dq[4] = {1.0, 0.5 * wx * dt, 0.5 * wy * dt, 0.5 * wz * dt};

  // Update orientation
  real_t q_j[4] = {0};
  quat_mul(q_i, dq, q_j);
  quat_normalize(q_j);

  // Accelerometer measurement
  const real_t a_ii[3] = {a_i[0] - ba_i[0], a_i[1] - ba_i[1], a_i[2] - ba_i[2]};
  const real_t a_jj[3] = {a_j[0] - ba_i[0], a_j[1] - ba_i[1], a_j[2] - ba_i[2]};
  real_t acc_i[3] = {0};
  real_t acc_j[3] = {0};
  quat_transform(q_i, a_ii, acc_i);
  quat_transform(q_j, a_jj, acc_j);
  real_t a[3] = {0};
  a[0] = 0.5 * (acc_i[0] + acc_j[0]);
  a[1] = 0.5 * (acc_i[1] + acc_j[1]);
  a[2] = 0.5 * (acc_i[2] + acc_j[2]);

  // Update position:
  // r_j = r_i + (v_i * dt) + (0.5 * a * dt_sq)
  real_t r_j[3] = {0};
  r_j[0] = r_i[0] + (v_i[0] * dt) + (0.5 * a[0] * dt_sq);
  r_j[1] = r_i[1] + (v_i[1] * dt) + (0.5 * a[1] * dt_sq);
  r_j[2] = r_i[2] + (v_i[2] * dt) + (0.5 * a[2] * dt_sq);

  // Update velocity:
  // v_j = v_i + a * dt
  real_t v_j[3] = {0};
  v_j[0] = v_i[0] + a[0] * dt;
  v_j[1] = v_i[1] + a[1] * dt;
  v_j[2] = v_i[2] + a[2] * dt;

  // Update biases
  // ba_j = ba_i;
  // bg_j = bg_i;
  real_t ba_j[3] = {0};
  real_t bg_j[3] = {0};
  vec_copy(ba_i, 3, ba_j);
  vec_copy(bg_i, 3, bg_j);

  // Write outputs
  vec_copy(r_j, 3, factor->r_j);
  vec_copy(v_j, 3, factor->v_j);
  vec_copy(q_j, 4, factor->q_j);
  vec_copy(ba_j, 3, factor->ba_j);
  vec_copy(bg_j, 3, factor->bg_j);

  vec_copy(r_j, 3, factor->dr);
  vec_copy(v_j, 3, factor->dv);
  vec_copy(q_j, 4, factor->dq);
  vec_copy(ba_j, 3, factor->ba);
  vec_copy(bg_j, 3, factor->bg);
}

/**
 * Form IMU Noise Matrix Q
 */
static void imu_factor_form_Q_matrix(const imu_params_t *imu_params,
                                     real_t Q[18 * 18]) {
  assert(imu_params != NULL);
  assert(Q != NULL);

  const real_t sigma_a_sq = imu_params->sigma_a * imu_params->sigma_a;
  const real_t sigma_g_sq = imu_params->sigma_g * imu_params->sigma_g;
  const real_t sigma_ba_sq = imu_params->sigma_aw * imu_params->sigma_aw;
  const real_t sigma_bg_sq = imu_params->sigma_gw * imu_params->sigma_gw;

  real_t q[18] = {0};
  q[0] = sigma_a_sq;
  q[1] = sigma_a_sq;
  q[2] = sigma_a_sq;

  q[3] = sigma_g_sq;
  q[4] = sigma_g_sq;
  q[5] = sigma_g_sq;

  q[6] = sigma_a_sq;
  q[7] = sigma_a_sq;
  q[8] = sigma_a_sq;

  q[9] = sigma_g_sq;
  q[10] = sigma_g_sq;
  q[11] = sigma_g_sq;

  q[12] = sigma_ba_sq;
  q[13] = sigma_ba_sq;
  q[14] = sigma_ba_sq;

  q[15] = sigma_bg_sq;
  q[16] = sigma_bg_sq;
  q[17] = sigma_bg_sq;

  zeros(Q, 18, 18);
  mat_diag_set(Q, 18, 18, q);
}

// F11 = eye(3)
#define IMU_FACTOR_F11(void)                                                   \
  real_t F11[3 * 3] = {0};                                                     \
  eye(F11, 3, 3);

// F12 = -0.25 * dC_i @ acc_i_x * dt_sq
// F12 += -0.25 * dC_j @ acc_j_x @ (eye(3) - gyr_x * dt) * dt_sq
#define IMU_FACTOR_F12(dCi_acc_i_x, dCj_acc_j_x, I_m_gyr_x_dt, dt_sq)          \
  real_t F12_A[3 * 3] = {0};                                                   \
  mat_copy(dCi_acc_i_x, 3, 3, F12_A);                                          \
  mat_scale(F12_A, 3, 3, -0.25);                                               \
  mat_scale(F12_A, 3, 3, dt_sq);                                               \
                                                                               \
  real_t F12_B[3 * 3] = {0};                                                   \
  mat_copy(dCj_acc_j_x, 3, 3, F12_B);                                          \
  mat_scale(F12_B, 3, 3, -0.25);                                               \
                                                                               \
  real_t F12_C[3 * 3] = {0};                                                   \
  mat_copy(I_m_gyr_x_dt, 3, 3, F12_C);                                         \
  mat_scale(F12_C, 3, 3, dt_sq);                                               \
                                                                               \
  real_t F12_D[3 * 3] = {0};                                                   \
  dot(F12_B, 3, 3, F12_C, 3, 3, F12_D);                                        \
                                                                               \
  real_t F12[3 * 3] = {0};                                                     \
  mat_add(F12_A, F12_D, F12, 3, 3);

// F13 = eye(3) * dt
#define IMU_FACTOR_F13(dt)                                                     \
  real_t F13[3 * 3] = {0};                                                     \
  eye(F13, 3, 3);                                                              \
  mat_scale(F13, 3, 3, dt);

// F14 = -0.25 * (dC_i + dC_j) * dt_sq
#define IMU_FACTOR_F14(dCi_dCj, dt_sq)                                         \
  real_t F14[3 * 3] = {0};                                                     \
  mat_copy(dCi_dCj, 3, 3, F14);                                                \
  mat_scale(F14, 3, 3, -0.25);                                                 \
  mat_scale(F14, 3, 3, dt_sq);

// F15 = 0.25 * -dC_j @ acc_j_x * dt_sq * -dt
#define IMU_FACTOR_F15(dCj_acc_j_x, dt, dt_sq)                                 \
  real_t F15[3 * 3] = {0};                                                     \
  mat_copy(dCj_acc_j_x, 3, 3, F15);                                            \
  mat_scale(F15, 3, 3, -1.0 * 0.25 * dt_sq * -dt);

// F22 = eye(3) - gyr_x * dt
#define IMU_FACTOR_F22(I_m_gyr_x_dt)                                           \
  real_t F22[3 * 3] = {0};                                                     \
  mat_copy(I_m_gyr_x_dt, 3, 3, F22);

// F25 = -eye(3) * dt
#define IMU_FACTOR_F25(dt)                                                     \
  real_t F25[3 * 3] = {0};                                                     \
  F25[0] = -dt;                                                                \
  F25[4] = -dt;                                                                \
  F25[8] = -dt;

// F32 = -0.5 * dC_i @ acc_i_x * dt
// F32 += -0.5 * dC_j @ acc_j_x @ (eye(3) - gyr_x * dt)* dt
#define IMU_FACTOR_F32(dCi_acc_i_x, dCj_acc_j_x, I_m_gyr_x_dt, dt)             \
  real_t F32_A[3 * 3] = {0};                                                   \
  mat_copy(dCi_acc_i_x, 3, 3, F32_A);                                          \
  for (int i = 0; i < 9; i++) {                                                \
    F32_A[i] = -0.5 * F32_A[i] * dt;                                           \
  }                                                                            \
                                                                               \
  real_t F32_B[3 * 3] = {0};                                                   \
  dot(dCj_acc_j_x, 3, 3, I_m_gyr_x_dt, 3, 3, F32_B);                           \
  for (int i = 0; i < 9; i++) {                                                \
    F32_B[i] = -0.5 * F32_B[i] * dt;                                           \
  }                                                                            \
                                                                               \
  real_t F32[3 * 3] = {0};                                                     \
  mat_add(F32_A, F32_B, F32, 3, 3);

// F33 = eye(3)
#define IMU_FACTOR_F33(void)                                                   \
  real_t F33[3 * 3] = {0};                                                     \
  F33[0] = 1.0;                                                                \
  F33[4] = 1.0;                                                                \
  F33[8] = 1.0;

// F34 = -0.5 * (dC_i + dC_j) * dt
#define IMU_FACTOR_F34(dC_i, dC_j, dt)                                         \
  real_t F34[3 * 3] = {0};                                                     \
  for (int i = 0; i < 9; i++) {                                                \
    F34[i] = -0.5 * dCi_dCj[i] * dt;                                           \
  }

// F35 = 0.5 * -dC_j @ acc_j_x * dt * -dt
#define IMU_FACTOR_F35(dCj_acc_j_x, dt)                                        \
  real_t F35[3 * 3] = {0};                                                     \
  for (int i = 0; i < 9; i++) {                                                \
    F35[i] = 0.5 * -1.0 * dCj_acc_j_x[i] * dt * -dt;                           \
  }

// F44 = eye(3)
#define IMU_FACTOR_F44(void)                                                   \
  real_t F44[3 * 3] = {0};                                                     \
  F44[0] = 1.0;                                                                \
  F44[4] = 1.0;                                                                \
  F44[8] = 1.0;

// F55 = eye(3)
#define IMU_FACTOR_F55(void)                                                   \
  real_t F55[3 * 3] = {0};                                                     \
  F55[0] = 1.0;                                                                \
  F55[4] = 1.0;                                                                \
  F55[8] = 1.0;

/**
 * Form IMU Transition Matrix F
 */
void imu_factor_F_matrix(const real_t q_i[4],
                         const real_t q_j[4],
                         const real_t ba_i[3],
                         const real_t bg_i[3],
                         const real_t a_i[3],
                         const real_t w_i[3],
                         const real_t a_j[3],
                         const real_t w_j[3],
                         const real_t dt,
                         real_t F_dt[15 * 15]) {
  // Setup
  const real_t dt_sq = dt * dt;

  // gyr_x = hat(0.5 * (imu_buf.gyr[k] + imu_buf.gyr[k + 1]) - bg_i)
  real_t gyr[3] = {0};
  real_t gyr_x[3 * 3] = {0};
  gyr[0] = 0.5 * (w_i[0] + w_j[0]) - bg_i[0];
  gyr[1] = 0.5 * (w_i[1] + w_j[1]) - bg_i[1];
  gyr[2] = 0.5 * (w_i[2] + w_j[2]) - bg_i[2];
  hat(gyr, gyr_x);

  // acc_i_x = hat(imu_buf.acc[k] - ba_i)
  // acc_j_x = hat(imu_buf.acc[k + 1] - ba_i)
  real_t acc_i[3] = {a_i[0] - ba_i[0], a_i[1] - ba_i[1], a_i[2] - ba_i[2]};
  real_t acc_j[3] = {a_j[0] - ba_i[0], a_j[1] - ba_i[1], a_j[2] - ba_i[2]};
  real_t acc_i_x[3 * 3] = {0};
  real_t acc_j_x[3 * 3] = {0};
  hat(acc_i, acc_i_x);
  hat(acc_j, acc_j_x);

  // dC_i = quat2rot(q_i)
  // dC_j = quat2rot(q_j)
  real_t dC_i[3 * 3] = {0};
  real_t dC_j[3 * 3] = {0};
  quat2rot(q_i, dC_i);
  quat2rot(q_j, dC_j);

  // (dC_i + dC_j)
  real_t dCi_dCj[3 * 3] = {0};
  mat_add(dC_i, dC_j, dCi_dCj, 3, 3);

  // dC_i @ acc_i_x
  real_t dCi_acc_i_x[3 * 3] = {0};
  dot(dC_i, 3, 3, acc_i_x, 3, 3, dCi_acc_i_x);

  // dC_j @ acc_j_x
  real_t dCj_acc_j_x[3 * 3] = {0};
  dot(dC_j, 3, 3, acc_j_x, 3, 3, dCj_acc_j_x);

  // (eye(3) - gyr_x * dt)
  real_t I_m_gyr_x_dt[3 * 3] = {0};
  I_m_gyr_x_dt[0] = 1.0 - gyr_x[0] * dt;
  I_m_gyr_x_dt[1] = 0.0 - gyr_x[1] * dt;
  I_m_gyr_x_dt[2] = 0.0 - gyr_x[2] * dt;

  I_m_gyr_x_dt[3] = 0.0 - gyr_x[3] * dt;
  I_m_gyr_x_dt[4] = 1.0 - gyr_x[4] * dt;
  I_m_gyr_x_dt[5] = 0.0 - gyr_x[5] * dt;

  I_m_gyr_x_dt[6] = 0.0 - gyr_x[6] * dt;
  I_m_gyr_x_dt[7] = 0.0 - gyr_x[7] * dt;
  I_m_gyr_x_dt[8] = 1.0 - gyr_x[8] * dt;

  IMU_FACTOR_F11();
  IMU_FACTOR_F12(dCi_acc_i_x, dCj_acc_j_x, I_m_gyr_x_dt, dt_sq);
  IMU_FACTOR_F13(dt);
  IMU_FACTOR_F14(dCi_dCj, dt_sq);
  IMU_FACTOR_F15(dCj_acc_j_x, dt, dt_sq);
  IMU_FACTOR_F22(I_m_gyr_x_dt);
  IMU_FACTOR_F25(dt);
  IMU_FACTOR_F32(dCi_acc_i_x, dCj_acc_j_x, I_m_gyr_x_dt, dt);
  IMU_FACTOR_F33();
  IMU_FACTOR_F34(dC_i, dC_j, dt);
  IMU_FACTOR_F35(dCj_acc_j_x, dt);
  IMU_FACTOR_F44();
  IMU_FACTOR_F55();

  // Fill matrix F
  zeros(F_dt, 15, 15);

  // -- Row block 1
  mat_block_set(F_dt, 15, 0, 2, 0, 2, F11);
  mat_block_set(F_dt, 15, 0, 2, 3, 5, F12);
  mat_block_set(F_dt, 15, 0, 2, 6, 8, F13);
  mat_block_set(F_dt, 15, 0, 2, 9, 11, F14);
  mat_block_set(F_dt, 15, 0, 2, 12, 14, F15);

  // -- Row block 2
  mat_block_set(F_dt, 15, 3, 5, 3, 5, F22);
  mat_block_set(F_dt, 15, 3, 5, 12, 14, F25);

  // -- Row block 3
  mat_block_set(F_dt, 15, 6, 8, 3, 5, F32);
  mat_block_set(F_dt, 15, 6, 8, 6, 8, F33);
  mat_block_set(F_dt, 15, 6, 8, 9, 11, F34);
  mat_block_set(F_dt, 15, 6, 8, 12, 14, F35);

  // -- Row block 4
  mat_block_set(F_dt, 15, 9, 11, 9, 11, F44);

  // -- Row block 5
  mat_block_set(F_dt, 15, 12, 14, 12, 14, F55);
}

/**
 * Form IMU Input Matrix G
 */
void imu_factor_form_G_matrix(const imu_factor_t *factor,
                              const real_t a_i[3],
                              const real_t a_j[3],
                              const real_t dt,
                              real_t G_dt[15 * 18]) {

  // dt_sq = dt * dt
  const real_t dt_sq = dt * dt;

  // dC_i = quat2rot(q_i)
  // dC_j = quat2rot(q_j)
  real_t dC_i[3 * 3] = {0};
  real_t dC_j[3 * 3] = {0};
  quat2rot(factor->q_i, dC_i);
  quat2rot(factor->q_j, dC_j);

  // acc_i_x = hat(imu_buf.acc[k] - ba_i)
  // acc_j_x = hat(imu_buf.acc[k + 1] - ba_i)
  const real_t *ba_i = factor->ba_i;
  real_t acc_i[3] = {a_i[0] - ba_i[0], a_i[1] - ba_i[1], a_i[2] - ba_i[2]};
  real_t acc_j[3] = {a_j[0] - ba_i[0], a_j[1] - ba_i[1], a_j[2] - ba_i[2]};
  real_t acc_i_x[3 * 3] = {0};
  real_t acc_j_x[3 * 3] = {0};
  hat(acc_i, acc_i_x);
  hat(acc_j, acc_j_x);

  // dC_j @ acc_j_x
  real_t dC_j_acc_j_x[3 * 3] = {0};
  dot(dC_j, 3, 3, acc_j_x, 3, 3, dC_j_acc_j_x);

  // G11 = 0.25 * dC_i * dt_sq
  real_t G11[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G11[i] = 0.25 * dC_i[i] * dt_sq;
  }

  // G12 = 0.25 * -dC_j @ acc_j_x * dt_sq * 0.5 * dt
  real_t G12[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G12[i] = 0.25 * -dC_j_acc_j_x[i] * dt_sq * 0.5 * dt;
  }

  // G13 = 0.25 * dC_j @ acc_j_x * dt_sq
  real_t G13[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G13[i] = 0.25 * dC_j_acc_j_x[i] * dt_sq;
  }

  // G14 = 0.25 * -dC_j @ acc_j_x * dt_sq * 0.5 * dt
  real_t G14[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G14[i] = 0.25 * -dC_j_acc_j_x[i] * dt_sq * 0.5 * dt;
  }

  // G22 = eye(3) * dt
  real_t G22[3 * 3] = {0};
  G22[0] = dt;
  G22[4] = dt;
  G22[8] = dt;

  // G24 = eye(3) * dt
  real_t G24[3 * 3] = {0};
  G24[0] = dt;
  G24[4] = dt;
  G24[8] = dt;

  // G31 = 0.5 * dC_i * dt
  real_t G31[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G31[i] = 0.5 * dC_i[i] * dt;
  }

  // G32 = 0.5 * -dC_j @ acc_j_x * dt * 0.5 * dt
  real_t G32[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G32[i] = 0.5 * -dC_j_acc_j_x[i] * dt * 0.5 * dt;
  }

  // G33 = 0.5 * dC_j * dt
  real_t G33[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G33[i] = 0.5 * dC_j[i] * dt;
  }

  // G34 = 0.5 * -dC_j @ acc_j_x * dt * 0.5 * dt
  real_t G34[3 * 3] = {0};
  for (int i = 0; i < 9; i++) {
    G34[i] = 0.5 * -dC_j_acc_j_x[i] * dt * 0.5 * dt;
  }

  // G45 = eye(3) * dt
  real_t G45[3 * 3] = {0};
  G45[0] = dt;
  G45[4] = dt;
  G45[8] = dt;

  // G56 = eye(3) * dt
  real_t G56[3 * 3] = {0};
  G56[0] = dt;
  G56[4] = dt;
  G56[8] = dt;

  // Fill matrix G
  zeros(G_dt, 15, 18);
  mat_block_set(G_dt, 18, 0, 2, 0, 2, G11);
  mat_block_set(G_dt, 18, 0, 2, 3, 5, G12);
  mat_block_set(G_dt, 18, 0, 2, 6, 8, G13);
  mat_block_set(G_dt, 18, 0, 2, 9, 11, G14);
  mat_block_set(G_dt, 18, 3, 5, 3, 5, G22);
  mat_block_set(G_dt, 18, 3, 5, 9, 11, G24);
  mat_block_set(G_dt, 18, 6, 8, 0, 2, G31);
  mat_block_set(G_dt, 18, 6, 8, 3, 5, G32);
  mat_block_set(G_dt, 18, 6, 8, 6, 8, G33);
  mat_block_set(G_dt, 18, 6, 8, 9, 11, G34);
  mat_block_set(G_dt, 18, 9, 11, 12, 14, G45);
  mat_block_set(G_dt, 18, 12, 14, 15, 17, G56);
}

/**
 * IMU Factor setup
 */
void imu_factor_setup(imu_factor_t *factor,
                      const imu_params_t *imu_params,
                      const imu_buffer_t *imu_buf,
                      const timestamp_t ts_i,
                      const timestamp_t ts_j,
                      real_t *pose_i,
                      real_t *vel_i,
                      real_t *biases_i,
                      real_t *pose_j,
                      real_t *vel_j,
                      real_t *biases_j) {
  assert(ts_j > ts_i);

  // IMU buffer and parameters
  factor->imu_params = imu_params;
  imu_buffer_copy(imu_buf, &factor->imu_buf);

  // Parameters
  factor->ts_i = ts_i;
  factor->pose_i = pose_i;
  factor->vel_i = vel_i;
  factor->biases_i = biases_i;

  factor->ts_j = ts_j;
  factor->pose_j = pose_j;
  factor->vel_j = vel_j;
  factor->biases_j = biases_j;

  factor->num_params = 6;
  factor->params[0] = factor->pose_i;
  factor->params[1] = factor->vel_i;
  factor->params[2] = factor->biases_i;
  factor->params[3] = factor->pose_j;
  factor->params[4] = factor->vel_j;
  factor->params[5] = factor->biases_j;
  factor->param_types[0] = POSE_PARAM;
  factor->param_types[1] = VELOCITY_PARAM;
  factor->param_types[2] = IMU_BIASES_PARAM;
  factor->param_types[3] = POSE_PARAM;
  factor->param_types[4] = VELOCITY_PARAM;
  factor->param_types[5] = IMU_BIASES_PARAM;

  // Residuals
  factor->r_size = 15;

  // Jacobians
  factor->jacs[0] = factor->J_pose_i;
  factor->jacs[1] = factor->J_vel_i;
  factor->jacs[2] = factor->J_biases_i;
  factor->jacs[3] = factor->J_pose_j;
  factor->jacs[4] = factor->J_vel_j;
  factor->jacs[5] = factor->J_biases_j;

  // Preintegrate
  imu_factor_preintegrate(factor);
}

/**
 * Reset IMU Factor
 */
void imu_factor_reset(imu_factor_t *factor) {
  // Residuals
  zeros(factor->r, 15, 1);

  // Jacobians
  zeros(factor->J_pose_i, 15, 6);
  zeros(factor->J_vel_i, 15, 3);
  zeros(factor->J_biases_i, 15, 6);
  zeros(factor->J_pose_j, 15, 6);
  zeros(factor->J_vel_j, 15, 3);
  zeros(factor->J_biases_j, 15, 6);

  // Pre-integration variables
  factor->Dt = 0.0;
  eye(factor->F, 15, 15);                                  // State jacobian
  zeros(factor->P, 15, 15);                                // State covariance
  imu_factor_form_Q_matrix(factor->imu_params, factor->Q); // Noise matrix
  zeros(factor->dr, 3, 1);                                 // Rel position
  zeros(factor->dv, 3, 1);                                 // Rel velocity
  quat_setup(factor->dq);                                  // Rel rotation
  vec_copy(factor->biases_i + 0, 3, factor->ba);           // Accel bias
  vec_copy(factor->biases_i + 3, 3, factor->bg);           // Gyro bias
  zeros(factor->ba_ref, 3, 1);
  zeros(factor->bg_ref, 3, 1);

  // Preintegration step variables
  zeros(factor->r_i, 3, 1);
  zeros(factor->v_i, 3, 1);
  quat_setup(factor->q_i);
  zeros(factor->ba_i, 3, 1);
  zeros(factor->bg_i, 3, 1);

  zeros(factor->r_j, 3, 1);
  zeros(factor->v_j, 3, 1);
  quat_setup(factor->q_j);
  zeros(factor->ba_j, 3, 1);
  zeros(factor->bg_j, 3, 1);
}

void imu_factor_preintegrate(imu_factor_t *factor) {
  // Reset variables
  imu_factor_reset(factor);

  // Pre-integrate imu measuremenets
  // -------------------------------
  // This step is essentially like a Kalman Filter whereby you propagate the
  // system inputs (in this case the system is an IMU model with
  // acceleration and angular velocity as inputs. In this step we are
  // interested in the:
  //
  // - Relative position between pose i and pose j
  // - Relative rotation between pose i and pose j
  // - Relative velocity between pose i and pose j
  // - Relative accelerometer bias between pose i and pose j
  // - Relative gyroscope bias between pose i and pose j
  // - Covariance
  //
  // The covariance can be square-rooted to form the square-root information
  // matrix used by the non-linear least squares algorithm to weigh the
  // parameters
  for (int k = 1; k < factor->imu_buf.size; k++) {
    const timestamp_t ts_i = factor->imu_buf.ts[k - 1];
    const timestamp_t ts_j = factor->imu_buf.ts[k];
    const real_t dt = ts2sec(ts_j) - ts2sec(ts_i);
    const real_t *a_i = factor->imu_buf.acc[k - 1];
    const real_t *w_i = factor->imu_buf.gyr[k - 1];
    const real_t *a_j = factor->imu_buf.acc[k];
    const real_t *w_j = factor->imu_buf.gyr[k];

    if (ts_i < factor->ts_i) {
      continue;
    } else if (ts_j > factor->ts_j) {
      break;
    }

    // Propagate
    imu_factor_propagate_step(factor, a_i, w_i, a_j, w_j, dt);

    // Form transition Matrix F
    const real_t *q_i = factor->q_i;
    const real_t *q_j = factor->q_j;
    const real_t *ba_i = factor->ba_i;
    const real_t *bg_i = factor->bg_i;
    real_t F_dt[15 * 15] = {0};
    imu_factor_F_matrix(q_i, q_j, ba_i, bg_i, a_i, w_i, a_j, w_j, dt, F_dt);

    // Input Jacobian G
    real_t G_dt[15 * 18] = {0};
    imu_factor_form_G_matrix(factor, a_i, a_j, dt, G_dt);

    // Update state matrix F
    // F = F_dt * F;
    real_t state_F[15 * 15] = {0};
    mat_copy(factor->F, 15, 15, state_F);
    dot(F_dt, 15, 15, state_F, 15, 15, factor->F);

    // Update covariance matrix P
    // P = F * P * F' + G * Q * G';
    real_t A[15 * 15] = {0};
    real_t B[15 * 15] = {0};
    dot_XAXt(F_dt, 15, 15, factor->P, 15, 15, A);
    dot_XAXt(G_dt, 15, 18, factor->Q, 18, 18, B);
    mat_add(A, B, factor->P, 15, 15);

    // Update overall dt
    factor->Dt += dt;
  }

  // Keep track of linearized accel / gyro biases
  vec3_copy(factor->biases_i + 0, factor->ba_ref);
  vec3_copy(factor->biases_i + 3, factor->bg_ref);

  // Covariance
  enforce_spd(factor->P, 15, 15);
  mat_copy(factor->P, 15, 15, factor->covar);

  // Square root information
  real_t info[15 * 15] = {0};
  real_t sqrt_info[15 * 15] = {0};

  pinv(factor->covar, 15, 15, info);
  assert(check_inv(info, factor->covar, 15) == 0);
  zeros(factor->sqrt_info, 15, 15);
  chol(info, 15, sqrt_info);
  mat_transpose(sqrt_info, 15, 15, factor->sqrt_info);
}

static void imu_factor_pose_i_jac(imu_factor_t *factor,
                                  const real_t dr_est[3],
                                  const real_t dv_est[3],
                                  const real_t dq[4]) {
  // Setup
  real_t C_i[3 * 3] = {0};
  real_t C_j[3 * 3] = {0};
  real_t C_it[3 * 3] = {0};
  real_t C_jt[3 * 3] = {0};
  pose_get_rot(factor->pose_i, C_i);
  pose_get_rot(factor->pose_j, C_j);
  mat_transpose(C_i, 3, 3, C_it);
  mat_transpose(C_j, 3, 3, C_jt);

  // Jacobian w.r.t pose_i
  real_t J_pose_i[15 * 6] = {0};

  // -- Jacobian w.r.t. r_i
  real_t drij_dri[3 * 3] = {0};

  for (int idx = 0; idx < 9; idx++) {
    drij_dri[idx] = -1.0 * C_it[idx];
  }
  mat_block_set(J_pose_i, 6, 0, 2, 0, 2, drij_dri);

  // -- Jacobian w.r.t. q_i
  HAT(dr_est, drij_dCi);
  HAT(dv_est, dvij_dCi);

  // -(quat_left(rot2quat(C_j.T @ C_i)) @ quat_right(dq))[1:4, 1:4]
  real_t dtheta_dCi[3 * 3] = {0};
  {
    DOT(C_jt, 3, 3, C_i, 3, 3, C_ji);
    ROT2QUAT(C_ji, q_ji);

    real_t Left[4 * 4] = {0};
    real_t Right[4 * 4] = {0};
    quat_left(q_ji, Left);
    quat_right(dq, Right);
    DOT(Left, 4, 4, Right, 4, 4, LR);

    mat_block_get(LR, 4, 1, 3, 1, 3, dtheta_dCi);
    mat_scale(dtheta_dCi, 3, 3, -1.0);
  }

  mat_block_set(J_pose_i, 6, 0, 2, 3, 5, drij_dCi);
  mat_block_set(J_pose_i, 6, 3, 5, 3, 5, dvij_dCi);
  mat_block_set(J_pose_i, 6, 6, 8, 3, 5, dtheta_dCi);

  // -- Multiply with sqrt_info
  dot(factor->sqrt_info, 15, 15, J_pose_i, 15, 6, factor->jacs[0]);
}

void imu_factor_velocity_i_jac(imu_factor_t *factor) {
  real_t q_i[4] = {0};
  real_t C_i[3 * 3] = {0};
  real_t C_it[3 * 3] = {0};
  real_t drij_dvi[3 * 3] = {0};
  real_t dvij_dvi[3 * 3] = {0};

  pose_get_quat(factor->pose_i, q_i);
  quat2rot(q_i, C_i);
  mat_transpose(C_i, 3, 3, C_it);

  for (int idx = 0; idx < 9; idx++) {
    drij_dvi[idx] = -1.0 * C_it[idx] * factor->Dt;
    dvij_dvi[idx] = -1.0 * C_it[idx];
  }

  real_t J_vel_i[15 * 3] = {0};
  mat_block_set(J_vel_i, 3, 0, 2, 0, 2, drij_dvi);
  mat_block_set(J_vel_i, 3, 3, 5, 0, 2, dvij_dvi);
  dot(factor->sqrt_info, 15, 15, J_vel_i, 15, 3, factor->jacs[1]);
}

void imu_factor_biases_i_jac(imu_factor_t *factor,
                             const real_t dq_dbg[3 * 3],
                             const real_t dr_dba[3 * 3],
                             const real_t dv_dba[3 * 3],
                             const real_t dr_dbg[3 * 3],
                             const real_t dv_dbg[3 * 3]) {
  real_t q_i[4] = {0};
  real_t q_j[4] = {0};
  pose_get_quat(factor->pose_i, q_i);
  pose_get_quat(factor->pose_j, q_j);

  QUAT2ROT(factor->dq, dC);
  QUAT2ROT(q_i, C_i);
  QUAT2ROT(q_j, C_j);

  MAT_TRANSPOSE(dC, 3, 3, dCt);
  MAT_TRANSPOSE(C_j, 3, 3, C_jt);
  DOT(C_jt, 3, 3, C_i, 3, 3, C_ji);
  DOT(C_ji, 3, 3, dC, 3, 3, C_ji_dC);
  ROT2QUAT(C_ji_dC, qji_dC);

  real_t left_xyz[3 * 3] = {0};
  quat_left_xyz(qji_dC, left_xyz);

  // Jacobian w.r.t IMU biases
  real_t J_biases_i[15 * 6] = {0};
  real_t mI3[3 * 3] = {0};
  mI3[0] = -1.0;
  mI3[4] = -1.0;
  mI3[8] = -1.0;

  // -- Jacobian w.r.t ba_i
  real_t drij_dbai[3 * 3] = {0};
  real_t dvij_dbai[3 * 3] = {0};
  for (int idx = 0; idx < 9; idx++) {
    drij_dbai[idx] = -1.0 * dr_dba[idx];
    dvij_dbai[idx] = -1.0 * dv_dba[idx];
  }
  mat_block_set(J_biases_i, 6, 0, 2, 0, 2, drij_dbai);
  mat_block_set(J_biases_i, 6, 3, 5, 0, 2, dvij_dbai);

  // -- Jacobian w.r.t bg_i
  real_t drij_dbgi[3 * 3] = {0};
  real_t dvij_dbgi[3 * 3] = {0};
  for (int idx = 0; idx < 9; idx++) {
    drij_dbgi[idx] = -1.0 * dr_dbg[idx];
    dvij_dbgi[idx] = -1.0 * dv_dbg[idx];
  }

  real_t dtheta_dbgi[3 * 3] = {0};
  dot(left_xyz, 3, 3, dq_dbg, 3, 3, dtheta_dbgi);
  for (int i = 0; i < 9; i++) {
    dtheta_dbgi[i] *= -1.0;
  }

  mat_block_set(J_biases_i, 6, 0, 2, 3, 5, drij_dbgi);
  mat_block_set(J_biases_i, 6, 3, 5, 3, 5, dvij_dbgi);
  mat_block_set(J_biases_i, 6, 6, 8, 3, 5, dtheta_dbgi);
  mat_block_set(J_biases_i, 6, 9, 11, 0, 2, mI3);
  mat_block_set(J_biases_i, 6, 12, 14, 3, 5, mI3);

  // -- Multiply with sqrt info
  dot(factor->sqrt_info, 15, 15, J_biases_i, 15, 6, factor->jacs[2]);
}

void imu_factor_pose_j_jac(imu_factor_t *factor, const real_t dq[4]) {
  // Setup
  real_t q_i[4] = {0};
  real_t q_j[4] = {0};
  real_t C_i[3 * 3] = {0};
  real_t C_j[3 * 3] = {0};
  real_t C_it[3 * 3] = {0};

  pose_get_quat(factor->pose_i, q_i);
  pose_get_quat(factor->pose_j, q_j);
  quat2rot(q_i, C_i);
  quat2rot(q_j, C_j);
  mat_transpose(C_i, 3, 3, C_it);

  // Jacobian w.r.t. pose_j
  real_t J_pose_j[15 * 6] = {0};

  // -- Jacobian w.r.t. r_j
  real_t drij_drj[3 * 3] = {0};
  mat_copy(C_it, 3, 3, drij_drj);
  mat_block_set(J_pose_j, 6, 0, 2, 0, 2, drij_drj);

  // -- Jacobian w.r.t. q_j
  // quat_left_xyz(rot2quat(dC.T @ C_i.T @ C_j))
  QUAT2ROT(dq, dC);
  MAT_TRANSPOSE(dC, 3, 3, dCt);
  DOT3(dCt, 3, 3, C_it, 3, 3, C_j, 3, 3, dCt_C_it_C_j);
  ROT2QUAT(dCt_C_it_C_j, dqij_dqj);

  real_t dtheta_dqj[3 * 3] = {0};
  quat_left_xyz(dqij_dqj, dtheta_dqj);

  mat_block_set(J_pose_j, 6, 6, 8, 3, 5, dtheta_dqj);
  dot(factor->sqrt_info, 15, 15, J_pose_j, 15, 6, factor->jacs[3]);
}

void imu_factor_velocity_j_jac(imu_factor_t *factor) {
  real_t q_i[4] = {0};
  real_t C_i[3 * 3] = {0};
  real_t C_it[3 * 3] = {0};

  pose_get_quat(factor->pose_i, q_i);
  quat2rot(q_i, C_i);
  mat_transpose(C_i, 3, 3, C_it);

  real_t dv_dvj[3 * 3] = {0};
  mat_copy(C_it, 3, 3, dv_dvj);
  real_t J_vel_j[15 * 3] = {0};
  mat_block_set(J_vel_j, 3, 3, 5, 0, 2, dv_dvj);
  dot(factor->sqrt_info, 15, 15, J_vel_j, 15, 3, factor->jacs[4]);
}

void imu_factor_biases_j_jac(imu_factor_t *factor) {
  real_t J_biases_j[15 * 6] = {0};
  real_t I3[3 * 3] = {0};
  eye(I3, 3, 3);
  mat_block_set(J_biases_j, 6, 9, 11, 0, 2, I3);
  mat_block_set(J_biases_j, 6, 12, 14, 3, 5, I3);
  dot(factor->sqrt_info, 15, 15, J_biases_j, 15, 6, factor->jacs[5]);
}

/**
 * Evaluate IMU factor
 * @returns `0` for success, `-1` for failure
 */
int imu_factor_eval(void *factor_ptr) {
  imu_factor_t *factor = (imu_factor_t *) factor_ptr;
  assert(factor != NULL);
  assert(factor->pose_i);
  assert(factor->pose_j);
  assert(factor->vel_i);
  assert(factor->vel_j);
  assert(factor->biases_i);
  assert(factor->biases_j);

  // Map params
  real_t r_i[3] = {0};
  real_t q_i[4] = {0};
  real_t v_i[3] = {0};
  real_t ba_i[3] = {0};
  real_t bg_i[3] = {0};

  real_t r_j[3] = {0};
  real_t q_j[4] = {0};
  real_t v_j[3] = {0};
  real_t ba_j[3] = {0};
  real_t bg_j[3] = {0};

  pose_get_trans(factor->pose_i, r_i);
  pose_get_quat(factor->pose_i, q_i);
  vec_copy(factor->vel_i, 3, v_i);
  vec_copy(factor->biases_i + 0, 3, ba_i);
  vec_copy(factor->biases_i + 3, 3, bg_i);

  pose_get_trans(factor->pose_j, r_j);
  pose_get_quat(factor->pose_j, q_j);
  vec_copy(factor->vel_j, 3, v_j);
  vec_copy(factor->biases_j + 0, 3, ba_j);
  vec_copy(factor->biases_j + 3, 3, bg_j);

  // Correct the relative position, velocity and rotation
  // -- Extract Jacobians from error-state jacobian
  real_t dr_dba[3 * 3] = {0};
  real_t dr_dbg[3 * 3] = {0};
  real_t dv_dba[3 * 3] = {0};
  real_t dv_dbg[3 * 3] = {0};
  real_t dq_dbg[3 * 3] = {0};
  mat_block_get(factor->F, 15, 0, 2, 9, 11, dr_dba);
  mat_block_get(factor->F, 15, 0, 2, 12, 14, dr_dbg);
  mat_block_get(factor->F, 15, 6, 8, 9, 11, dv_dba);
  mat_block_get(factor->F, 15, 6, 8, 12, 14, dv_dbg);
  mat_block_get(factor->F, 15, 3, 5, 12, 14, dq_dbg);

  real_t dba[3] = {0};
  dba[0] = ba_i[0] - factor->ba[0];
  dba[1] = ba_i[1] - factor->ba[1];
  dba[2] = ba_i[2] - factor->ba[2];

  real_t dbg[3] = {0};
  dbg[0] = bg_i[0] - factor->bg[0];
  dbg[1] = bg_i[1] - factor->bg[1];
  dbg[2] = bg_i[2] - factor->bg[2];

  // -- Correct relative position
  // dr = dr + dr_dba * dba + dr_dbg * dbg
  real_t dr[3] = {0};
  {
    DOT(dr_dba, 3, 3, dba, 3, 1, ba_correction);
    DOT(dr_dbg, 3, 3, dbg, 3, 1, bg_correction);
    dr[0] = factor->dr[0] + ba_correction[0] + bg_correction[0];
    dr[1] = factor->dr[1] + ba_correction[1] + bg_correction[1];
    dr[2] = factor->dr[2] + ba_correction[2] + bg_correction[2];
  }
  // -- Correct relative velocity
  // dv = dv + dv_dba * dba + dv_dbg * dbg
  real_t dv[3] = {0};
  {
    DOT(dv_dba, 3, 3, dba, 3, 1, ba_correction);
    DOT(dv_dbg, 3, 3, dbg, 3, 1, bg_correction);
    dv[0] = factor->dv[0] + ba_correction[0] + bg_correction[0];
    dv[1] = factor->dv[1] + ba_correction[1] + bg_correction[1];
    dv[2] = factor->dv[2] + ba_correction[2] + bg_correction[2];
  }
  // -- Correct relative rotation
  // dq = quat_mul(dq, [1.0, 0.5 * dq_dbg * dbg])
  real_t dq[4] = {0};
  {
    real_t theta[3] = {0};
    dot(dq_dbg, 3, 3, dbg, 3, 1, theta);

    real_t q_correction[4] = {0};
    q_correction[0] = 1.0;
    q_correction[1] = 0.5 * theta[0];
    q_correction[2] = 0.5 * theta[1];
    q_correction[3] = 0.5 * theta[2];

    quat_mul(factor->dq, q_correction, dq);
    quat_normalize(dq);
  }

  // Form residuals
  const real_t g_W[3] = {0.0, 0.0, 9.81};
  const real_t Dt = factor->Dt;
  const real_t Dt_sq = Dt * Dt;
  QUAT2ROT(q_i, C_i);
  MAT_TRANSPOSE(C_i, 3, 3, C_it);

  // dr_est = C_i.T @ ((r_j - r_i) - (v_i * Dt) + (0.5 * g_W * Dt_sq))
  real_t dr_est[3] = {0};
  {
    real_t dr_tmp[3] = {0};
    dr_tmp[0] = (r_j[0] - r_i[0]) - (v_i[0] * Dt) + (0.5 * g_W[0] * Dt_sq);
    dr_tmp[1] = (r_j[1] - r_i[1]) - (v_i[1] * Dt) + (0.5 * g_W[1] * Dt_sq);
    dr_tmp[2] = (r_j[2] - r_i[2]) - (v_i[2] * Dt) + (0.5 * g_W[2] * Dt_sq);

    dot(C_it, 3, 3, dr_tmp, 3, 1, dr_est);
  }

  // dv_est = C_i.T @ ((v_j - v_i) + (g_W * Dt))
  real_t dv_est[3] = {0};
  {
    real_t dv_tmp[3] = {0};
    dv_tmp[0] = (v_j[0] - v_i[0]) + (g_W[0] * Dt);
    dv_tmp[1] = (v_j[1] - v_i[1]) + (g_W[1] * Dt);
    dv_tmp[2] = (v_j[2] - v_i[2]) + (g_W[2] * Dt);

    dot(C_it, 3, 3, dv_tmp, 3, 1, dv_est);
  }

  // err_pos = dr_est - dr
  real_t err_pos[3] = {0.0, 0.0, 0.0};
  err_pos[0] = dr_est[0] - dr[0];
  err_pos[1] = dr_est[1] - dr[1];
  err_pos[2] = dr_est[2] - dr[2];

  // err_vel = dv_est - dv
  real_t err_vel[3] = {0.0, 0.0, 0.0};
  err_vel[0] = dv_est[0] - dv[0];
  err_vel[1] = dv_est[1] - dv[1];
  err_vel[2] = dv_est[2] - dv[2];

  // err_rot = (2.0 * quat_mul(quat_inv(dq), quat_mul(quat_inv(q_i), q_j)))[1:4]
  real_t err_rot[3] = {0.0, 0.0, 0.0};
  {
    real_t dq_inv[4] = {0};
    real_t q_i_inv[4] = {0};
    real_t q_i_inv_j[4] = {0};
    real_t err_quat[4] = {0};

    quat_inv(dq, dq_inv);
    quat_inv(q_i, q_i_inv);
    quat_mul(q_i_inv, q_j, q_i_inv_j);
    quat_mul(dq_inv, q_i_inv_j, err_quat);

    err_rot[0] = 2.0 * err_quat[1];
    err_rot[1] = 2.0 * err_quat[2];
    err_rot[2] = 2.0 * err_quat[3];
  }

  // err_ba = ba_j - ba_i
  real_t err_ba[3] = {ba_j[0] - ba_i[0], ba_j[1] - ba_i[1], ba_j[2] - ba_i[2]};

  // err_bg = bg_j - bg_i
  real_t err_bg[3] = {bg_j[0] - bg_i[0], bg_j[1] - bg_i[1], bg_j[2] - bg_i[2]};

  // Residual vector
  // r = sqrt_info * [err_pos; err_vel; err_rot; err_ba; err_bg]
  {
    real_t r_raw[15] = {0};
    r_raw[0] = err_pos[0];
    r_raw[1] = err_pos[1];
    r_raw[2] = err_pos[2];

    r_raw[3] = err_vel[0];
    r_raw[4] = err_vel[1];
    r_raw[5] = err_vel[2];

    r_raw[6] = err_rot[0];
    r_raw[7] = err_rot[1];
    r_raw[8] = err_rot[2];

    r_raw[9] = err_ba[0];
    r_raw[10] = err_ba[1];
    r_raw[11] = err_ba[2];

    r_raw[12] = err_bg[0];
    r_raw[13] = err_bg[1];
    r_raw[14] = err_bg[2];

    dot(factor->sqrt_info, 15, 15, r_raw, 15, 1, factor->r);
  }

  // Form Jacobians
  imu_factor_pose_i_jac(factor, dr_est, dv_est, dq);
  imu_factor_velocity_i_jac(factor);
  imu_factor_biases_i_jac(factor, dq_dbg, dr_dba, dv_dba, dr_dbg, dv_dbg);
  imu_factor_pose_j_jac(factor, dq);
  imu_factor_velocity_j_jac(factor);
  imu_factor_biases_j_jac(factor);

  return 0;
}

//////////////////
// LIDAR FACTOR //
//////////////////

pcd_t *pcd_malloc(const timestamp_t ts_start,
                  const timestamp_t ts_end,
                  const float *data,
                  const float *time_diffs,
                  const size_t num_points) {
  assert(data != NULL);

  pcd_t *pcd = malloc(sizeof(pcd_t));
  pcd->ts_start = ts_start;
  pcd->ts_end = ts_end;

  pcd->data = malloc(sizeof(float) * 3 * num_points);
  for (size_t i = 0; i < num_points; ++i) {
    pcd->data[i * 3 + 0] = data[i * 3 + 0];
    pcd->data[i * 3 + 1] = data[i * 3 + 1];
    pcd->data[i * 3 + 2] = data[i * 3 + 2];
  }

  if (time_diffs) {
    pcd->time_diffs = malloc(sizeof(float) * num_points);
    for (size_t i = 0; i < num_points; ++i) {
      pcd->time_diffs[i] = time_diffs[i];
    }
  } else {
    pcd->time_diffs = NULL;
  }

  pcd->num_points = num_points;
  pcd->kdtree = kdtree_malloc(pcd->data, pcd->num_points);

  return pcd;
}

void pcd_free(pcd_t *pcd) {
  if (pcd == NULL) {
    return;
  }

  free(pcd->data);
  free(pcd->time_diffs);
  kdtree_free(pcd->kdtree);
  free(pcd);
}

void pcd_deskew(pcd_t *pcd,
                const real_t T_WL_km1[4 * 4],
                const real_t T_WL_km2[4 * 4]) {
  assert(pcd);
  assert(T_WL_km1);
  assert(T_WL_km2);

  // Setup
  const real_t ts_start = ts2sec(pcd->ts_start);
  const real_t ts_end = ts2sec(pcd->ts_end);
  const real_t dt = ts_end - ts_start;
  TF_ROT(T_WL_km2, C_WL_km2);
  TF_ROT(T_WL_km1, C_WL_km1);
  TF_TRANS(T_WL_km2, r_WL_km2);
  TF_TRANS(T_WL_km1, r_WL_km1);

  // v_WL = (C_WL_km2' * (r_WL_km1 - r_WL_km2)) / dt
  VEC_SUB(r_WL_km1, r_WL_km2, dr, 3);
  MAT_TRANSPOSE(C_WL_km2, 3, 3, C_WL_t_km2);
  DOT(C_WL_t_km2, 3, 3, dr, 3, 1, v_WL);
  vec_scale(v_WL, 3, 1.0 / dt);

  // w_WL = (Log(C_WL_km2' * C_WL_km1)) / dt
  real_t w_WL[3] = {0};
  DOT(C_WL_t_km2, 3, 3, C_WL_km1, 3, 3, dC);
  lie_Log(dC, w_WL);
  vec_scale(w_WL, 3, 1.0 / dt);

  // Deskew point cloud
  // p = Exp(s_i * w_WL) * p + s_i * v_WL
  for (size_t i = 0; i < pcd->num_points; ++i) {
    real_t p[3] = {
        pcd->data[i * 3 + 0],
        pcd->data[i * 3 + 1],
        pcd->data[i * 3 + 2],
    };

    const real_t s_i = pcd->time_diffs[i];
    const real_t v_WL_i[3] = {s_i * v_WL[0], s_i * v_WL[1], s_i * v_WL[2]};
    const real_t w_WL_i[3] = {s_i * w_WL[0], s_i * w_WL[1], s_i * w_WL[2]};

    real_t dC[3 * 3] = {0};
    lie_Exp(w_WL_i, dC);
    DOT(dC, 3, 3, p, 3, 1, p_new);
    vec_add(p_new, v_WL_i, p, 3);
  }
}

// /**
//  * Setup lidar factor
//  */
// void lidar_factor_setup(lidar_factor_t *factor,
//                         pcd_t *pcd,
//                         pose_t *pose,
//                         const real_t var[3]) {
//   assert(factor != NULL);
//   assert(pcd != NULL);
//   assert(pose != NULL);
//   assert(var != NULL);
//
//   // Parameters
//   factor->pcd = pcd;
//   factor->pose = pose;
//
//   // Measurement covariance
//   zeros(factor->covar, 3, 3);
//   factor->covar[0] = var[0];
//   factor->covar[4] = var[1];
//   factor->covar[8] = var[2];
//
//   // Square-root information matrix
//   zeros(factor->sqrt_info, 3, 3);
//   factor->sqrt_info[0] = sqrt(1.0 / factor->covar[0]);
//   factor->sqrt_info[4] = sqrt(1.0 / factor->covar[1]);
//   factor->sqrt_info[8] = sqrt(1.0 / factor->covar[2]);
//
//   // Factor parameters, residuals and Jacobians
//   factor->r_size = pcd->num_points * 3;
//   factor->num_params = 1;
//   factor->param_types[0] = POSE_PARAM;
//   factor->params[0] = factor->pose->data;
//   factor->jacs[0] = factor->J_pose;
// }
//
// static float *pcd_transform(pcd_t *pcd, const real_t *T_WL) {
//   TF_INV(T_WL, T_LW);
//   TF_ROT(T_LW, C_LW);
//   TF_TRANS(T_LW, r_LW);
//
//   float C_LW_f[3 * 3] = {0};
//   float r_LW_f[3 * 3] = {0};
//   for (int i = 0; i < 9; ++i) {
//     C_LW_f[i] = C_LW[i];
//   }
//   for (int i = 0; i < 3; ++i) {
//     r_LW_f[i] = r_LW[i];
//   }
//
//   float *points_W = malloc(sizeof(real_t) * 3 * pcd->num_points);
//   dotf(pcd->data, pcd->num_points, 3, C_LW_f, 3, 3, points_W);
//   for (size_t i = 0; i < pcd->num_points; ++i) {
//     points_W[i * 3 + 0] += r_LW_f[0];
//     points_W[i * 3 + 1] += r_LW_f[1];
//     points_W[i * 3 + 2] += r_LW_f[2];
//   }
//
//   return points_W;
// }
//
// void lidar_factor_jacobian(real_t *C_WL, float *p_W_est, real_t J_pose[3 *
// 6]) {
//   // J_pos = -1.0 * eye(3)
//   real_t J_pos[3 * 3] = {0};
//   J_pos[0] = -1.0;
//   J_pos[4] = -1.0;
//   J_pos[8] = -1.0;
//
//   // J_rot = C_WL @ hat(p_W_est)
//   // clang-format off
//   real_t xP[3 * 3] = {0};
//   xP[0] = 0.0;         xP[1] = -p_W_est[2]; xP[2] = p_W_est[1];
//   xP[3] = p_W_est[2];  xP[4] = 0.0;         xP[5] = -p_W_est[0];
//   xP[6] = -p_W_est[1]; xP[7] = p_W_est[0];  xP[8] = 0.0;
//   // clang-format on
//   DOT(C_WL, 3, 3, xP, 3, 3, J_rot);
//
//   // J_pose = zeros((3, 6))
//   // J[0:3, 0:3] = J_pos
//   // J[0:3, 3:6] = J_rot
//   mat_block_set(J_pose, 6, 0, 2, 0, 2, J_pos);
//   mat_block_set(J_pose, 6, 0, 2, 3, 5, J_rot);
// }
//
// /**
//  * Evaluate lidar factor
//  */
// void lidar_factor_eval(void *factor_ptr) {
//   lidar_factor_t *factor = (lidar_factor_t *) factor_ptr;
//   assert(factor != NULL);
//   assert(factor->pose);
//   assert(factor->pcd);
//
//   // Map params
//   TF(factor->params[0], T_WB);
//   TF(factor->params[1], T_BL);
//   TF_CHAIN(T_WL, 2, T_WB, T_BL);
//   TF_ROT(T_WL, C_WL);
//   TF_INV(T_WL, T_LW);
//   TF_ROT(T_LW, C_LW);
//
//   // Transform lidar scan points to world frame and obtain closest points
//   float *points_est = pcd_transform(factor->pcd, T_WL);
//   float *points_map = malloc(sizeof(real_t) * 3 * factor->pcd->num_points);
//   for (size_t i = 0; i < factor->pcd->num_points; ++i) {
//     float point[3] = {0};
//     float dist = 0.0f;
//     kdtree_nn(factor->kdtree, &points_est[i * 3], point, &dist);
//     points_map[i * 3 + 0] = point[0];
//     points_map[i * 3 + 1] = point[1];
//     points_map[i * 3 + 2] = point[2];
//   }
//
//   // Calculate residuals
//   factor->r = malloc(sizeof(real_t) * 3 * factor->pcd->num_points);
//   for (size_t i = 0; i < factor->pcd->num_points; ++i) {
//     factor->r[i * 3 + 0] = points_map[i * 3 + 0] - points_est[i * 3 + 0];
//     factor->r[i * 3 + 1] = points_map[i * 3 + 1] - points_est[i * 3 + 1];
//     factor->r[i * 3 + 2] = points_map[i * 3 + 2] - points_est[i * 3 + 2];
//   }
//
//   // Calculate jacobians
//   // -- Form: -1 * sqrt_info
//   real_t neg_sqrt_info[3 * 3] = {0};
//   mat_copy(factor->sqrt_info, 3, 3, neg_sqrt_info);
//   mat_scale(neg_sqrt_info, 3, 3, -1.0);
//   // -- Fill jacobians
//   const size_t num_rows = 3 * factor->pcd->num_points;
//   const size_t num_cols = 6;
//   if (factor->J_pose) {
//     free(factor->J_pose);
//   }
//   factor->J_pose = malloc(sizeof(real_t) * num_rows * num_cols);
//
//   const int stride = 6;
//   for (size_t i = 0; i < factor->pcd->num_points; ++i) {
//     const int rs = i * 3 + 0;
//     const int re = i * 3 + 2;
//     const int cs = 0;
//     const int ce = 5;
//
//     real_t J_pose[3 * 6] = {0};
//     lidar_factor_jacobian(C_WL, &points_est[i * 3], J_pose);
//     mat_block_set(factor->J_pose, stride, rs, re, cs, ce, J_pose);
//   }
// }

////////////////////////
// JOINT-ANGLE FACTOR //
////////////////////////

/**
 * Setup joint-angle factor
 */
void joint_factor_setup(joint_factor_t *factor,
                        real_t *joint,
                        const real_t z,
                        const real_t var) {
  assert(factor != NULL);
  assert(joint != NULL);

  // Parameters
  factor->joint = joint;
  factor->num_params = 1;

  // Measurement
  factor->z[0] = z;

  // Measurement covariance matrix
  factor->covar[0] = var;

  // Square-root information matrix
  factor->sqrt_info[0] = sqrt(1.0 / factor->covar[0]);

  // Factor residuals, parameters and Jacobians
  factor->r_size = 1;
  factor->num_params = 1;
  factor->param_types[0] = JOINT_PARAM;
  factor->params[0] = factor->joint;
  factor->jacs[0] = factor->J_joint;
}

/**
 * Copy joint factor.
 */
void joint_factor_copy(const joint_factor_t *src, joint_factor_t *dst) {
  assert(src != NULL);
  assert(dst != NULL);

  dst->joint = src->joint;
  dst->z[0] = src->z[0];
  dst->covar[0] = src->covar[0];
  dst->sqrt_info[0] = src->sqrt_info[0];

  dst->r_size = src->r_size;
  dst->num_params = src->num_params;
  dst->param_types[0] = src->param_types[0];

  dst->params[0] = src->params[0];
  dst->r[0] = src->r[0];
  dst->J_joint[0] = src->J_joint[0];
}

/**
 * Evaluate joint-angle factor
 * @returns `0` for success, `-1` for failure
 */
int joint_factor_eval(void *factor_ptr) {
  assert(factor_ptr != NULL);

  // Map factor
  joint_factor_t *factor = (joint_factor_t *) factor_ptr;

  // Calculate residuals
  factor->r[0] = factor->sqrt_info[0] * (factor->z[0] - factor->joint[0]);

  // Calculate Jacobians
  factor->jacs[0][0] = -1 * factor->sqrt_info[0];

  return 0;
}

/**
 * Check if two joint factors are equal in value.
 */
int joint_factor_equals(const joint_factor_t *j0, const joint_factor_t *j1) {
  CHECK(vec_equals(j0->z, j1->z, 1));
  CHECK(mat_equals(j0->covar, j1->covar, 1, 1, 1e-8));
  CHECK(mat_equals(j0->sqrt_info, j1->sqrt_info, 1, 1, 1e-8));

  CHECK(j0->r_size == j1->r_size);
  CHECK(j0->num_params == j1->num_params);
  CHECK(j0->param_types[0] == j1->param_types[0]);

  CHECK(vec_equals(j0->params[0], j1->params[0], 1));
  CHECK(vec_equals(j0->r, j1->r, 1));
  CHECK(vec_equals(j0->jacs[0], j1->jacs[0], 1));
  CHECK(mat_equals(j0->J_joint, j1->J_joint, 1, 1, 1e-8));

  return 1;
error:
  return 0;
}

//////////////
// CAMCHAIN //
//////////////

/**
 * Allocate memory for the camchain initialzer.
 */
camchain_t *camchain_malloc(const int num_cams) {
  camchain_t *cc = malloc(sizeof(camchain_t) * 1);

  // Flags
  cc->analyzed = 0;
  cc->num_cams = num_cams;

  // Allocate memory for the adjacency list and extrinsics
  cc->adj_list = calloc(cc->num_cams, sizeof(int *));
  cc->adj_exts = calloc(cc->num_cams, sizeof(real_t *));
  for (int cam_idx = 0; cam_idx < cc->num_cams; cam_idx++) {
    cc->adj_list[cam_idx] = calloc(cc->num_cams, sizeof(int));
    cc->adj_exts[cam_idx] = calloc(cc->num_cams * (4 * 4), sizeof(real_t));
  }

  // Allocate memory for camera poses
  cc->cam_poses = calloc(num_cams, sizeof(rbt_t *));
  for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
    cc->cam_poses[cam_idx] = rbt_malloc(int_cmp);
  }

  return cc;
}

/**
 * Free camchain initialzer.
 */
void camchain_free(camchain_t *cc) {
  // Adjacency list and extrinsic
  for (int cam_idx = 0; cam_idx < cc->num_cams; cam_idx++) {
    free(cc->adj_list[cam_idx]);
    free(cc->adj_exts[cam_idx]);
  }
  free(cc->adj_list);
  free(cc->adj_exts);

  // Camera poses
  for (int cam_idx = 0; cam_idx < cc->num_cams; cam_idx++) {
    const size_t n = rbt_size(cc->cam_poses[cam_idx]);
    if (n == 0) {
      rbt_free(cc->cam_poses[cam_idx]);
      continue;
    }

    arr_t *keys = arr_malloc(n);
    arr_t *vals = arr_malloc(n);
    rbt_keys_values(cc->cam_poses[cam_idx], keys, vals);
    for (size_t i = 0; i < n; ++i) {
      free(keys->data[i]);
      free(vals->data[i]);
    }
    arr_free(keys);
    arr_free(vals);
    rbt_free(cc->cam_poses[cam_idx]);
  }
  free(cc->cam_poses);

  // Finish
  free(cc);
}

/**
 * Add camera pose to camchain.
 */
void camchain_add_pose(camchain_t *cc,
                       const int cam_idx,
                       const timestamp_t ts,
                       const real_t T_CiF[4 * 4]) {
  void *key = timestamp_malloc(ts);
  void *val = vector_malloc(T_CiF, 16);
  rbt_insert(cc->cam_poses[cam_idx], key, val);
}

/**
 * Form camchain adjacency list.
 */
void camchain_adjacency(camchain_t *cc) {
  // Iterate through camera i data
  for (int cam_i = 0; cam_i < cc->num_cams; ++cam_i) {
    const size_t n = rbt_size(cc->cam_poses[cam_i]);
    if (n == 0) {
      continue;
    }
    arr_t *cam_i_ts = arr_malloc(n);
    arr_t *cam_i_poses = arr_malloc(n);
    rbt_keys_values(cc->cam_poses[cam_i], cam_i_ts, cam_i_poses);

    for (size_t k = 0; k < n; ++k) {
      const timestamp_t ts_i = *(timestamp_t *) cam_i_ts->data[k];
      const real_t *T_CiF = cam_i_poses->data[k];

      // Iterate through camera j data
      for (int cam_j = cam_i + 1; cam_j < cc->num_cams; cam_j++) {
        // Check if a link has already been discovered
        if (cc->adj_list[cam_i][cam_j] == 1) {
          continue;
        }

        // Check if a link exists between camera i and j in the data
        if (rbt_contains(cc->cam_poses[cam_j], &ts_i) == false) {
          continue;
        }

        // Form T_CiCj and T_CjCi
        const real_t *T_CjF = rbt_search(cc->cam_poses[cam_j], &ts_i);
        if (T_CjF == NULL) {
          continue;
        }
        TF_INV(T_CjF, T_FCj);
        TF_INV(T_CiF, T_FCi);
        TF_CHAIN(T_CiCj, 2, T_CiF, T_FCj);
        TF_CHAIN(T_CjCi, 2, T_CjF, T_FCi);

        // Add link between camera i and j
        // TODO: Maybe add to a list and then get the median
        cc->adj_list[cam_i][cam_j] = 1;
        cc->adj_list[cam_j][cam_i] = 1;
        mat_copy(T_CiCj, 4, 4, &cc->adj_exts[cam_i][cam_j * (4 * 4)]);
        mat_copy(T_CjCi, 4, 4, &cc->adj_exts[cam_j][cam_i * (4 * 4)]);
      }
    }

    arr_free(cam_i_ts);
    arr_free(cam_i_poses);
  }

  // Mark camchain as analyzed
  cc->analyzed = 1;
}

/**
 * Print camchain adjacency matrix.
 */
void camchain_adjacency_print(const camchain_t *cc) {
  for (int i = 0; i < cc->num_cams; i++) {
    printf("%d: ", i);
    for (int j = 0; j < cc->num_cams; j++) {
      printf("%d ", cc->adj_list[i][j]);
    }
    printf("\n");
  }
}

/**
 * The purpose of camchain initializer is to find the initial camera to camera
 * extrinsic of arbitrary cameras. Lets suppose you are calibrating a
 * multi-camera system with N cameras observing the same calibration fiducial
 * target (F). The idea is as you add the relative pose between the i-th camera
 * (Ci) and fiducial target (F), the camchain initialzer will build an adjacency
 * matrix and form all possible camera-camera extrinsic combinations. This is
 * useful for multi-camera extrinsics where you need to initialize the
 * camera-extrinsic parameter.
 *
 * Usage:
 *
 *   camchain_t *camchain = camchain_malloc(num_cams);
 *   for (int cam_idx = 0; cam_idx < num_cams; ++cam_idx) {
 *     for (int ts_idx = 0; ts_idx < len(camera_poses); ++ts_idx) {
 *       timestamp_t ts = camera_timestamps[ts_idx];
 *       real_t *T_CiF = camera_poses[cam_idx][ts_idx];
 *       camchain_add_pose(camchain, cam_idx, ts, T_CiF);
 *     }
 *   }
 *   camchain_adjacency(camchain);
 *   camchain_adjacency_print(camchain);
 *   camchain_find(camchain, cam_i, cam_j, T_CiCj);
 *
 */
int camchain_find(camchain_t *cc,
                  const int cam_i,
                  const int cam_j,
                  real_t T_CiCj[4 * 4]) {
  // Form adjacency
  if (cc->analyzed == 0) {
    camchain_adjacency(cc);
  }

  // Straight forward case where extrinsic of itself is identity
  if (cam_i == cam_j) {
    if (rbt_size(cc->cam_poses[cam_i])) {
      eye(T_CiCj, 4, 4);
      return 0;
    } else {
      return -1;
    }
  }

  // Check if T_CiCj was formed before
  if (cc->adj_list[cam_i][cam_j] == 1) {
    mat_copy(&cc->adj_exts[cam_i][cam_j * (4 * 4)], 4, 4, T_CiCj);
    return 0;
  }

  return -1;
}

/////////////////////////
// CALIB-CAMERA FACTOR //
/////////////////////////

/**
 * Setup camera calibration factor
 */
void calib_camera_factor_setup(calib_camera_factor_t *factor,
                               real_t *pose,
                               real_t *cam_ext,
                               camera_t *cam_params,
                               const int cam_idx,
                               const int tag_id,
                               const int corner_idx,
                               const real_t p_FFi[3],
                               const real_t z[2],
                               const real_t var[2]) {
  assert(factor != NULL);
  assert(pose != NULL);
  assert(cam_ext != NULL);
  assert(cam_params != NULL);
  assert(z != NULL);
  assert(var != NULL);

  // Parameters
  factor->pose = pose;
  factor->cam_ext = cam_ext;
  factor->cam_params = cam_params;
  factor->num_params = 3;

  // Measurement
  factor->cam_idx = cam_idx;
  factor->tag_id = tag_id;
  factor->corner_idx = corner_idx;
  factor->p_FFi[0] = p_FFi[0];
  factor->p_FFi[1] = p_FFi[1];
  factor->p_FFi[2] = p_FFi[2];
  factor->z[0] = z[0];
  factor->z[1] = z[1];

  // Measurement covariance matrix
  factor->covar[0] = var[0];
  factor->covar[1] = 0.0;
  factor->covar[2] = 0.0;
  factor->covar[3] = var[1];

  // Square-root information matrix
  factor->sqrt_info[0] = sqrt(1.0 / factor->covar[0]);
  factor->sqrt_info[1] = 0.0;
  factor->sqrt_info[2] = 0.0;
  factor->sqrt_info[3] = sqrt(1.0 / factor->covar[3]);

  // Factor residuals, parameters and Jacobians
  factor->r_size = 2;
  factor->param_types[0] = POSE_PARAM;
  factor->param_types[1] = EXTRINSIC_PARAM;
  factor->param_types[2] = CAMERA_PARAM;

  factor->params[0] = factor->pose;
  factor->params[1] = factor->cam_ext;
  factor->params[2] = factor->cam_params->data;

  factor->jacs[0] = factor->J_pose;
  factor->jacs[1] = factor->J_cam_ext;
  factor->jacs[2] = factor->J_cam_params;
}

int calib_camera_factor_eval(void *factor_ptr) {
  // Map factor
  calib_camera_factor_t *factor = (calib_camera_factor_t *) factor_ptr;
  assert(factor != NULL);

  // Map params
  const real_t *p_FFi = factor->p_FFi;
  TF(factor->params[0], T_BF);                  // Relative pose T_BF
  TF(factor->params[1], T_BCi);                 // Camera extrinsic T_BCi
  const real_t *cam_params = factor->params[2]; // Camera parameters

  // Form T_CiF
  TF_INV(T_BCi, T_CiB);
  TF_CHAIN(T_CiF, 2, T_CiB, T_BF);

  // Project to image plane
  int status = 0;
  real_t z_hat[2];
  TF_POINT(T_CiF, factor->p_FFi, p_CiFi);
  camera_project(factor->cam_params, p_CiFi, z_hat);
  const int res_x = factor->cam_params->resolution[0];
  const int res_y = factor->cam_params->resolution[1];
  const int x_ok = (z_hat[0] > 0 && z_hat[0] < res_x);
  const int y_ok = (z_hat[1] > 0 && z_hat[1] < res_y);
  const int z_ok = p_CiFi[2] > 0;
  if (x_ok && y_ok && z_ok) {
    status = 1;
  }

  // Calculate residuals
  real_t r[2] = {0, 0};
  r[0] = factor->z[0] - z_hat[0];
  r[1] = factor->z[1] - z_hat[1];
  dot(factor->sqrt_info, 2, 2, r, 2, 1, factor->r);

  // Calculate Jacobians
  // -- Zero out jacobians if reprojection is not valid
  if (status == 0) {
    zeros(factor->J_pose, 2, 6);
    zeros(factor->J_cam_ext, 2, 6);
    zeros(factor->J_cam_params, 2, 8);
    return 0;
  }
  // Form: -1 * sqrt_info
  real_t neg_sqrt_info[2 * 2] = {0};
  mat_copy(factor->sqrt_info, 2, 2, neg_sqrt_info);
  mat_scale(neg_sqrt_info, 2, 2, -1.0);
  // Form: Jh_w = -1 * sqrt_info * Jh
  real_t Jh[2 * 3] = {0};
  real_t Jh_w[2 * 3] = {0};
  pinhole_radtan4_project_jacobian(cam_params, p_CiFi, Jh);
  dot(neg_sqrt_info, 2, 2, Jh, 2, 3, Jh_w);
  // Form: J_cam_params
  real_t J_cam_params[2 * 8] = {0};
  pinhole_radtan4_params_jacobian(cam_params, p_CiFi, J_cam_params);

  // -- Jacobians w.r.t relative camera pose T_BF
  {
    // J_pos = Jh * C_CiB
    real_t J_pos[2 * 3] = {0};
    real_t C_CiB[3 * 3] = {0};
    tf_rot_get(T_CiB, C_CiB);
    dot(Jh_w, 2, 3, C_CiB, 3, 3, J_pos);
    factor->jacs[0][0] = J_pos[0];
    factor->jacs[0][1] = J_pos[1];
    factor->jacs[0][2] = J_pos[2];

    factor->jacs[0][6] = J_pos[3];
    factor->jacs[0][7] = J_pos[4];
    factor->jacs[0][8] = J_pos[5];

    // J_rot = Jh * C_CiB * -C_BF @ hat(p_FFi)
    real_t C_BF[3 * 3] = {0};
    real_t C_CiF[3 * 3] = {0};
    tf_rot_get(T_BF, C_BF);
    dot(C_CiB, 3, 3, C_BF, 3, 3, C_CiF);
    mat_scale(C_CiF, 3, 3, -1);

    real_t J_rot[2 * 3] = {0};
    real_t p_FFi_x[3 * 3] = {0};
    hat(p_FFi, p_FFi_x);
    dot3(Jh_w, 2, 3, C_CiF, 3, 3, p_FFi_x, 3, 3, J_rot);

    factor->jacs[0][3] = J_rot[0];
    factor->jacs[0][4] = J_rot[1];
    factor->jacs[0][5] = J_rot[2];

    factor->jacs[0][9] = J_rot[3];
    factor->jacs[0][10] = J_rot[4];
    factor->jacs[0][11] = J_rot[5];
  }

  // -- Jacobians w.r.t camera extrinsic T_BCi
  {
    // J_pos = Jh * -C_CiB
    real_t J_pos[2 * 3] = {0};
    real_t nC_CiB[3 * 3] = {0};
    tf_rot_get(T_CiB, nC_CiB);
    mat_scale(nC_CiB, 3, 3, -1.0);
    dot(Jh_w, 2, 3, nC_CiB, 3, 3, J_pos);
    factor->jacs[1][0] = J_pos[0];
    factor->jacs[1][1] = J_pos[1];
    factor->jacs[1][2] = J_pos[2];

    factor->jacs[1][6] = J_pos[3];
    factor->jacs[1][7] = J_pos[4];
    factor->jacs[1][8] = J_pos[5];

    // J_rot = Jh * -C_CiB * hat(r_BFi - r_BCi) * -C_BCi
    real_t J_rot[2 * 3] = {0};
    real_t r_BFi[3] = {0};
    real_t r_BCi[3] = {0};
    real_t dr[3] = {0};
    real_t hdr[3 * 3] = {0};
    real_t nC_BCi[3 * 3] = {0};

    tf_point(T_BF, p_FFi, r_BFi);
    tf_trans_get(T_BCi, r_BCi);
    dr[0] = r_BFi[0] - r_BCi[0];
    dr[1] = r_BFi[1] - r_BCi[1];
    dr[2] = r_BFi[2] - r_BCi[2];
    hat(dr, hdr);
    tf_rot_get(T_BCi, nC_BCi);
    mat_scale(nC_BCi, 3, 3, -1.0);

    real_t B[3 * 3] = {0};
    dot(hdr, 3, 3, nC_BCi, 3, 3, B);
    dot(J_pos, 2, 3, B, 3, 3, J_rot);

    factor->jacs[1][3] = J_rot[0];
    factor->jacs[1][4] = J_rot[1];
    factor->jacs[1][5] = J_rot[2];

    factor->jacs[1][9] = J_rot[3];
    factor->jacs[1][10] = J_rot[4];
    factor->jacs[1][11] = J_rot[5];
  }

  // -- Jacobians w.r.t. camera parameters
  dot(neg_sqrt_info, 2, 2, J_cam_params, 2, 8, factor->jacs[2]);

  return 0;
}

/////////////////////////
// CALIB-IMUCAM FACTOR //
/////////////////////////

/**
 * Setup imu-camera time-delay calibration factor
 */
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
                               const real_t var[2]) {
  assert(factor != NULL);
  assert(fiducial != NULL);
  assert(imu_pose != NULL);
  assert(imu_ext != NULL);
  assert(cam_ext != NULL);
  assert(camera != NULL);
  assert(time_delay != NULL);
  assert(p_FFi != NULL);
  assert(z != NULL);
  assert(v != NULL);
  assert(var != NULL);

  // Parameters
  factor->fiducial = fiducial;
  factor->imu_pose = imu_pose;
  factor->imu_ext = imu_ext;
  factor->cam_ext = cam_ext;
  factor->camera = camera;
  factor->time_delay = time_delay;
  factor->num_params = 6;

  // Measurement
  factor->cam_idx = cam_idx;
  factor->tag_id = tag_id;
  factor->corner_idx = corner_idx;
  factor->p_FFi[0] = p_FFi[0];
  factor->p_FFi[1] = p_FFi[1];
  factor->p_FFi[2] = p_FFi[2];
  factor->z[0] = z[0];
  factor->z[1] = z[1];
  factor->v[0] = v[0];
  factor->v[1] = v[1];

  // Measurement covariance matrix
  factor->covar[0] = var[0];
  factor->covar[1] = 0.0;
  factor->covar[2] = 0.0;
  factor->covar[3] = var[1];

  // Square-root information matrix
  factor->sqrt_info[0] = sqrt(1.0 / factor->covar[0]);
  factor->sqrt_info[1] = 0.0;
  factor->sqrt_info[2] = 0.0;
  factor->sqrt_info[3] = sqrt(1.0 / factor->covar[3]);

  // Factor residuals, parameters and Jacobians
  factor->r_size = 2;
  factor->param_types[0] = POSE_PARAM;
  factor->param_types[1] = POSE_PARAM;
  factor->param_types[2] = EXTRINSIC_PARAM;
  factor->param_types[3] = EXTRINSIC_PARAM;
  factor->param_types[4] = CAMERA_PARAM;
  factor->param_types[5] = TIME_DELAY_PARAM;

  factor->params[0] = factor->fiducial;
  factor->params[1] = factor->imu_pose;
  factor->params[2] = factor->imu_ext;
  factor->params[3] = factor->cam_ext;
  factor->params[4] = factor->camera->data;
  factor->params[5] = factor->time_delay;

  factor->jacs[0] = factor->J_fiducial;
  factor->jacs[1] = factor->J_imu_pose;
  factor->jacs[2] = factor->J_imu_ext;
  factor->jacs[3] = factor->J_cam_ext;
  factor->jacs[4] = factor->J_camera;
  factor->jacs[5] = factor->J_time_delay;
}

int calib_imucam_factor_eval(void *factor_ptr) {
  // Map factor
  calib_imucam_factor_t *factor = (calib_imucam_factor_t *) factor_ptr;
  assert(factor != NULL);

  // Map params
  TF(factor->params[0], T_WF);                  // Fiducial T_WF
  TF(factor->params[1], T_WS);                  // IMU Pose T_WS
  TF(factor->params[2], T_SC0);                 // IMU extrinsic T_SC0
  TF(factor->params[3], T_C0Ci);                // Camera extrinsic T_C0Ci
  const real_t *cam_params = factor->params[4]; // Camera parameters
  const real_t td = factor->params[5][0];       // Time delay

  // Form T_CiW and T_CiF
  // T_CiW = inv(T_C0Ci) * inv(T_SC0) * inv(T_WS) * T_WF
  TF_INV(T_WS, T_SW);
  TF_INV(T_SC0, T_C0S);
  TF_INV(T_C0Ci, T_CiC0);
  TF_CHAIN(T_CiS, 2, T_CiC0, T_C0S);
  TF_CHAIN(T_CiW, 2, T_CiS, T_SW);
  TF_CHAIN(T_CiF, 2, T_CiW, T_WF);

  // Project to image plane
  real_t z_hat[2];
  TF_POINT(T_CiF, factor->p_FFi, p_CiFi);
  camera_project(factor->camera, p_CiFi, z_hat);

  // Calculate residuals
  real_t r[2] = {0, 0};
  r[0] = (factor->z[0] + td * factor->v[0]) - z_hat[0];
  r[1] = (factor->z[1] + td * factor->v[1]) - z_hat[1];
  dot(factor->sqrt_info, 2, 2, r, 2, 1, factor->r);

  // Calculate Jacobians
  // Form: -1 * sqrt_info
  real_t sqrt_info[2 * 2] = {0};
  real_t neg_sqrt_info[2 * 2] = {0};
  mat_copy(factor->sqrt_info, 2, 2, sqrt_info);
  mat_copy(factor->sqrt_info, 2, 2, neg_sqrt_info);
  mat_scale(neg_sqrt_info, 2, 2, -1.0);
  // Form: Jh_w = -1 * sqrt_info * Jh
  real_t Jh[2 * 3] = {0};
  real_t Jh_w[2 * 3] = {0};
  pinhole_radtan4_project_jacobian(cam_params, p_CiFi, Jh);
  dot(neg_sqrt_info, 2, 2, Jh, 2, 3, Jh_w);
  // Form: J_cam_params
  real_t J_cam_params[2 * 8] = {0};
  pinhole_radtan4_params_jacobian(cam_params, p_CiFi, J_cam_params);

  // -- Jacobians w.r.t fiducial pose T_WF
  {
    // J_pos = Jh * C_CiW
    real_t J_pos[2 * 3] = {0};
    real_t C_CiW[3 * 3] = {0};
    tf_rot_get(T_CiW, C_CiW);
    dot(Jh_w, 2, 3, C_CiW, 3, 3, J_pos);
    factor->jacs[0][0] = J_pos[0];
    factor->jacs[0][1] = J_pos[1];
    factor->jacs[0][2] = J_pos[2];

    factor->jacs[0][6] = J_pos[3];
    factor->jacs[0][7] = J_pos[4];
    factor->jacs[0][8] = J_pos[5];

    // J_rot = Jh * C_CiW * -C_WF @ hat(p_FFi)
    real_t C_WF[3 * 3] = {0};
    real_t C_CiF[3 * 3] = {0};
    tf_rot_get(T_WF, C_WF);
    dot(C_CiW, 3, 3, C_WF, 3, 3, C_CiF);
    mat_scale(C_CiF, 3, 3, -1);

    real_t J_rot[2 * 3] = {0};
    real_t p_FFi_x[3 * 3] = {0};
    hat(factor->p_FFi, p_FFi_x);
    dot3(Jh_w, 2, 3, C_CiF, 3, 3, p_FFi_x, 3, 3, J_rot);

    factor->jacs[0][3] = J_rot[0];
    factor->jacs[0][4] = J_rot[1];
    factor->jacs[0][5] = J_rot[2];

    factor->jacs[0][9] = J_rot[3];
    factor->jacs[0][10] = J_rot[4];
    factor->jacs[0][11] = J_rot[5];
  }

  // -- Jacobians w.r.t IMU pose T_WS
  {
    // J_pos = Jh * -C_CiW
    real_t J_pos[2 * 3] = {0};
    real_t nC_CiW[3 * 3] = {0};
    tf_rot_get(T_CiW, nC_CiW);
    mat_scale(nC_CiW, 3, 3, -1.0);
    dot(Jh_w, 2, 3, nC_CiW, 3, 3, J_pos);
    factor->jacs[1][0] = J_pos[0];
    factor->jacs[1][1] = J_pos[1];
    factor->jacs[1][2] = J_pos[2];

    factor->jacs[1][6] = J_pos[3];
    factor->jacs[1][7] = J_pos[4];
    factor->jacs[1][8] = J_pos[5];

    // J_rot = Jh * -C_CiW * hat(p_WFi - r_WS) * -C_WS
    real_t r_WS[3] = {0};
    real_t dp[3] = {0};
    real_t dp_x[3 * 3] = {0};
    real_t p_WFi[3] = {0};
    tf_trans_get(T_WS, r_WS);
    tf_point(T_WF, factor->p_FFi, p_WFi);
    dp[0] = p_WFi[0] - r_WS[0];
    dp[1] = p_WFi[1] - r_WS[1];
    dp[2] = p_WFi[2] - r_WS[2];
    hat(dp, dp_x);

    real_t nC_WS[3 * 3] = {0};
    tf_rot_get(T_WS, nC_WS);
    mat_scale(nC_WS, 3, 3, -1.0);

    real_t J_rot[2 * 3] = {0};
    dot3(J_pos, 2, 3, dp_x, 3, 3, nC_WS, 3, 3, J_rot);

    factor->jacs[1][3] = J_rot[0];
    factor->jacs[1][4] = J_rot[1];
    factor->jacs[1][5] = J_rot[2];

    factor->jacs[1][9] = J_rot[3];
    factor->jacs[1][10] = J_rot[4];
    factor->jacs[1][11] = J_rot[5];
  }

  // -- Jacobians w.r.t IMU extrinsic T_SC0
  {
    // J_pos = Jh * -C_CiS
    real_t J_pos[2 * 3] = {0};
    real_t nC_CiS[3 * 3] = {0};
    tf_rot_get(T_CiS, nC_CiS);
    mat_scale(nC_CiS, 3, 3, -1.0);
    dot(Jh_w, 2, 3, nC_CiS, 3, 3, J_pos);
    factor->jacs[2][0] = J_pos[0];
    factor->jacs[2][1] = J_pos[1];
    factor->jacs[2][2] = J_pos[2];

    factor->jacs[2][6] = J_pos[3];
    factor->jacs[2][7] = J_pos[4];
    factor->jacs[2][8] = J_pos[5];

    // J_rot = Jh * -C_CiS * hat(p_SFi - r_SC0) * -C_SC0
    real_t r_SC0[3] = {0};
    real_t dp[3] = {0};
    real_t dp_x[3 * 3] = {0};
    real_t p_SFi[3] = {0};
    tf_trans_get(T_SC0, r_SC0);
    TF_CHAIN(T_SF, 2, T_SW, T_WF);
    tf_point(T_SF, factor->p_FFi, p_SFi);
    dp[0] = p_SFi[0] - r_SC0[0];
    dp[1] = p_SFi[1] - r_SC0[1];
    dp[2] = p_SFi[2] - r_SC0[2];
    hat(dp, dp_x);

    real_t nC_SC0[3 * 3] = {0};
    tf_rot_get(T_SC0, nC_SC0);
    mat_scale(nC_SC0, 3, 3, -1.0);

    real_t J_rot[2 * 3] = {0};
    dot3(J_pos, 2, 3, dp_x, 3, 3, nC_SC0, 3, 3, J_rot);

    factor->jacs[2][3] = J_rot[0];
    factor->jacs[2][4] = J_rot[1];
    factor->jacs[2][5] = J_rot[2];

    factor->jacs[2][9] = J_rot[3];
    factor->jacs[2][10] = J_rot[4];
    factor->jacs[2][11] = J_rot[5];
  }

  // -- Jacobians w.r.t camera extrinsic T_C0Ci
  {
    // J_pos = Jh * -C_CiC0
    real_t J_pos[2 * 3] = {0};
    real_t nC_CiC0[3 * 3] = {0};
    tf_rot_get(T_CiC0, nC_CiC0);
    mat_scale(nC_CiC0, 3, 3, -1.0);
    dot(Jh_w, 2, 3, nC_CiC0, 3, 3, J_pos);
    factor->jacs[3][0] = J_pos[0];
    factor->jacs[3][1] = J_pos[1];
    factor->jacs[3][2] = J_pos[2];

    factor->jacs[3][6] = J_pos[3];
    factor->jacs[3][7] = J_pos[4];
    factor->jacs[3][8] = J_pos[5];

    // J_rot = Jh * -C_CiC0 * hat(p_C0Fi - r_C0Ci) * -C_C0Ci
    real_t J_rot[2 * 3] = {0};
    real_t r_C0Fi[3] = {0};
    real_t r_C0Ci[3] = {0};
    real_t dr[3] = {0};
    real_t hdr[3 * 3] = {0};
    real_t nC_C0Ci[3 * 3] = {0};

    TF_CHAIN(T_C0F, 2, T_C0Ci, T_CiF);
    tf_point(T_C0F, factor->p_FFi, r_C0Fi);
    tf_trans_get(T_C0Ci, r_C0Ci);
    dr[0] = r_C0Fi[0] - r_C0Ci[0];
    dr[1] = r_C0Fi[1] - r_C0Ci[1];
    dr[2] = r_C0Fi[2] - r_C0Ci[2];
    hat(dr, hdr);
    tf_rot_get(T_C0Ci, nC_C0Ci);
    mat_scale(nC_C0Ci, 3, 3, -1.0);

    real_t B[3 * 3] = {0};
    dot(hdr, 3, 3, nC_C0Ci, 3, 3, B);
    dot(J_pos, 2, 3, B, 3, 3, J_rot);

    factor->jacs[3][3] = J_rot[0];
    factor->jacs[3][4] = J_rot[1];
    factor->jacs[3][5] = J_rot[2];

    factor->jacs[3][9] = J_rot[3];
    factor->jacs[3][10] = J_rot[4];
    factor->jacs[3][11] = J_rot[5];
  }

  // -- Jacobians w.r.t. camera parameters
  dot(neg_sqrt_info, 2, 2, J_cam_params, 2, 8, factor->jacs[4]);

  // -- Jacobians w.r.t. time delay
  real_t J_time_delay[2 * 1] = {factor->v[0], factor->v[1]};
  dot(sqrt_info, 2, 2, J_time_delay, 2, 1, factor->jacs[5]);

  return 0;
}

//////////////////
// MARGINALIZER //
//////////////////

/**
 * Malloc marginalization factor.
 */
marg_factor_t *marg_factor_malloc(void) {
  marg_factor_t *marg = malloc(sizeof(marg_factor_t) * 1);

  // Settings
  marg->debug = 1;
  marg->cond_hessian = 1;

  // Flags
  marg->marginalized = 0;
  marg->schur_complement_ok = 0;
  marg->eigen_decomp_ok = 0;

  // Parameters
  // -- Remain parameters
  marg->r_positions = list_malloc();
  marg->r_rotations = list_malloc();
  marg->r_poses = list_malloc();
  marg->r_velocities = list_malloc();
  marg->r_imu_biases = list_malloc();
  marg->r_fiducials = list_malloc();
  marg->r_joints = list_malloc();
  marg->r_extrinsics = list_malloc();
  marg->r_features = list_malloc();
  marg->r_cam_params = list_malloc();
  marg->r_time_delays = list_malloc();
  // -- Marginal parameters
  marg->m_positions = list_malloc();
  marg->m_rotations = list_malloc();
  marg->m_poses = list_malloc();
  marg->m_velocities = list_malloc();
  marg->m_imu_biases = list_malloc();
  marg->m_fiducials = list_malloc();
  marg->m_joints = list_malloc();
  marg->m_extrinsics = list_malloc();
  marg->m_features = list_malloc();
  marg->m_cam_params = list_malloc();
  marg->m_time_delays = list_malloc();

  // Factors
  marg->ba_factors = list_malloc();
  marg->camera_factors = list_malloc();
  marg->imu_factors = list_malloc();
  marg->calib_camera_factors = list_malloc();
  marg->calib_imucam_factors = list_malloc();
  marg->marg_factor = NULL;

  // Hessian and residuals
  marg->param_seen = param_index_malloc();
  marg->param_index = param_index_malloc();
  marg->m_lsize = 0;
  marg->r_lsize = 0;
  marg->m_gsize = 0;
  marg->r_gsize = 0;

  marg->x0 = NULL;
  marg->r0 = NULL;
  marg->J0 = NULL;
  marg->dchi = NULL;
  marg->J0_dchi = NULL;

  marg->J0 = NULL;
  marg->J0_inv = NULL;
  marg->H = NULL;
  marg->b = NULL;
  marg->H_marg = NULL;
  marg->b_marg = NULL;

  // Parameters, residuals and Jacobians
  marg->num_params = 0;
  marg->param_types = NULL;
  marg->param_ptrs = NULL;
  marg->params = NULL;
  marg->r = NULL;
  marg->jacs = NULL;

  // Profiling
  marg->time_hessian_form = 0;
  marg->time_schur_complement = 0;
  marg->time_hessian_decomp = 0;
  marg->time_fejs = 0;
  marg->time_total = 0;

  return marg;
}

/**
 * Free marginalization factor.
 */
void marg_factor_free(marg_factor_t *marg) {
  if (marg == NULL) {
    return;
  }

  // Parameters
  // -- Remain parameters
  list_free(marg->r_positions);
  list_free(marg->r_rotations);
  list_free(marg->r_poses);
  list_free(marg->r_velocities);
  list_free(marg->r_imu_biases);
  list_free(marg->r_features);
  list_free(marg->r_joints);
  list_free(marg->r_extrinsics);
  list_free(marg->r_fiducials);
  list_free(marg->r_cam_params);
  list_free(marg->r_time_delays);
  // -- Marginal parameters
  list_free(marg->m_positions);
  list_free(marg->m_rotations);
  list_free(marg->m_poses);
  list_free(marg->m_velocities);
  list_free(marg->m_imu_biases);
  list_free(marg->m_features);
  list_free(marg->m_joints);
  list_free(marg->m_extrinsics);
  list_free(marg->m_fiducials);
  list_free(marg->m_cam_params);
  list_free(marg->m_time_delays);

  // Factors
  list_free(marg->ba_factors);
  list_free(marg->camera_factors);
  list_free(marg->imu_factors);
  list_free(marg->calib_camera_factors);
  list_free(marg->calib_imucam_factors);

  // Residuals
  rbt_free(marg->param_seen);
  param_index_free(marg->param_index);
  free(marg->x0);
  free(marg->r0);
  free(marg->J0);
  free(marg->J0_inv);
  free(marg->dchi);
  free(marg->J0_dchi);

  free(marg->H);
  free(marg->b);
  free(marg->H_marg);
  free(marg->b_marg);

  // Jacobians
  free(marg->param_types);
  if (marg->param_ptrs) {
    free(marg->param_ptrs);
  }
  free(marg->params);
  free(marg->r);
  for (int i = 0; i < marg->num_params; i++) {
    free(marg->jacs[i]);
  }
  free(marg->jacs);

  free(marg);
}

void marg_factor_print_stats(const marg_factor_t *marg) {
  printf("Parameters to be marginalized:\n");
  printf("------------------------------\n");
  printf("m_positions: %ld\n", marg->m_positions->length);
  printf("m_rotations: %ld\n", marg->m_rotations->length);
  printf("m_poses: %ld\n", marg->m_poses->length);
  printf("m_velocities: %ld\n", marg->m_velocities->length);
  printf("m_imu_biases: %ld\n", marg->m_imu_biases->length);
  printf("m_features: %ld\n", marg->m_features->length);
  printf("m_joints: %ld\n", marg->m_joints->length);
  printf("m_extrinsics: %ld\n", marg->m_extrinsics->length);
  printf("m_fiducials: %ld\n", marg->m_fiducials->length);
  printf("m_cam_params: %ld\n", marg->m_cam_params->length);
  printf("m_time_delays: %ld\n", marg->m_time_delays->length);
  printf("\n");

  printf("Parameters to remain:\n");
  printf("---------------------\n");
  printf("r_positions: %ld\n", marg->r_positions->length);
  printf("r_rotations: %ld\n", marg->r_rotations->length);
  printf("r_poses: %ld\n", marg->r_poses->length);
  printf("r_velocities: %ld\n", marg->r_velocities->length);
  printf("r_imu_biases: %ld\n", marg->r_imu_biases->length);
  printf("r_features: %ld\n", marg->r_features->length);
  printf("r_joints: %ld\n", marg->r_joints->length);
  printf("r_extrinsics: %ld\n", marg->r_extrinsics->length);
  printf("r_fiducials: %ld\n", marg->r_fiducials->length);
  printf("r_cam_params: %ld\n", marg->r_cam_params->length);
  printf("r_time_delays: %ld\n", marg->r_time_delays->length);
  printf("\n");
}

/**
 * Add factor to marginalization factor.
 */
void marg_factor_add(marg_factor_t *marg, int factor_type, void *factor_ptr) {
  assert(marg != NULL);
  assert(factor_ptr != NULL);

  switch (factor_type) {
    case MARG_FACTOR:
      assert(marg->marg_factor != NULL); // Implementation error!
      marg->marg_factor = factor_ptr;
      break;
    case BA_FACTOR:
      list_push(marg->ba_factors, factor_ptr);
      break;
    case CAMERA_FACTOR:
      list_push(marg->camera_factors, factor_ptr);
      break;
    case IMU_FACTOR:
      list_push(marg->imu_factors, factor_ptr);
      break;
    case CALIB_CAMERA_FACTOR:
      list_push(marg->calib_camera_factors, factor_ptr);
      break;
    case CALIB_IMUCAM_FACTOR:
      list_push(marg->calib_imucam_factors, factor_ptr);
      break;
    default:
      FATAL("Implementation Error!\n");
      break;
  };
}

#define MARG_TRACK_FN(FUNC_NAME, RLIST, MLIST)                                 \
  void FUNC_NAME(marg_factor_t *marg, real_t *p) {                             \
    if (rbt_contains(marg->param_seen, p)) {                                   \
      return;                                                                  \
    }                                                                          \
    if (rbt_contains(marg->marg_params, p)) {                                  \
      list_push(marg->MLIST, p);                                               \
    } else {                                                                   \
      list_push(marg->RLIST, p);                                               \
    }                                                                          \
    rbt_insert(marg->param_seen, p, NULL);                                     \
  }

MARG_TRACK_FN(marg_track_pos, r_positions, m_positions)
MARG_TRACK_FN(marg_track_rot, r_rotations, m_rotations)
MARG_TRACK_FN(marg_track_pose, r_poses, m_poses)
MARG_TRACK_FN(marg_track_velocity, r_velocities, m_velocities)
MARG_TRACK_FN(marg_track_imu_biases, r_imu_biases, m_imu_biases)
MARG_TRACK_FN(marg_track_feature, r_features, m_features)
MARG_TRACK_FN(marg_track_fiducial, r_fiducials, m_fiducials)
MARG_TRACK_FN(marg_track_extrinsic, r_extrinsics, m_extrinsics)
MARG_TRACK_FN(marg_track_joint, r_joints, m_joints)
MARG_TRACK_FN(marg_track_camera, r_cam_params, m_cam_params)
MARG_TRACK_FN(marg_track_time_delay, r_time_delays, m_time_delays)

static void marg_track_factor(marg_factor_t *marg,
                              const int param_type,
                              real_t *param) {
  switch (param_type) {
    case POSITION_PARAM:
      marg_track_pos(marg, param);
      break;
    case ROTATION_PARAM:
      marg_track_rot(marg, param);
      break;
    case POSE_PARAM:
      marg_track_pose(marg, param);
      break;
    case VELOCITY_PARAM:
      marg_track_velocity(marg, param);
      break;
    case IMU_BIASES_PARAM:
      marg_track_imu_biases(marg, param);
      break;
    case FEATURE_PARAM:
      marg_track_feature(marg, param);
      break;
    case FIDUCIAL_PARAM:
      marg_track_fiducial(marg, param);
      break;
    case EXTRINSIC_PARAM:
      marg_track_extrinsic(marg, param);
      break;
    case JOINT_PARAM:
      marg_track_joint(marg, param);
      break;
    case CAMERA_PARAM:
      marg_track_camera(marg, param);
      break;
    case TIME_DELAY_PARAM:
      marg_track_time_delay(marg, param);
      break;
    default:
      FATAL("Implementation Error!\n");
      break;
  }
}

#define MARG_RINDEX(PARAM_LIST, PARAM_ENUM, COL_IDX)                           \
  {                                                                            \
    list_node_t *node = PARAM_LIST->first;                                     \
    while (node != NULL) {                                                     \
      real_t *p = node->value;                                                 \
      const int fix = rbt_contains(marg->fix_params, p);                       \
      if (fix == 0) {                                                          \
        marg->r_lsize += param_local_size(PARAM_ENUM);                         \
        marg->r_gsize += param_global_size(PARAM_ENUM);                        \
        marg->num_params += 1;                                                 \
      }                                                                        \
      param_index_add(marg->param_index, PARAM_ENUM, fix, p, &COL_IDX);        \
      node = node->next;                                                       \
    }                                                                          \
  }

#define MARG_MINDEX(PARAM_LIST, PARAM_ENUM, COL_IDX)                           \
  {                                                                            \
    list_node_t *node = PARAM_LIST->first;                                     \
    while (node != NULL) {                                                     \
      real_t *p = node->value;                                                 \
      const int fix = rbt_contains(marg->fix_params, p);                       \
      if (fix == 0) {                                                          \
        marg->m_lsize += param_local_size(PARAM_ENUM);                         \
        marg->m_gsize += param_global_size(PARAM_ENUM);                        \
      }                                                                        \
      param_index_add(marg->param_index, PARAM_ENUM, fix, p, &COL_IDX);        \
      node = node->next;                                                       \
    }                                                                          \
  }

#define MARG_PARAMS(MARG, PARAM_LIST, PARAM_ENUM, PARAM_IDX, X0_IDX)           \
  {                                                                            \
    list_node_t *node = PARAM_LIST->first;                                     \
    while (node != NULL) {                                                     \
      real_t *param_ptr = node->value;                                         \
      const size_t param_size = param_global_size(PARAM_ENUM);                 \
      if (rbt_contains(marg->fix_params, param_ptr)) {                         \
        node = node->next;                                                     \
        continue;                                                              \
      }                                                                        \
      MARG->param_types[PARAM_IDX] = PARAM_ENUM;                               \
      MARG->params[PARAM_IDX] = param_ptr;                                     \
      PARAM_IDX++;                                                             \
                                                                               \
      vec_copy(param_ptr, param_size, MARG->x0 + X0_IDX);                      \
      X0_IDX += param_size;                                                    \
      node = node->next;                                                       \
    }                                                                          \
  }

#define MARG_H(MARG, FACTOR_TYPE, FACTORS, H, G, LOCAL_SIZE)                   \
  {                                                                            \
    list_node_t *node = FACTORS->first;                                        \
    while (node != NULL) {                                                     \
      FACTOR_TYPE *factor = (FACTOR_TYPE *) node->value;                       \
      solver_fill_hessian(marg->param_index,                                   \
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

/**
 * Form Hessian matrix using data in marginalization factor.
 */
static void marg_factor_hessian_form(marg_factor_t *marg) {
  // Track Factor Params
  // -- Track marginalization factor params
  if (marg->marg_factor) {
    for (int i = 0; i < marg->marg_factor->num_params; i++) {
      void *param = marg->marg_factor->param_ptrs[i];
      int param_type = marg->marg_factor->param_types[i];
      marg_track_factor(marg, param_type, param);
    }
  }
  // -- Track BA factor params
  {
    list_node_t *node = marg->ba_factors->first;
    while (node != NULL) {
      ba_factor_t *factor = (ba_factor_t *) node->value;
      marg_track_pose(marg, factor->pose);
      marg_track_feature(marg, factor->feature);
      marg_track_camera(marg, factor->camera->data);
      node = node->next;
    }
  }
  // -- Track camera factor params
  {
    list_node_t *node = marg->camera_factors->first;
    while (node != NULL) {
      camera_factor_t *factor = (camera_factor_t *) node->value;
      marg_track_pose(marg, factor->pose);
      marg_track_extrinsic(marg, factor->extrinsic);
      marg_track_feature(marg, factor->feature);
      marg_track_camera(marg, factor->camera->data);
      node = node->next;
    }
  }
  // printf("num params seen:              %ld\n", rbt_size(marg->param_seen));
  // printf("num_poses      [remain]     : %ld\n", marg->r_poses->length);
  // printf("num_poses      [marginalize]: %ld\n", marg->m_poses->length);
  // printf("num_extrinsics [remain]     : %ld\n", marg->r_extrinsics->length);
  // printf("num_extrinsics [marginalize]: %ld\n", marg->m_extrinsics->length);
  // printf("num_features   [remain]     : %ld\n", marg->r_features->length);
  // printf("num_features   [marginalize]: %ld\n", marg->m_features->length);
  // printf("num_cam_params [remain]     : %ld\n", marg->r_cam_params->length);
  // printf("num_cam_params [marginalize]: %ld\n", marg->m_cam_params->length);
  // -- Track IMU factor params
  {
    list_node_t *node = marg->imu_factors->first;
    while (node != NULL) {
      imu_factor_t *factor = (imu_factor_t *) node->value;
      marg_track_pose(marg, factor->pose_i);
      marg_track_velocity(marg, factor->vel_i);
      marg_track_imu_biases(marg, factor->biases_i);
      marg_track_pose(marg, factor->pose_j);
      marg_track_velocity(marg, factor->vel_j);
      marg_track_imu_biases(marg, factor->biases_j);
      node = node->next;
    }
  }
  // -- Track calib camera factor params
  {
    list_node_t *node = marg->calib_camera_factors->first;
    while (node != NULL) {
      calib_camera_factor_t *factor = (calib_camera_factor_t *) node->value;
      marg_track_pose(marg, factor->pose);
      marg_track_extrinsic(marg, factor->cam_ext);
      marg_track_camera(marg, factor->cam_params->data);
      node = node->next;
    }
  }
  // -- Track calib imucam factor params
  {
    list_node_t *node = marg->calib_imucam_factors->first;
    while (node != NULL) {
      calib_imucam_factor_t *factor = (calib_imucam_factor_t *) node->value;
      marg_track_fiducial(marg, factor->fiducial);
      marg_track_pose(marg, factor->imu_pose);
      marg_track_extrinsic(marg, factor->imu_ext);
      marg_track_extrinsic(marg, factor->cam_ext);
      marg_track_camera(marg, factor->camera->data);
      marg_track_time_delay(marg, factor->time_delay);
      node = node->next;
    }
  }

  // Determine parameter block column indicies for Hessian matrix H
  int H_idx = 0; // Column / row index of Hessian matrix H
  // -- Column indices for parameter blocks to be marginalized
  MARG_MINDEX(marg->m_positions, POSITION_PARAM, H_idx);
  MARG_MINDEX(marg->m_rotations, ROTATION_PARAM, H_idx);
  MARG_MINDEX(marg->m_poses, POSE_PARAM, H_idx);
  MARG_MINDEX(marg->m_velocities, VELOCITY_PARAM, H_idx);
  MARG_MINDEX(marg->m_imu_biases, IMU_BIASES_PARAM, H_idx);
  MARG_MINDEX(marg->m_features, FEATURE_PARAM, H_idx);
  MARG_MINDEX(marg->m_joints, JOINT_PARAM, H_idx);
  MARG_MINDEX(marg->m_extrinsics, EXTRINSIC_PARAM, H_idx);
  MARG_MINDEX(marg->m_fiducials, FIDUCIAL_PARAM, H_idx);
  MARG_MINDEX(marg->m_cam_params, CAMERA_PARAM, H_idx);
  MARG_MINDEX(marg->m_time_delays, TIME_DELAY_PARAM, H_idx);
  // -- Column indices for parameter blocks to remain
  MARG_RINDEX(marg->r_positions, POSITION_PARAM, H_idx);
  MARG_RINDEX(marg->r_rotations, ROTATION_PARAM, H_idx);
  MARG_RINDEX(marg->r_poses, POSE_PARAM, H_idx);
  MARG_RINDEX(marg->r_velocities, VELOCITY_PARAM, H_idx);
  MARG_RINDEX(marg->r_imu_biases, IMU_BIASES_PARAM, H_idx);
  MARG_RINDEX(marg->r_features, FEATURE_PARAM, H_idx);
  MARG_RINDEX(marg->r_joints, JOINT_PARAM, H_idx);
  MARG_RINDEX(marg->r_extrinsics, EXTRINSIC_PARAM, H_idx);
  MARG_RINDEX(marg->r_fiducials, FIDUCIAL_PARAM, H_idx);
  MARG_RINDEX(marg->r_cam_params, CAMERA_PARAM, H_idx);
  MARG_RINDEX(marg->r_time_delays, TIME_DELAY_PARAM, H_idx);

  // Track linearization point x0 and parameter pointers
  assert(marg->m_lsize > 0);
  assert(marg->m_gsize > 0);
  assert(marg->r_lsize > 0);
  assert(marg->r_gsize > 0);

  int param_idx = 0;
  int x0_idx = 0;
  marg->x0 = malloc(sizeof(real_t) * marg->r_gsize);
  marg->param_types = malloc(sizeof(int) * marg->num_params);
  marg->params = malloc(sizeof(real_t *) * marg->num_params);
  MARG_PARAMS(marg, marg->r_positions, POSITION_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_rotations, ROTATION_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_poses, POSE_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_velocities, VELOCITY_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_imu_biases, IMU_BIASES_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_features, FEATURE_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_joints, JOINT_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_extrinsics, EXTRINSIC_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_fiducials, FIDUCIAL_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_cam_params, CAMERA_PARAM, param_idx, x0_idx);
  MARG_PARAMS(marg, marg->r_time_delays, TIME_DELAY_PARAM, param_idx, x0_idx);

  // Allocate memory LHS and RHS of Gauss newton
  const int ls = marg->m_lsize + marg->r_lsize;
  real_t *H = calloc(ls * ls, sizeof(real_t));
  real_t *b = calloc(ls * 1, sizeof(real_t));

  // Fill Hessian
  if (marg->marg_factor) {
    solver_fill_hessian(marg->param_index,
                        marg->marg_factor->num_params,
                        marg->marg_factor->params,
                        marg->marg_factor->jacs,
                        marg->marg_factor->r,
                        marg->marg_factor->r_lsize,
                        ls,
                        H,
                        b);
  }

  // param_index_print(marg->param_index);
  MARG_H(marg, ba_factor_t, marg->ba_factors, H, b, ls);
  MARG_H(marg, camera_factor_t, marg->camera_factors, H, b, ls);
  MARG_H(marg, imu_factor_t, marg->imu_factors, H, b, ls);
  MARG_H(marg, calib_camera_factor_t, marg->calib_camera_factors, H, b, ls);
  MARG_H(marg, calib_imucam_factor_t, marg->calib_imucam_factors, H, b, ls);
  marg->H = H;
  marg->b = b;
}

/**
 * Perform Schur-Complement.
 */
static void marg_factor_schur_complement(marg_factor_t *marg) {
  assert(marg);
  assert(marg->m_lsize > 0);
  assert(marg->r_lsize > 0);

  // Compute Schurs Complement
  const int m = marg->m_lsize;
  const int r = marg->r_lsize;
  const int ls = m + r;
  const real_t *H = marg->H;
  const real_t *b = marg->b;
  real_t *H_marg = malloc(sizeof(real_t) * r * r);
  real_t *b_marg = malloc(sizeof(real_t) * r * 1);
  if (schur_complement(H, b, ls, m, r, H_marg, b_marg) == 0) {
    marg->schur_complement_ok = 1;
  }
  marg->H_marg = H_marg;
  marg->b_marg = b_marg;

  // Enforce symmetry: H_marg = 0.5 * (H_marg + H_marg')
  if (marg->cond_hessian) {
    enforce_spd(marg->H_marg, r, r);
  }
}

/**
 * Decompose Hessian into two Jacobians.
 */
static void marg_factor_hessian_decomp(marg_factor_t *marg) {
  // Decompose H_marg into Jt and J, and in the process also obtain inv(J).
  // Hessian H_marg can be decomposed via Eigen-decomposition:
  //
  //   H_marg = J' * J = V * diag(w) * V'
  //   J = diag(w^{0.5}) * V'
  //   J_inv = diag(w^-0.5) * V'
  //
  // -- Setup
  const int r = marg->r_lsize;
  real_t *J = calloc(r * r, sizeof(real_t));
  real_t *J_inv = calloc(r * r, sizeof(real_t));
  real_t *V = calloc(r * r, sizeof(real_t));
  real_t *Vt = calloc(r * r, sizeof(real_t));
  real_t *w = calloc(r, sizeof(real_t));
  real_t *W_sqrt = calloc(r * r, sizeof(real_t));
  real_t *W_inv_sqrt = calloc(r * r, sizeof(real_t));

  // -- Eigen decomposition
  if (eig_sym(marg->H_marg, r, r, V, w) != 0) {
    free(J);
    free(J_inv);
    free(V);
    free(Vt);
    free(w);
    free(W_sqrt);
    free(W_inv_sqrt);
    return;
  }
  mat_transpose(V, r, r, Vt);

  // -- Form J and J_inv:
  //
  //   J = diag(w^0.5) * V'
  //   J_inv = diag(w^-0.5) * V'
  //
  const real_t tol = 1e-18;
  for (int i = 0; i < r; i++) {
    if (w[i] > tol) {
      W_sqrt[(i * r) + i] = sqrt(w[i]);
      W_inv_sqrt[(i * r) + i] = sqrt(1.0 / w[i]);
    } else {
      W_sqrt[(i * r) + i] = 0.0;
      W_inv_sqrt[(i * r) + i] = 0.0;
    }
  }
  dot(W_sqrt, r, r, Vt, r, r, J);
  dot(W_inv_sqrt, r, r, Vt, r, r, J_inv);
  mat_scale(J_inv, r, r, -1.0);
  marg->eigen_decomp_ok = 1;

  // Check J' * J == H_marg
  if (marg->debug) {
    real_t *Jt = calloc(r * r, sizeof(real_t));
    real_t *H_ = calloc(r * r, sizeof(real_t));
    mat_transpose(J, r, r, Jt);
    dot(Jt, r, r, J, r, r, H_);

    real_t diff = 0.0;
    for (int i = 0; i < (r * r); i++) {
      diff += pow(H_[i] - marg->H_marg[i], 2);
    }

    if (diff > 1e-2) {
      marg->eigen_decomp_ok = 0;
      LOG_WARN("J' * J != H_marg. Diff is %.2e\n", diff);
      LOG_WARN("This is bad ... Usually means marginalization "
               "is bad!\n");
    }

    free(Jt);
    free(H_);
  }

  // Check J_inv * J == eye
  // if (marg->debug) {
  //   if (check_inv(J, J_inv, r) != 0) {
  //     marg->eigen_decomp_ok = 0;
  //     LOG_WARN("inv(J) * J != eye\n");
  //   }
  // }

  // Update
  marg->J0 = J;
  marg->J0_inv = J_inv;

  // Clean up
  free(V);
  free(Vt);
  free(w);
  free(W_sqrt);
  free(W_inv_sqrt);
}

static void marg_factor_form_fejs(marg_factor_t *marg) {
  // Track Linearized residuals, jacobians
  // -- Linearized residuals: r0 = -J0_inv * b_marg;
  marg->r0 = malloc(sizeof(real_t) * marg->r_lsize);
  dot(marg->J0_inv,
      marg->r_lsize,
      marg->r_lsize,
      marg->b_marg,
      marg->r_lsize,
      1,
      marg->r0);
  // -- Linearized jacobians: J0 = J;
  marg->dchi = malloc(sizeof(real_t) * marg->r_lsize);
  marg->J0_dchi = malloc(sizeof(real_t) * marg->r_lsize);

  // Form First-Estimate Jacobians (FEJ)
  const size_t m = marg->r_lsize;
  const int col_offset = -marg->m_lsize;
  const int rs = 0;
  const int re = m - 1;
  marg->r = malloc(sizeof(real_t) * m);
  marg->jacs = malloc(sizeof(real_t *) * marg->num_params);

  char param_type[100] = {0};
  for (size_t i = 0; i < marg->num_params; i++) {
    real_t *param_ptr = marg->params[i];
    const param_info_t *param_info = rbt_search(marg->param_index, param_ptr);
    param_type_string(param_info->type, param_type);
    const int n = param_local_size(param_info->type);
    const int cs = param_info->idx + col_offset;
    const int ce = cs + n - 1;

    marg->jacs[i] = malloc(sizeof(real_t) * m * n);
    mat_block_get(marg->J0, m, rs, re, cs, ce, marg->jacs[i]);
  }
}

void marg_factor_marginalize(marg_factor_t *marg,
                             const rbt_t *marg_params,
                             const rbt_t *fix_params) {
  assert(marg);
  assert(marg_params);
  assert(fix_params);

  // Setup
  marg->marg_params = marg_params;
  marg->fix_params = fix_params;

  // Form Hessian and RHS of Gauss newton
  tic();
  marg_factor_hessian_form(marg);
  marg->time_hessian_form = toc();
  marg->time_total += marg->time_hessian_form;

  // Apply Schur Complement
  tic();
  marg_factor_schur_complement(marg);
  marg->time_schur_complement = toc();
  marg->time_total += marg->time_schur_complement;

  // Decompose marginalized Hessian
  tic();
  marg_factor_hessian_decomp(marg);
  marg->time_hessian_decomp = toc();
  marg->time_total += marg->time_hessian_decomp;

  // Form FEJs
  tic();
  marg_factor_form_fejs(marg);
  marg->time_fejs = toc();
  marg->time_total += marg->time_fejs;

  // Update state
  marg->marginalized = 1;
}

int marg_factor_eval(void *marg_ptr) {
  assert(marg_ptr);

  // Map factor
  marg_factor_t *marg = (marg_factor_t *) marg_ptr;
  assert(marg->marginalized == 1);

  // Compute residuals
  // -- Compute dchi vector
  int param_row_idx = 0;
  int dchi_row_idx = 0;
  for (size_t i = 0; i < marg->num_params; i++) {
    const int param_type = marg->param_types[i];
    const int param_size = param_global_size(param_type);
    const int local_size = param_local_size(param_type);
    const real_t *x0 = marg->x0 + param_row_idx;
    const real_t *x = marg->params[i];

    // Calculate i-th dchi
    switch (param_type) {
      case POSE_PARAM:
      case FIDUCIAL_PARAM:
      case EXTRINSIC_PARAM: {
        // Pose minus
        // dr = r - r0
        const real_t dr[3] = {x[0] - x0[0], x[1] - x0[1], x[2] - x0[2]};

        // dq = q0.inverse() * q
        const real_t q[4] = {x[3], x[4], x[5], x[6]};
        const real_t q0[4] = {x0[3], x0[4], x0[5], x0[6]};
        real_t q0_inv[4] = {0};
        real_t dq[4] = {0};
        quat_inv(q0, q0_inv);
        quat_mul(q0_inv, q, dq);

        marg->dchi[dchi_row_idx + 0] = dr[0];
        marg->dchi[dchi_row_idx + 1] = dr[1];
        marg->dchi[dchi_row_idx + 2] = dr[2];
        marg->dchi[dchi_row_idx + 3] = 2.0 * dq[1];
        marg->dchi[dchi_row_idx + 4] = 2.0 * dq[2];
        marg->dchi[dchi_row_idx + 5] = 2.0 * dq[3];
      } break;
      default:
        // Trivial minus: x - x0
        vec_sub(x, x0, marg->dchi + dchi_row_idx, param_size);
        break;
    }
    param_row_idx += param_size;
    dchi_row_idx += local_size;
  }
  // -- Compute residuals: r = r0 + J0 * dchi;
  dot(marg->J0,
      marg->r_lsize,
      marg->r_lsize,
      marg->dchi,
      marg->r_lsize,
      1,
      marg->J0_dchi);
  for (int i = 0; i < marg->r_lsize; i++) {
    marg->r[i] = marg->r0[i] + marg->J0_dchi[i];
  }

  return 0;
}

////////////////
// DATA UTILS //
////////////////

/**
 * Save poses to `save_path`.
 */
int save_poses(const char *save_path,
               const timestamp_t *timestamps,
               const real_t *poses,
               const int num_poses) {
  assert(save_path != NULL);
  assert(timestamps != NULL);
  assert(poses != NULL);

  FILE *fp = fopen(save_path, "w");
  if (fp == NULL) {
    return -1;
  }

  fprintf(fp, "num_poses: %d\n", num_poses);
  fprintf(fp, "ts x y z qw qx qy qz\n");
  for (int i = 0; i < num_poses; ++i) {
    fprintf(fp, "%ld ", timestamps[i]);
#if PRECISION == 1
    fprintf(fp, "%f ", poses[i * 7 + 0]);
    fprintf(fp, "%f ", poses[i * 7 + 1]);
    fprintf(fp, "%f ", poses[i * 7 + 2]);
    fprintf(fp, "%f ", poses[i * 7 + 3]);
    fprintf(fp, "%f ", poses[i * 7 + 4]);
    fprintf(fp, "%f ", poses[i * 7 + 5]);
    fprintf(fp, "%f\n", poses[i * 7 + 6]);
#elif PRECISION == 2
    fprintf(fp, "%lf ", poses[i * 7 + 0]);
    fprintf(fp, "%lf ", poses[i * 7 + 1]);
    fprintf(fp, "%lf ", poses[i * 7 + 2]);
    fprintf(fp, "%lf ", poses[i * 7 + 3]);
    fprintf(fp, "%lf ", poses[i * 7 + 4]);
    fprintf(fp, "%lf ", poses[i * 7 + 5]);
    fprintf(fp, "%lf\n", poses[i * 7 + 6]);
#else
#error "Invalid precision!"
#endif
  }
  fclose(fp);

  return 0;
}

/**
 * Load poses from file `data_path`. The number of poses in file
 * will be outputted to `num_poses`.
 */
int load_poses(const char *data_path,
               timestamp_t **timestamps,
               real_t **poses,
               int *num_poses) {
  assert(data_path != NULL);
  assert(timestamps != NULL);
  assert(poses != NULL);
  assert(num_poses != NULL);

  // Load file
  FILE *fp = fopen(data_path, "r");
  if (fp == NULL) {
    return -1;
  }

  // Parse number of poses
  fscanf(fp, "num_poses: %d\n", num_poses);
  skip_line(fp);

  // Initialize memory for pose data
  *timestamps = malloc(sizeof(timestamp_t) * *num_poses);
  *poses = malloc(sizeof(real_t) * 7 * *num_poses);

  for (int i = 0; i < *num_poses; ++i) {
    timestamp_t ts = 0;
    real_t rx, ry, rz = 0;
    real_t qx, qy, qz, qw = 0;
#if PRECISION == 1
    const char *fmt = "%ld %f %f %f %f %f %f %f";
#elif PRECISION == 2
    const char *fmt = "%ld %lf %lf %lf %lf %lf %lf %lf";
#else
#error "Invalid precision!"
#endif
    if (fscanf(fp, fmt, &ts, &rx, &ry, &rz, &qw, &qx, &qy, &qz) != 8) {
      fclose(fp);
      return -1;
    }

    (*timestamps)[i] = ts;
    (*poses)[i * 7 + 0] = rx;
    (*poses)[i * 7 + 1] = ry;
    (*poses)[i * 7 + 2] = rz;
    (*poses)[i * 7 + 3] = qw;
    (*poses)[i * 7 + 4] = qx;
    (*poses)[i * 7 + 5] = qy;
    (*poses)[i * 7 + 6] = qz;
  }

  // Clean up
  fclose(fp);

  return 0;
}

// /**
//  * Associate pose data
//  */
// int **assoc_pose_data(pose_t *gnd_poses,
//                       size_t num_gnd_poses,
//                       pose_t *est_poses,
//                       size_t num_est_poses,
//                       double threshold,
//                       size_t *num_matches) {
//   assert(gnd_poses != NULL);
//   assert(est_poses != NULL);
//   assert(num_gnd_poses != 0);
//   assert(num_est_poses != 0);
//
//   size_t gnd_idx = 0;
//   size_t est_idx = 0;
//   size_t k_end =
//       (num_gnd_poses > num_est_poses) ? num_est_poses : num_gnd_poses;
//
//   size_t match_idx = 0;
//   int **matches = malloc(sizeof(int *) * k_end);
//
//   while ((gnd_idx + 1) < num_gnd_poses && (est_idx + 1) < num_est_poses) {
//     // Calculate time difference between ground truth and
//     // estimate
//     double gnd_k_time = ts2sec(gnd_poses[gnd_idx].ts);
//     double est_k_time = ts2sec(est_poses[est_idx].ts);
//     double t_k_diff = fabs(gnd_k_time - est_k_time);
//
//     // Check to see if next ground truth timestamp forms
//     // a smaller time diff
//     double t_kp1_diff = threshold;
//     if ((gnd_idx + 1) < num_gnd_poses) {
//       double gnd_kp1_time = ts2sec(gnd_poses[gnd_idx + 1].ts);
//       t_kp1_diff = fabs(gnd_kp1_time - est_k_time);
//     }
//
//     // Conditions to call this pair (ground truth and
//     // estimate) a match
//     int threshold_met = t_k_diff < threshold;
//     int smallest_diff = t_k_diff < t_kp1_diff;
//
//     // Mark pairs as a match or increment appropriate
//     // indices
//     if (threshold_met && smallest_diff) {
//       matches[match_idx] = malloc(sizeof(int) * 2);
//       matches[match_idx][0] = gnd_idx;
//       matches[match_idx][1] = est_idx;
//       match_idx++;
//
//       gnd_idx++;
//       est_idx++;
//
//     } else if (gnd_k_time > est_k_time) {
//       est_idx++;
//
//     } else if (gnd_k_time < est_k_time) {
//       gnd_idx++;
//     }
//   }
//
//   // Clean up
//   if (match_idx == 0) {
//     free(matches);
//     matches = NULL;
//   }
//
//   *num_matches = match_idx;
//   return matches;
// }

////////////
// SOLVER //
////////////

/**
 * Setup Solver
 */
void solver_setup(solver_t *solver) {
  assert(solver);

  // Settings
  solver->verbose = 0;
  solver->max_iter = 10;
  solver->lambda = 1e4;
  solver->lambda_factor = 10.0;

  // Data
  solver->param_index = NULL;
  solver->linearize = 0;
  solver->r_size = 0;
  solver->sv_size = 0;
  solver->H_damped = NULL;
  solver->H = NULL;
  solver->g = NULL;
  solver->r = NULL;
  solver->dx = NULL;

  // SuiteSparse
#ifdef SOLVER_USE_SUITESPARSE
  solver->common = NULL;
#endif

  // Callbacks
  solver->param_index_func = NULL;
  solver->cost_func = NULL;
  solver->linearize_func = NULL;
  solver->linsolve_func = NULL;
}

/**
 * Calculate cost with residual vector `r` of length `r_size`.
 */
real_t solver_cost(const solver_t *solver, const void *data) {
  solver->cost_func(data, solver->r);
  real_t r_sq = {0};
  dot(solver->r, 1, solver->r_size, solver->r, solver->r_size, 1, &r_sq);
  return 0.5 * r_sq;
}

/**
 * Fill Jacobian matrix
 */
void solver_fill_jacobian(rbt_t *param_index,
                          int num_params,
                          real_t **params,
                          real_t **jacs,
                          real_t *r,
                          int r_size,
                          int sv_size,
                          int J_row_idx,
                          real_t *J,
                          real_t *g) {
  for (int i = 0; i < num_params; i++) {
    // Check if i-th parameter is fixed
    param_info_t *info = rbt_search(param_index, params[i]);
    if (info->fix) {
      continue;
    }

    // Get i-th parameter and corresponding Jacobian
    int idx_i = info->idx;
    int size_i = param_local_size(info->type);
    const real_t *J_i = jacs[i];

    // Fill in the Jacobian
    const int rs = J_row_idx;
    const int re = rs + r_size - 1;
    const int cs = idx_i;
    const int ce = idx_i + size_i - 1;
    mat_block_set(J, sv_size, rs, re, cs, ce, J_i);

    // Fill in the R.H.S of H dx = g, where g = -J_i' * r
    real_t *Jt_i = malloc(sizeof(real_t) * r_size * size_i);
    real_t *g_i = malloc(sizeof(real_t) * size_i);
    mat_transpose(J_i, r_size, size_i, Jt_i);
    mat_scale(Jt_i, size_i, r_size, -1);
    dot(Jt_i, size_i, r_size, r, r_size, 1, g_i);
    for (int g_idx = 0; g_idx < size_i; g_idx++) {
      g[idx_i + g_idx] += g_i[g_idx];
    }

    // Clean up
    free(g_i);
    free(Jt_i);
  }
}

/**
 * Fill Hessian matrix
 */
void solver_fill_hessian(rbt_t *param_index,
                         int num_params,
                         real_t **params,
                         real_t **jacs,
                         real_t *r,
                         int r_size,
                         int sv_size,
                         real_t *H,
                         real_t *g) {
  if (H == NULL || g == NULL) {
    return;
  }

  for (int i = 0; i < num_params; ++i) {
    // Check if i-th parameter is fixed
    assert(rbt_contains(param_index, params[i]));
    param_info_t *info_i = rbt_search(param_index, params[i]);
    if (info_i->fix) {
      continue;
    }

    // Get i-th parameter and corresponding Jacobian
    int idx_i = info_i->idx;
    int size_i = param_local_size(info_i->type);
    const real_t *J_i = jacs[i];
    real_t *Jt_i = malloc(sizeof(real_t) * r_size * size_i);
    mat_transpose(J_i, r_size, size_i, Jt_i);

    for (int j = i; j < num_params; ++j) {
      // Check if j-th parameter is fixed
      assert(rbt_contains(param_index, params[j]));
      param_info_t *info_j = rbt_search(param_index, params[j]);
      if (info_j->fix) {
        continue;
      }

      // Get j-th parameter and corresponding Jacobian
      int idx_j = info_j->idx;
      int size_j = param_local_size(info_j->type);
      const real_t *J_j = jacs[j];
      real_t *H_ij = malloc(sizeof(real_t) * size_i * size_j);
      dot(Jt_i, size_i, r_size, J_j, r_size, size_j, H_ij);

      // Fill Hessian H
      int rs = idx_i;
      int re = idx_i + size_i - 1;
      int cs = idx_j;
      int ce = idx_j + size_j - 1;

      if (i == j) {
        // Fill diagonal
        mat_block_add(H, sv_size, rs, re, cs, ce, H_ij);
      } else {
        // Fill off-diagonal
        real_t *H_ji = malloc(sizeof(real_t) * size_j * size_i);
        mat_transpose(H_ij, size_i, size_j, H_ji);
        mat_block_add(H, sv_size, rs, re, cs, ce, H_ij);
        mat_block_add(H, sv_size, cs, ce, rs, re, H_ji);
        free(H_ji);
      }

      // Clean up
      free(H_ij);
    }

    // Fill in the R.H.S of H dx = g, where g = -J_i' * r
    real_t *g_i = malloc(sizeof(real_t) * size_i);
    mat_scale(Jt_i, size_i, r_size, -1);
    dot(Jt_i, size_i, r_size, r, r_size, 1, g_i);
    for (int g_idx = 0; g_idx < size_i; g_idx++) {
      g[idx_i + g_idx] += g_i[g_idx];
    }

    // Clean up
    free(g_i);
    free(Jt_i);
  }
}

/**
 * Create a copy of the parameter vector
 */
real_t **solver_params_copy(const solver_t *solver) {
  real_t **x = malloc(sizeof(real_t *) * rbt_size(solver->param_index));
  const size_t n = rbt_size(solver->param_index);
  arr_t *keys = arr_malloc(n);
  arr_t *vals = arr_malloc(n);
  rbt_keys_values(solver->param_index, keys, vals);

  for (size_t idx = 0; idx < n; ++idx) {
    const param_info_t *info = vals->data[idx];
    const int global_size = param_global_size(info->type);
    x[idx] = malloc(sizeof(real_t) * global_size);

    for (int i = 0; i < global_size; i++) {
      x[idx][i] = ((real_t *) info->data)[i];
    }
  }

  arr_free(keys);
  arr_free(vals);

  return x;
}

/**
 * Restore parameter values
 */
void solver_params_restore(solver_t *solver, real_t **x) {
  const size_t n = rbt_size(solver->param_index);
  arr_t *keys = arr_malloc(n);
  arr_t *vals = arr_malloc(n);
  rbt_keys_values(solver->param_index, keys, vals);

  for (size_t idx = 0; idx < n; ++idx) {
    const param_info_t *info = vals->data[idx];
    for (int i = 0; i < param_global_size(info->type); ++i) {
      ((real_t *) info->data)[i] = x[idx][i];
    }
  }

  arr_free(keys);
  arr_free(vals);
}

/**
 * Free params
 */
void solver_params_free(const solver_t *solver, real_t **x) {
  for (int idx = 0; idx < rbt_size(solver->param_index); ++idx) {
    free(x[idx]);
  }
  free(x);
}

/**
 * Update parameter
 */
void solver_update(solver_t *solver, real_t *dx, int sv_size) {
  const size_t n = rbt_size(solver->param_index);
  arr_t *keys = arr_malloc(n);
  arr_t *vals = arr_malloc(n);
  rbt_keys_values(solver->param_index, keys, vals);

  for (int i = 0; i < n; ++i) {
    const param_info_t *info = vals->data[i];
    if (info->fix) {
      continue;
    }

    real_t *data = info->data;
    int idx = info->idx;
    switch (info->type) {
      case POSITION_PARAM:
        for (int i = 0; i < 3; i++) {
          data[i] += dx[idx + i];
        }
        break;
      case POSE_PARAM:
      case FIDUCIAL_PARAM:
      case EXTRINSIC_PARAM:
        pose_update(data, dx + idx);
        break;
      case VELOCITY_PARAM:
        for (int i = 0; i < 3; i++) {
          data[i] += dx[idx + i];
        }
        break;
      case IMU_BIASES_PARAM:
        for (int i = 0; i < 6; i++) {
          data[i] += dx[idx + i];
        }
        break;
      case FEATURE_PARAM:
        for (int i = 0; i < 3; i++) {
          data[i] += dx[idx + i];
        }
        break;
      case JOINT_PARAM:
        data[0] += dx[idx];
        break;
      case CAMERA_PARAM:
        for (int i = 0; i < 8; i++) {
          data[i] += dx[idx + i];
        }
        break;
      case TIME_DELAY_PARAM:
        data[0] += dx[idx];
        break;
      default:
        FATAL("Invalid param type [%d]!\n", info->type);
        break;
    }
  }

  arr_free(keys);
  arr_free(vals);
}

/**
 * Step nonlinear least squares problem.
 */
real_t **solver_step(solver_t *solver, const real_t lambda_k, void *data) {
  // Linearize non-linear system
  if (solver->linearize) {
    // Linearize
    zeros(solver->H, solver->sv_size, solver->sv_size);
    zeros(solver->g, solver->sv_size, 1);
    zeros(solver->r, solver->r_size, 1);

    solver->linearize_func(data,
                           solver->sv_size,
                           solver->param_index,
                           solver->H,
                           solver->g,
                           solver->r);
  }

  // Damp Hessian: H = H + lambda * I
  mat_copy(solver->H, solver->sv_size, solver->sv_size, solver->H_damped);
  for (int i = 0; i < solver->sv_size; i++) {
    solver->H_damped[(i * solver->sv_size) + i] += lambda_k;
  }

  // Solve non-linear system
  if (solver->linsolve_func) {
    solver->linsolve_func(data,
                          solver->sv_size,
                          solver->param_index,
                          solver->H_damped,
                          solver->g,
                          solver->dx);
  } else {
    // Solve: H * dx = g
#ifdef SOLVER_USE_SUITESPARSE
    suitesparse_chol_solve(solver->common,
                           solver->H_damped,
                           solver->sv_size,
                           solver->sv_size,
                           solver->g,
                           solver->sv_size,
                           solver->dx);
#else
    chol_solve(solver->H_damped, solver->g, solver->dx, solver->sv_size);
#endif
  }

  // Update
  real_t **x_copy = solver_params_copy(solver);
  solver_update(solver, solver->dx, solver->sv_size);

  return x_copy;
}

/**
 * Solve nonlinear least squares problem.
 */
int solver_solve(solver_t *solver, void *data) {
  assert(solver != NULL);
  assert(solver->param_index_func != NULL);
  assert(solver->cost_func != NULL);
  assert(solver->linearize_func != NULL);
  assert(data != NULL);

  // Determine parameter order
  int sv_size = 0;
  int r_size = 0;
  solver->param_index = solver->param_index_func(data, &sv_size, &r_size);
  assert(sv_size > 0);
  assert(r_size > 0);

  // Calculate initial cost
  solver->linearize = 1;
  solver->r_size = r_size;
  solver->sv_size = sv_size;
  solver->H_damped = calloc(sv_size * sv_size, sizeof(real_t));
  solver->H = calloc(sv_size * sv_size, sizeof(real_t));
  solver->g = calloc(sv_size, sizeof(real_t));
  solver->r = calloc(r_size, sizeof(real_t));
  solver->dx = calloc(sv_size, sizeof(real_t));
  real_t J_km1 = solver_cost(solver, data);
  if (solver->verbose) {
    printf("iter 0: lambda_k: %.2e, J: %.4e\n", solver->lambda, J_km1);
  }

  // Start cholmod workspace
#ifdef SOLVER_USE_SUITESPARSE
  solver->common = malloc(sizeof(cholmod_common) * 1);
  cholmod_start(solver->common);
#endif

  // Solve
  int max_iter = solver->max_iter;
  real_t lambda_k = solver->lambda;
  real_t J_k = 0.0;

  for (int iter = 0; iter < max_iter; iter++) {
    // Linearize and calculate cost
    real_t **x_copy = solver_step(solver, lambda_k, data);
    J_k = solver_cost(solver, data);

    // Accept or reject update*/
    const real_t dJ = J_k - J_km1;
    const real_t dx_norm = vec_norm(solver->dx, solver->sv_size);
    if (J_k < J_km1) {
      // Accept update
      J_km1 = J_k;
      lambda_k /= solver->lambda_factor;
      solver->linearize = 1;
    } else {
      // Reject update
      lambda_k *= solver->lambda_factor;
      solver_params_restore(solver, x_copy);
      solver->linearize = 0;
    }
    lambda_k = clip_value(lambda_k, 1e-8, 1e8);
    solver_params_free(solver, x_copy);

    // Display
    if (solver->verbose) {
      printf("iter %d: lambda_k: %.2e, J: %.4e, dJ: %.2e, "
             "norm(dx): %.2e\n",
             iter + 1,
             lambda_k,
             J_km1,
             dJ,
             dx_norm);
    }

    // Termination criteria
    if (solver->linearize && fabs(dJ) < fabs(-1e-10)) {
      // printf("dJ < -1e-10\n");
      break;
    } else if (solver->linearize && dx_norm < 1e-10) {
      // printf("dx_norm < 1e-10\n");
      break;
    }
  }

  // Clean up
#ifdef SOLVER_USE_SUITESPARSE
  cholmod_finish(solver->common);
  free(solver->common);
  solver->common = NULL;
#endif
  param_index_free(solver->param_index);
  free(solver->H_damped);
  free(solver->H);
  free(solver->g);
  free(solver->r);
  free(solver->dx);

  return 0;
}

/*******************************************************************************
 * TIMELINE
 ******************************************************************************/

/**
 * Sort timestamps `timestamps` of length `n` with insertion sort.
 */
static void timestamps_insertion_sort(timestamp_t *timestamps, const size_t n) {
  for (size_t i = 1; i < n; i++) {
    timestamp_t key = timestamps[i];
    size_t j = i - 1;

    while (j >= 0 && timestamps[j] > key) {
      timestamps[j + 1] = timestamps[j];
      j = j - 1;
    }
    timestamps[j + 1] = key;
  }
}

/**
 * This function only adds unique timestamps to `set` if it does not already
 * exists.
 */
static void timestamps_unique(timestamp_t *set,
                              size_t *set_len,
                              const timestamp_t *timestamps,
                              const size_t num_timestamps) {
  for (size_t i = 0; i < num_timestamps; i++) {
    const timestamp_t ts_k = timestamps[i];

    // Check duplicate in set
    int dup = 0;
    for (size_t j = 0; j < *set_len; j++) {
      if (set[j] == ts_k) {
        dup = 1;
        break;
      }
    }

    // Add to set if no duplicate
    if (dup == 0) {
      set[*set_len] = ts_k;
      (*set_len)++;
    }
  }

  // Sort timestamps (just to be sure)
  timestamps_insertion_sort(set, *set_len);
}

/**
 * Print camera event.
 */
void print_camera_event(const camera_event_t *event) {
  printf("camera_event:\n");
  printf("  ts: %ld\n", event->ts);
  printf("  cam_idx: %d\n", event->cam_idx);
  if (event->image_path) {
    printf("  image_path: %s\n", event->image_path);
  }
  printf("\n");
  printf("  num_features: %d\n", event->num_features);
  printf("  features: [\n");
  for (size_t i = 0; i < event->num_features; i++) {
    const size_t feature_id = event->feature_ids[i];
    const real_t *kps = &event->keypoints[i * 2 + 0];
    printf("    %zu: [%.2f, %.2f]\n", feature_id, kps[0], kps[1]);
  }
  printf("  ]\n");
}

/**
 * Print IMU event.
 */
void print_imu_event(const imu_event_t *event) {
  printf("imu_event:\n");
  printf("  ts: %ld\n", event->ts);
  printf("  acc: [%.4f, %.4f, %.4f]\n",
         event->acc[0],
         event->acc[1],
         event->acc[2]);
  printf("  gyr: [%.4f, %.4f, %.4f]\n",
         event->gyr[0],
         event->gyr[1],
         event->gyr[2]);
  printf("\n");
}

/**
 * Print Fiducial event.
 */
void print_fiducial_event(const fiducial_event_t *event) {
  printf("fiducial_event:\n");
  printf("  ts: %ld\n", event->ts);
  printf("  cam_idx: %d\n", event->cam_idx);
  printf("  num_corners: %d\n", event->num_corners);
  printf("\n");
  printf("  #tag_id, corner_idx, kp_x, kp_y, p_x, p_y, p_z\n");
  for (int i = 0; i < event->num_corners; i++) {
    const int tag_id = event->tag_ids[i];
    const int corner_idx = event->corner_indices[i];
    printf("  ");
    printf("%d, ", tag_id);
    printf("%d, ", corner_idx);
    printf("%.2f, ", event->keypoints[i * 2 + 0]);
    printf("%.2f, ", event->keypoints[i * 2 + 1]);
    printf("%.2f, ", event->object_points[i * 3 + 0]);
    printf("%.2f, ", event->object_points[i * 3 + 1]);
    printf("%.2f", event->object_points[i * 3 + 2]);
    printf("\n");
  }
  printf("\n");
}

/**
 * Malloc timeline.
 */
timeline_t *timeline_malloc(void) {
  timeline_t *timeline = malloc(sizeof(timeline_t) * 1);

  // Stats
  timeline->num_cams = 0;
  timeline->num_imus = 0;
  timeline->num_event_types = 0;

  // Events
  timeline->events = NULL;
  timeline->events_timestamps = NULL;
  timeline->events_lengths = NULL;
  timeline->events_types = NULL;

  // Timeline
  timeline->timeline_length = 0;
  timeline->timeline_timestamps = 0;
  timeline->timeline_events = 0;
  timeline->timeline_events_lengths = 0;

  return timeline;
}

/**
 * Free timeline.
 */
void timeline_free(timeline_t *timeline) {
  // Pre-check
  if (timeline == NULL) {
    return;
  }

  // Free events
  for (size_t type_idx = 0; type_idx < timeline->num_event_types; type_idx++) {
    for (int k = 0; k < timeline->events_lengths[type_idx]; k++) {
      timeline_event_t *event = &timeline->events[type_idx][k];
      if (event == NULL) {
        continue;
      }

      switch (event->type) {
        case CAMERA_EVENT:
          free(event->data.camera.image_path);
          free(event->data.camera.keypoints);
          break;
        case IMU_EVENT:
          // Do nothing
          break;
        case FIDUCIAL_EVENT:
          free(event->data.fiducial.tag_ids);
          free(event->data.fiducial.corner_indices);
          free(event->data.fiducial.object_points);
          free(event->data.fiducial.keypoints);
          break;
      }
    }
    free(timeline->events[type_idx]);
    free(timeline->events_timestamps[type_idx]);
  }
  free(timeline->events);
  free(timeline->events_timestamps);
  free(timeline->events_lengths);
  free(timeline->events_types);

  // Free timeline
  free(timeline->timeline_timestamps);
  for (int k = 0; k < timeline->timeline_length; k++) {
    free(timeline->timeline_events[k]);
  }
  free(timeline->timeline_events);
  free(timeline->timeline_events_lengths);

  // Free timeline
  free(timeline);
}

/**
 * Load timeline fiducial data.
 */
timeline_event_t *timeline_load_fiducial(const char *data_dir,
                                         const int cam_idx,
                                         int *num_events) {
  // Load fiducial files
  *num_events = 0;
  char **files = list_files(data_dir, num_events);

  // Exit if no data
  if (*num_events == 0) {
    for (int view_idx = 0; view_idx < *num_events; view_idx++) {
      free(files[view_idx]);
    }
    free(files);
    return NULL;
  }

  // Load fiducial events
  timeline_event_t *events = malloc(sizeof(timeline_event_t) * *num_events);

  for (int view_idx = 0; view_idx < *num_events; view_idx++) {
    // Load aprilgrid
    aprilgrid_t *grid = aprilgrid_load(files[view_idx]);

    // Get aprilgrid measurements
    const timestamp_t ts = grid->timestamp;
    const int num_corners = grid->corners_detected;
    int *tag_ids = malloc(sizeof(int) * num_corners);
    int *corner_indices = malloc(sizeof(int) * num_corners);
    real_t *kps = malloc(sizeof(real_t) * num_corners * 2);
    real_t *pts = malloc(sizeof(real_t) * num_corners * 3);
    aprilgrid_measurements(grid, tag_ids, corner_indices, kps, pts);

    // Create event
    events[view_idx].type = FIDUCIAL_EVENT;
    events[view_idx].ts = ts;
    events[view_idx].data.fiducial.ts = ts;
    events[view_idx].data.fiducial.cam_idx = cam_idx;
    events[view_idx].data.fiducial.num_corners = num_corners;
    events[view_idx].data.fiducial.corner_indices = corner_indices;
    events[view_idx].data.fiducial.tag_ids = tag_ids;
    events[view_idx].data.fiducial.object_points = pts;
    events[view_idx].data.fiducial.keypoints = kps;

    // Clean up
    free(files[view_idx]);
    aprilgrid_free(grid);
  }
  free(files);

  return events;
}

/**
 * Load timeline IMU data.
 */
timeline_event_t *timeline_load_imu(const char *csv_path, int *num_events) {
  // Open file for loading
  const int num_rows = file_rows(csv_path);
  FILE *fp = fopen(csv_path, "r");
  if (fp == NULL) {
    FATAL("Failed to open [%s]!\n", csv_path);
  }
  skip_line(fp);

  // Malloc
  assert(num_rows > 0);
  *num_events = num_rows - 1;
  timeline_event_t *events = malloc(sizeof(timeline_event_t) * *num_events);

  // Parse file
  for (size_t k = 0; k < *num_events; k++) {
    // Parse line
    timestamp_t ts = 0;
    double w[3] = {0};
    double a[3] = {0};
    int retval = fscanf(fp,
                        "%" SCNd64 ",%lf,%lf,%lf,%lf,%lf,%lf",
                        &ts,
                        &w[0],
                        &w[1],
                        &w[2],
                        &a[0],
                        &a[1],
                        &a[2]);
    if (retval != 7) {
      FATAL("Failed to parse line in [%s]\n", csv_path);
    }

    // Add data
    events[k].type = IMU_EVENT;
    events[k].ts = ts;
    events[k].data.imu.ts = ts;
    events[k].data.imu.acc[0] = a[0];
    events[k].data.imu.acc[1] = a[1];
    events[k].data.imu.acc[2] = a[2];
    events[k].data.imu.gyr[0] = w[0];
    events[k].data.imu.gyr[1] = w[1];
    events[k].data.imu.gyr[2] = w[2];
  }
  fclose(fp);

  return events;
}

/**
 * Load events.
 */
static void timeline_load_events(timeline_t *timeline, const char *data_dir) {
  // Load events
  const int num_event_types = timeline->num_event_types;
  timeline_event_t **events =
      malloc(sizeof(timeline_event_t *) * num_event_types);
  int *events_lengths = calloc(num_event_types, sizeof(int));
  int *events_types = calloc(num_event_types, sizeof(int));
  timestamp_t **events_timestamps =
      malloc(sizeof(timestamp_t *) * num_event_types);
  int type_idx = 0;

  // -- Load fiducial events
  for (int cam_idx = 0; cam_idx < timeline->num_cams; cam_idx++) {
    // Form events
    int num_events = 0;
    char dir[1024] = {0};
    sprintf(dir, "%s/cam%d", data_dir, cam_idx);
    events[type_idx] = timeline_load_fiducial(dir, cam_idx, &num_events);
    events_lengths[type_idx] = num_events;
    events_types[type_idx] = FIDUCIAL_EVENT;

    // Form timestamps
    events_timestamps[type_idx] = calloc(num_events, sizeof(timestamp_t));
    for (int k = 0; k < num_events; k++) {
      events_timestamps[type_idx][k] = events[type_idx][k].ts;
    }

    // Update
    type_idx++;
  }

  // -- Load imu events
  for (int imu_idx = 0; imu_idx < timeline->num_imus; imu_idx++) {
    // Form events
    int num_events = 0;
    char csv_path[1024] = {0};
    sprintf(csv_path, "%s/imu%d/data.csv", data_dir, imu_idx);
    events[type_idx] = timeline_load_imu(csv_path, &num_events);
    events_lengths[type_idx] = num_events;
    events_types[type_idx] = IMU_EVENT;

    // Form timestamps
    events_timestamps[type_idx] = calloc(num_events, sizeof(timestamp_t));
    for (int k = 0; k < num_events; k++) {
      events_timestamps[type_idx][k] = events[type_idx][k].ts;
    }

    // Update
    type_idx++;
  }

  // Set timeline events
  timeline->events = events;
  timeline->events_timestamps = events_timestamps;
  timeline->events_lengths = events_lengths;
  timeline->events_types = events_types;
}

/**
 * Form timeline.
 */
void timeline_form_timeline(timeline_t *tl) {
  // Determine timeline timestamps
  int max_timeline_length = 0;
  for (int type_idx = 0; type_idx < tl->num_event_types; type_idx++) {
    max_timeline_length += tl->events_lengths[type_idx];
  }

  tl->timeline_length = 0;
  tl->timeline_timestamps = calloc(max_timeline_length, sizeof(timestamp_t));
  for (int type_idx = 0; type_idx < tl->num_event_types; type_idx++) {
    timestamps_unique(tl->timeline_timestamps,
                      &tl->timeline_length,
                      tl->events_timestamps[type_idx],
                      tl->events_lengths[type_idx]);
  }

  // Form timeline events
  tl->timeline_events =
      calloc(tl->timeline_length, sizeof(timeline_event_t **));
  tl->timeline_events_lengths = calloc(tl->timeline_length, sizeof(int));

  int *indices = calloc(tl->num_event_types, sizeof(int));
  for (int k = 0; k < tl->timeline_length; k++) {
    // Allocate memory
    tl->timeline_events[k] =
        calloc(tl->num_event_types, sizeof(timeline_event_t *));

    // Add events at k
    int k_len = 0; // Number of events at k
    for (int type_idx = 0; type_idx < tl->num_event_types; type_idx++) {
      // Find timestamp index
      int ts_found = 0;
      int ts_idx = 0;
      for (int i = indices[type_idx]; i < tl->events_lengths[type_idx]; i++) {
        timeline_event_t *event = &tl->events[type_idx][i];
        if (event->ts == tl->timeline_timestamps[k]) {
          indices[type_idx] = i;
          ts_found = 1;
          ts_idx = i;
          break;
        } else if (event->ts > tl->timeline_timestamps[k]) {
          break;
        }
      }

      // Add event to timeline
      if (ts_found) {
        tl->timeline_events[k][k_len] = &tl->events[type_idx][ts_idx];
        k_len++;
      }
    }

    // Set number of events at timestep k
    tl->timeline_events_lengths[k] = k_len;
  }

  // Clean-up
  free(indices);
}

/**
 * Load timeline
 */
timeline_t *timeline_load_data(const char *data_dir,
                               const int num_cams,
                               const int num_imus) {
  assert(num_cams >= 0);
  assert(num_imus >= 0 && num_imus <= 1);

  // Form timeline
  timeline_t *timeline = timeline_malloc();
  timeline->num_cams = num_cams;
  timeline->num_imus = num_imus;
  timeline->num_event_types = num_cams + num_imus;
  // -- Events
  timeline_load_events(timeline, data_dir);
  // -- Timeline
  timeline_form_timeline(timeline);

  return timeline;
}

/*******************************************************************************
 * SIMULATION
 ******************************************************************************/

////////////////
// TORUS KNOT //
////////////////

/* Base knot K(t). Adjust p, q, R, r for different shapes. */
void torus_knot(real_t t, int p, int q, real_t R, real_t r, real_t out[3]) {
  real_t cosqt = cos((real_t) q * t / (real_t) p);
  real_t sinqt = sin((real_t) q * t / (real_t) p);
  out[0] = (R + r * cosqt) * cos(t);
  out[1] = (R + r * cosqt) * sin(t);
  out[2] = r * sinqt;
}

/* Central difference derivative of K */
void torus_knot_deriv(real_t t,
                      int p,
                      int q,
                      real_t R,
                      real_t r,
                      real_t out[3]) {
  real_t h = 1e-5;
  real_t a[3], b[3];
  torus_knot(t + h, p, q, R, r, a);
  torus_knot(t - h, p, q, R, r, b);
  out[0] = (a[0] - b[0]) / (2 * h);
  out[1] = (a[1] - b[1]) / (2 * h);
  out[2] = (a[2] - b[2]) / (2 * h);
}

float *torus_knot_points(size_t *num_points) {
  // Parameters — tweak for different complexity
  const int p = 3, q = 2;  // knot type
  const real_t R = 2.0;    // major radius of base
  const real_t r = 0.7;    // minor radius of base
  const real_t a = 0.25;   // tube radius (thickness)
  const int t_steps = 600; // along knot
  const int s_steps = 48;  // around tube

  *num_points = t_steps * s_steps;
  float *points = malloc(sizeof(float) * 3 * *num_points);

  // Generate points
  int point_index = 0;
  for (int i = 0; i < t_steps; i++) {
    real_t t = 2.0 * M_PI * (real_t) i / (real_t) t_steps;
    real_t Kp[3], K0[3];
    torus_knot(t, p, q, R, r, K0);
    torus_knot_deriv(t, p, q, R, r, Kp);
    vec3_normalize(Kp); /* tangent T */

    // choose reference vector not parallel to tangent
    real_t ref[3] = {0.0, 0.0, 1.0};
    if (fabs(vec3_dot(ref, Kp)) > 0.99) {
      ref[0] = 0.0;
      ref[1] = 1.0;
      ref[2] = 0.0;
    }

    // N = normalize(ref - T * dot(ref, T))
    real_t proj[3];
    real_t N[3];
    vec3_scale(Kp, vec3_dot(ref, Kp), proj);
    vec3_sub(ref, proj, N);
    vec3_normalize(N);

    // B = T x N
    real_t B[3];
    vec3_cross(Kp, N, B);
    vec3_normalize(B);

    for (int j = 0; j < s_steps; j++) {
      real_t s = 2.0 * M_PI * (real_t) j / (real_t) s_steps;
      real_t c = cos(s), si = sin(s);
      real_t offset1[3], offset2[3], off[3], P[3];

      vec3_scale(N, a * c, offset1);
      vec3_scale(B, a * si, offset2);
      vec3_add(offset1, offset2, off);
      vec3_add(K0, off, P);

      points[point_index * 3 + 0] = P[0];
      points[point_index * 3 + 1] = P[1];
      points[point_index * 3 + 2] = P[2];
      point_index++;
    }
  }

  return points;
}

int torus_knot_save(const char *csv_path) {
  size_t num_points = 0;
  float *points = torus_knot_points(&num_points);

  FILE *f = fopen(csv_path, "w");
  if (!f) {
    perror(csv_path);
    return -1;
  }

  fprintf(f, "x y z\n");
  for (int i = 0; i < num_points; ++i) {
    fprintf(f, "%f ", points[i * 3 + 0]);
    fprintf(f, "%f ", points[i * 3 + 1]);
    fprintf(f, "%f\n", points[i * 3 + 2]);
  }

  fclose(f);
  free(points);

  return 0;
}

////////////////
// SIM CIRCLE //
////////////////

/**
 * Default circle trajectory settings.
 */
void sim_circle_defaults(sim_circle_t *conf) {
  conf->imu_rate = 200.0;
  conf->cam_rate = 10.0;
  conf->circle_r = 5.0;
  conf->circle_v = 1.0;
  conf->theta_init = M_PI;
  conf->yaw_init = M_PI / 2.0;
}

//////////////////
// SIM FEATURES //
//////////////////

/**
 * Save simulation feature data.
 */
void sim_features_save(sim_features_t *features, const char *csv_path) {
  FILE *features_file = fopen(csv_path, "w");
  for (int i = 0; i < features->num_features; ++i) {
    fprintf(features_file, "%f,", features->features[i][0]);
    fprintf(features_file, "%f,", features->features[i][1]);
    fprintf(features_file, "%f\n", features->features[i][2]);
  }
  fclose(features_file);
}

/**
 * Load simulation feature data.
 */
sim_features_t *sim_features_load(const char *csv_path) {
  sim_features_t *features_data = malloc(sizeof(sim_features_t) * 1);
  int num_rows = 0;
  int num_cols = 0;
  features_data->features = csv_data(csv_path, &num_rows, &num_cols);
  features_data->num_features = num_rows;
  return features_data;
}

/**
 * Free simulation feature data.
 */
void sim_features_free(sim_features_t *feature_data) {
  // Pre-check
  if (feature_data == NULL) {
    return;
  }

  // Free data
  for (int i = 0; i < feature_data->num_features; i++) {
    free(feature_data->features[i]);
  }
  free(feature_data->features);
  free(feature_data);
}

//////////////////
// SIM IMU DATA //
//////////////////

/**
 * Setup sim imu data.
 */
void sim_imu_data_setup(sim_imu_data_t *imu_data) {
  imu_data->num_measurements = 0;
  imu_data->timestamps = NULL;
  imu_data->poses = NULL;
  imu_data->velocities = NULL;
  imu_data->imu_acc = NULL;
  imu_data->imu_gyr = NULL;
}

/**
 * Malloc sim imu data.
 */
sim_imu_data_t *sim_imu_data_malloc(void) {
  sim_imu_data_t *imu_data = malloc(sizeof(sim_imu_data_t) * 1);
  sim_imu_data_setup(imu_data);
  return imu_data;
}

/**
 * Free simulation imu data.
 */
void sim_imu_data_free(sim_imu_data_t *imu_data) {
  // Pre-check
  if (imu_data == NULL) {
    return;
  }

  // Free data
  free(imu_data->timestamps);
  free(imu_data->poses);
  free(imu_data->velocities);
  free(imu_data->imu_acc);
  free(imu_data->imu_gyr);
  free(imu_data);
}

/**
 * Save simulation imu data.
 */
void sim_imu_data_save(sim_imu_data_t *imu_data, const char *csv_path) {
  assert(imu_data);
  assert(csv_path);

  FILE *csv_file = fopen(csv_path, "w");
  if (csv_file == NULL) {
    FATAL("Failed to open [%s]!\n", csv_path);
  }

  for (int i = 0; i < imu_data->num_measurements; ++i) {
    fprintf(csv_file, "%ld,", imu_data->timestamps[i]);
    fprintf(csv_file, "%lf,", imu_data->poses[i * 7 + 0]);
    fprintf(csv_file, "%lf,", imu_data->poses[i * 7 + 1]);
    fprintf(csv_file, "%lf,", imu_data->poses[i * 7 + 2]);
    fprintf(csv_file, "%lf,", imu_data->poses[i * 7 + 3]);
    fprintf(csv_file, "%lf,", imu_data->poses[i * 7 + 4]);
    fprintf(csv_file, "%lf,", imu_data->poses[i * 7 + 5]);
    fprintf(csv_file, "%lf,", imu_data->poses[i * 7 + 6]);
    fprintf(csv_file, "%lf,", imu_data->velocities[i * 3 + 0]);
    fprintf(csv_file, "%lf,", imu_data->velocities[i * 3 + 1]);
    fprintf(csv_file, "%lf,", imu_data->velocities[i * 3 + 2]);
    fprintf(csv_file, "%lf,", imu_data->imu_acc[i * 3 + 0]);
    fprintf(csv_file, "%lf,", imu_data->imu_acc[i * 3 + 1]);
    fprintf(csv_file, "%lf,", imu_data->imu_acc[i * 3 + 2]);
    fprintf(csv_file, "%lf,", imu_data->imu_gyr[i * 3 + 0]);
    fprintf(csv_file, "%lf,", imu_data->imu_gyr[i * 3 + 1]);
    fprintf(csv_file, "%lf\n", imu_data->imu_gyr[i * 3 + 2]);
  }

  fclose(csv_file);
}

/**
 * Load simulation imu data.
 */
sim_imu_data_t *sim_imu_data_load(const char *csv_path) {
  assert(csv_path);
  FILE *csv_file = fopen(csv_path, "r");
  if (csv_file == NULL) {
    FATAL("Failed to open [%s]!\n", csv_path);
  }

  const size_t num_rows = file_lines(csv_path);
  char fmt[1024] = {0};
  strcat(fmt, "%" SCNd64 ",");                 // Timestamp
  strcat(fmt, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,"); // Pose
  strcat(fmt, "%lf,%lf,%lf,");                 // Velocity
  strcat(fmt, "%lf,%lf,%lf,");                 // Imu acc
  strcat(fmt, "%lf,%lf,%lf");                  // Imu gyr
  sim_imu_data_t *imu_data = malloc(sizeof(sim_imu_data_t));
  imu_data->num_measurements = num_rows;
  imu_data->timestamps = malloc(sizeof(timestamp_t) * num_rows);
  imu_data->poses = malloc(sizeof(real_t) * 7 * num_rows);
  imu_data->velocities = malloc(sizeof(real_t) * 3 * num_rows);
  imu_data->imu_acc = malloc(sizeof(real_t) * 3 * num_rows);
  imu_data->imu_gyr = malloc(sizeof(real_t) * 3 * num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    timestamp_t ts = 0;
    double pose[7] = {0};
    double v_WS[3] = {0};
    double imu_acc[3] = {0};
    double imu_gyr[3] = {0};
    int retval = fscanf(csv_file,
                        fmt,
                        &ts,
                        &pose[0],
                        &pose[1],
                        &pose[2],
                        &pose[3],
                        &pose[4],
                        &pose[5],
                        &pose[6],
                        &v_WS[0],
                        &v_WS[1],
                        &v_WS[2],
                        &imu_acc[0],
                        &imu_acc[1],
                        &imu_acc[2],
                        &imu_gyr[0],
                        &imu_gyr[1],
                        &imu_gyr[2]);
    if (retval != 17) {
      FATAL("Failed to parse line %ld in [%s]\n", i, csv_path);
    }

    imu_data->timestamps[i] = ts;
    vec_copy(pose, 7, &imu_data->poses[i * 7]);
    vec_copy(v_WS, 3, &imu_data->velocities[i * 3]);
    vec_copy(imu_acc, 3, &imu_data->imu_acc[i * 3]);
    vec_copy(imu_gyr, 3, &imu_data->imu_gyr[i * 3]);
  }
  fclose(csv_file);

  return imu_data;
}

/**
 * Simulate IMU circle trajectory.
 */
sim_imu_data_t *sim_imu_circle_trajectory(const sim_circle_t *conf) {
  // Setup
  const int imu_rate = conf->imu_rate;
  const real_t circle_r = conf->circle_r;
  const real_t circle_v = conf->circle_r;
  const real_t theta_init = conf->circle_r;
  const real_t yaw_init = conf->yaw_init;

  // Circle trajectory configurations
  const real_t circle_dist = 2.0 * M_PI * circle_r;
  const real_t time_taken = circle_dist / circle_v;
  const real_t w = -2.0 * M_PI * (1.0 / time_taken);

  // Allocate memory for test data
  sim_imu_data_t *imu_data = sim_imu_data_malloc();
  imu_data->num_measurements = time_taken * imu_rate;
  imu_data->timestamps =
      calloc(imu_data->num_measurements, sizeof(timestamp_t));
  imu_data->poses = calloc(imu_data->num_measurements * 7, sizeof(real_t));
  imu_data->velocities = calloc(imu_data->num_measurements * 3, sizeof(real_t));
  imu_data->imu_acc = calloc(imu_data->num_measurements * 3, sizeof(real_t));
  imu_data->imu_gyr = calloc(imu_data->num_measurements * 3, sizeof(real_t));

  // Simulate IMU poses
  const real_t dt = 1.0 / imu_rate;
  timestamp_t ts = 0;
  real_t theta = theta_init;
  real_t yaw = yaw_init;

  for (size_t k = 0; k < imu_data->num_measurements; k++) {
    // IMU pose
    // -- Position
    const real_t rx = circle_r * cos(theta);
    const real_t ry = circle_r * sin(theta);
    const real_t rz = 0.0;
    // -- Orientation
    const real_t ypr[3] = {yaw, 0.0, 0.0};
    real_t q[4] = {0};
    euler2quat(ypr, q);
    // -- Pose vector
    const real_t pose[7] = {rx, ry, rz, q[0], q[1], q[2], q[3]};

    // Velocity
    const real_t vx = -circle_r * w * sin(theta);
    const real_t vy = circle_r * w * cos(theta);
    const real_t vz = 0.0;
    const real_t v_WS[3] = {vx, vy, vz};

    // Acceleration
    const real_t ax = -circle_r * w * w * cos(theta);
    const real_t ay = -circle_r * w * w * sin(theta);
    const real_t az = 0.0;
    const real_t a_WS[3] = {ax, ay, az};

    // Angular velocity
    const real_t wx = 0.0;
    const real_t wy = 0.0;
    const real_t wz = w;
    const real_t w_WS[3] = {wx, wy, wz};

    // IMU measurements
    real_t C_WS[3 * 3] = {0};
    real_t C_SW[3 * 3] = {0};
    quat2rot(q, C_WS);
    mat_transpose(C_WS, 3, 3, C_SW);
    // -- Accelerometer measurement
    real_t acc[3] = {0};
    dot(C_SW, 3, 3, a_WS, 3, 1, acc);
    acc[2] += 9.81;
    // -- Gyroscope measurement
    real_t gyr[3] = {0};
    dot(C_SW, 3, 3, w_WS, 3, 1, gyr);

    // Update
    imu_data->timestamps[k] = ts;
    vec_copy(pose, 7, imu_data->poses + k * 7);
    vec_copy(v_WS, 3, imu_data->velocities + k * 3);
    vec_copy(acc, 3, imu_data->imu_acc + k * 3);
    vec_copy(gyr, 3, imu_data->imu_gyr + k * 3);

    theta += w * dt;
    yaw += w * dt;
    ts += sec2ts(dt);
  }

  return imu_data;
}

void sim_imu_measurements(const sim_imu_data_t *data,
                          const int64_t ts_i,
                          const int64_t ts_j,
                          imu_buffer_t *imu_buf) {
  imu_buffer_setup(imu_buf);

  for (size_t k = 0; k < data->num_measurements; k++) {
    const int64_t ts = data->timestamps[k];
    if (ts <= ts_i) {
      continue;
    } else if (ts >= ts_j) {
      break;
    }

    imu_buffer_add(imu_buf, ts, &data->imu_acc[k * 3], &data->imu_gyr[k * 3]);
  }
}

/////////////////////
// SIM CAMERA DATA //
/////////////////////

/**
 * Simulate 3D features.
 */
void sim_create_features(const real_t origin[3],
                         const real_t dim[3],
                         const int num_features,
                         real_t *features) {
  assert(origin != NULL);
  assert(dim != NULL);
  assert(num_features > 0);
  assert(features != NULL);

  // Setup
  const real_t w = dim[0];
  const real_t l = dim[1];
  const real_t h = dim[2];
  const int features_per_side = num_features / 4.0;
  int feature_idx = 0;

  // Features in the east side
  {
    const real_t x_bounds[2] = {origin[0] - w, origin[0] + w};
    const real_t y_bounds[2] = {origin[1] + l, origin[1] + l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx * 3 + 0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx * 3 + 1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx * 3 + 2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }

  // Features in the north side
  {
    const real_t x_bounds[2] = {origin[0] + w, origin[0] + w};
    const real_t y_bounds[2] = {origin[1] - l, origin[1] + l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx * 3 + 0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx * 3 + 1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx * 3 + 2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }

  // Features in the west side
  {
    const real_t x_bounds[2] = {origin[0] - w, origin[0] + w};
    const real_t y_bounds[2] = {origin[1] - l, origin[1] - l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx * 3 + 0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx * 3 + 1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx * 3 + 2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }

  // Features in the south side
  {
    const real_t x_bounds[2] = {origin[0] - w, origin[0] - w};
    const real_t y_bounds[2] = {origin[1] - l, origin[1] + l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx * 3 + 0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx * 3 + 1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx * 3 + 2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }
}

/**
 * Setup simulated camera frame.
 */
void sim_camera_frame_setup(sim_camera_frame_t *frame,
                            const timestamp_t ts,
                            const int camera_index) {
  frame->ts = ts;
  frame->camera_index = camera_index;
  frame->n = 0;
  frame->feature_ids = NULL;
  frame->keypoints = NULL;
}

/**
 * Malloc simulated camera frame.
 */
sim_camera_frame_t *sim_camera_frame_malloc(const timestamp_t ts,
                                            const int camera_index) {
  sim_camera_frame_t *frame = malloc(sizeof(sim_camera_frame_t) * 1);
  sim_camera_frame_setup(frame, ts, camera_index);
  return frame;
}

/**
 * Free simulated camera frame.
 */
void sim_camera_frame_free(sim_camera_frame_t *frame_data) {
  // Pre-check
  if (frame_data == NULL) {
    return;
  }

  // Free data
  free(frame_data->keypoints);
  free(frame_data->feature_ids);
  free(frame_data);
}

/**
 * Add keypoint measurement to camera frame.
 */
void sim_camera_frame_add_keypoint(sim_camera_frame_t *frame,
                                   const size_t fid,
                                   const real_t kp[2]) {
  const int N = frame->n + 1;
  frame->n = N;
  frame->feature_ids = realloc(frame->feature_ids, sizeof(real_t) * N);
  frame->keypoints = realloc(frame->keypoints, sizeof(real_t) * N * 2);
  frame->feature_ids[N - 1] = fid;
  frame->keypoints[(N - 1) * 2 + 0] = kp[0];
  frame->keypoints[(N - 1) * 2 + 1] = kp[1];
}

/**
 * Save simulated camera frame.
 */
void sim_camera_frame_save(const sim_camera_frame_t *frame_data,
                           const char *csv_path) {
  assert(frame_data);
  assert(csv_path);

  FILE *csv_file = fopen(csv_path, "w");
  if (csv_file == NULL) {
    FATAL("Failed to open [%s]!\n", csv_path);
  }

  fprintf(csv_file, "timestamp: %ld\n", frame_data->ts);
  fprintf(csv_file, "camera_index: %d\n", frame_data->camera_index);
  fprintf(csv_file, "num_keypoints: %d\n", frame_data->n);
  for (size_t i = 0; i < frame_data->n; ++i) {
    fprintf(csv_file, "%ld,", frame_data->feature_ids[i]);
    fprintf(csv_file, "%.17g,", frame_data->keypoints[i * 2 + 0]);
    fprintf(csv_file, "%.17g\n", frame_data->keypoints[i * 2 + 1]);
  }
  fclose(csv_file);
}

/**
 * Load simulated camera frame.
 */
sim_camera_frame_t *sim_camera_frame_load(const char *csv_path) {
  assert(csv_path);

  FILE *csv_file = fopen(csv_path, "r");
  if (csv_file == NULL) {
    FATAL("Failed to open [%s]!\n", csv_path);
  }

  sim_camera_frame_t *frame = sim_camera_frame_malloc(0, 0);
  char key[1024] = {0};
  fscanf(csv_file, "%s %ld", key, &frame->ts);
  fscanf(csv_file, "%s %d", key, &frame->camera_index);
  fscanf(csv_file, "%s %d", key, &frame->n);

  frame->feature_ids = malloc(sizeof(size_t) * frame->n);
  frame->keypoints = malloc(sizeof(real_t) * 2 * frame->n);
  for (int i = 0; i < frame->n; ++i) {
    size_t fid = 0;
    real_t kx = 0;
    real_t ky = 0;
    fscanf(csv_file, "%ld,%lf,%lf", &fid, &kx, &ky);
    frame->feature_ids[i] = fid;
    frame->keypoints[i * 2 + 0] = kx;
    frame->keypoints[i * 2 + 1] = ky;
  }

  return frame;
}

/**
 * Print camera frame.
 */
void sim_camera_frame_print(const sim_camera_frame_t *frame_data) {
  printf("ts: %ld\n", frame_data->ts);
  printf("num_measurements: %d\n", frame_data->n);
  for (int i = 0; i < frame_data->n; i++) {
    const int feature_id = frame_data->feature_ids[i];
    const real_t *kp = frame_data->keypoints + i * 2;
    printf("- ");
    printf("feature_id: [%d], ", feature_id);
    printf("kp: [%.2f, %.2f]\n", kp[0], kp[1]);
  }
  printf("\n");
}

/**
 * Setup simulated camera frames.
 */
void sim_camera_data_setup(sim_camera_data_t *data) {
  data->frames = NULL;
  data->num_frames = 0;
  data->timestamps = NULL;
  data->poses = NULL;
}

/**
 * Malloc simulated camera frames.
 */
sim_camera_data_t *sim_camerea_data_malloc(void) {
  sim_camera_data_t *data = malloc(sizeof(sim_camera_data_t) * 1);
  sim_camera_data_setup(data);
  return data;
}

/**
 * Free simulated camera data.
 */
void sim_camera_data_free(sim_camera_data_t *cam_data) {
  // Pre-check
  if (cam_data == NULL) {
    return;
  }

  // Free data
  for (size_t k = 0; k < cam_data->num_frames; k++) {
    sim_camera_frame_free(cam_data->frames[k]);
  }
  free(cam_data->frames);
  free(cam_data->timestamps);
  free(cam_data->poses);
  free(cam_data);
}

/**
 * Save simulated camera data.
 */
void sim_camera_data_save(sim_camera_data_t *cam_data, const char *data_dir) {
  assert(cam_data);
  assert(data_dir);

  // Create output directory
  int retval = mkdir(data_dir, 0755);
  if (retval != 0 && errno != EEXIST) {
    FATAL("Failed to create directory [%s]", data_dir);
  }

  // Output data.csv
  char *csv_path = path_join(data_dir, "/data.csv");
  FILE *csv_file = fopen(csv_path, "w");
  fprintf(csv_file, "camera_index: %d\n", cam_data->camera_index);
  fprintf(csv_file, "num_frames: %d\n", cam_data->num_frames);
  for (int i = 0; i < cam_data->num_frames; ++i) {
    fprintf(csv_file, "%ld,", cam_data->timestamps[i]);
    fprintf(csv_file, "%.17g,", cam_data->poses[i * 7 + 0]);
    fprintf(csv_file, "%.17g,", cam_data->poses[i * 7 + 1]);
    fprintf(csv_file, "%.17g,", cam_data->poses[i * 7 + 2]);
    fprintf(csv_file, "%.17g,", cam_data->poses[i * 7 + 3]);
    fprintf(csv_file, "%.17g,", cam_data->poses[i * 7 + 4]);
    fprintf(csv_file, "%.17g,", cam_data->poses[i * 7 + 5]);
    fprintf(csv_file, "%.17g\n", cam_data->poses[i * 7 + 6]);
  }
  free(csv_path);
  fclose(csv_file);

  // Output frame data
  for (int i = 0; i < cam_data->num_frames; ++i) {
    char ts_str[40] = {0};
    const timestamp_t ts = cam_data->timestamps[i];
    snprintf(ts_str, sizeof(ts_str), "/%ld.csv", ts);
    char *csv_path = path_join(data_dir, ts_str);
    sim_camera_frame_save(cam_data->frames[i], csv_path);
    free(csv_path);
  }
}

/**
 * Load simulated camera data.
 */
sim_camera_data_t *sim_camera_data_load(const char *data_dir) {
  assert(data_dir != NULL);

  // Form csv file path
  char *csv_path = path_join(data_dir, "/data.csv");
  if (file_exists(csv_path) == 0) {
    free(csv_path);
    return NULL;
  }

  // Check number of rows
  const int num_rows = dsv_rows(csv_path);
  if (num_rows == 0) {
    free(csv_path);
    return NULL;
  }

  // Open csv file
  FILE *csv_file = fopen(csv_path, "r");
  if (csv_file == NULL) {
    free(csv_path);
    return NULL;
  }

  // Form sim_camera_frame_t
  sim_camera_data_t *data = malloc(sizeof(sim_camera_data_t));
  char key[20] = {0};
  fscanf(csv_file, "%s %d", key, &data->camera_index);
  assert(strcmp(key, "camera_index:") == 0);
  fscanf(csv_file, "%s %d", key, &data->num_frames);
  assert(strcmp(key, "num_frames:") == 0);

  data->timestamps = malloc(sizeof(timestamp_t) * data->num_frames);
  data->poses = malloc(sizeof(real_t *) * data->num_frames * 7);
  data->frames = malloc(sizeof(sim_camera_frame_t *) * data->num_frames);
  char fmt[1024] = {0};
  strcat(fmt, "%" SCNd64 ",");                // Timestamp
  strcat(fmt, "%lf,%lf,%lf,%lf,%lf,%lf,%lf"); // Pose
  for (size_t i = 0; i < data->num_frames; ++i) {
    timestamp_t ts = 0;
    real_t pose[7] = {0};
    int retval = fscanf(csv_file,
                        fmt,
                        &ts,
                        &pose[0],
                        &pose[1],
                        &pose[2],
                        &pose[3],
                        &pose[4],
                        &pose[5],
                        &pose[6]);
    if (retval != 8) {
      FATAL("Failed to parse line %ld in [%s]\n", i, csv_path);
    }

    // Add camera frame to sim_camera_frame_t
    char fname[128] = {0};
    sprintf(fname, "/%ld.csv", ts);
    char *frame_csv = path_join(data_dir, fname);
    data->frames[i] = sim_camera_frame_load(frame_csv);
    free(frame_csv);

    // Add pose to sim_camera_frame_t
    data->timestamps[i] = ts;
    data->poses[i * 7 + 0] = pose[0];
    data->poses[i * 7 + 1] = pose[1];
    data->poses[i * 7 + 2] = pose[2];
    data->poses[i * 7 + 3] = pose[3];
    data->poses[i * 7 + 4] = pose[4];
    data->poses[i * 7 + 5] = pose[5];
    data->poses[i * 7 + 6] = pose[6];
  }

  // Clean up
  free(csv_path);
  fclose(csv_file);

  return data;
}

/**
 * Simulate camera going round in a circle.
 */
sim_camera_data_t *sim_camera_circle_trajectory(const sim_circle_t *conf,
                                                const real_t T_BC[4 * 4],
                                                const camera_t *cam_params,
                                                const real_t *features,
                                                const int num_features) {
  // Settings
  const real_t cam_rate = conf->cam_rate;
  const real_t circle_r = conf->circle_r;
  const real_t circle_v = conf->circle_v;
  const real_t theta_init = conf->theta_init;
  const real_t yaw_init = conf->yaw_init;

  // Circle trajectory configurations
  const real_t circle_dist = 2.0 * M_PI * circle_r;
  const real_t time_taken = circle_dist / circle_v;
  const real_t w = -2.0 * M_PI * (1.0 / time_taken);

  // Allocate memory for test data
  const int camera_index = cam_params->cam_idx;
  const int num_frames = time_taken * cam_rate;
  sim_camera_data_t *data = sim_camerea_data_malloc();
  data->camera_index = camera_index;
  data->frames = calloc(num_frames, sizeof(sim_camera_frame_t *));
  data->num_frames = num_frames;
  data->timestamps = calloc(data->num_frames, sizeof(real_t));
  data->poses = calloc(data->num_frames * 7, sizeof(real_t));

  // Simulate camera
  const real_t dt = 1.0 / cam_rate;
  timestamp_t ts = 0.0;
  real_t theta = theta_init;
  real_t yaw = yaw_init;

  for (size_t k = 0; k < data->num_frames; k++) {
    // Body pose T_WB
    // -- Position
    const real_t rx = circle_r * cos(theta);
    const real_t ry = circle_r * sin(theta);
    const real_t rz = 0.0;
    const real_t r_WB[3] = {rx, ry, rz};
    // -- Orientation
    const real_t ypr_WB[3] = {yaw, 0.0, 0.0};
    real_t q_WB[4] = {0};
    euler2quat(ypr_WB, q_WB);
    // -- Body Pose
    TF_QR(q_WB, r_WB, T_WB);

    // Camera pose
    TF_CHAIN(T_WC, 2, T_WB, T_BC);
    TF_VECTOR(T_WC, cam_pose);
    TF_INV(T_WC, T_CW);

    // Simulate camera frame
    sim_camera_frame_t *frame = sim_camera_frame_malloc(ts, camera_index);
    for (size_t fid = 0; fid < num_features; ++fid) {
      // Check point is infront of camera
      const real_t *p_W = &features[fid * 3];
      TF_POINT(T_CW, p_W, p_C);
      if (p_C[2] < 0) {
        continue;
      }

      // Project image point to image plane
      real_t z[2] = {0};
      pinhole_radtan4_project(cam_params->data, p_C, z);

      // Check projection
      const int x_ok = (z[0] < cam_params->resolution[0] && z[0] > 0);
      const int y_ok = (z[1] < cam_params->resolution[1] && z[1] > 0);
      if (x_ok == 0 || y_ok == 0) {
        continue;
      }

      // Add keypoint to camera frame
      sim_camera_frame_add_keypoint(frame, fid, z);
    }
    data->frames[k] = frame;

    // Update
    data->timestamps[k] = ts;
    vec_copy(cam_pose, 7, &data->poses[k * 7]);

    theta += w * dt;
    yaw += w * dt;
    ts += sec2ts(dt);
  }

  return data;
}

/////////////////////////
// SIM CAMERA IMU DATA //
/////////////////////////

sim_circle_camera_imu_t *sim_circle_camera_imu(void) {
  // Malloc
  sim_circle_camera_imu_t *sim_data =
      malloc(sizeof(sim_circle_camera_imu_t) * 1);

  // Simulate features
  const real_t origin[3] = {0.0, 0.0, 0.0};
  const real_t dim[3] = {5.0, 5.0, 5.0};
  sim_data->num_features = 1000;
  sim_create_features(origin,
                      dim,
                      sim_data->num_features,
                      sim_data->feature_data);

  // Camera configuration
  const int res[2] = {640, 480};
  const real_t fov = 90.0;
  const real_t fx = pinhole_focal(res[0], fov);
  const real_t fy = pinhole_focal(res[0], fov);
  const real_t cx = res[0] / 2.0;
  const real_t cy = res[1] / 2.0;
  const real_t cam_vec[8] = {fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0};
  const char *pmodel = "pinhole";
  const char *dmodel = "radtan4";
  camera_t *cam0_params = &sim_data->cam0_params;
  camera_t *cam1_params = &sim_data->cam1_params;
  camera_setup(cam0_params, 0, res, pmodel, dmodel, cam_vec);
  camera_setup(cam1_params, 1, res, pmodel, dmodel, cam_vec);

  // IMU-Camera0 extrinsic
  const real_t cam0_ext_ypr[3] = {-M_PI / 2.0, 0.0, -M_PI / 2.0};
  const real_t cam0_ext_r[3] = {0.05, 0.0, 0.0};
  const real_t cam1_ext_ypr[3] = {-M_PI / 2.0, 0.0, -M_PI / 2.0};
  const real_t cam1_ext_r[3] = {-0.05, 0.0, 0.0};
  TF_ER(cam0_ext_ypr, cam0_ext_r, T_SC0);
  TF_ER(cam1_ext_ypr, cam1_ext_r, T_SC1);
  tf_vector(T_SC0, sim_data->cam0_ext);
  tf_vector(T_SC1, sim_data->cam1_ext);

  // IMU extrinsic
  TF_IDENTITY(T_BS);
  tf_vector(T_BS, sim_data->imu0_ext);

  // Simulate data
  sim_circle_defaults(&sim_data->conf);
  sim_data->imu_data = sim_imu_circle_trajectory(&sim_data->conf);
  sim_data->cam0_data = sim_camera_circle_trajectory(&sim_data->conf,
                                                     T_SC0,
                                                     &sim_data->cam0_params,
                                                     sim_data->feature_data,
                                                     sim_data->num_features);
  sim_data->cam1_data = sim_camera_circle_trajectory(&sim_data->conf,
                                                     T_SC1,
                                                     &sim_data->cam1_params,
                                                     sim_data->feature_data,
                                                     sim_data->num_features);

  // Form timeline
  const int num_event_types = 1;
  timeline_event_t **events =
      malloc(sizeof(timeline_event_t *) * num_event_types);
  timestamp_t **events_timestamps =
      malloc(sizeof(timestamp_t *) * num_event_types);
  int *events_lengths = calloc(num_event_types, sizeof(int));
  int *events_types = calloc(num_event_types, sizeof(int));
  int type_idx = 0;

  // -- IMU data to timeline
  const size_t num_imu_events = sim_data->imu_data->num_measurements;
  timeline_event_t *imu_events =
      malloc(sizeof(timeline_event_t) * num_imu_events);
  for (size_t k = 0; k < sim_data->imu_data->num_measurements; k++) {
    imu_events[k].type = IMU_EVENT;
    imu_events[k].ts = sim_data->imu_data->timestamps[k];
    imu_events[k].data.imu.ts = sim_data->imu_data->timestamps[k];
    imu_events[k].data.imu.acc[0] = sim_data->imu_data->imu_acc[k * 3 + 0];
    imu_events[k].data.imu.acc[1] = sim_data->imu_data->imu_acc[k * 3 + 1];
    imu_events[k].data.imu.acc[2] = sim_data->imu_data->imu_acc[k * 3 + 2];
    imu_events[k].data.imu.gyr[0] = sim_data->imu_data->imu_gyr[k * 3 + 0];
    imu_events[k].data.imu.gyr[1] = sim_data->imu_data->imu_gyr[k * 3 + 1];
    imu_events[k].data.imu.gyr[2] = sim_data->imu_data->imu_gyr[k * 3 + 2];
  }
  events[type_idx] = imu_events;
  events_timestamps[type_idx] = calloc(num_imu_events, sizeof(timestamp_t));
  for (int k = 0; k < num_imu_events; k++) {
    events_timestamps[type_idx][k] = events[type_idx][k].ts;
  }
  events_lengths[type_idx] = num_imu_events;
  events_types[type_idx] = IMU_EVENT;
  type_idx++;

  // -- Add to timeline
  sim_data->timeline = timeline_malloc();
  sim_data->timeline->num_imus = 1;
  sim_data->timeline->num_event_types = num_event_types;
  sim_data->timeline->events = events;
  sim_data->timeline->events_timestamps = events_timestamps;
  sim_data->timeline->events_lengths = events_lengths;
  sim_data->timeline->events_types = events_types;

  // -- Form timeline
  timeline_form_timeline(sim_data->timeline);

  return sim_data;
}

void sim_circle_camera_imu_free(sim_circle_camera_imu_t *sim_data) {
  sim_imu_data_free(sim_data->imu_data);
  sim_camera_data_free(sim_data->cam0_data);
  sim_camera_data_free(sim_data->cam1_data);
  timeline_free(sim_data->timeline);
  free(sim_data);
}

/******************************************************************************
 * EUROC
 ******************************************************************************/

/**
 * Print vector
 */
static void print_int_vector(const char *prefix, const int *v, const int n) {
  printf("%s: [", prefix);
  for (int i = 0; i < n; i++) {
    printf("%d ", v[i]);
  }
  printf("\b]\n");
}

/**
 * Print YAML Token
 */
inline void yaml_print_token(const yaml_token_t token) {
  switch (token.type) {
    case YAML_NO_TOKEN:
      printf("YAML_NO_TOKEN\n");
      break;
    case YAML_STREAM_START_TOKEN:
      printf("YAML_STREAM_START_TOKEN\n");
      break;
    case YAML_STREAM_END_TOKEN:
      printf("YAML_STREAM_END_TOKEN\n");
      break;

    case YAML_VERSION_DIRECTIVE_TOKEN:
      printf("YAML_VERSION_DIRECTIVE_TOKEN\n");
      break;
    case YAML_TAG_DIRECTIVE_TOKEN:
      printf("YAML_TAG_DIRECTIVE_TOKEN\n");
      break;
    case YAML_DOCUMENT_START_TOKEN:
      printf("YAML_DOCUMENT_START_TOKEN\n");
      break;
    case YAML_DOCUMENT_END_TOKEN:
      printf("YAML_DOCUMENT_END_TOKEN\n");
      break;

    case YAML_BLOCK_SEQUENCE_START_TOKEN:
      printf("YAML_BLOCK_SEQUENCE_START_TOKEN\n");
      break;
    case YAML_BLOCK_MAPPING_START_TOKEN:
      printf("YAML_BLOCK_MAPPING_START_TOKEN\n");
      break;
    case YAML_BLOCK_END_TOKEN:
      printf("YAML_BLOCK_END_TOKEN\n");
      break;

    case YAML_FLOW_SEQUENCE_START_TOKEN:
      printf("YAML_FLOW_SEQUENCE_START_TOKEN\n");
      break;
    case YAML_FLOW_SEQUENCE_END_TOKEN:
      printf("YAML_FLOW_SEQUENCE_END_TOKEN\n");
      break;
    case YAML_FLOW_MAPPING_START_TOKEN:
      printf("YAML_FLOW_MAPPING_START_TOKEN\n");
      break;
    case YAML_FLOW_MAPPING_END_TOKEN:
      printf("YAML_FLOW_MAPPING_END_TOKEN\n");
      break;

    case YAML_BLOCK_ENTRY_TOKEN:
      printf("YAML_BLOCK_ENTRY_TOKEN\n");
      break;
    case YAML_FLOW_ENTRY_TOKEN:
      printf("YAML_FLOW_ENTRY_TOKEN\n");
      break;
    case YAML_KEY_TOKEN:
      printf("YAML_KEY_TOKEN\n");
      break;
    case YAML_VALUE_TOKEN:
      printf("YAML_VALUE_TOKEN\n");
      break;

    case YAML_ALIAS_TOKEN:
      printf("YAML_ALIAS_TOKEN\n");
      break;
    case YAML_ANCHOR_TOKEN:
      printf("YAML_ANCHOR_TOKEN\n");
      break;
    case YAML_TAG_TOKEN:
      printf("YAML_TAG_TOKEN\n");
      break;
    case YAML_SCALAR_TOKEN:
      printf("YAML_SCALAR_TOKEN [%s]\n", token.data.scalar.value);
      break;

    default:
      printf("-\n");
      break;
  }
}

/**
 * Get key-value from yaml file
 */
static int
yaml_get(const char *yaml_file, const char *key, char value_type, void *value) {
  // Load calibration data
  yaml_parser_t parser;
  yaml_token_t token;

  // Open sensor file
  FILE *fp = fopen(yaml_file, "r");
  if (fp == NULL) {
    EUROC_FATAL("YAML file [%s] not found!\n", yaml_file);
    return -1;
  }

  // Initialize YAML parser
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_file(&parser, fp);

  // Parse YAML data
  int state = 0;
  int match = 0;
  int found = 0;
  do {
    yaml_parser_scan(&parser, &token);

    switch (token.type) {
      case YAML_KEY_TOKEN:
        state = 0;
        break;
      case YAML_VALUE_TOKEN:
        state = 1;
        break;
      case YAML_SCALAR_TOKEN: {
        char *tk = (char *) token.data.scalar.value;

        // Check key
        if (state == 0 && strcmp(tk, key) == 0) {
          match = 1;
        }

        // Parse value
        if (state == 1 && match == 1) {
          if (value_type == 'd') {
            *(double *) value = strtod(tk, NULL);
          } else if (value_type == 's') {
            strcpy((char *) value, tk);
          } else if (value_type == 'i') {
            *(int *) value = strtol(tk, NULL, 10);
          } else {
            EUROC_FATAL("Unrecognized value type: '%c'!\n", value_type);
          }
          found = 1;
        }
        break;
      }
      default:
        break;
    }

    if (token.type != YAML_STREAM_END_TOKEN) {
      yaml_token_delete(&token);
    }
  } while (token.type != YAML_STREAM_END_TOKEN && found == 0);

  // Clean up
  yaml_token_delete(&token);
  yaml_parser_delete(&parser);
  fclose(fp);

  return (found) ? 0 : -1;
}

/**
 * Get vector from yaml file
 */
static int yaml_get_vector(const char *yaml_file,
                           const char *key,
                           const char value_type,
                           const int n,
                           void *v) {
  // Load calibration data
  yaml_parser_t parser;
  yaml_token_t token;

  // Open sensor file
  FILE *fp = fopen(yaml_file, "r");
  if (fp == NULL) {
    return -1;
  }

  // Initialize YAML parser
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_file(&parser, fp);

  // Parse YAML data
  int done = 0;
  int state = 0;
  int match_key = 0;
  int found_key = 0;
  int v_idx = 0;

  do {
    yaml_parser_scan(&parser, &token);

    switch (token.type) {
      case YAML_KEY_TOKEN:
        state = 0;
        break;
      case YAML_VALUE_TOKEN:
        state = 1;
        break;
      case YAML_SCALAR_TOKEN: {
        char *tk = (char *) token.data.scalar.value;

        // Check key
        if (state == 0 && strcmp(tk, key) == 0) {
          match_key = 1;
          found_key = 1;
        }

        // Parse data
        if (match_key == 1 && state == 1) {
          if (value_type == 'd') {
            ((double *) v)[v_idx++] = strtod(tk, NULL);
          } else if (value_type == 'i') {
            ((int *) v)[v_idx++] = strtol(tk, NULL, 10);
          } else {
            EUROC_FATAL("Unrecognized value type: '%c'!\n", value_type);
          }
        }
        break;
      }
      default:
        break;
    }

    if (token.type != YAML_STREAM_END_TOKEN) {
      yaml_token_delete(&token);
    }
  } while (token.type != YAML_STREAM_END_TOKEN && done == 0);

  // Clean up
  yaml_token_delete(&token);
  yaml_parser_delete(&parser);
  fclose(fp);

  return (found_key && (n == v_idx)) ? 0 : -1;
}

/**
 * Get matrix from yaml file
 */
static int yaml_get_matrix(const char *yaml_file,
                           const char *key,
                           const int m,
                           const int n,
                           double *A) {
  // Load calibration data
  yaml_parser_t parser;
  yaml_token_t token;

  // Open sensor file
  FILE *fp = fopen(yaml_file, "r");
  if (fp == NULL) {
    return -1;
  }

  // Initialize YAML parser
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_file(&parser, fp);

  // Parse YAML data
  int done = 0;
  int state = 0;
  int match_key = 0;
  int parse_rows = 0;
  int parse_cols = 0;
  int parse_data = 0;

  int num_rows = 0;
  int num_cols = 0;
  int tf_idx = 0;

  do {
    yaml_parser_scan(&parser, &token);
    // yaml_print_token(token);

    switch (token.type) {
      case YAML_KEY_TOKEN:
        state = 0;
        break;
      case YAML_VALUE_TOKEN:
        state = 1;
        break;
      case YAML_SCALAR_TOKEN: {
        char *tk = (char *) token.data.scalar.value;

        // Check key
        if (state == 0 && strcmp(tk, key) == 0) {
          match_key = 1;
        }

        // Parse rows
        if (match_key == 1 && state == 0 && strcmp(tk, "rows") == 0) {
          parse_rows = 1;
        } else if (match_key == 1 && state == 1 && parse_rows == 1) {
          num_rows = strtol(tk, NULL, 10);
          parse_rows = 0;
        }

        // Parse cols
        if (match_key == 1 && state == 0 && strcmp(tk, "cols") == 0) {
          parse_cols = 1;
        } else if (match_key == 1 && state == 1 && parse_cols == 1) {
          num_cols = strtol(tk, NULL, 10);
          parse_cols = 0;
        }

        // Parse data
        if (match_key == 1 && state == 0 && strcmp(tk, "data") == 0) {
          parse_data = 1;
        } else if (match_key == 1 && state == 1 && parse_data == 1) {
          // Pre-check
          if (num_rows != m || num_cols != n) {
            EUROC_LOG("Number of rows or columns expected != "
                      "parsed\n");
            EUROC_LOG("rows expected: %d, got: %d\n", m, num_rows);
            EUROC_LOG("cols expected: %d, got: %d\n", m, num_cols);
          }

          // Set matrix
          A[tf_idx++] = strtod(tk, NULL);
          if (tf_idx >= (num_rows * num_cols)) {
            parse_data = 0;
            done = 1;
          }
        }
        break;
      }
      default:
        break;
    }

    if (token.type != YAML_STREAM_END_TOKEN) {
      yaml_token_delete(&token);
    }
  } while (token.type != YAML_STREAM_END_TOKEN && done == 0);

  // Clean up
  yaml_token_delete(&token);
  yaml_parser_delete(&parser);
  fclose(fp);

  return ((num_rows * num_cols) == tf_idx) ? 0 : -1;
}

/////////////////
// euroc_imu_t //
/////////////////

/**
 * Load EuRoC IMU data
 */
euroc_imu_t *euroc_imu_load(const char *data_dir) {
  // Setup
  euroc_imu_t *data = malloc(sizeof(euroc_imu_t) * 1);

  // Form data and sensor paths
  char data_path[1024] = {0};
  char conf[1024] = {0};
  strcat(data_path, data_dir);
  strcat(data_path, "/data.csv");
  strcat(conf, data_dir);
  strcat(conf, "/sensor.yaml");

  // Open file for loading
  const size_t num_rows = file_lines(data_path);
  FILE *fp = fopen(data_path, "r");
  if (fp == NULL) {
    EUROC_FATAL("Failed to open [%s]!\n", data_path);
  }

  // Malloc
  assert(num_rows > 0);
  data->num_timestamps = 0;
  data->timestamps = malloc(sizeof(timestamp_t) * num_rows);
  data->w_B = malloc(sizeof(double *) * num_rows);
  data->a_B = malloc(sizeof(double *) * num_rows);

  // Parse file
  for (size_t i = 0; i < num_rows; i++) {
    // Skip first line
    if (i == 0) {
      skip_line(fp);
      continue;
    }

    // Parse line
    timestamp_t ts = 0;
    double w[3] = {0};
    double a[3] = {0};
    int retval = fscanf(fp,
                        "%" SCNd64 ",%lf,%lf,%lf,%lf,%lf,%lf",
                        &ts,
                        &w[0],
                        &w[1],
                        &w[2],
                        &a[0],
                        &a[1],
                        &a[2]);
    if (retval != 7) {
      EUROC_FATAL("Failed to parse line in [%s]\n", data_path);
    }

    // Add data
    data->timestamps[data->num_timestamps] = ts;
    data->w_B[data->num_timestamps] = vec_malloc(w, 3);
    data->a_B[data->num_timestamps] = vec_malloc(a, 3);
    data->num_timestamps++;
  }
  fclose(fp);

  // Load sensor configuration
  // clang-format off
  yaml_get(conf, "sensor_type", 's', &data->sensor_type);
  yaml_get(conf, "comment", 's', &data->comment);
  yaml_get_matrix(conf, "T_BS", 4, 4, data->T_BS);
  yaml_get(conf, "rate_hz", 'd', &data->rate_hz);
  yaml_get(conf, "gyroscope_noise_density", 'd', &data->gyro_noise_density);
  yaml_get(conf, "gyroscope_random_walk", 'd', &data->gyro_random_walk);
  yaml_get(conf, "accelerometer_noise_density", 'd', &data->accel_noise_density);
  yaml_get(conf, "accelerometer_random_walk", 'd', &data->accel_random_walk);
  // clang-format on

  return data;
}

/**
 * Free EuRoC IMU data
 */
void euroc_imu_free(euroc_imu_t *data) {
  assert(data != NULL);
  free(data->timestamps);
  for (size_t k = 0; k < data->num_timestamps; k++) {
    free(data->w_B[k]);
    free(data->a_B[k]);
  }
  free(data->w_B);
  free(data->a_B);
  free(data);
}

/**
 * Print EuRoC IMU data
 */
void euroc_imu_print(const euroc_imu_t *data) {
  printf("sensor_type: %s\n", data->sensor_type);
  printf("comment: %s\n", data->comment);
  print_matrix("T_BS", data->T_BS, 4, 4);
  printf("rate_hz: %f\n", data->rate_hz);
  printf("gyroscope_noise_density: %f\n", data->gyro_noise_density);
  printf("gyroscope_random_walk: %f\n", data->gyro_random_walk);
  printf("accelerometer_noise_density: %f\n", data->accel_noise_density);
  printf("accelerometer_random_walk: %f\n", data->accel_random_walk);
}

////////////////////
// euroc_camera_t //
////////////////////

/**
 * Load EuRoC camera data
 */
euroc_camera_t *euroc_camera_load(const char *data_dir, int is_calib_data) {
  // Setup
  euroc_camera_t *data = malloc(sizeof(euroc_camera_t) * 1);
  data->is_calib_data = is_calib_data;

  // Form data and sensor paths
  char data_path[1024] = {0};
  char conf[1024] = {0};
  strcat(data_path, data_dir);
  strcat(data_path, "/data.csv");
  strcat(conf, data_dir);
  strcat(conf, "/sensor.yaml");

  // Open file for loading
  const int num_rows = file_lines(data_path);
  FILE *fp = fopen(data_path, "r");
  if (fp == NULL) {
    EUROC_FATAL("Failed to open [%s]!\n", data_path);
  }

  // Malloc
  assert(num_rows > 0);
  data->num_timestamps = 0;
  data->timestamps = malloc(sizeof(timestamp_t) * num_rows);
  data->image_paths = malloc(sizeof(char *) * num_rows);

  // Parse file
  for (size_t i = 0; i < num_rows; i++) {
    // Skip first line
    if (i == 0) {
      skip_line(fp);
      continue;
    }

    // Parse line
    timestamp_t ts = 0;
    char filename[50] = {0};
    int retval = fscanf(fp, "%" SCNd64 ",%s", &ts, filename);
    if (retval != 2) {
      EUROC_FATAL("Failed to parse line in [%s]\n", data_path);
    }

    // Check if file exists
    char image_path[9046] = {0};
    strcat(image_path, data_dir);
    strcat(image_path, "/data/");
    strcat(image_path, filename);
    if (file_exists(image_path) == 0) {
      EUROC_FATAL("File [%s] does not exist!\n", image_path);
    }

    // Add data
    const int k = data->num_timestamps;
    data->timestamps[k] = ts;
    data->image_paths[k] = string_malloc(image_path);
    data->num_timestamps++;
  }
  fclose(fp);

  // Load sensor configuration
  yaml_get(conf, "sensor_type", 's', &data->sensor_type);
  yaml_get(conf, "comment", 's', &data->comment);
  yaml_get_matrix(conf, "T_BS", 4, 4, data->T_BS);
  yaml_get(conf, "rate_hz", 'd', &data->rate_hz);
  yaml_get_vector(conf, "resolution", 'i', 2, data->resolution);

  if (is_calib_data) {
    // Camera data is calibration data, thus there are no calibration
    // parameters
    data->camera_model[0] = '\0';
    memset(data->intrinsics, 0, 4 * sizeof(double));
    data->distortion_model[0] = '\0';
    memset(data->distortion_coefficients, 0, 4 * sizeof(double));

  } else {
    // Camera data is calibrated
    yaml_get(conf, "camera_model", 's', &data->camera_model);
    yaml_get_vector(conf, "intrinsics", 'd', 4, data->intrinsics);
    yaml_get(conf, "distortion_model", 's', &data->distortion_model);
    yaml_get_vector(conf,
                    "distortion_coefficients",
                    'd',
                    4,
                    data->distortion_coefficients);
  }

  return data;
}

/**
 * Free EuRoC camera data
 */
void euroc_camera_free(euroc_camera_t *data) {
  free(data->timestamps);
  for (size_t k = 0; k < data->num_timestamps; k++) {
    free(data->image_paths[k]);
  }
  free(data->image_paths);
  free(data);
}

/**
 * EuRoC camera to output stream
 */
void euroc_camera_print(const euroc_camera_t *data) {
  printf("sensor_type: %s\n", data->sensor_type);
  printf("comment: %s\n", data->comment);
  print_matrix("T_BS", data->T_BS, 4, 4);
  printf("rate_hz: %f\n", data->rate_hz);
  print_int_vector("resolution", data->resolution, 2);
  if (data->is_calib_data == 0) {
    printf("camera_model: %s\n", data->camera_model);
    print_vector("intrinsics", data->intrinsics, 4);
    printf("distortion_model: %s\n", data->distortion_model);
    print_vector("distortion_coefficients", data->distortion_coefficients, 4);
  }
}

//////////////////////////
// euroc_ground_truth_t //
//////////////////////////

/**
 * Load EuRoC ground truth data
 */
euroc_ground_truth_t *euroc_ground_truth_load(const char *data_dir) {
  // Setup
  euroc_ground_truth_t *data = malloc(sizeof(euroc_ground_truth_t) * 1);

  // Form data path
  char data_path[9046] = {0};
  strcat(data_path, data_dir);
  strcat(data_path, "/data.csv");

  // Open file for loading
  const size_t num_rows = file_lines(data_path);
  FILE *fp = fopen(data_path, "r");
  if (fp == NULL) {
    EUROC_FATAL("Failed to open [%s]!\n", data_path);
  }

  // Malloc
  assert(num_rows > 0);
  data->num_timestamps = 0;
  data->timestamps = malloc(sizeof(timestamp_t) * num_rows);
  data->p_RS_R = malloc(sizeof(double *) * num_rows);
  data->q_RS = malloc(sizeof(double *) * num_rows);
  data->v_RS_R = malloc(sizeof(double *) * num_rows);
  data->b_w_RS_S = malloc(sizeof(double *) * num_rows);
  data->b_a_RS_S = malloc(sizeof(double *) * num_rows);

  // Parse file
  char str_format[9046] = {0};
  strcat(str_format, "%" SCNd64 ",");     // Timestamp
  strcat(str_format, "%lf,%lf,%lf,");     // Position
  strcat(str_format, "%lf,%lf,%lf,%lf,"); // Quaternion
  strcat(str_format, "%lf,%lf,%lf,");     // Velocity
  strcat(str_format, "%lf,%lf,%lf,");     // Gyro bias
  strcat(str_format, "%lf,%lf,%lf");      // Accel bias

  for (size_t i = 0; i < num_rows; i++) {
    // Skip first line
    if (i == 0) {
      skip_line(fp);
      continue;
    }

    // Parse line
    timestamp_t ts = 0;
    double p[3] = {0};
    double q[4] = {0};
    double v[3] = {0};
    double w[3] = {0};
    double a[3] = {0};
    int retval = fscanf(fp,
                        str_format,
                        &ts,
                        &p[0],
                        &p[1],
                        &p[2],
                        &q[0],
                        &q[1],
                        &q[2],
                        &q[3],
                        &v[0],
                        &v[1],
                        &v[2],
                        &w[0],
                        &w[1],
                        &w[2],
                        &a[0],
                        &a[1],
                        &a[2]);
    if (retval != 17) {
      EUROC_FATAL("Failed to parse line in [%s]", data_path);
    }

    // Add data
    const int k = data->num_timestamps;
    data->timestamps[k] = ts;
    data->p_RS_R[k] = vec_malloc(p, 3);
    data->q_RS[k] = vec_malloc(q, 4);
    data->v_RS_R[k] = vec_malloc(v, 3);
    data->b_w_RS_S[k] = vec_malloc(w, 3);
    data->b_a_RS_S[k] = vec_malloc(a, 3);
    data->num_timestamps++;
  }
  fclose(fp);

  return data;
}

/**
 * Free EuRoC ground truth data
 */
void euroc_ground_truth_free(euroc_ground_truth_t *data) {
  free(data->timestamps);

  for (size_t k = 0; k < data->num_timestamps; k++) {
    free(data->p_RS_R[k]);
    free(data->q_RS[k]);
    free(data->v_RS_R[k]);
    free(data->b_w_RS_S[k]);
    free(data->b_a_RS_S[k]);
  }
  free(data->p_RS_R);
  free(data->q_RS);
  free(data->v_RS_R);
  free(data->b_w_RS_S);
  free(data->b_a_RS_S);

  free(data);
}

//////////////////////
// euroc_timeline_t //
//////////////////////

/**
 * Create EuRoC timeline
 */
euroc_timeline_t *euroc_timeline_create(const euroc_imu_t *imu0_data,
                                        const euroc_camera_t *cam0_data,
                                        const euroc_camera_t *cam1_data) {
  // Determine unique set of timestamps
  int max_len = 0;
  max_len += imu0_data->num_timestamps;
  max_len += cam0_data->num_timestamps;
  max_len += cam1_data->num_timestamps;

  size_t ts_set_len = 0;
  timestamp_t *ts_set = malloc(sizeof(timestamp_t) * max_len);
  timestamps_unique(ts_set,
                    &ts_set_len,
                    imu0_data->timestamps,
                    imu0_data->num_timestamps);
  timestamps_unique(ts_set,
                    &ts_set_len,
                    cam0_data->timestamps,
                    cam0_data->num_timestamps);
  timestamps_unique(ts_set,
                    &ts_set_len,
                    cam1_data->timestamps,
                    cam1_data->num_timestamps);

  // Create timeline
  euroc_timeline_t *timeline = malloc(sizeof(euroc_timeline_t) * 1);
  timeline->num_timestamps = ts_set_len;
  timeline->timestamps = ts_set;
  timeline->events = malloc(sizeof(euroc_event_t) * ts_set_len);

  size_t imu0_idx = 0;
  size_t cam0_idx = 0;
  size_t cam1_idx = 0;

  for (size_t k = 0; k < timeline->num_timestamps; k++) {
    const timestamp_t ts = timeline->timestamps[k];

    // imu0 event
    int has_imu0 = 0;
    double *acc = NULL;
    double *gyr = NULL;
    for (size_t i = imu0_idx; i < imu0_data->num_timestamps; i++) {
      if (imu0_data->timestamps[i] == ts) {
        has_imu0 = 1;
        acc = imu0_data->a_B[imu0_idx];
        gyr = imu0_data->w_B[imu0_idx];
        imu0_idx++;
        break;
      }
    }

    // cam0 event
    int has_cam0 = 0;
    char *cam0_image = NULL;
    for (size_t i = cam0_idx; i < cam0_data->num_timestamps; i++) {
      if (cam0_data->timestamps[i] == ts) {
        has_cam0 = 1;
        cam0_image = cam0_data->image_paths[cam0_idx];
        cam0_idx++;
        break;
      }
    }

    // cam1 event
    int has_cam1 = 0;
    char *cam1_image = NULL;
    for (size_t i = cam1_idx; i < cam1_data->num_timestamps; i++) {
      if (cam1_data->timestamps[i] == ts) {
        has_cam1 = 1;
        cam1_image = cam1_data->image_paths[cam1_idx];
        cam1_idx++;
        break;
      }
    }

    // Add to event
    euroc_event_t *event = &timeline->events[k];
    event->has_imu0 = has_imu0;
    event->has_cam0 = has_cam0;
    event->has_cam1 = has_cam1;
    event->ts = ts;
    event->imu0_idx = imu0_idx - 1;
    event->acc = acc;
    event->gyr = gyr;
    event->cam0_idx = cam0_idx - 1;
    event->cam0_image = cam0_image;
    event->cam1_idx = cam1_idx - 1;
    event->cam1_image = cam1_image;
  }

  return timeline;
}

/**
 * Free EuRoC timeline
 */
void euroc_timeline_free(euroc_timeline_t *timeline) {
  free(timeline->timestamps);
  free(timeline->events);
  free(timeline);
}

//////////////////
// euroc_data_t //
//////////////////

/**
 * Load EuRoC data
 */
euroc_data_t *euroc_data_load(const char *data_path) {
  // Setup
  euroc_data_t *data = malloc(sizeof(euroc_data_t) * 1);

  // Load IMU data
  char imu0_path[9046] = {0};
  strcat(imu0_path, data_path);
  strcat(imu0_path, "/mav0/imu0");
  data->imu0_data = euroc_imu_load(imu0_path);

  // Load cam0 data
  char cam0_path[9046] = {0};
  strcat(cam0_path, data_path);
  strcat(cam0_path, "/mav0/cam0");
  data->cam0_data = euroc_camera_load(cam0_path, 0);

  // Load cam1 data
  char cam1_path[9046] = {0};
  strcat(cam1_path, data_path);
  strcat(cam1_path, "/mav0/cam1");
  data->cam1_data = euroc_camera_load(cam1_path, 0);

  // Load ground truth
  char gnd_path[9046] = {0};
  strcat(gnd_path, data_path);
  strcat(gnd_path, "/mav0/state_groundtruth_estimate0");
  data->ground_truth = euroc_ground_truth_load(gnd_path);

  // Create timeline
  data->timeline =
      euroc_timeline_create(data->imu0_data, data->cam0_data, data->cam1_data);

  return data;
}

/**
 * Free EuRoC data
 */
void euroc_data_free(euroc_data_t *data) {
  assert(data != NULL);
  euroc_imu_free(data->imu0_data);
  euroc_camera_free(data->cam0_data);
  euroc_camera_free(data->cam1_data);
  euroc_ground_truth_free(data->ground_truth);
  euroc_timeline_free(data->timeline);
  free(data);
}

//////////////////////////
// euroc_calib_target_t //
//////////////////////////

/**
 * Load EuRoC calibration target configuration
 */
euroc_calib_target_t *euroc_calib_target_load(const char *conf) {
  euroc_calib_target_t *data = malloc(sizeof(euroc_calib_target_t) * 1);
  yaml_get(conf, "target_type", 's', &data->type);
  yaml_get(conf, "tagRows", 'i', &data->tag_rows);
  yaml_get(conf, "tagCols", 'i', &data->tag_cols);
  yaml_get(conf, "tagSize", 'd', &data->tag_size);
  yaml_get(conf, "tagSpacing", 'd', &data->tag_spacing);
  return data;
}

/**
 * Free EuRoC calibration target
 */
void euroc_calib_target_free(euroc_calib_target_t *target) { free(target); }

/**
 * EuRoC calibration target to output stream
 */
void euroc_calib_target_print(const euroc_calib_target_t *target) {
  printf("target_type: %s\n", target->type);
  printf("tag_rows: %d\n", target->tag_rows);
  printf("tag_cols: %d\n", target->tag_cols);
  printf("tag_size: %f\n", target->tag_size);
  printf("tag_spacing: %f\n", target->tag_spacing);
}

///////////////////
// euroc_calib_t //
///////////////////

/**
 * Load EuRoC calibration data
 */
euroc_calib_t *euroc_calib_load(const char *data_path) {
  // Setup
  euroc_calib_t *data = malloc(sizeof(euroc_calib_t) * 1);

  // Load IMU data
  char imu0_path[9046] = {0};
  strcat(imu0_path, data_path);
  strcat(imu0_path, "/mav0/imu0");
  data->imu0_data = euroc_imu_load(imu0_path);

  // Load cam0 data
  char cam0_path[9046] = {0};
  strcat(cam0_path, data_path);
  strcat(cam0_path, "/mav0/cam0");
  data->cam0_data = euroc_camera_load(cam0_path, 0);

  // Load cam1 data
  char cam1_path[9046] = {0};
  strcat(cam1_path, data_path);
  strcat(cam1_path, "/mav0/cam1");
  data->cam1_data = euroc_camera_load(cam1_path, 0);

  // Load calibration target data
  char target_path[9046] = {0};
  strcat(target_path, data_path);
  strcat(target_path, "/april_6x6.yaml");
  data->calib_target = euroc_calib_target_load(target_path);

  // Create timeline
  data->timeline =
      euroc_timeline_create(data->imu0_data, data->cam0_data, data->cam1_data);

  return data;
}

/**
 * Free EuRoC calibration data
 */
void euroc_calib_free(euroc_calib_t *data) {
  euroc_imu_free(data->imu0_data);
  euroc_camera_free(data->cam0_data);
  euroc_camera_free(data->cam1_data);
  euroc_calib_target_free(data->calib_target);
  euroc_timeline_free(data->timeline);
  free(data);
}

/******************************************************************************
 * KITTI
 ******************************************************************************/

/**
 * Fatal
 *
 * @param[in] M Message
 * @param[in] ... Varadic arguments
 */
#ifndef KITTI_FATAL
#define KITTI_FATAL(...)                                                       \
  do {                                                                         \
    fprintf(stderr,                                                            \
            "[KITTI_FATAL] [%s:%d:%s()]: ",                                    \
            __FILE__,                                                          \
            __LINE__,                                                          \
            __func__);                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0);                                                                 \
  exit(-1)
#endif

#ifndef KITTI_LOG
#define KITTI_LOG(...)                                                         \
  do {                                                                         \
    fprintf(stderr,                                                            \
            "[KITTI_LOG] [%s:%d:%s()]: ",                                      \
            __FILE__,                                                          \
            __LINE__,                                                          \
            __func__);                                                         \
    fprintf(stderr, __VA_ARGS__);                                              \
  } while (0);
#endif

/**
 * Parse date time string to timestamp in nanoseconds (Unix time)
 */
static timestamp_t parse_dateline(const char *dt_str) {
  // Parse
  struct tm tm = {0};
  int fractional_seconds;
  sscanf(dt_str,
         "%d-%d-%d %d:%d:%d.%d",
         &tm.tm_year,
         &tm.tm_mon,
         &tm.tm_mday,
         &tm.tm_hour,
         &tm.tm_min,
         &tm.tm_sec,
         &fractional_seconds);
  tm.tm_year -= 1900; // Adjust for tm_year (years since 1900)
  tm.tm_mon -= 1;     // Adjust for 0-based months

  // Convert to time_t (Unix timestamp)
  time_t timestamp = mktime(&tm);
  if (timestamp == -1) {
    KITTI_FATAL("Failed to convert time to timestamp");
  }

  // Convert to uint64_t: Combine seconds and fractional part
  return (uint64_t) timestamp * 1000000000LL + fractional_seconds;
}

/**
 * Parse line
 */
static void
parse_value(FILE *fp, const char *key, const char *value_type, void *value) {
  assert(fp != NULL);
  assert(key != NULL);
  assert(value_type != NULL);
  assert(value != NULL);

  // Parse line
  const size_t buf_len = 1024;
  char buf[1024] = {0};
  if (fgets(buf, buf_len, fp) == NULL) {
    KITTI_FATAL("Failed to parse [%s]\n", key);
  }

  // Split key-value
  char delim[2] = ":";
  char *key_str = strtok(buf, delim);
  char *value_str = strtok(NULL, "\n");

  // Check key matches
  if (strcmp(key_str, key) != 0) {
    KITTI_FATAL("Failed to parse [%s]\n", key);
  }

  // Typecase value
  if (value_type == NULL) {
    KITTI_FATAL("Value type not set!\n");
  }

  if (strcmp(value_type, "string") == 0) {
    strcpy((char *) value, value_str);
  } else if (strcmp(value_type, "uint64_t") == 0) {
    *(uint64_t *) value = atol(value_str);
  } else if (strcmp(value_type, "int") == 0) {
    *(int *) value = atoi(value_str);
  } else if (strcmp(value_type, "float") == 0) {
    *(float *) value = atof(value_str);
  } else if (strcmp(value_type, "double") == 0) {
    *(double *) value = atof(value_str);
  } else {
    KITTI_FATAL("Invalid value type [%s]\n", value_type);
  }
}

/**
 * Parse double array
 */
static void
parse_double_array(FILE *fp, const char *key, double *array, int size) {
  assert(fp != NULL);
  assert(key != NULL);
  assert(array != NULL);

  // Parse line
  const size_t buf_len = 1024;
  char buf[1024] = {0};
  if (fgets(buf, buf_len, fp) == NULL) {
    KITTI_FATAL("Failed to parse [%s]\n", key);
  }

  // Split key-value
  char delim[2] = ":";
  char *key_str = strtok(buf, ":");
  char *value_str = strtok(NULL, delim);

  // Check key matches
  if (strcmp(key_str, key) != 0) {
    KITTI_FATAL("Failed to parse [%s]\n", key);
  }

  int i = 0;
  char *token = strtok(value_str, " ");
  while (token != NULL && i < size) {
    array[i++] = strtof(token, NULL);
    token = strtok(NULL, " ");
  }
}

////////////////////
// kitti_camera_t //
////////////////////

kitti_camera_t *kitti_camera_load(const char *data_dir) {
  // Form timestamps path
  char timestamps_path[1024] = {0};
  sprintf(timestamps_path, "%s/timestamps.txt", data_dir);

  // Open timestamps file
  const size_t num_rows = file_lines(timestamps_path);
  FILE *fp = fopen(timestamps_path, "r");
  if (fp == NULL) {
    KITTI_FATAL("Failed to open [%s]!\n", timestamps_path);
  }

  // Parse
  assert(num_rows > 0);
  kitti_camera_t *data = malloc(sizeof(kitti_camera_t));
  data->num_timestamps = num_rows;
  data->timestamps = malloc(sizeof(timestamp_t) * num_rows);
  data->image_paths = malloc(sizeof(char *) * num_rows);

  for (int i = 0; i < num_rows; ++i) {
    // Timestamp
    char line[1024] = {0};
    if (fgets(line, sizeof(line), fp) == NULL) {
      KITTI_FATAL("Failed to parse line %d in [%s]!\n", i, timestamps_path);
    }

    timestamp_t ts = parse_dateline(line);
    data->timestamps[i] = ts;

    // Image path
    char image_path[1024] = {0};
    sprintf(image_path, "%s/%s/%010d.png", data_dir, "data", i);
    data->image_paths[i] = malloc(sizeof(char) * strlen(image_path) + 1);
    strcpy(data->image_paths[i], image_path);
  }
  fclose(fp);

  return data;
}

void kitti_camera_free(kitti_camera_t *data) {
  // Timestamps
  free(data->timestamps);

  // Image paths
  for (size_t i = 0; i < data->num_timestamps; ++i) {
    free(data->image_paths[i]);
  }
  free(data->image_paths);

  free(data);
}

//////////////////
// kitti_oxts_t //
//////////////////

kitti_oxts_t *kitti_oxts_load(const char *data_dir) {
  // Get number of measurements
  char timestamps_path[1024] = {0};
  sprintf(timestamps_path, "%s/timestamps.txt", data_dir);
  const size_t num_rows = file_lines(timestamps_path);

  // Parse setup
  assert(num_rows > 0);
  kitti_oxts_t *data = malloc(sizeof(kitti_oxts_t));
  // -- Timestamps
  data->num_timestamps = num_rows;
  data->timestamps = malloc(sizeof(timestamp_t) * num_rows);
  // -- GPS
  data->lat = malloc(sizeof(real_t) * num_rows);
  data->lon = malloc(sizeof(real_t) * num_rows);
  data->alt = malloc(sizeof(real_t) * num_rows);
  // -- Attitude
  data->roll = malloc(sizeof(real_t) * num_rows);
  data->pitch = malloc(sizeof(real_t) * num_rows);
  data->yaw = malloc(sizeof(real_t) * num_rows);
  // -- Velocity
  data->vn = malloc(sizeof(real_t) * num_rows);
  data->ve = malloc(sizeof(real_t) * num_rows);
  data->vf = malloc(sizeof(real_t) * num_rows);
  data->vl = malloc(sizeof(real_t) * num_rows);
  data->vu = malloc(sizeof(real_t) * num_rows);
  // -- Acceleration
  data->ax = malloc(sizeof(real_t) * num_rows);
  data->ay = malloc(sizeof(real_t) * num_rows);
  data->az = malloc(sizeof(real_t) * num_rows);
  data->af = malloc(sizeof(real_t) * num_rows);
  data->al = malloc(sizeof(real_t) * num_rows);
  data->au = malloc(sizeof(real_t) * num_rows);
  // -- Angular velocity
  data->wx = malloc(sizeof(real_t) * num_rows);
  data->wy = malloc(sizeof(real_t) * num_rows);
  data->wz = malloc(sizeof(real_t) * num_rows);
  data->wf = malloc(sizeof(real_t) * num_rows);
  data->wl = malloc(sizeof(real_t) * num_rows);
  data->wu = malloc(sizeof(real_t) * num_rows);
  // -- Satellite tracking
  data->pos_accuracy = malloc(sizeof(real_t) * num_rows);
  data->vel_accuracy = malloc(sizeof(real_t) * num_rows);
  data->navstat = malloc(sizeof(int) * num_rows);
  data->numsats = malloc(sizeof(int) * num_rows);
  data->posmode = malloc(sizeof(int) * num_rows);
  data->velmode = malloc(sizeof(int) * num_rows);
  data->orimode = malloc(sizeof(int) * num_rows);

  // -- Parse timestamps
  {
    FILE *fp = fopen(timestamps_path, "r");
    if (fp == NULL) {
      KITTI_FATAL("Failed to open [%s]!\n", timestamps_path);
    }
    for (int i = 0; i < num_rows; ++i) {
      char line[1024] = {0};
      if (fgets(line, sizeof(line), fp) == NULL) {
        KITTI_FATAL("Failed to parse line %d in [%s]!\n", i, timestamps_path);
      }
      data->timestamps[i] = parse_dateline(line);
    }
    fclose(fp);
  }

  // -- Parse oxts data
  {
    char format[1024] = {0};
    strcat(format, "%lf %lf %lf ");             // GPS
    strcat(format, "%lf %lf %lf ");             // Attitude
    strcat(format, "%lf %lf %lf %lf %lf ");     // Velocity
    strcat(format, "%lf %lf %lf %lf %lf %lf "); // Acceleration
    strcat(format,
           "%lf %lf %lf %lf %lf %lf "); // Angular velocity
    strcat(format,
           "%lf %lf %s %s %s %s %s"); // Satellite tracking

    for (int i = 0; i < num_rows; ++i) {
      // Open oxts entry
      char entry_path[1024] = {0};
      sprintf(entry_path, "%s/%s/%010d.txt", data_dir, "data", i);
      FILE *fp = fopen(entry_path, "r");
      if (fp == NULL) {
        KITTI_FATAL("Failed to open [%s]!\n", timestamps_path);
      }

      // Parse
      char navstat_str[30] = {0}; // Navigation status
      char numsats_str[30] = {0}; // Number of satelllites tracked by GPS
      char posmode_str[30] = {0}; // Position mode
      char velmode_str[30] = {0}; // Velocity mode
      char orimode_str[30] = {0}; // Orientation mode
      int retval = fscanf(fp,
                          format,
                          // GPS
                          &data->lat[i],
                          &data->lon[i],
                          &data->alt[i],
                          // Attitude
                          &data->roll[i],
                          &data->pitch[i],
                          &data->yaw[i],
                          // Velocity
                          &data->vn[i],
                          &data->ve[i],
                          &data->vf[i],
                          &data->vl[i],
                          &data->vu[i],
                          // Acceleration
                          &data->ax[i],
                          &data->ay[i],
                          &data->az[i],
                          &data->af[i],
                          &data->al[i],
                          &data->au[i],
                          // Angular velocity
                          &data->wx[i],
                          &data->wy[i],
                          &data->wz[i],
                          &data->wf[i],
                          &data->wl[i],
                          &data->wu[i],
                          // Satellite tracking
                          &data->pos_accuracy[i],
                          &data->vel_accuracy[i],
                          navstat_str,
                          numsats_str,
                          posmode_str,
                          velmode_str,
                          orimode_str);

      // There's a bug in the KITTI OXTS data where in should be integer
      // but sometimes its float. Here we are parsing the satellite
      // tracking data as a string and converting it to integers.
      data->navstat[i] = strtol(navstat_str, NULL, 10);
      data->numsats[i] = strtol(numsats_str, NULL, 10);
      data->posmode[i] = strtol(posmode_str, NULL, 10);
      data->velmode[i] = strtol(velmode_str, NULL, 10);
      data->orimode[i] = strtol(orimode_str, NULL, 10);

      if (retval != 30) {
        KITTI_FATAL("Failed to parse [%s]\n", entry_path);
      }
      fclose(fp);
    }
  }

  return data;
}

void kitti_oxts_free(kitti_oxts_t *data) {
  // Timestamps
  free(data->timestamps);

  // GPS
  free(data->lat);
  free(data->lon);
  free(data->alt);

  // Attitude
  free(data->roll);
  free(data->pitch);
  free(data->yaw);

  // Velocity
  free(data->vn);
  free(data->ve);
  free(data->vf);
  free(data->vl);
  free(data->vu);

  // Acceleration
  free(data->ax);
  free(data->ay);
  free(data->az);
  free(data->af);
  free(data->al);
  free(data->au);

  // Angular velocity
  free(data->wx);
  free(data->wy);
  free(data->wz);
  free(data->wf);
  free(data->wl);
  free(data->wu);

  // Satellite tracking
  free(data->pos_accuracy);
  free(data->vel_accuracy);
  free(data->navstat);
  free(data->numsats);
  free(data->posmode);
  free(data->velmode);
  free(data->orimode);

  free(data);
}

//////////////////////
// kitti_velodyne_t //
//////////////////////

static timestamp_t *load_timestamps(const char *file_path) {
  const size_t num_rows = file_lines(file_path);
  if (num_rows == 0) {
    return NULL;
  }

  FILE *fp = fopen(file_path, "r");
  if (fp == NULL) {
    KITTI_FATAL("Failed to open [%s]!\n", file_path);
  }

  timestamp_t *timestamps = malloc(sizeof(timestamp_t) * num_rows);
  for (int i = 0; i < num_rows; ++i) {
    char line[1024] = {0};
    if (fgets(line, sizeof(line), fp) == NULL) {
      KITTI_FATAL("Failed to parse line %d in [%s]!\n", i, file_path);
    }
    timestamps[i] = parse_dateline(line);
  }

  return timestamps;
}

float *kitti_load_points(const char *pcd_path, size_t *num_points) {
  // Load pcd file
  FILE *pcd_file = fopen(pcd_path, "rb");
  if (!pcd_file) {
    KITTI_LOG("Failed to open [%s]", pcd_path);
    return NULL;
  }

  // Get the size of the file to know how many points
  fseek(pcd_file, 0, SEEK_END);
  const long int file_size = ftell(pcd_file);
  rewind(pcd_file);

  // Allocate memory for the points
  *num_points = file_size / (sizeof(float) * 4);
  float *points = malloc(sizeof(float) * 4 * *num_points);
  if (!points) {
    KITTI_LOG("Failed to allocate memory for points");
    fclose(pcd_file);
    return NULL;
  }

  // Read points from the file
  const size_t point_size = sizeof(float) * 4;
  const size_t read_count = fread(points, point_size, *num_points, pcd_file);
  if (read_count != *num_points) {
    KITTI_LOG("Failed to read all points");
    free(points);
    fclose(pcd_file);
    return NULL;
  }

  // Clean up
  fclose(pcd_file);

  return points;
}

kitti_velodyne_t *kitti_velodyne_load(const char *data_dir) {
  // Setup timestamp paths
  char timestamps_path[1024] = {0};
  char timestamps_start_path[1024] = {0};
  char timestamps_end_path[1024] = {0};
  sprintf(timestamps_path, "%s/timestamps.txt", data_dir);
  sprintf(timestamps_start_path, "%s/timestamps_start.txt", data_dir);
  sprintf(timestamps_end_path, "%s/timestamps_end.txt", data_dir);

  // Load data
  kitti_velodyne_t *data = malloc(sizeof(kitti_velodyne_t));
  data->num_timestamps = file_lines(timestamps_path);
  data->timestamps = load_timestamps(timestamps_path);
  data->timestamps_start = load_timestamps(timestamps_start_path);
  data->timestamps_end = load_timestamps(timestamps_end_path);
  data->pcd_paths = malloc(sizeof(char *) * data->num_timestamps);
  for (int i = 0; i < data->num_timestamps; ++i) {
    char pcd_path[1024] = {0};
    sprintf(pcd_path, "%s/%s/%010d.bin", data_dir, "data", i);
    data->pcd_paths[i] = malloc(sizeof(char) * strlen(pcd_path) + 1);
    strcpy(data->pcd_paths[i], pcd_path);
  }

  return data;
}

void kitti_velodyne_free(kitti_velodyne_t *data) {
  free(data->timestamps);
  free(data->timestamps_start);
  free(data->timestamps_end);
  for (int i = 0; i < data->num_timestamps; ++i) {
    free(data->pcd_paths[i]);
  }
  free(data->pcd_paths);
  free(data);
}

float *kitti_lidar_xyz(const char *pcd_path,
                       const float voxel_size,
                       size_t *nout) {
  // Load Kitti LIDAR points [x, y, z, intensity]
  size_t n = 0;
  float *raw_points = kitti_load_points(pcd_path, &n);

  // Extract only the relevant parts (x, y, z)
  float *points_xyz = malloc(sizeof(float) * 3 * n);
  for (size_t i = 0; i < n; ++i) {
    points_xyz[i * 3 + 0] = raw_points[i * 4 + 0];
    points_xyz[i * 3 + 1] = raw_points[i * 4 + 1];
    points_xyz[i * 3 + 2] = raw_points[i * 4 + 2];
  }

  // Downsample
  float *points_out = voxel_grid_downsample(points_xyz, n, voxel_size, nout);

  // Clean up
  free(raw_points);
  free(points_xyz);

  return points_out;
}

///////////////////
// kitti_calib_t //
///////////////////

kitti_calib_t *kitti_calib_load(const char *data_dir) {
  kitti_calib_t *data = malloc(sizeof(kitti_calib_t));

  // Load camera calibrations
  {
    char calib_cam_to_cam_path[1024] = {0};
    sprintf(calib_cam_to_cam_path, "%s/calib_cam_to_cam.txt", data_dir);
    FILE *fp = fopen(calib_cam_to_cam_path, "r");
    if (fp == NULL) {
      KITTI_FATAL("Failed to load [%s]\n", calib_cam_to_cam_path);
    }

    parse_value(fp, "calib_time", "string", &data->calib_time_cam_to_cam);
    parse_value(fp, "corner_dist", "double", &data->corner_dist);

    parse_double_array(fp, "S_00", data->S_00, 2);
    parse_double_array(fp, "K_00", data->K_00, 9);
    parse_double_array(fp, "D_00", data->D_00, 5);
    parse_double_array(fp, "R_00", data->D_00, 9);
    parse_double_array(fp, "T_00", data->T_00, 3);
    parse_double_array(fp, "S_rect_00", data->S_rect_00, 2);
    parse_double_array(fp, "R_rect_00", data->R_rect_00, 9);
    parse_double_array(fp, "P_rect_00", data->P_rect_00, 12);

    parse_double_array(fp, "S_01", data->S_01, 2);
    parse_double_array(fp, "K_01", data->K_01, 9);
    parse_double_array(fp, "D_01", data->D_01, 5);
    parse_double_array(fp, "R_01", data->D_01, 9);
    parse_double_array(fp, "T_01", data->T_01, 3);
    parse_double_array(fp, "S_rect_01", data->S_rect_01, 2);
    parse_double_array(fp, "R_rect_01", data->R_rect_01, 9);
    parse_double_array(fp, "P_rect_01", data->P_rect_01, 12);

    parse_double_array(fp, "S_02", data->S_02, 2);
    parse_double_array(fp, "K_02", data->K_02, 9);
    parse_double_array(fp, "D_02", data->D_02, 5);
    parse_double_array(fp, "R_02", data->D_02, 9);
    parse_double_array(fp, "T_02", data->T_02, 3);
    parse_double_array(fp, "S_rect_02", data->S_rect_02, 2);
    parse_double_array(fp, "R_rect_02", data->R_rect_02, 9);
    parse_double_array(fp, "P_rect_02", data->P_rect_02, 12);

    parse_double_array(fp, "S_03", data->S_03, 2);
    parse_double_array(fp, "K_03", data->K_03, 9);
    parse_double_array(fp, "D_03", data->D_03, 5);
    parse_double_array(fp, "R_03", data->D_03, 9);
    parse_double_array(fp, "T_03", data->T_03, 3);
    parse_double_array(fp, "S_rect_03", data->S_rect_03, 2);
    parse_double_array(fp, "R_rect_03", data->R_rect_03, 9);
    parse_double_array(fp, "P_rect_03", data->P_rect_03, 12);

    fclose(fp);
  }

  // Load IMU to Velodyne extrinsics
  {
    char calib_imu_to_velo_path[1024] = {0};
    sprintf(calib_imu_to_velo_path, "%s/calib_imu_to_velo.txt", data_dir);
    FILE *fp = fopen(calib_imu_to_velo_path, "r");
    if (fp == NULL) {
      KITTI_FATAL("Failed to load [%s]\n", calib_imu_to_velo_path);
    }

    parse_value(fp, "calib_time", "string", &data->calib_time_imu_to_velo);
    parse_double_array(fp, "R", data->R_velo_imu, 9);
    parse_double_array(fp, "T", data->T_velo_imu, 3);
    fclose(fp);
  }

  // Load Velodyne to camera extrinsics
  {
    char calib_velo_to_cam_path[1024] = {0};
    sprintf(calib_velo_to_cam_path, "%s/calib_velo_to_cam.txt", data_dir);
    FILE *fp = fopen(calib_velo_to_cam_path, "r");
    if (fp == NULL) {
      KITTI_FATAL("Failed to load [%s]\n", calib_velo_to_cam_path);
    }

    parse_value(fp, "calib_time", "string", &data->calib_time_velo_to_cam);
    parse_double_array(fp, "R", data->R_cam_velo, 9);
    parse_double_array(fp, "T", data->T_cam_velo, 3);
    parse_double_array(fp, "delta_f", data->delta_f, 2);
    parse_double_array(fp, "delta_c", data->delta_c, 2);
    fclose(fp);
  }

  return data;
}

void kitti_calib_free(kitti_calib_t *data) { free(data); }

void kitti_calib_print(const kitti_calib_t *data) {
  printf("calib_time_cam_to_cam: %s\n", data->calib_time_cam_to_cam);
  printf("calib_time_imu_to_velo: %s\n", data->calib_time_imu_to_velo);
  printf("calib_time_velo_to_cam: %s\n", data->calib_time_velo_to_cam);
  printf("corner_dist: %f\n", data->corner_dist);
  print_double_array("S_00", data->S_00, 2);
  print_double_array("K_00", data->K_00, 9);
  print_double_array("D_00", data->D_00, 5);
  print_double_array("R_00", data->R_00, 9);
  print_double_array("T_00", data->T_00, 3);
  print_double_array("S_rect_00", data->S_rect_00, 2);
  print_double_array("R_rect_00", data->R_rect_00, 9);
  print_double_array("P_rect_00", data->P_rect_00, 12);
  printf("\n");

  print_double_array("S_01", data->S_01, 2);
  print_double_array("K_01", data->K_01, 9);
  print_double_array("D_01", data->D_01, 5);
  print_double_array("R_01", data->R_01, 9);
  print_double_array("T_01", data->T_01, 3);
  print_double_array("S_rect_01", data->S_rect_01, 2);
  print_double_array("R_rect_01", data->R_rect_01, 9);
  print_double_array("P_rect_01", data->P_rect_01, 12);
  printf("\n");

  print_double_array("S_02", data->S_02, 2);
  print_double_array("K_02", data->K_02, 9);
  print_double_array("D_02", data->D_02, 5);
  print_double_array("R_02", data->R_02, 9);
  print_double_array("T_02", data->T_02, 3);
  print_double_array("S_rect_02", data->S_rect_02, 2);
  print_double_array("R_rect_02", data->R_rect_02, 9);
  print_double_array("P_rect_02", data->P_rect_02, 12);
  printf("\n");

  print_double_array("S_03", data->S_03, 2);
  print_double_array("K_03", data->K_03, 9);
  print_double_array("D_03", data->D_03, 5);
  print_double_array("R_03", data->R_03, 9);
  print_double_array("T_03", data->T_03, 3);
  print_double_array("S_rect_03", data->S_rect_03, 2);
  print_double_array("R_rect_03", data->R_rect_03, 9);
  print_double_array("P_rect_03", data->P_rect_03, 12);
  printf("\n");

  print_double_array("R_velo_imu", data->R_velo_imu, 9);
  print_double_array("T_velo_imu", data->T_velo_imu, 3);
  printf("\n");

  print_double_array("R_cam_velo", data->R_cam_velo, 9);
  print_double_array("T_cam_velo", data->T_cam_velo, 3);
  print_double_array("delta_f", data->delta_f, 2);
  print_double_array("delta_c", data->delta_c, 2);
  printf("\n");
}

/////////////////
// kitti_raw_t //
/////////////////

kitti_raw_t *kitti_raw_load(const char *data_dir, const char *seq_name) {
  // Setup paths
  char image_00_path[1024] = {0};
  char image_01_path[1024] = {0};
  char image_02_path[1024] = {0};
  char image_03_path[1024] = {0};
  char oxts_path[1024] = {0};
  char velodyne_points_path[1024] = {0};
  sprintf(image_00_path, "%s/%s/image_00", data_dir, seq_name);
  sprintf(image_01_path, "%s/%s/image_01", data_dir, seq_name);
  sprintf(image_02_path, "%s/%s/image_02", data_dir, seq_name);
  sprintf(image_03_path, "%s/%s/image_03", data_dir, seq_name);
  sprintf(oxts_path, "%s/%s/oxts", data_dir, seq_name);
  sprintf(velodyne_points_path, "%s/%s/velodyne_points", data_dir, seq_name);

  // Load data
  kitti_raw_t *data = malloc(sizeof(kitti_raw_t));
  strcpy(data->seq_name, seq_name);
  data->image_00 = kitti_camera_load(image_00_path);
  data->image_01 = kitti_camera_load(image_01_path);
  data->image_02 = kitti_camera_load(image_02_path);
  data->image_03 = kitti_camera_load(image_03_path);
  data->oxts = kitti_oxts_load(oxts_path);
  data->velodyne = kitti_velodyne_load(velodyne_points_path);
  data->calib = kitti_calib_load(data_dir);

  return data;
}

void kitti_raw_free(kitti_raw_t *data) {
  kitti_camera_free(data->image_00);
  kitti_camera_free(data->image_01);
  kitti_camera_free(data->image_02);
  kitti_camera_free(data->image_03);
  kitti_oxts_free(data->oxts);
  kitti_velodyne_free(data->velodyne);
  kitti_calib_free(data->calib);
  free(data);
}
