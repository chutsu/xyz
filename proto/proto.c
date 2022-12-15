#include "proto.h"

#ifdef USE_STB
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_DS_IMPLEMENTATION
#include <stb_ds.h>
#endif

// #ifdef USE_APRILGRID
// #define APRILGRID_IMPLEMENTATION
// #include "aprilgrid.h"
// #endif

// #ifdef USE_SBGC
// #define SBGC_IMPLEMENTATION
// #include "sbgc.h"
// #endif

/******************************************************************************
 * FILE SYSTEM
 ******************************************************************************/

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
    retval = MALLOC(char, (strlen(x) + strlen(y)) + 1);
    string_copy(retval, x);
    string_copy(retval + strlen(retval), (y[0] == '/') ? y + 1 : y);
  } else {
    retval = MALLOC(char, (strlen(x) + strlen(y)) + 2);
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
  int nb_files = scandir(path, &namelist, 0, alphasort);
  if (nb_files < 0) {
    return NULL;
  }

  // The first two are '.' and '..'
  free(namelist[0]);
  free(namelist[1]);

  // Allocate memory for list of files
  char **files = MALLOC(char *, nb_files - 2);
  *n = 0;

  // Create list of files
  for (int i = 2; i < nb_files; i++) {
    char fp[9046] = {0};
    const char *c = (path[strlen(path) - 1] == '/') ? "" : "/";
    string_cat(fp, path);
    string_cat(fp, c);
    string_cat(fp, namelist[i]->d_name);

    files[*n] = MALLOC(char, strlen(fp) + 1);
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

  char *buf = MALLOC(char, len + 1);
  if (buf == NULL) {
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
int file_exists(const char *fp) {
  return (access(fp, F_OK) == 0) ? 1 : 0;
}

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
    fclose(file);
    return -1;
  }

  // Obtain number of lines
  int nb_rows = 0;
  char *line = NULL;
  size_t len = 0;
  while (getline(&line, &len, file) != -1) {
    nb_rows++;
  }
  free(line);

  // Clean up
  fclose(file);

  return nb_rows;
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
    fclose(src_file);
    return -1;
  }

  FILE *dst_file = fopen(dst, "wb");
  if (dst_file == NULL) {
    fclose(src_file);
    fclose(dst_file);
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

/******************************************************************************
 * DATA
 ******************************************************************************/

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
  char *retval = MALLOC(char, strlen(s) + 1);
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
        data = CALLOC(int, array_size + 1);
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
int **load_iarrays(const char *csv_path, int *nb_arrays) {
  assert(csv_path != NULL);
  FILE *csv_file = fopen(csv_path, "r");
  *nb_arrays = dsv_rows(csv_path);
  int **array = CALLOC(int *, *nb_arrays);

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
static real_t *parse_darray_line(char *line) {
  assert(line != NULL);
  char entry[MAX_LINE_LENGTH] = {0};
  int index = 0;
  real_t *data = NULL;

  for (size_t i = 0; i < strlen(line); i++) {
    char c = line[i];
    if (c == ' ') {
      continue;
    }

    if (c == ',' || c == '\n') {
      if (data == NULL) {
        size_t array_size = strtod(entry, NULL);
        data = CALLOC(real_t, array_size);
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
 * Parse 2D real arrays from csv file at `csv_path`, on success `nb_arrays`
 * will return number of arrays.
 * @returns
 * - List of 1D vector of reals
 * - NULL for failure
 */
real_t **load_darrays(const char *csv_path, int *nb_arrays) {
  assert(csv_path != NULL);
  assert(nb_arrays != NULL);
  FILE *csv_file = fopen(csv_path, "r");
  *nb_arrays = dsv_rows(csv_path);
  real_t **array = CALLOC(real_t *, *nb_arrays);

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
  int *i = MALLOC(int, 1);
  *i = val;
  return i;
}

/**
 * Allocate heap memory for float `val`.
 */
float *float_malloc(const float val) {
  float *f = MALLOC(float, 1);
  *f = val;
  return f;
}

/**
 * Allocate heap memory for double `val`.
 */
double *double_malloc(const double val) {
  double *d = MALLOC(double, 1);
  *d = val;
  return d;
}

/**
 * Allocate heap memory for vector `vec` with length `N`.
 */
real_t *vector_malloc(const real_t *vec, const real_t N) {
  real_t *retval = MALLOC(real_t, N);
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
  int nb_rows = 0;
  char line[MAX_LINE_LENGTH] = {0};
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    if (line[0] != '#') {
      nb_rows++;
    }
  }

  // Cleanup
  fclose(infile);

  return nb_rows;
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
  int nb_elements = 1;
  int found_separator = 0;
  for (size_t i = 0; i < MAX_LINE_LENGTH; i++) {
    if (line[i] == delim) {
      found_separator = 1;
      nb_elements++;
    }
  }

  // Cleanup
  fclose(infile);

  return (found_separator) ? nb_elements : -1;
}

/**
 * Get the fields of the delimited file at `fp`, where `delim` is the value
 * separated symbol and `nb_fields` returns the length of the fields returned.
 * @returns
 * - List of field strings
 * - NULL for failure
 */
char **dsv_fields(const char *fp, const char delim, int *nb_fields) {
  assert(fp != NULL);

  // Load file
  FILE *infile = fopen(fp, "r");
  if (infile == NULL) {
    fclose(infile);
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
  *nb_fields = dsv_cols(fp, delim);
  char **fields = MALLOC(char *, *nb_fields);
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
real_t **
dsv_data(const char *fp, const char delim, int *nb_rows, int *nb_cols) {
  assert(fp != NULL);

  // Obtain number of rows and columns in dsv data
  *nb_rows = dsv_rows(fp);
  *nb_cols = dsv_cols(fp, delim);
  if (*nb_rows == -1 || *nb_cols == -1) {
    return NULL;
  }

  // Load file
  FILE *infile = fopen(fp, "r");
  if (infile == NULL) {
    fclose(infile);
    return NULL;
  }

  // Loop through data
  char line[MAX_LINE_LENGTH] = {0};
  int row_idx = 0;
  int col_idx = 0;

  // Loop through data line by line
  real_t **data = MALLOC(real_t *, *nb_rows);
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    // Ignore if comment line
    if (line[0] == '#') {
      continue;
    }

    // Iterate through values in line separated by commas
    data[row_idx] = MALLOC(real_t, *nb_cols);
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
void dsv_free(real_t **data, const int nb_rows) {
  assert(data != NULL);
  for (int i = 0; i < nb_rows; i++) {
    free(data[i]);
  }
  free(data);
}

/**
 * Load comma separated data as a matrix, where `fp` is the csv file path, on
 * success `nb_rows` and `nb_cols` will be filled.
 * @returns
 * - Matrix of CSV data
 * - NULL for failure
 */
real_t **csv_data(const char *fp, int *nb_rows, int *nb_cols) {
  assert(fp != NULL);
  return dsv_data(fp, ',', nb_rows, nb_cols);
}

/**
 * Free CSV data.
 */
void csv_free(real_t **data, const int nb_rows) {
  for (int i = 0; i < nb_rows; i++) {
    free(data[i]);
  }
  free(data);
}

/******************************************************************************
 * DATA-STRUCTURES
 ******************************************************************************/

////////////
// DARRAY //
////////////

darray_t *darray_new(size_t element_size, size_t initial_max) {
  assert(element_size > 0);
  assert(initial_max > 0);

  darray_t *array = MALLOC(darray_t, 1);
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
    FREE_MEM(array->contents[i], free);
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

//////////
// LIST //
//////////

list_t *list_new() {
  list_t *list = calloc(1, sizeof(list_t));
  list->length = 0;
  list->first = NULL;
  list->last = NULL;
  return list;
}

void list_destroy(list_t *list) {
  assert(list != NULL);

  list_node_t *node;
  list_node_t *next_node;

  // Destroy
  node = list->first;
  while (node != NULL) {
    next_node = node->next;
    FREE_MEM(node, free);
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

void list_clear_destroy(list_t *list) {
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

  if (before_last == NULL) {
    return NULL;
  }
  before_last->next = NULL;

  return value;
}

void *list_pop_front(list_t *list) {
  assert(list != NULL);
  assert(list->first != NULL);

  // pop front
  list_node_t *first_node = list->first;
  void *data = first_node->value;
  list_node_t *next_node = first_node->next;

  if (next_node != NULL) {
    list->first = next_node;
  } else {
    list->first = NULL;
  }
  list->length--;

  // clean up
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

///////////
// STACK //
///////////

mstack_t *stack_new() {
  mstack_t *s = MALLOC(mstack_t, 1);
  s->size = 0;
  s->root = NULL;
  s->end = NULL;
  return s;
}

void mstack_destroy_traverse(mstack_node_t *n, void (*free_func)(void *)) {
  if (n->next) {
    mstack_destroy_traverse(n->next, free_func);
  }
  if (free_func) {
    free_func(n->value);
  }
  free(n);
  n = NULL;
}

void mstack_clear_destroy(mstack_t *s, void (*free_func)(void *)) {
  if (s->root) {
    mstack_destroy_traverse(s->root, free_func);
  }
  free(s);
  s = NULL;
}

void mstack_destroy(mstack_t *s) {
  if (s->root) {
    mstack_destroy_traverse(s->root, NULL);
  }
  free(s);
  s = NULL;
}

int mstack_push(mstack_t *s, void *value) {
  mstack_node_t *n = MALLOC(mstack_node_t, 1);
  if (n == NULL) {
    return -1;
  }

  mstack_node_t *prev_end = s->end;
  n->value = value;
  n->next = NULL;
  n->prev = prev_end;

  if (s->size == 0) {
    s->root = n;
    s->end = n;
  } else {
    prev_end->next = n;
    s->end = n;
  }
  s->size++;

  return 0;
}

void *mstack_pop(mstack_t *s) {
  void *value = s->end->value;
  mstack_node_t *previous = s->end->prev;

  free(s->end);
  if (s->size > 1) {
    previous->next = NULL;
    s->end = previous;
  } else {
    s->root = NULL;
    s->end = NULL;
  }
  s->size--;

  return value;
}

///////////
// QUEUE //
///////////

queue_t *queue_new() {
  queue_t *q = calloc(1, sizeof(queue_t));
  q->queue = list_new();
  q->count = 0;
  return q;
}

void queue_destroy(queue_t *q) {
  assert(q != NULL);
  list_destroy(q->queue);
  free(q);
  q = NULL;
}

int queue_enqueue(queue_t *q, void *data) {
  assert(q != NULL);
  list_push(q->queue, data);
  q->count++;
  return 0;
}

void *queue_dequeue(queue_t *q) {
  assert(q != NULL);
  void *data = list_pop_front(q->queue);
  q->count--;

  return data;
}

int queue_empty(queue_t *q) {
  assert(q != NULL);
  return (q->count == 0) ? 1 : 0;
}

void *queue_first(queue_t *q) {
  assert(q != NULL);
  if (q->count != 0) {
    return q->queue->first->value;
  }
  return NULL;
}

void *queue_last(queue_t *q) {
  assert(q != NULL);
  if (q->count != 0) {
    return q->queue->last->value;
  }
  return NULL;
}

/////////////
// HASHMAP //
/////////////

static inline int default_cmp(void *a, void *b) {
  return strcmp(a, b);
}

static uint32_t default_hash(void *a) {
  // Simple bob jenkins's hash algorithm
  char *k = a;
  uint32_t hash = 0;
  for (uint32_t i = 0; i < strlen(a); i++) {
    hash += (uint32_t) k[i];
    hash += (hash << 10);
    hash ^= (hash >> 6);
  }

  hash += (hash << 3);
  hash ^= (hash >> 11);
  hash += (hash << 15);

  return hash;
}

static inline void *default_key_copy(void *target) {
  return string_malloc(target);
}

static inline void *default_value_copy(void *target) {
  return string_malloc(target);
}

hashmap_t *hashmap_new() {
  hashmap_t *map = MALLOC(hashmap_t, 1);
  if (map == NULL) {
    return NULL;
  }

  // Create bucket
  map->buckets = darray_new(sizeof(darray_t *), DEFAULT_NUMBER_OF_BUCKETS);
  map->buckets->end = map->buckets->max; // fake out expanding it
  if (map->buckets == NULL) {
    free(map);
    return NULL;
  }

  // Set comparator and hash functions
  map->cmp = default_cmp;
  map->hash = default_hash;

  // Set key and value copy functions
  map->copy_kv = 1;
  map->k_copy = default_key_copy;
  map->v_copy = default_value_copy;
  map->k_free = free;
  map->v_free = free;

  return map;
}

static void free_bucket(darray_t *bucket) {
  assert(bucket != NULL);

  for (int i = 0; i < bucket->end; i++) {
    hashmap_node_t *n = darray_get(bucket, i);
    free(n);
  }

  darray_destroy(bucket);
}

static void clear_free_bucket(hashmap_t *map, darray_t *bucket) {
  assert(map != NULL);
  assert(bucket != NULL);

  // Clear free bucket
  for (int i = 0; i < bucket->end; i++) {
    hashmap_node_t *n = darray_get(bucket, i);
    map->k_free(n->key);
    map->k_free(n->value);
    free(n);
  }

  darray_destroy(bucket);
}

static void free_buckets(hashmap_t *map) {
  assert(map != NULL);

  // Free buckets
  for (int i = 0; i < map->buckets->end; i++) {
    darray_t *bucket = darray_get(map->buckets, i);

    if (bucket) {
      if (map->copy_kv) {
        clear_free_bucket(map, bucket);
      } else {
        free_bucket(bucket);
      }
    }
  }

  darray_destroy(map->buckets);
}

void hashmap_clear_destroy(hashmap_t *map) {
  if (map) {
    if (map->buckets) {
      free_buckets(map);
    }
    free(map);
  }
}

void hashmap_destroy(hashmap_t *map) {
  if (map) {
    if (map->buckets) {
      free_buckets(map);
    }
    free(map);
  }
}

static hashmap_node_t *hashmap_node_new(uint32_t h, void *k, void *v) {
  assert(k != NULL);
  assert(v != NULL);

  // Setup
  hashmap_node_t *node = calloc(1, sizeof(hashmap_node_t));
  if (node == NULL) {
    return NULL;
  }

  // Create hashmap node
  node->key = k;
  node->value = v;
  node->hash = h;

  return node;
}

static darray_t *
hashmap_find_bucket(hashmap_t *map, void *k, int create, uint32_t *hash_out) {
  assert(map != NULL);
  assert(k != NULL);
  assert(hash_out != NULL);

  // Pre-check
  uint32_t hash = map->hash(k);
  int bucket_n = hash % DEFAULT_NUMBER_OF_BUCKETS;
  if ((bucket_n >= 0) == 0) {
    return NULL;
  }
  *hash_out = hash; // Store it for return so caller can use it

  // Find bucket
  darray_t *bucket = darray_get(map->buckets, bucket_n);

  // Coundn't find bucket, create one instead
  if (!bucket && create) {
    // New bucket, set it up
    bucket = darray_new(sizeof(void *), DEFAULT_NUMBER_OF_BUCKETS);
    if (bucket == NULL) {
      return NULL;
    }
    darray_set(map->buckets, bucket_n, bucket);
  }

  return bucket;
}

int hashmap_set(hashmap_t *map, void *k, void *v) {
  assert(map != NULL);
  assert(map->k_copy != NULL);
  assert(map->v_copy != NULL);
  assert(k != NULL);
  assert(v != NULL);

  // Pre-check
  uint32_t hash = 0;
  darray_t *bucket = hashmap_find_bucket(map, k, 1, &hash);
  if (bucket == NULL) {
    return -1;
  }

  // Set hashmap
  hashmap_node_t *node = hashmap_node_new(hash, map->k_copy(k), map->v_copy(v));
  if (node == NULL) {
    return -1;
  }
  darray_push(bucket, node);

  return 0;
}

static inline int
hashmap_get_node(hashmap_t *map, uint32_t hash, darray_t *bucket, void *k) {
  assert(map != NULL);
  assert(bucket != NULL);
  assert(k != NULL);

  for (int i = 0; i < bucket->end; i++) {
    hashmap_node_t *node = darray_get(bucket, i);
    if (node->hash == hash && map->cmp(node->key, k) == 0) {
      return i;
    }
  }

  return -1;
}

void *hashmap_get(hashmap_t *map, void *k) {
  assert(map != NULL);
  assert(k != NULL);

  // Find bucket
  uint32_t hash = 0;
  darray_t *bucket = hashmap_find_bucket(map, k, 0, &hash);
  if (bucket == NULL) {
    return NULL;
  }

  // Find hashmap node
  int i = hashmap_get_node(map, hash, bucket, k);
  if (i == -1) {
    return NULL;
  }

  // Get value
  hashmap_node_t *node = darray_get(bucket, i);
  if (node == NULL) {
    return NULL;
  }

  return node->value;
}

int hashmap_traverse(hashmap_t *map,
                     int (*hashmap_traverse_cb)(hashmap_node_t *)) {
  assert(map != NULL);
  assert(hashmap_traverse_cb != NULL);

  // Traverse
  int rc = 0;
  for (int i = 0; i < map->buckets->end; i++) {
    darray_t *bucket = darray_get(map->buckets, i);

    if (bucket) {
      for (int j = 0; j < bucket->end; j++) {
        hashmap_node_t *node = darray_get(bucket, j);
        rc = hashmap_traverse_cb(node);

        if (rc != 0) {
          return rc;
        }
      }
    }
  }

  return 0;
}

void *hashmap_delete(hashmap_t *map, void *k) {
  assert(map != NULL);
  assert(k != NULL);

  // Find bucket containing hashmap node
  uint32_t hash = 0;
  darray_t *bucket = hashmap_find_bucket(map, k, 0, &hash);
  if (bucket == NULL) {
    return NULL;
  }

  // From bucket get hashmap node and free it
  int i = hashmap_get_node(map, hash, bucket, k);
  if (i == -1) {
    return NULL;
  }

  // Get node
  hashmap_node_t *node = darray_get(bucket, i);
  void *v = node->value;
  if (map->copy_kv) {
    map->k_free(node->key);
  }
  free(node);

  // Check to see if last element in bucket is a node
  hashmap_node_t *ending = darray_pop(bucket);
  if (ending != node) {
    // Alright looks like it's not the last one, swap it
    darray_set(bucket, i, ending);
  }

  return v;
}

/******************************************************************************
 * TIME
 ******************************************************************************/

/**
 * Tic, start timer.
 * @returns A timespec encapsulating the time instance when tic() is called
 */
struct timespec tic() {
  struct timespec time_start;
  clock_gettime(CLOCK_MONOTONIC, &time_start);
  return time_start;
}

/**
 * Toc, stop timer.
 * @returns Time elapsed in seconds
 */
float toc(struct timespec *tic) {
  assert(tic != NULL);
  struct timespec toc;
  float time_elasped;

  clock_gettime(CLOCK_MONOTONIC, &toc);
  time_elasped = (toc.tv_sec - tic->tv_sec);
  time_elasped += (toc.tv_nsec - tic->tv_nsec) / 1000000000.0;

  return time_elasped;
}

/**
 * Toc, stop timer.
 * @returns Time elapsed in milli-seconds
 */
float mtoc(struct timespec *tic) {
  assert(tic != NULL);
  return toc(tic) * 1000.0;
}

/**
 * Get time now since epoch.
 * @return Time now in nano-seconds since epoch
 */
timestamp_t time_now() {
  struct timespec spec;
  clock_gettime(CLOCK_REALTIME, &spec);

  const time_t sec = spec.tv_sec;
  const long int ns = spec.tv_nsec;
  const uint64_t BILLION = 1000000000L;

  return (uint64_t) sec * BILLION + (uint64_t) ns;
}

/**
 * Convert timestamp to seconds
 */
real_t ts2sec(const timestamp_t ts) {
  return ts * 1e-9;
}

/**
 * Convert seconds to timestamp
 */
timestamp_t sec2ts(const real_t time_s) {
  return time_s * 1e9;
}

/******************************************************************************
 * NETWORK
 ******************************************************************************/

/**
 * Return IP and Port info from socket file descriptor `sockfd` to `ip` and
 * `port`. Returns `0` for success and `-1` for failure.
 * @returns
 * - 0 for success
 * - -1 for failure
 */
int ip_port_info(const int sockfd, char *ip, int *port) {
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
int tcp_server_setup(tcp_server_t *server, const int port) {
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
  if (setsockopt(server->sockfd, SOL_SOCKET, SO_REUSEPORT, &en, int_sz) < 0) {
    LOG_ERROR("setsockopt(SO_REUSEPORT) failed");
  }

  // Assign IP, PORT
  struct sockaddr_in addr;
  bzero(&addr, sizeof(addr));
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
int tcp_server_loop(tcp_server_t *server) {
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
int tcp_client_setup(tcp_client_t *client,
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
int tcp_client_loop(tcp_client_t *client) {
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

/******************************************************************************
 *                                 MATHS
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
 * Degrees to radians.
 * @returns Radians
 */
real_t deg2rad(const real_t d) {
  return d * (M_PI / 180.0);
}

/**
 * Radians to degrees.
 * @returns Degrees
 */
real_t rad2deg(const real_t r) {
  return r * (180.0 / M_PI);
}

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
  real_t *vals = MALLOC(real_t, n);
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

/******************************************************************************
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
      // printf("%f  ", A[idx]);
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
    // printf("%e ", v[i]);
    printf("%f ", v[i]);
    // printf("%.4f ", v[i]);
    // printf("%.10f ", v[i]);
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
 * Malloc matrix of size `m x n`.
 */
real_t *mat_malloc(const size_t m, const size_t n) {
  assert(m > 0);
  assert(n > 0);
  return CALLOC(real_t, m * n);
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

/**
 * Save matrix `A` of size `m x n` to `save_path`.
 * @returns `0` for success, `-1` for failure
 */
int mat_save(const char *save_path, const real_t *A, const int m, const int n) {
  assert(save_path != NULL);
  assert(A != NULL);
  assert(m > 0);
  assert(n > 0);

  FILE *csv_file = fopen(save_path, "w");
  if (csv_file == NULL) {
    return -1;
  }

  int idx = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      fprintf(csv_file, "%.18e", A[idx]);
      idx++;
      if ((j + 1) != n) {
        fprintf(csv_file, ",");
      }
    }
    fprintf(csv_file, "\n");
  }
  fclose(csv_file);

  return 0;
}

/**
 * Load matrix from file in `mat_path`, on success `nb_rows` and `nb_cols` will
 * be set respectively.
 */
real_t *mat_load(const char *mat_path, int *nb_rows, int *nb_cols) {
  assert(mat_path != NULL);
  assert(nb_rows != NULL);
  assert(nb_cols != NULL);

  // Obtain number of rows and columns in csv data
  *nb_rows = dsv_rows(mat_path);
  *nb_cols = dsv_cols(mat_path, ',');
  if (*nb_rows == -1 || *nb_cols == -1) {
    return NULL;
  }

  // Initialize memory for csv data
  real_t *A = MALLOC(real_t, *nb_rows * *nb_cols);

  // Load file
  FILE *infile = fopen(mat_path, "r");
  if (infile == NULL) {
    fclose(infile);
    free(A);
    return NULL;
  }

  // Loop through data
  char line[MAX_LINE_LENGTH] = {0};
  int row_idx = 0;
  int col_idx = 0;
  int idx = 0;

  // Loop through data line by line
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    // Ignore if comment line
    if (line[0] == '#') {
      continue;
    }

    // Iterate through values in line separated by commas
    char entry[100] = {0};
    for (size_t i = 0; i < strlen(line); i++) {
      char c = line[i];
      if (c == ' ') {
        continue;
      }

      if (c == ',' || c == '\n') {
        A[idx] = strtod(entry, NULL);
        idx++;

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

  return A;
}

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
                 const int nb_rows,
                 const int col_idx,
                 const real_t *x) {
  int vec_idx = 0;
  for (int i = 0; i < nb_rows; i++) {
    A[i * stride + col_idx] = x[vec_idx++];
  }
}

/**
 * Get matrix column.
 */
void mat_col_get(
    const real_t *A, const int m, const int n, const int col_idx, real_t *x) {
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
 * Create new vector of length `n` in heap memory.
 * @returns Heap allocated vector
 */
real_t *vec_malloc(const real_t *x, const size_t n) {
  assert(n > 0);
  real_t *vec = CALLOC(real_t, n);
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
 * Load vector.
 *
 * @param vec_path Path to csv containing vector values
 * @param m Number of rows
 * @param n Number of cols
 *
 * @returns Vector or `NULL` for failure
 */
real_t *vec_load(const char *vec_path, int *m, int *n) {
  assert(vec_path != NULL);
  assert(m != NULL);
  assert(n != NULL);

  // Obtain number of rows and columns in csv data
  *m = dsv_rows(vec_path);
  *n = dsv_cols(vec_path, ',');
  if (*m > 0 && *n == -1) {
    // Load file
    FILE *infile = fopen(vec_path, "r");
    if (infile == NULL) {
      fclose(infile);
      return NULL;
    }

    // Loop through data line by line
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
      // Ignore if comment line
      if (line[0] == '#') {
        continue;
      }

      if (strlen(line) == 0) {
        fclose(infile);
        return NULL;
      }
    }

    *n = 1;
  } else if (*m == -1 || *n == -1) {
    return NULL;
  }

  // Initialize memory for csv data
  real_t *x = MALLOC(real_t, *m * *n);

  // Load file
  FILE *infile = fopen(vec_path, "r");
  if (infile == NULL) {
    fclose(infile);
    free(x);
    return NULL;
  }

  // Loop through data
  char line[MAX_LINE_LENGTH] = {0};
  int row_idx = 0;
  int col_idx = 0;
  int idx = 0;

  // Loop through data line by line
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    // Ignore if comment line
    if (line[0] == '#') {
      continue;
    }

    // Iterate through values in line separated by commas
    char entry[100] = {0};
    for (size_t i = 0; i < strlen(line); i++) {
      char c = line[i];
      if (c == ' ') {
        continue;
      }

      if (c == ',' || c == '\n') {
        x[idx] = strtod(entry, NULL);
        idx++;

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

  return x;
}

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
  real_t *AB = MALLOC(real_t, A_m * B_n);
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

  real_t *XtA = MALLOC(real_t, (X_m * A_m));
  real_t *Xt = MALLOC(real_t, (X_m * X_n));

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

  real_t *Xt = MALLOC(real_t, (X_m * X_n));
  real_t *XA = MALLOC(real_t, (X_m * A_n));

  mat_transpose(X, X_m, X_n, Xt);
  dot(X, X_m, X_n, A, A_m, A_n, XA);
  dot(XA, X_m, A_n, Xt, X_n, X_m, Y);

  free(XA);
  free(Xt);
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
 * Check inverted matrix A by multiplying by its inverse.
 * @returns `0` for succces, `-1` for failure.
 */
int check_inv(const real_t *A, const real_t *A_inv, const int m) {
  real_t *inv_check = MALLOC(real_t, m * m);
  dot(A, m, m, A_inv, m, m, inv_check);

  for (int i = 0; i < m; i++) {
    if (fltcmp(inv_check[i * m + i], 1.0) != 0) {
      free(inv_check);
      return -1;
    }
  }

  free(inv_check);
  return 0;
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
/**
 * Decompose matrix A with SVD
 */
int __lapack_svd(
    real_t *A, const int m, const int n, real_t *s, real_t *U, real_t *Vt) {
  const int lda = n;
#if PRECISION == 1
  const int info =
      LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'A', m, n, A, lda, s, U, n, Vt, n);
  return (info == 0) ? 0 : -1;
#elif PRECISION == 2
  const int info =
      LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', m, n, A, lda, s, U, n, Vt, n);
  return (info == 0) ? 0 : -1;
#endif
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

  rv1 = MALLOC(double, n);
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
      for (k = i; k < m; k++)
        scale += fabs(A[k * n + i]);
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
          for (s = 0.0, k = i; k < m; k++)
            s += A[k * n + i] * A[k * n + j];
          f = s / h;
          for (k = i; k < m; k++)
            A[k * n + j] += f * A[k * n + i];
        }
        for (k = i; k < m; k++)
          A[k * n + i] *= scale;
      }
    }
    w[i] = scale * g;
    g = s = scale = 0.0;
    if (i < m && i != n - 1) {
      for (k = l; k < n; k++)
        scale += fabs(A[i * n + k]);
      if (scale) {
        for (k = l; k < n; k++) {
          A[i * n + k] /= scale;
          s += A[i * n + k] * A[i * n + k];
        }
        f = A[i * n + l];
        g = -SIGN2(sqrt(s), f);
        h = f * g - s;
        A[i * n + l] = f - g;
        for (k = l; k < n; k++)
          rv1[k] = A[i * n + k] / h;
        for (j = l; j < m; j++) {
          for (s = 0.0, k = l; k < n; k++)
            s += A[j * n + k] * A[i * n + k];
          for (k = l; k < n; k++)
            A[j * n + k] += s * rv1[k];
        }
        for (k = l; k < n; k++)
          A[i * n + k] *= scale;
      }
    }
    anorm = PMAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
  }

  for (i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g) {
        for (j = l; j < n; j++)
          V[j * n + i] = (A[i * n + j] / A[i * n + l]) / g;
        for (j = l; j < n; j++) {
          for (s = 0.0, k = l; k < n; k++)
            s += A[i * n + k] * V[k * n + j];
          for (k = l; k < n; k++)
            V[k * n + j] += s * V[k * n + i];
        }
      }
      for (j = l; j < n; j++)
        V[i * n + j] = V[j * n + i] = 0.0;
    }
    V[i * n + i] = 1.0;
    g = rv1[i];
    l = i;
  }

  for (i = PMIN(m, n) - 1; i >= 0; i--) {
    l = i + 1;
    g = w[i];
    for (j = l; j < n; j++)
      A[i * n + j] = 0.0;
    if (g) {
      g = 1.0 / g;
      for (j = l; j < n; j++) {
        for (s = 0.0, k = l; k < m; k++)
          s += A[k * n + i] * A[k * n + j];
        f = (s / A[i * n + i]) * g;
        for (k = i; k < m; k++)
          A[k * n + j] += f * A[k * n + i];
      }
      for (j = i; j < m; j++)
        A[j * n + i] *= g;
    } else
      for (j = i; j < m; j++)
        A[j * n + i] = 0.0;
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
        if ((fabs(w[nm]) + anorm) == anorm)
          break;
      }
      if (flag) {
        c = 0.0;
        s = 1.0;
        for (i = l; i <= k; i++) {
          f = s * rv1[i];
          rv1[i] = c * rv1[i];
          if ((fabs(f) + anorm) == anorm)
            break;
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
          for (j = 0; j < n; j++)
            V[j * n + k] = -V[j * n + k];
        }
        break;
      }
      if (its == 29)
        printf("no convergence in 30 svd iterations\n");
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
  real_t *A_copy = MALLOC(real_t, m * n);
  real_t *Vt = MALLOC(real_t, n * n);
  mat_copy(A, m, n, A_copy);
  const int retval = __lapack_svd(A_copy, m, n, s, U, Vt);
  mat_transpose(Vt, n, n, V);
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
  real_t *s = MALLOC(real_t, diag_size);
  real_t *U = MALLOC(real_t, m * n);
  real_t *V = MALLOC(real_t, n * n);
  svd(A, m, n, U, s, V);

  // Form Sinv diagonal matrix
  real_t *Si = MALLOC(real_t, m * n);
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
  // real_t *V_Si = MALLOC(real_t, m * m);
  // zeros(A_inv, m, n);
  // dot(V, m, n, Si, n, m, V_Si);
  // dot(V_Si, m, m, Ut, m, n, A_inv);

  // A_inv = U * Si * Ut
  real_t *Ut = MALLOC(real_t, m * n);
  real_t *Si_Ut = MALLOC(real_t, diag_size * n);
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
  real_t *U = MALLOC(real_t, m * n);
  real_t *s = MALLOC(real_t, k);
  real_t *V = MALLOC(real_t, k * k);
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
  real_t *U = MALLOC(real_t, m * n);
  real_t *s = MALLOC(real_t, k);
  real_t *V = MALLOC(real_t, k * k);
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
/**
 * Decompose matrix A to lower triangular matrix L
 */
void __lapack_chol(const real_t *A, const size_t m, real_t *L) {
  assert(A != NULL);
  assert(m > 0);
  assert(L != NULL);

  // Cholesky Decomposition
  int info = 0;
  int lda = m;
  int n = m;
  char uplo = 'L';
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
  char uplo = 'L';
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
  real_t *L = CALLOC(real_t, n * n);
  real_t *Lt = CALLOC(real_t, n * n);
  real_t *y = CALLOC(real_t, n);

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

void __lapack_qr(real_t *A, const int m, const int n, real_t *R) {
  int lda = m;
  real_t *tau = MALLOC(real_t, (m < n) ? m : n);
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, m, n, A, lda, tau);
  free(tau);
}

void qr(real_t *A, const int m, const int n, real_t *R) {
#ifdef USE_LAPACK
  __lapack_qr(A, m, n, R);
#endif // USE_LAPACK
}

/////////
// EIG //
/////////

int __lapack_eig(
    const real_t *A, const int m, const int n, real_t *V, real_t *w) {
  assert(A != NULL);
  assert(m > 0 && n > 0);
  assert(m == n);
  const int lda = n;

  //   mat_triu(A, n, V);
  // #if PRECISION == 1
  //   const int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, lda, w);
  // #elif PRECISION == 2
  //   const int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', n, A, lda, w);
  // #endif // Precision
  //   // Check for convergence
  //   if (info > 0) {
  //     LOG_ERROR("The algorithm failed to compute eigenvalues.\n");
  //     return -1;
  //   }

  // Copy matrix A to output matrix V
  mat_triu(A, n, V);

  // Query and allocate the optimal workspace
  int lwork = -1;
  int info = 0;
  real_t wkopt;
#if PRECISION == 1
  ssyev_("Vectors", "Lower", &n, V, &lda, w, &wkopt, &lwork, &info);
#elif PRECISION == 2
  dsyev_("Vectors", "Lower", &n, V, &lda, w, &wkopt, &lwork, &info);
#endif // Precision
  lwork = (int) wkopt;
  real_t *work = MALLOC(double, lwork);

  // Solve eigenproblem
#if PRECISION == 1
  ssyev_("Vectors", "Lower", &n, V, &lda, w, work, &lwork, &info);
#elif PRECISION == 2
  dsyev_("Vectors", "Lower", &n, V, &lda, w, work, &lwork, &info);
#endif // Precision
  if (info > 0) {
    LOG_ERROR("The algorithm failed to compute eigenvalues.\n");
    free(work);
    return -1;
  }

  // Clean up
  real_t *Vt = MALLOC(real_t, n * n);
  mat_transpose(V, n, n, Vt);
  mat_copy(Vt, n, n, V);
  free(Vt);
  free(work);

  return 0;
}

/**
 * Perform Eigen-Decomposition of a symmetric matrix `A` of size `m` x `n`.
 */
int eig_sym(const real_t *A, const int m, const int n, real_t *V, real_t *w) {
  assert(A != NULL);
  assert(m > 0 && n > 0);
  assert(m == n);
  assert(V != NULL && w != NULL);
  return __lapack_eig(A, m, n, V, w);
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
  real_t *V = MALLOC(real_t, m * m);
  real_t *Vt = MALLOC(real_t, m * m);
  real_t *Lambda_inv = MALLOC(real_t, m * m);
  real_t *w = MALLOC(real_t, m);

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

  real_t *V = MALLOC(real_t, m * n);
  real_t *w = MALLOC(real_t, m);
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

/******************************************************************************
 * SUITE-SPARSE
 *****************************************************************************/

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

/******************************************************************************
 * Lie
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

/******************************************************************************
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
 * Pose difference between `pose0` and `pose`, returns difference in
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
 * Pose difference between `pose0` and `pose`, returns difference in
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
void pose_vector_update(real_t pose[7], const real_t dx[6]) {
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
 * Print pose vector
 */
void print_pose_vector(const char *prefix, const real_t pose[7]) {
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
  left[4]  = qx; left[5]  = qw;  left[6]  = -qz; left[7]  = qy;
  left[8]  = qy; left[9]  = qz;  left[10] = qw;  left[11] = -qx;
  left[12] = qz; left[13] = -qy; left[14] = qx;  left[15] = qw;
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
  left_xyz[0]  = qw; left_xyz[1] = -qz; left_xyz[2]  = qy;
  left_xyz[3]  = qz; left_xyz[4] = qw;  left_xyz[5] = -qx;
  left_xyz[6] = -qy; left_xyz[7] = qx;  left_xyz[8] = qw;
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
  right[0] = qw;
  right[1] = -qx;
  right[2] = -qy;
  right[3] = -qz;
  right[4] = qx;
  right[5] = qw;
  right[6] = qz;
  right[7] = -qy;
  right[8] = qy;
  right[9] = -qz;
  right[10] = qw;
  right[11] = qx;
  right[12] = qz;
  right[13] = qy;
  right[14] = -qx;
  right[15] = qw;
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
}

/**
 * Update quaternion with small update dalpha.
 */
void quat_update(real_t q[4], const real_t dalpha[3]) {
  const real_t dq[4] = {1.0, 0.5 * dalpha[0], 0.5 * dalpha[1], 0.5 * dalpha[2]};
  real_t q_new[4] = {0};
  quat_mul(q, dq, q_new);
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

/******************************************************************************
 * CV
 ******************************************************************************/

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

  image_t *img = MALLOC(image_t, 1);
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

  // Get last row of V_t and normalize the scale to obtain the 3D point
  const real_t x = V[3];
  const real_t y = V[7];
  const real_t z = V[11];
  const real_t w = V[15];
  p[0] = x / w;
  p[1] = y / w;
  p[2] = z / w;
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

/******************************************************************************
 * SENSOR FUSION
 ******************************************************************************/

///////////
// UTILS //
///////////

int schurs_complement(const real_t *H,
                      const int H_m,
                      const int H_n,
                      const real_t *b,
                      const int b_m,
                      const int m,
                      const int r,
                      real_t *H_marg,
                      real_t *b_marg) {
  assert(H != NULL);
  assert(H_m > 0 && H_n > 0);
  assert(H_m == H_n);
  assert(b);
  assert(b_m > 0);
  assert((m + r) == H_m);
  assert(H_marg != NULL && b_marg != NULL);

  // Extract sub-blocks of matrix H
  // H = [Hmm, Hmr,
  //      Hrm, Hrr]
  real_t *Hmm = MALLOC(real_t, m * m);
  real_t *Hmr = MALLOC(real_t, m * r);
  real_t *Hrm = MALLOC(real_t, m * r);
  real_t *Hrr = MALLOC(real_t, r * r);
  real_t *Hmm_inv = MALLOC(real_t, m * m);
  mat_block_get(H, H_n, 0, m, 0, m, Hmm);
  mat_block_get(H, H_n, 0, m, m, H_n, Hmr);
  mat_block_get(H, H_n, m, 0, 0, 0, Hrm);
  mat_block_get(H, H_n, m, m + r, m, m + r, Hrr);

  // Extract sub-blocks of vector b
  // b = [b_mm, b_rr]
  real_t *bmm = MALLOC(real_t, m);
  real_t *brr = MALLOC(real_t, r);
  vec_copy(b, m, bmm);
  vec_copy(b + m, r, brr);

  // Invert Hmm
  int status = 0;
  const int c = 1;
  if (eig_inv(Hmm, m, m, c, Hmm_inv) != 0) {
    status = -1;
  }

  // Shur-Complement
  // H_marg = H_rr - H_rm * H_mm_inv * H_mr
  // b_marg = b_rr - H_rm * H_mm_inv * b_mm
  if (status == 0) {
    dot3(Hrm, r, m, Hmm_inv, m, m, Hmr, m, r, H_marg);
    dot3(Hrm, r, m, Hmm_inv, m, m, bmm, m, r, b_marg);
    for (int i = 0; i < (m * m); i++) {
      H_marg[i] = Hrr[i] - H_marg[i];
    }
    for (int i = 0; i < m; i++) {
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

//////////
// POSE //
//////////

void pose_setup(pose_t *pose, const timestamp_t ts, const real_t *data) {
  assert(pose != NULL);
  assert(data != NULL);

  // Timestamp
  pose->ts = ts;

  // Translation
  pose->data[0] = data[0]; // rx
  pose->data[1] = data[1]; // ry
  pose->data[2] = data[2]; // rz

  // Rotation (Quaternion)
  pose->data[3] = data[3]; // qw
  pose->data[4] = data[4]; // qx
  pose->data[5] = data[5]; // qy
  pose->data[6] = data[6]; // qz
}

/**
 * Print pose
 */
void pose_print(const char *prefix, const pose_t *pose) {
  const timestamp_t ts = pose->ts;

  const real_t x = pose->data[0];
  const real_t y = pose->data[1];
  const real_t z = pose->data[2];

  const real_t qw = pose->data[3];
  const real_t qx = pose->data[4];
  const real_t qy = pose->data[5];
  const real_t qz = pose->data[6];

  printf("[%s] ", prefix);
  printf("ts: %19ld, ", ts);
  printf("pos: (%f, %f, %f), ", x, y, z);
  printf("quat: (%f, %f, %f, %f)\n", qw, qx, qy, qz);
}

//////////////
// VELOCITY //
//////////////

/**
 * Setup velocity
 */
void velocity_setup(velocity_t *vel, const timestamp_t ts, const real_t v[3]) {
  assert(vel != NULL);
  assert(v != NULL);

  // Timestamp
  vel->ts = ts;

  // Accel biases
  vel->data[0] = v[0];
  vel->data[1] = v[1];
  vel->data[2] = v[2];
}

void velocity_print(const velocity_t *vel);

////////////////
// IMU-BIASES //
////////////////

/**
 * Setup speed and biases
 */
void imu_biases_setup(imu_biases_t *sb,
                      const timestamp_t ts,
                      const real_t ba[3],
                      const real_t bg[3]) {
  assert(sb != NULL);
  assert(ba != NULL);
  assert(bg != NULL);

  // Timestamp
  sb->ts = ts;

  // Accel biases
  sb->data[0] = ba[0];
  sb->data[1] = ba[1];
  sb->data[2] = ba[2];

  // Gyro biases
  sb->data[3] = bg[0];
  sb->data[4] = bg[1];
  sb->data[5] = bg[2];
}

/**
 * Get IMU accelerometer biases
 */
void imu_biases_get_accel_bias(const imu_biases_t *biases, real_t ba[3]) {
  ba[0] = biases->data[0];
  ba[1] = biases->data[1];
  ba[2] = biases->data[2];
}

/**
 * Get IMU gyroscope biases
 */
void imu_biases_get_gyro_bias(const imu_biases_t *biases, real_t bg[3]) {
  bg[0] = biases->data[3];
  bg[1] = biases->data[4];
  bg[2] = biases->data[5];
}

/////////////
// FEATURE //
/////////////

/**
 * Setup feature
 */
void feature_setup(feature_t *feature, const real_t *data) {
  assert(feature != NULL);
  assert(data != NULL);

  feature->data[0] = data[0];
  feature->data[1] = data[1];
  feature->data[2] = data[2];
}

/**
 * Setup features
 */
void features_setup(features_t *features) {
  assert(features);
  features->nb_features = 0;
  for (int i = 0; i < MAX_FEATURES; i++) {
    features->status[i] = 0;
  }
}

/**
 * Check whether feature with `feature_id` exists
 * @returns 1 for yes, 0 for no
 */
int features_exists(const features_t *features, const int feature_id) {
  return features->status[feature_id];
}

/**
 * @returns pointer to feature with `feature_id`
 */
feature_t *features_get(features_t *features, const int feature_id) {
  return &features->data[feature_id];
}

/**
 * Add feature
 * @returns Pointer to feature
 */
feature_t *features_add(features_t *features,
                        const int feature_id,
                        const real_t *param) {
  feature_setup(&features->data[feature_id], param);
  features->status[feature_id] = 1;
  return &features->data[feature_id];
}

/**
 * Remove feature with `feature_id`
 */
void features_remove(features_t *features, const int feature_id) {
  const real_t param[3] = {0};
  feature_setup(&features->data[feature_id], param);
  features->status[feature_id] = 0;
}

////////////////
// extrinsic //
////////////////

/**
 * Setup extrinsic
 */
void extrinsic_setup(extrinsic_t *exts, const real_t *data) {
  assert(exts != NULL);
  assert(data != NULL);

  // Translation
  exts->data[0] = data[0]; // rx
  exts->data[1] = data[1]; // ry
  exts->data[2] = data[2]; // rz

  // Rotation (Quaternion)
  exts->data[3] = data[3]; // qw
  exts->data[4] = data[4]; // qx
  exts->data[5] = data[5]; // qy
  exts->data[6] = data[6]; // qz
}

/**
 * Print extrinsic
 */
void extrinsic_print(const char *prefix, const extrinsic_t *exts) {
  const real_t x = exts->data[0];
  const real_t y = exts->data[1];
  const real_t z = exts->data[2];

  const real_t qw = exts->data[3];
  const real_t qx = exts->data[4];
  const real_t qy = exts->data[5];
  const real_t qz = exts->data[6];

  printf("[%s] ", prefix);
  printf("pos: (%.2f, %.2f, %.2f), ", x, y, z);
  printf("quat: (%.2f, %.2f, %.2f, %.2f)\n", qw, qx, qy, qz);
}

//////////////////
// JOINT-ANGLES //
//////////////////

/**
 * Joint Angle Setup
 */
void joint_setup(joint_t *joint,
                 const timestamp_t ts,
                 const int joint_idx,
                 const real_t theta) {
  assert(joint != NULL);
  joint->ts = ts;
  joint->joint_idx = joint_idx;
  joint->data[0] = theta;
}

/**
 * Print Joint Angle
 */
void joint_print(const char *prefix, const joint_t *joint) {
  printf("[%s] ", prefix);
  printf("ts: %ld ", joint->ts);
  printf("data: %f\n", joint->data[0]);
}

///////////////////////
// CAMERA-PARAMETERS //
///////////////////////

/**
 * Setup camera parameters
 */
void camera_params_setup(camera_params_t *camera,
                         const int cam_idx,
                         const int cam_res[2],
                         const char *proj_model,
                         const char *dist_model,
                         const real_t data[8]) {
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
}

/**
 * Print camera parameters
 */
void camera_params_print(const camera_params_t *cam) {
  assert(cam != NULL);

  printf("cam%d:\n", cam->cam_idx);
  printf("  resolution: [%d, %d]\n", cam->resolution[0], cam->resolution[1]);
  printf("  proj_model: %s\n", cam->proj_model);
  printf("  dist_model: %s\n", cam->dist_model);
  printf("  data: [");
  for (int i = 0; i < 8; i++) {
    if ((i + 1) < 8) {
      printf("%.4f, ", cam->data[i]);
    } else {
      printf("%.4f", cam->data[i]);
    }
  }
  printf("]\n");
}

/////////////////
// POSE-FACTOR //
/////////////////

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
  real_t *r = CALLOC(real_t, r_size);
  real_t *J_numdiff = CALLOC(real_t, r_size * param_size);

  // Evaluate factor
  if (factor_eval(factor, params, r, NULL) != 0) {
    return -2;
  }

  // Numerical diff - forward finite difference
  for (int i = 0; i < param_size; i++) {
    real_t *r_fwd = CALLOC(real_t, r_size);
    real_t *r_diff = CALLOC(real_t, r_size);

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
  real_t *r = CALLOC(real_t, r_size);
  real_t *J_numdiff = CALLOC(real_t, r_size * param_size);

  // Evaluate factor
  if (factor_eval(factor, params, r, NULL) != 0) {
    return -2;
  }

  for (int i = 0; i < param_size; i++) {
    real_t *r_fwd = CALLOC(real_t, r_size);
    real_t *r_diff = CALLOC(real_t, r_size);

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

/**
 * Setup pose factor
 */
void pose_factor_setup(pose_factor_t *factor,
                       pose_t *pose,
                       const real_t var[6]) {
  assert(factor != NULL);
  assert(pose != NULL);
  assert(var != NULL);

  // Parameters
  factor->pose_est = pose;

  // Measurement
  factor->pos_meas[0] = pose->data[0];
  factor->pos_meas[1] = pose->data[1];
  factor->pos_meas[2] = pose->data[2];
  factor->quat_meas[0] = pose->data[3];
  factor->quat_meas[1] = pose->data[4];
  factor->quat_meas[2] = pose->data[5];
  factor->quat_meas[3] = pose->data[6];

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
  factor->params[0] = factor->pose_est->data;
  factor->jacs[0] = factor->J_pose;
}

/**
 * Evaluate pose factor
 *
 * Parameter `params`:
 * 1. Position r_est
 * 2. Quaternion q_est
 *
 * Residuals `r_out` represents the pose error [6x1]
 * Jacobians `J_out` with respect to `params`
 *
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
// BA-FACTOR //
///////////////

/**
 * Setup bundle adjustment factor
 */
void ba_factor_setup(ba_factor_t *factor,
                     pose_t *pose,
                     feature_t *feature,
                     camera_params_t *camera,
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

  factor->params[0] = factor->pose->data;
  factor->params[1] = factor->feature->data;
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
 *
 * Parameter `params`:
 * 1. Position r_WCi
 * 2. Quaternion q_WCi
 * 3. Feature p_W
 * 4. Camera i parameters
 *
 * Residuals `r_out` represents the reprojection error [2x1]
 * Jacobians `J_out` with respect to `params`
 *
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
  pinhole_radtan4_project(cam_params, p_Ci, z_hat);
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
// VISION-FACTOR //
///////////////////

/**
 * Setup vision factor
 */
void vision_factor_setup(vision_factor_t *factor,
                         pose_t *pose,
                         extrinsic_t *extrinsic,
                         feature_t *feature,
                         camera_params_t *camera,
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

  factor->params[0] = factor->pose->data;
  factor->params[1] = factor->extrinsic->data;
  factor->params[2] = factor->feature->data;
  factor->params[3] = factor->camera->data;

  factor->jacs[0] = factor->J_pose;
  factor->jacs[1] = factor->J_extrinsic;
  factor->jacs[2] = factor->J_feature;
  factor->jacs[3] = factor->J_camera;
}

/**
 * Pose jacobian
 */
static void vision_factor_pose_jacobian(const real_t Jh_weighted[2 * 3],
                                        const real_t T_WB[3 * 3],
                                        const real_t T_BC[3 * 3],
                                        const real_t p_W[3],
                                        real_t J[2 * 6]) {
  assert(Jh_weighted != NULL);
  assert(T_BC != NULL);
  assert(T_WB != NULL);
  assert(p_W != NULL);
  assert(J != NULL);

  // Jh_weighted = -1 * sqrt_info * Jh;
  // J_pos = Jh_weighted * C_CB * -C_BW;
  // J_rot = Jh_weighted * C_CB * C_BW * hat(p_W - r_WB) * -C_WB;
  // J = [J_pos, J_rot];

  // Setup
  real_t C_WB[3 * 3] = {0};
  real_t C_BC[3 * 3] = {0};
  real_t C_BW[3 * 3] = {0};
  real_t C_CB[3 * 3] = {0};
  real_t C_CW[3 * 3] = {0};

  tf_rot_get(T_WB, C_WB);
  tf_rot_get(T_BC, C_BC);
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
  real_t r_WB[3] = {0};
  real_t p[3] = {0};
  real_t S[3 * 3] = {0};
  tf_trans_get(T_WB, r_WB);
  vec_sub(p_W, r_WB, p, 3);
  hat(p, S);

  real_t A[3 * 3] = {0};
  real_t B[3 * 3] = {0};
  dot(neg_C_CW, 3, 3, S, 3, 3, A);
  dot(A, 3, 3, neg_C_WB, 3, 3, B);

  // Form: J_pos = Jh_weighted * C_CB * -C_BW;
  real_t J_pos[2 * 3] = {0};
  dot(Jh_weighted, 2, 3, neg_C_CW, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // Form: J_rot = Jh_weighted * C_CB * -C_BW * hat(p_W - r_WB) * -C_WB;
  real_t J_rot[2 * 3] = {0};
  dot(Jh_weighted, 2, 3, B, 3, 3, J_rot);

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
static void vision_factor_extrinsic_jacobian(const real_t Jh_weighted[2 * 3],
                                             const real_t T_BC[3 * 3],
                                             const real_t p_C[3],
                                             real_t J[2 * 6]) {
  assert(Jh_weighted != NULL);
  assert(T_BC != NULL);
  assert(p_C != NULL);
  assert(J != NULL);

  // Jh_weighted = -1 * sqrt_info * Jh;
  // J_pos = Jh_weighted * -C_CB;
  // J_rot = Jh_weighted * C_CB * hat(C_BC * p_C);

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

  // Form: J_rot = Jh_weighted * -C_CB;
  real_t J_pos[2 * 3] = {0};
  dot(Jh_weighted, 2, 3, neg_C_CB, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // Form: J_rot = Jh_weighted * -C_CB * hat(C_BC * p_C) * -C_BC;
  real_t J_rot[2 * 3] = {0};
  dot(Jh_weighted, 2, 3, B, 3, 3, J_rot);

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
static void vision_factor_camera_jacobian(const real_t neg_sqrt_info[2 * 2],
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
static void vision_factor_feature_jacobian(const real_t Jh_weighted[2 * 3],
                                           const real_t T_WB[4 * 4],
                                           const real_t T_BC[4 * 4],
                                           real_t J[2 * 3]) {
  if (J == NULL) {
    return;
  }
  assert(Jh_weighted != NULL);
  assert(T_WB != NULL);
  assert(T_BC != NULL);
  assert(J != NULL);

  // Jh_weighted = -1 * sqrt_info * Jh;
  // J = Jh_weighted * C_CW;

  // Setup
  real_t T_WC[4 * 4] = {0};
  real_t C_WC[3 * 3] = {0};
  real_t C_CW[3 * 3] = {0};
  dot(T_WB, 4, 4, T_BC, 4, 4, T_WC);
  tf_rot_get(T_WC, C_WC);
  mat_transpose(C_WC, 3, 3, C_CW);

  // Form: J = -1 * sqrt_info * Jh * C_CW;
  dot(Jh_weighted, 2, 3, C_CW, 3, 3, J);
}

/**
 * Evaluate vision factor
 *
 * Parameter `params`:
 * 1. Position r_WB
 * 2. Quaternion q_WB
 * 3. Body-Camera i extrinsic - Translation r_BCi
 * 4. Body-Camera i extrinsic - Quaternion q_BCi
 * 5. Camera i parameters
 * 6. Feature p_W
 *
 * Residuals `r_out` camera reprojection error
 * Jacobians `J_out` with respect to `params`
 *
 * @returns `0` for success, `-1` for failure
 */
int vision_factor_eval(void *factor_ptr) {
  vision_factor_t *factor = (vision_factor_t *) factor_ptr;
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
  real_t T_WCi[4 * 4] = {0};
  real_t T_CiW[4 * 4] = {0};
  dot(T_WB, 4, 4, T_BCi, 4, 4, T_WCi);
  tf_inv(T_WCi, T_CiW);

  // Transform feature from world to camera frame
  real_t p_Ci[3] = {0};
  tf_point(T_CiW, p_W, p_Ci);

  // Calculate residuals
  // -- Project point from world to image plane
  real_t z_hat[2];
  pinhole_radtan4_project(cam_params, p_Ci, z_hat);
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
  vision_factor_pose_jacobian(Jh_, T_WB, T_BCi, p_W, factor->jacs[0]);
  vision_factor_extrinsic_jacobian(Jh_, T_BCi, p_Ci, factor->jacs[1]);
  vision_factor_feature_jacobian(Jh_, T_WB, T_BCi, factor->jacs[2]);
  vision_factor_camera_jacobian(neg_sqrt_info, J_cam_params, factor->jacs[3]);

  return 0;
}

////////////////////////
// JOINT-ANGLE FACTOR //
////////////////////////

/**
 * Setup joint-angle factor
 */
void joint_factor_setup(joint_factor_t *factor,
                        joint_t *joint,
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
  factor->params[0] = factor->joint->data;
  factor->jacs[0] = factor->J_joint;
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
  factor->r[0] = factor->sqrt_info[0] * (factor->z[0] - factor->joint->data[0]);

  // Calculate Jacobians
  factor->jacs[0][0] = -1 * factor->sqrt_info[0];

  return 0;
}

/**
 * Check if two joint factors are equal in value.
 */
int joint_factor_equals(const joint_factor_t *j0, const joint_factor_t *j1) {
  CHECK(vec_equals(j0->z, j1->z, 2));
  CHECK(mat_equals(j0->covar, j1->covar, 2, 2, 1e-8));
  CHECK(mat_equals(j0->sqrt_info, j1->sqrt_info, 2, 2, 1e-8));

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

/////////////////////////
// CALIB-GIMBAL-FACTOR //
/////////////////////////

#define GIMBAL_JOINT_TF(THETA, T_JOINT)                                        \
  real_t T_JOINT[4 * 4] = {0};                                                 \
  {                                                                            \
    real_t C_joint[3 * 3] = {0};                                               \
    const real_t r_joint[3] = {0.0, 0.0, 0.0};                                 \
    rotz(THETA, C_joint);                                                      \
    tf_cr(C_joint, r_joint, T_JOINT);                                          \
  }

void gimbal_setup_extrinsic(const real_t ypr[3],
                            const real_t r[3],
                            real_t T[4 * 4],
                            extrinsic_t *link) {
  real_t C[3 * 3] = {0};
  euler321(ypr, C);
  tf_cr(C, r, T);

  real_t q[4] = {0};
  rot2quat(C, q);

  real_t x[7] = {0};
  x[0] = r[0];
  x[1] = r[1];
  x[2] = r[2];
  x[3] = q[0];
  x[4] = q[1];
  x[5] = q[2];
  x[6] = q[3];

  extrinsic_setup(link, x);
}

void gimbal_setup_joint(const timestamp_t ts,
                        const int joint_idx,
                        const real_t theta,
                        real_t T_joint[4 * 4],
                        joint_t *joint) {
  real_t C_joint[3 * 3] = {0};
  rotz(theta, C_joint);

  const real_t r_joint[3] = {0.0, 0.0, 0.0};
  tf_cr(C_joint, r_joint, T_joint);

  joint_setup(joint, ts, joint_idx, theta);
}

void calib_gimbal_factor_setup(calib_gimbal_factor_t *factor,
                               extrinsic_t *fiducial_exts,
                               extrinsic_t *gimbal_exts,
                               pose_t *pose,
                               extrinsic_t *link0,
                               extrinsic_t *link1,
                               joint_t *joint0,
                               joint_t *joint1,
                               joint_t *joint2,
                               extrinsic_t *cam_exts,
                               camera_params_t *cam,
                               const timestamp_t ts,
                               const int cam_idx,
                               const int tag_id,
                               const int corner_idx,
                               const real_t p_FFi[3],
                               const real_t z[2],
                               const real_t var[2]) {
  assert(factor != NULL);
  assert(fiducial_exts != NULL);
  assert(gimbal_exts != NULL);
  assert(pose != NULL);
  assert(link0 != NULL && link1 != NULL);
  assert(joint0 != NULL && joint1 != NULL && joint2 != NULL);
  assert(cam_exts != NULL);
  assert(cam != NULL);
  assert(z != NULL);
  assert(var != NULL);

  // Parameters
  factor->fiducial_exts = fiducial_exts;
  factor->gimbal_exts = gimbal_exts;
  factor->pose = pose;
  factor->link0 = link0;
  factor->link1 = link1;
  factor->joint0 = joint0;
  factor->joint1 = joint1;
  factor->joint2 = joint2;
  factor->cam_exts = cam_exts;
  factor->cam = cam;
  factor->num_params = 10;

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
  factor->num_params = 10;
  factor->param_types[0] = EXTRINSIC_PARAM;
  factor->param_types[1] = EXTRINSIC_PARAM;
  factor->param_types[2] = POSE_PARAM;
  factor->param_types[3] = EXTRINSIC_PARAM;
  factor->param_types[4] = EXTRINSIC_PARAM;
  factor->param_types[5] = JOINT_PARAM;
  factor->param_types[6] = JOINT_PARAM;
  factor->param_types[7] = JOINT_PARAM;
  factor->param_types[8] = EXTRINSIC_PARAM;
  factor->param_types[9] = CAMERA_PARAM;

  factor->params[0] = factor->fiducial_exts->data;
  factor->params[1] = factor->gimbal_exts->data;
  factor->params[2] = factor->pose->data;
  factor->params[3] = factor->link0->data;
  factor->params[4] = factor->link1->data;
  factor->params[5] = factor->joint0->data;
  factor->params[6] = factor->joint1->data;
  factor->params[7] = factor->joint2->data;
  factor->params[8] = factor->cam_exts->data;
  factor->params[9] = factor->cam->data;

  factor->jacs[0] = factor->J_fiducial_exts;
  factor->jacs[1] = factor->J_gimbal_exts;
  factor->jacs[2] = factor->J_pose;
  factor->jacs[3] = factor->J_link0;
  factor->jacs[4] = factor->J_link1;
  factor->jacs[5] = factor->J_joint0;
  factor->jacs[6] = factor->J_joint1;
  factor->jacs[7] = factor->J_joint2;
  factor->jacs[8] = factor->J_cam_exts;
  factor->jacs[9] = factor->J_cam_params;
}

static void gimbal_factor_joint_tf(const real_t theta, real_t T[4 * 4]) {
  real_t C[3 * 3] = {0};
  real_t r[3] = {0.0, 0.0, 0.0};
  rotz(theta, C);
  tf_cr(C, r, T);
}

static void gimbal_factor_fiducial_jac(const real_t Jh_w[2 * 3],
                                       const real_t T_CiW[4 * 4],
                                       const real_t T_WF[4 * 4],
                                       const real_t p_FFi[3],
                                       real_t J[2 * 6]) {
  if (J == NULL) {
    return;
  }

  // J_pos = Jh * C_CiW
  real_t J_pos[2 * 3] = {0};
  real_t C_CiW[3 * 3] = {0};
  tf_rot_get(T_CiW, C_CiW);
  dot(Jh_w, 2, 3, C_CiW, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // J_rot = Jh * C_CiW * -C_WF @ hat(p_FFi)
  real_t C_WF[3 * 3] = {0};
  real_t C_CiF[3 * 3] = {0};
  tf_rot_get(T_WF, C_WF);
  dot(C_CiW, 3, 3, C_WF, 3, 3, C_CiF);
  mat_scale(C_CiF, 3, 3, -1);

  real_t J_rot[2 * 3] = {0};
  real_t p_FFi_x[3 * 3] = {0};
  hat(p_FFi, p_FFi_x);
  dot3(Jh_w, 2, 3, C_CiF, 3, 3, p_FFi_x, 3, 3, J_rot);

  J[3] = J_rot[0];
  J[4] = J_rot[1];
  J[5] = J_rot[2];

  J[9] = J_rot[3];
  J[10] = J_rot[4];
  J[11] = J_rot[5];
}

static void gimbal_factor_pose_jac(const real_t Jh_w[2 * 3],
                                   const real_t T_CiW[4 * 4],
                                   const real_t T_WB[4 * 4],
                                   const real_t p_WFi[3],
                                   real_t J[2 * 6]) {
  if (J == NULL) {
    return;
  }

  // Form: -C_CiW
  real_t nC_CiW[3 * 3] = {0};
  tf_rot_get(T_CiW, nC_CiW);
  mat_scale(nC_CiW, 3, 3, -1.0);

  // J_pos = Jh * -C_CiW
  real_t J_pos[2 * 3] = {0};
  dot(Jh_w, 2, 3, nC_CiW, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // J_rot = Jh * -C_CiW * hat(p_WFi - r_WB) * -C_WB
  real_t r_WB[3] = {0};
  real_t dp[3] = {0};
  real_t dp_x[3 * 3] = {0};
  tf_trans_get(T_WB, r_WB);
  dp[0] = p_WFi[0] - r_WB[0];
  dp[1] = p_WFi[1] - r_WB[1];
  dp[2] = p_WFi[2] - r_WB[2];
  hat(dp, dp_x);

  real_t nC_WB[3 * 3] = {0};
  tf_rot_get(T_WB, nC_WB);
  mat_scale(nC_WB, 3, 3, -1.0);

  real_t J_rot[2 * 3] = {0};
  dot3(J_pos, 2, 3, dp_x, 3, 3, nC_WB, 3, 3, J_rot);

  J[3] = J_rot[0];
  J[4] = J_rot[1];
  J[5] = J_rot[2];

  J[9] = J_rot[3];
  J[10] = J_rot[4];
  J[11] = J_rot[5];
}

static void gimbal_factor_link_jac(const real_t Jh_w[2 * 3],
                                   const real_t T_LaLb[4 * 4],
                                   const real_t T_CiLa[4 * 4],
                                   const real_t p_La[3],
                                   real_t J[2 * 6]) {
  if (J == NULL) {
    return;
  }

  // Form: -C_LaLb
  real_t nC_LaLb[3 * 3] = {0};
  tf_rot_get(T_LaLb, nC_LaLb);
  mat_scale(nC_LaLb, 3, 3, -1.0);

  // J_pos = Jh * -C_CiLa
  real_t J_pos[2 * 3] = {0};
  real_t nC_CiLa[3 * 3] = {0};
  tf_rot_get(T_CiLa, nC_CiLa);
  mat_scale(nC_CiLa, 3, 3, -1.0);
  dot(Jh_w, 2, 3, nC_CiLa, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // J_rot = Jh * -C_CiLa.T * hat(p_BFi - r_BM0) * -C_BM0
  real_t J_rot[2 * 3] = {0};
  real_t r_LaLb[3] = {0};
  real_t dp[3] = {0};
  real_t dp_x[3 * 3] = {0};
  tf_trans_get(T_LaLb, r_LaLb);
  dp[0] = p_La[0] - r_LaLb[0];
  dp[1] = p_La[1] - r_LaLb[1];
  dp[2] = p_La[2] - r_LaLb[2];
  hat(dp, dp_x);
  dot3(J_pos, 2, 3, dp_x, 3, 3, nC_LaLb, 3, 3, J_rot);

  J[3] = J_rot[0];
  J[4] = J_rot[1];
  J[5] = J_rot[2];

  J[9] = J_rot[3];
  J[10] = J_rot[4];
  J[11] = J_rot[5];
}

static void gimbal_factor_joint_jac(const real_t Jh_w[2 * 3],
                                    const real_t T_CiMe[4 * 4],
                                    const real_t p_MbFi[3],
                                    const real_t theta,
                                    real_t J[2 * 1]) {
  if (J == NULL) {
    return;
  }

  assert(Jh_w != NULL);
  assert(T_CiMe != NULL);
  assert(p_MbFi != NULL);
  assert(J != NULL);

  // C_CiMe = tf_rot(T_CiMe)
  real_t C_CiMe[3 * 3] = {0};
  tf_rot_get(T_CiMe, C_CiMe);

  // p = [-p_M0bFi[0] * sin(joints[0]) + p_M0bFi[1] * cos(joints[0]),
  //      -p_M0bFi[0] * cos(joints[0]) - p_M0bFi[1] * sin(joints[0]),
  //      0.0]
  // J = Jh * C_CiMe @ p
  const real_t p[3] = {-p_MbFi[0] * sin(theta) + p_MbFi[1] * cos(theta),
                       -p_MbFi[0] * cos(theta) - p_MbFi[1] * sin(theta),
                       0.0};
  dot3(Jh_w, 2, 3, C_CiMe, 3, 3, p, 3, 1, J);
}

static void gimbal_factor_cam_ext_jac(const real_t Jh_w[2 * 3],
                                      const real_t T_L2Ci[4 * 4],
                                      const real_t p_L2Fi[3],
                                      real_t J[2 * 6]) {
  if (J == NULL) {
    return;
  }

  assert(Jh_w != NULL);
  assert(T_L2Ci != NULL);
  assert(p_L2Fi != NULL);
  assert(J != NULL);

  // Form: -C_CiL2
  real_t nC_L2Ci[3 * 3] = {0};
  real_t nC_CiL2[3 * 3] = {0};
  tf_rot_get(T_L2Ci, nC_L2Ci);
  mat_scale(nC_L2Ci, 3, 3, -1.0);
  mat_transpose(nC_L2Ci, 3, 3, nC_CiL2);

  // J_pos = Jh * -C_L2Ci.T
  real_t J_pos[2 * 6] = {0};
  dot(Jh_w, 2, 3, nC_CiL2, 3, 3, J_pos);

  J[0] = J_pos[0];
  J[1] = J_pos[1];
  J[2] = J_pos[2];

  J[6] = J_pos[3];
  J[7] = J_pos[4];
  J[8] = J_pos[5];

  // J_rot = Jh * -C_L2Ci.T * hat(p_L2Fi - r_L2Ci) * -C_L2Ci
  real_t J_rot[2 * 6] = {0};
  real_t r_L2Ci[3] = {0};
  real_t dr[3] = {0};
  real_t dr_x[3 * 3] = {0};
  tf_trans_get(T_L2Ci, r_L2Ci);
  dr[0] = p_L2Fi[0] - r_L2Ci[0];
  dr[1] = p_L2Fi[1] - r_L2Ci[1];
  dr[2] = p_L2Fi[2] - r_L2Ci[2];
  hat(dr, dr_x);
  dot3(J_pos, 2, 3, dr_x, 3, 3, nC_L2Ci, 3, 3, J_rot);

  J[3] = J_rot[0];
  J[4] = J_rot[1];
  J[5] = J_rot[2];

  J[9] = J_rot[3];
  J[10] = J_rot[4];
  J[11] = J_rot[5];
}

static void gimbal_factor_camera_jac(const real_t neg_sqrt_info[2 * 2],
                                     const real_t J_cam_params[2 * 8],
                                     real_t J[2 * 8]) {
  if (J == NULL) {
    return;
  }

  assert(neg_sqrt_info != NULL);
  assert(J_cam_params != NULL);
  assert(J != NULL);
  dot(neg_sqrt_info, 2, 2, J_cam_params, 2, 8, J);
}

/**
 * Evaluate gimbal calibration factor
 *
 * Parameter `params`:
 * 1. Fiducial pose T_WF
 * 2. Gimbal pose T_WB
 * 3. Link0 pose T_BM0
 * 4. Link1 pose T_L0M1
 * 5. Link2 pose T_L1M2
 * 6. Joint0 angle
 * 7. Joint1 angle
 * 8. Joint2 angle
 * 9. Camera extrinsic pose T_L2Ci
 * 10. Camera parameters K_Ci
 *
 * Residuals `r_out` camera reprojection error
 * Jacobians `J_out` with respect to `params`
 *
 * @returns `0` for success, `-1` for failure
 */
int calib_gimbal_factor_eval(void *factor_ptr) {
  assert(factor_ptr != NULL);

  // Map factor
  calib_gimbal_factor_t *factor = (calib_gimbal_factor_t *) factor_ptr;

  // Map params
  const real_t *p_FFi = factor->p_FFi;
  TF(factor->params[0], T_WF);  // -- Fiducial Extriniscs
  TF(factor->params[1], T_BM0); // -- Gimbal extrinsic
  TF(factor->params[2], T_WB);  // -- Pose
  // -- Links
  TF(factor->params[3], T_L0M1);
  TF(factor->params[4], T_L1M2);
  // -- Joint angles
  real_t T_M0L0[4 * 4] = {0};
  real_t T_M1L1[4 * 4] = {0};
  real_t T_M2L2[4 * 4] = {0};
  const real_t th0 = factor->params[5][0];
  const real_t th1 = factor->params[6][0];
  const real_t th2 = factor->params[7][0];
  gimbal_factor_joint_tf(th0, T_M0L0);
  gimbal_factor_joint_tf(th1, T_M1L1);
  gimbal_factor_joint_tf(th2, T_M2L2);
  // -- Camera extrinsic
  TF(factor->params[8], T_L2Ci);
  // -- Camera parameters
  const real_t *cam_params = factor->params[9];

  // Form T_CiF
  TF_CHAIN(T_BCi,
           7,
           T_BM0,   // Gimbal extrinsic
           T_M0L0,  // Joint0
           T_L0M1,  // Link1
           T_M1L1,  // Joint1
           T_L1M2,  // Link2
           T_M2L2,  // Joint2
           T_L2Ci); // Camera extrinsic
  TF_INV(T_WB, T_BW);
  TF_INV(T_BCi, T_CiB);
  TF_CHAIN(T_BF, 2, T_BW, T_WF);
  TF_CHAIN(T_CiF, 2, T_CiB, T_BF);
  TF_CHAIN(T_CiW, 2, T_CiB, T_BW);

  // Project to image plane
  real_t z_hat[2];
  TF_POINT(T_CiF, factor->p_FFi, p_CiFi);
  pinhole_radtan4_project(cam_params, p_CiFi, z_hat);

  // Calculate residuals
  real_t r[2] = {0, 0};
  r[0] = factor->z[0] - z_hat[0];
  r[1] = factor->z[1] - z_hat[1];
  dot(factor->sqrt_info, 2, 2, r, 2, 1, factor->r);

  // Calculate Jacobians
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

  // -- Fill Jacobians
  TF_CHAIN(T_CiM0b, 2, T_CiB, T_BM0);
  TF_CHAIN(T_CiM0e, 2, T_CiM0b, T_M0L0);
  TF_CHAIN(T_CiM1b, 2, T_CiM0e, T_L0M1);
  TF_CHAIN(T_CiM1e, 2, T_CiM1b, T_M1L1);
  TF_CHAIN(T_CiM2b, 2, T_CiM1e, T_L1M2);
  TF_CHAIN(T_CiL2, 2, T_CiM2b, T_M2L2);

  TF_INV(T_CiM0b, T_M0bCi);
  TF_INV(T_CiM1b, T_M1bCi);
  TF_INV(T_CiM2b, T_M2bCi);
  TF_INV(T_CiM0e, T_M0eCi);
  TF_INV(T_CiM1e, T_M1eCi);

  TF_POINT(T_WF, p_FFi, p_WFi);
  TF_POINT(T_BF, p_FFi, p_BFi);
  TF_POINT(T_M0bCi, p_CiFi, p_M0bFi);
  TF_POINT(T_M1bCi, p_CiFi, p_M1bFi);
  TF_POINT(T_M2bCi, p_CiFi, p_M2bFi);
  TF_POINT(T_M0eCi, p_CiFi, p_M0eFi);
  TF_POINT(T_M1eCi, p_CiFi, p_M1eFi);
  TF_POINT(T_L2Ci, p_CiFi, p_L2Fi);

  gimbal_factor_fiducial_jac(Jh_w, T_CiW, T_WF, p_FFi, factor->jacs[0]);
  gimbal_factor_link_jac(Jh_w, T_BM0, T_CiB, p_BFi, factor->jacs[1]);
  gimbal_factor_pose_jac(Jh_w, T_CiW, T_WB, p_WFi, factor->jacs[2]);
  gimbal_factor_link_jac(Jh_w, T_L0M1, T_CiM0e, p_M0eFi, factor->jacs[3]);
  gimbal_factor_link_jac(Jh_w, T_L1M2, T_CiM1e, p_M1eFi, factor->jacs[4]);
  gimbal_factor_joint_jac(Jh_w, T_CiM0e, p_M0bFi, th0, factor->jacs[5]);
  gimbal_factor_joint_jac(Jh_w, T_CiM1e, p_M1bFi, th1, factor->jacs[6]);
  gimbal_factor_joint_jac(Jh_w, T_CiL2, p_M2bFi, th2, factor->jacs[7]);
  gimbal_factor_cam_ext_jac(Jh_w, T_L2Ci, p_L2Fi, factor->jacs[8]);
  gimbal_factor_camera_jac(neg_sqrt_info, J_cam_params, factor->jacs[9]);

  return 0;
}

int calib_gimbal_factor_ceres_eval(void *factor_ptr,
                                   real_t **params,
                                   real_t *r_out,
                                   real_t **J_out) {
  CERES_FACTOR_EVAL(calib_gimbal_factor,
                    ((calib_gimbal_factor_t *) factor_ptr),
                    calib_gimbal_factor_eval,
                    params,
                    r_out,
                    J_out);
}

/**
 * Check whether two `calib_gimbal_factor_t` are equal.
 */
int calib_gimbal_factor_equals(const calib_gimbal_factor_t *c0,
                               const calib_gimbal_factor_t *c1) {
  CHECK(c0->ts == c1->ts);
  CHECK(c0->cam_idx == c1->cam_idx);
  CHECK(c0->tag_id == c1->tag_id);
  CHECK(c0->corner_idx == c1->corner_idx);
  CHECK(vec_equals(c0->p_FFi, c1->p_FFi, 3));
  CHECK(vec_equals(c0->z, c1->z, 2));

  CHECK(c0->r_size == c1->r_size);
  CHECK(c0->num_params == c1->num_params);
  for (int i = 0; i < c0->num_params; i++) {
    const int m = c0->r_size;
    const int n = param_local_size(c0->param_types[i]);
    CHECK(c0->param_types[i] == c1->param_types[i]);
    CHECK(vec_equals(c0->params[i], c1->params[i], n));
    CHECK(mat_equals(c0->jacs[i], c1->jacs[i], m, n, 1e-8));
  }

  return 1;
error:
  return 0;
}

////////////////
// IMU FACTOR //
////////////////

/**
 * Setup IMU buffer
 */
void imu_buf_setup(imu_buf_t *imu_buf) {
  for (int k = 0; k < IMU_BUF_MAX_SIZE; k++) {
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
void imu_buf_print(const imu_buf_t *imu_buf) {
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
void imu_buf_add(imu_buf_t *imu_buf,
                 const timestamp_t ts,
                 const real_t acc[3],
                 const real_t gyr[3]) {
  assert(imu_buf->size < IMU_BUF_MAX_SIZE);
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
 * Clear IMU buffer
 */
void imu_buf_clear(imu_buf_t *imu_buf) {
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
void imu_buf_copy(const imu_buf_t *src, imu_buf_t *dst) {
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

/**
 * Propagate IMU measurement
 */
void imu_factor_propagate_step(real_t r[3],
                               real_t v[3],
                               real_t q[4],
                               real_t ba[3],
                               real_t bg[3],
                               const real_t a[3],
                               const real_t w[3],
                               const real_t dt) {
  // Compensate accelerometer and gyroscope measurements
  const real_t a_t[3] = {a[0] - ba[0], a[1] - ba[1], a[2] - ba[2]};
  const real_t w_t[3] = {w[0] - bg[0], w[1] - bg[1], w[2] - bg[2]};

  // Update position:
  // dr = dr + (dv * dt) + (0.5 * dC * a_t * dt_sq);
  real_t acc_dint[3] = {0};
  real_t C[3 * 3] = {0};
  quat2rot(q, C);
  dot(C, 3, 3, a_t, 3, 1, acc_dint);
  r[0] += (v[0] * dt) + (0.5 * acc_dint[0] * dt * dt);
  r[1] += (v[1] * dt) + (0.5 * acc_dint[1] * dt * dt);
  r[2] += (v[2] * dt) + (0.5 * acc_dint[2] * dt * dt);

  // Update velocity
  // dv = dv + dC * a_t * dt;
  real_t dv[3] = {0};
  real_t acc_int[3] = {a_t[0] * dt, a_t[1] * dt, a_t[2] * dt};
  dot(C, 3, 3, acc_int, 3, 1, dv);
  v[0] += dv[0];
  v[1] += dv[1];
  v[2] += dv[2];

  // Update rotation
  // quat_update_dt(q, w_t, dt);
  // quat_normalize(q);

  const real_t phi[3] = {w_t[0] * dt, w_t[1] * dt, w_t[2] * dt};
  real_t phi_SO3[3 * 3] = {0};
  real_t dC[3 * 3] = {0};
  lie_Exp(phi, phi_SO3);
  dot(C, 3, 3, phi_SO3, 3, 3, dC);

  rot2quat(dC, q);
  quat_normalize(q);

  // Update accelerometer biases
  // ba = ba;

  // Update gyroscope biases
  // bg = bg;
}

/**
 * Form IMU Noise Matrix Q
 */
static void imu_factor_form_Q_matrix(const imu_params_t *imu_params,
                                     real_t Q[12 * 12]) {
  assert(imu_params != NULL);
  assert(Q != NULL);

  const real_t sigma_a_sq = imu_params->sigma_a * imu_params->sigma_a;
  const real_t sigma_g_sq = imu_params->sigma_g * imu_params->sigma_g;
  const real_t sigma_ba_sq = imu_params->sigma_aw * imu_params->sigma_aw;
  const real_t sigma_bg_sq = imu_params->sigma_gw * imu_params->sigma_gw;

  zeros(Q, 12, 12);
  Q[0] = sigma_a_sq;
  Q[13] = sigma_a_sq;
  Q[26] = sigma_a_sq;
  Q[39] = sigma_g_sq;
  Q[52] = sigma_g_sq;
  Q[65] = sigma_g_sq;
  Q[78] = sigma_ba_sq;
  Q[91] = sigma_ba_sq;
  Q[104] = sigma_ba_sq;
  Q[117] = sigma_bg_sq;
  Q[130] = sigma_bg_sq;
  Q[143] = sigma_bg_sq;
}

/**
 * Form IMU Transition Matrix F
 */
static void imu_factor_form_F_matrix(const real_t dq[4],
                                     const real_t ba[3],
                                     const real_t bg[3],
                                     const real_t a[3],
                                     const real_t w[3],
                                     const real_t dt,
                                     real_t I_F_dt[15 * 15]) {
  // Convert quaternion to rotation matrix
  real_t dC[3 * 3] = {0};
  quat2rot(dq, dC);

  // Compensate accelerometer and gyroscope measurements
  const real_t a_t[3] = {a[0] - ba[0], a[1] - ba[1], a[2] - ba[2]};
  const real_t w_t[3] = {w[0] - bg[0], w[1] - bg[1], w[2] - bg[2]};

  // Form continuous time transition matrix F
  // -- Initialize memory for F
  zeros(I_F_dt, 15, 15);
  // -- F[0:3, 3:6] = eye(3);
  I_F_dt[3] = 1.0;
  I_F_dt[19] = 1.0;
  I_F_dt[35] = 1.0;
  // -- F[4:6, 7:9] = -dC * hat(a_t);
  real_t F1[3 * 3] = {0};
  real_t ndC[3 * 3] = {0};
  real_t a_t_x[3 * 3] = {0};
  mat_copy(dC, 3, 3, ndC);
  mat_scale(ndC, 3, 3, -1);
  hat(a_t, a_t_x);
  dot(ndC, 3, 3, a_t_x, 3, 3, F1);
  mat_block_set(I_F_dt, 15, 3, 5, 6, 8, F1);
  // -- F[4:6, 10:12] = -dC;
  mat_block_set(I_F_dt, 15, 3, 5, 9, 11, ndC);
  // -- F[7:9, 7:9] = -hat(w_t);
  real_t F3[3 * 3] = {0};
  F3[1] = w_t[2];
  F3[2] = -w_t[1];
  F3[3] = -w_t[2];
  F3[5] = w_t[0];
  F3[6] = w_t[1];
  F3[7] = -w_t[0];
  mat_block_set(I_F_dt, 15, 6, 8, 6, 8, F3);
  // -- F[7:9, 13:15] = -eye(3);
  real_t F4[3 * 3] = {-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0};
  mat_block_set(I_F_dt, 15, 6, 8, 12, 14, F4);

  // Discretize continuous time transition matrix
  // I_F_dt = eye(15) + F * dt;
  for (int idx = 0; idx < (15 * 15); idx++) {
    if (idx == 0 || idx % 16 == 0) {
      I_F_dt[idx] = 1.0 + I_F_dt[idx] * dt;
    } else {
      I_F_dt[idx] = I_F_dt[idx] * dt;
    }
  }
}

/**
 * Form IMU Input Matrix G
 */
static void imu_factor_form_G_matrix(const real_t dq[4],
                                     const real_t dt,
                                     real_t G_dt[15 * 12]) {
  // Setup
  real_t dC[3 * 3] = {0};
  quat2rot(dq, dC);
  zeros(G_dt, 15, 12);

  real_t neg_eye[3 * 3] = {0};
  real_t pos_eye[3 * 3] = {0};
  for (int idx = 0; idx < 9; idx += 4) {
    neg_eye[idx] = -1.0;
    pos_eye[idx] = 1.0;
  }

  // Form continuous input matrix G
  // -- G[4:6, 1:3] = -dC;
  real_t G0[3 * 3] = {0};
  for (int idx = 0; idx < 9; idx++) {
    G0[idx] = -1.0 * dC[idx];
  }
  mat_block_set(G_dt, 12, 3, 5, 0, 2, G0);
  // -- G[7:9, 4:6] = -eye(3);
  mat_block_set(G_dt, 12, 6, 8, 3, 5, neg_eye);
  // -- G[10:12, 7:9] = eye(3);
  mat_block_set(G_dt, 12, 9, 11, 6, 8, pos_eye);
  // -- G[13:15, 10:12] = eye(3);
  mat_block_set(G_dt, 12, 12, 14, 9, 11, pos_eye);

  // Discretize G
  // G_dt = G * dt;
  for (int idx = 0; idx < (15 * 12); idx++) {
    G_dt[idx] = G_dt[idx] * dt;
  }
}

/**
 * IMU Factor setup
 */
void imu_factor_setup(imu_factor_t *factor,
                      imu_params_t *imu_params,
                      imu_buf_t *imu_buf,
                      pose_t *pose_i,
                      velocity_t *vel_i,
                      imu_biases_t *biases_i,
                      pose_t *pose_j,
                      velocity_t *vel_j,
                      imu_biases_t *biases_j) {
  // IMU buffer and parameters
  factor->imu_params = imu_params;
  imu_buf_copy(imu_buf, &factor->imu_buf);

  // Parameters
  factor->num_params = 6;

  factor->pose_i = pose_i;
  factor->vel_i = vel_i;
  factor->biases_i = biases_i;

  factor->pose_j = pose_j;
  factor->vel_j = vel_j;
  factor->biases_j = biases_j;

  factor->params[0] = factor->pose_i->data;
  factor->params[1] = factor->vel_i->data;
  factor->params[2] = factor->biases_i->data;
  factor->params[3] = factor->pose_j->data;
  factor->params[4] = factor->vel_j->data;
  factor->params[5] = factor->biases_j->data;

  factor->param_types[0] = POSE_PARAM;
  factor->param_types[1] = VELOCITY_PARAM;
  factor->param_types[2] = IMU_BIASES_PARAM;
  factor->param_types[3] = POSE_PARAM;
  factor->param_types[4] = VELOCITY_PARAM;
  factor->param_types[5] = IMU_BIASES_PARAM;

  // Pre-integration variables
  factor->Dt = 0.0;
  eye(factor->F, 15, 15);                          // State jacobian
  zeros(factor->P, 15, 15);                        // State covariance
  imu_factor_form_Q_matrix(imu_params, factor->Q); // Noise matrix
  zeros(factor->dr, 3, 1);                         // Relative position
  zeros(factor->dv, 3, 1);                         // Relative velocity
  quat_setup(factor->dq);                          // Relative rotation
  imu_biases_get_accel_bias(biases_i, factor->ba); // Accelerometer bias
  imu_biases_get_gyro_bias(biases_i, factor->bg);  // Gyroscope bias

  // Pre-integrate imu measuremenets
  // -------------------------------
  // This step is essentially like a Kalman Filter whereby you propagate the
  // system inputs (in this case the system is an IMU model with acceleration
  // and angular velocity as inputs. In this step we are interested in the:
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
  real_t dt = 0.0;
  for (int k = 0; k < (imu_buf->size - 1); k++) {
    if (k + 1 < imu_buf->size) {
      const timestamp_t ts_i = imu_buf->ts[k];
      const timestamp_t ts_j = imu_buf->ts[k + 1];
      dt = ts2sec(ts_j - ts_i);
    }
    const real_t *a = imu_buf->acc[k];
    const real_t *w = imu_buf->gyr[k];

    // Propagate
    imu_factor_propagate_step(factor->dr,
                              factor->dv,
                              factor->dq,
                              factor->ba,
                              factor->bg,
                              a,
                              w,
                              dt);

    // Form transition Matrix F
    real_t I_F_dt[15 * 15] = {0};
    imu_factor_form_F_matrix(factor->dq,
                             factor->ba,
                             factor->bg,
                             a,
                             w,
                             dt,
                             I_F_dt);

    // Input Jacobian G
    real_t G_dt[15 * 12] = {0};
    imu_factor_form_G_matrix(factor->dq, dt, G_dt);

    // Update state matrix F
    // F = I_F_dt * F;
    real_t F_prev[15 * 15] = {0};
    mat_copy(factor->F, 15, 15, F_prev);
    dot(I_F_dt, 15, 15, F_prev, 15, 15, factor->F);

    // Update covariance matrix P
    // P = I_F_dt * P * I_F_dt' + G_dt * Q * G_dt';
    real_t A[15 * 15] = {0};
    real_t B[15 * 15] = {0};
    dot_XAXt(I_F_dt, 15, 15, factor->P, 15, 15, A);
    dot_XAXt(G_dt, 15, 12, factor->Q, 12, 12, B);
    mat_add(A, B, factor->P, 15, 15);

    // Update overall dt
    factor->Dt += dt;
  }

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

  // Residuals
  factor->r_size = 15;
  zeros(factor->r, 15, 1);

  // Jacobians
  factor->jacs[0] = factor->J_pose_i;
  factor->jacs[1] = factor->J_vel_i;
  factor->jacs[2] = factor->J_biases_i;
  factor->jacs[3] = factor->J_pose_j;
  factor->jacs[4] = factor->J_vel_j;
  factor->jacs[5] = factor->J_biases_j;
}

/**
 * Reset IMU Factor
 */
void imu_factor_reset(imu_factor_t *factor) {
  zeros(factor->r, 15, 1); // Residuals
  zeros(factor->dr, 3, 1); // Relative position
  zeros(factor->dv, 3, 1); // Relative velocity
  quat_setup(factor->dq);  // Relative rotation
  zeros(factor->ba, 3, 1); // Accel bias
  zeros(factor->bg, 3, 1); // Gyro bias
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

  pose_get_trans(factor->pose_i->data, r_i);
  pose_get_quat(factor->pose_i->data, q_i);
  vec_copy(factor->vel_i->data, 3, v_i);
  imu_biases_get_accel_bias(factor->biases_i, ba_i);
  imu_biases_get_gyro_bias(factor->biases_i, bg_i);

  pose_get_trans(factor->pose_j->data, r_j);
  pose_get_quat(factor->pose_j->data, q_j);
  vec_copy(factor->vel_j->data, 3, v_j);

  // Correct the relative position, velocity and rotation
  // -- Extract Jacobians from error-state jacobian
  real_t dr_dba[3 * 3] = {0};
  real_t dr_dbg[3 * 3] = {0};
  real_t dv_dba[3 * 3] = {0};
  real_t dv_dbg[3 * 3] = {0};
  real_t dq_dbg[3 * 3] = {0};
  mat_block_get(factor->F, 15, 0, 2, 9, 11, dr_dba);
  mat_block_get(factor->F, 15, 0, 2, 12, 14, dr_dbg);
  mat_block_get(factor->F, 15, 3, 5, 9, 11, dv_dba);
  mat_block_get(factor->F, 15, 3, 5, 12, 14, dv_dbg);
  mat_block_get(factor->F, 15, 6, 8, 12, 14, dq_dbg);

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
    real_t ba_correction[3] = {0};
    real_t bg_correction[3] = {0};
    dot(dr_dba, 3, 3, dba, 3, 1, ba_correction);
    dot(dr_dbg, 3, 3, dbg, 3, 1, bg_correction);
    dr[0] = factor->dr[0] + ba_correction[0] + bg_correction[0];
    dr[1] = factor->dr[1] + ba_correction[1] + bg_correction[1];
    dr[2] = factor->dr[2] + ba_correction[2] + bg_correction[2];
  }
  // -- Correct relative velocity
  // dv = dv + dv_dba * dba + dv_dbg * dbg
  real_t dv[3] = {0};
  {
    real_t ba_correction[3] = {0};
    real_t bg_correction[3] = {0};
    dot(dv_dba, 3, 3, dba, 3, 1, ba_correction);
    dot(dv_dbg, 3, 3, dbg, 3, 1, bg_correction);
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
  real_t C_i[3 * 3] = {0};
  real_t C_it[3 * 3] = {0};
  quat2rot(q_i, C_i);
  mat_transpose(C_i, 3, 3, C_it);

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

  // err_rot = (2.0 * quat_mul(quat_inv(dq), quat_mul(quat_inv(q_i),
  // q_j)))[1:4]
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

  // err_ba = [0.0, 0.0, 0.0]
  real_t err_ba[3] = {0.0, 0.0, 0.0};

  // err_bg = [0.0, 0.0, 0.0]
  real_t err_bg[3] = {0.0, 0.0, 0.0};

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
  // -- Setup
  real_t dC[3 * 3] = {0};
  real_t C_j[3 * 3] = {0};
  real_t C_jt[3 * 3] = {0};
  real_t C_ji[3 * 3] = {0};
  real_t C_ji_dC[3 * 3] = {0};
  real_t qji_dC[4] = {0};
  real_t left_xyz[3 * 3] = {0};
  quat2rot(factor->dq, dC);
  quat2rot(q_j, C_j);
  mat_transpose(C_j, 3, 3, C_jt);
  dot(C_jt, 3, 3, C_i, 3, 3, C_ji);
  dot(C_ji, 3, 3, dC, 3, 3, C_ji_dC);
  rot2quat(C_ji_dC, qji_dC);
  quat_left_xyz(qji_dC, left_xyz);

  // -- Jacobian w.r.t pose_i
  real_t J_pose_i[15 * 6] = {0};
  zeros(J_pose_i, 15, 6);
  // ---- Jacobian w.r.t. r_i
  real_t drij_dri[3 * 3] = {0};
  for (int idx = 0; idx < 9; idx++) {
    drij_dri[idx] = -1.0 * C_it[idx];
  }
  mat_block_set(J_pose_i, 6, 0, 2, 0, 2, drij_dri);
  // -- Jacobian w.r.t. q_i
  real_t drij_dCi[3 * 3] = {0};
  real_t dvij_dCi[3 * 3] = {0};
  hat(dr_est, drij_dCi);
  hat(dv_est, dvij_dCi);

  // -(quat_left(rot2quat(C_j.T @ C_i)) @ quat_right(dq))[1:4, 1:4]
  real_t dtheta_dCi[3 * 3] = {0};
  {
    real_t q_ji[4] = {0};
    real_t Left[4 * 4] = {0};
    real_t Right[4 * 4] = {0};
    rot2quat(C_ji, q_ji);
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

  // -- Jacobian w.r.t. v_i
  real_t drij_dvi[3 * 3] = {0};
  real_t dvij_dvi[3 * 3] = {0};
  for (int idx = 0; idx < 9; idx++) {
    drij_dvi[idx] = -1.0 * C_it[idx] * factor->Dt;
    dvij_dvi[idx] = -1.0 * C_it[idx];
  }
  real_t J_vel_i[15 * 3] = {0};
  mat_block_set(J_vel_i, 3, 0, 2, 0, 2, drij_dvi);
  mat_block_set(J_vel_i, 3, 3, 5, 0, 2, dvij_dvi);
  dot(factor->sqrt_info, 15, 15, J_vel_i, 15, 3, factor->jacs[1]);

  // Jacobian w.r.t IMU biases
  real_t J_biases_i[15 * 6] = {0};
  // ---- Jacobian w.r.t ba_i
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
  // -- Multiply with sqrt info
  dot(factor->sqrt_info, 15, 15, J_biases_i, 15, 6, factor->jacs[2]);

  // -- Jacobian w.r.t. pose_j
  real_t J_pose_j[15 * 6] = {0};
  // ---- Jacobian w.r.t. r_j
  real_t drij_drj[3 * 3] = {0};
  mat_copy(C_it, 3, 3, drij_drj);
  mat_block_set(J_pose_j, 6, 0, 2, 0, 2, drij_drj);
  // ---- Jacobian w.r.t. q_j
  real_t dtheta_dqj[3 * 3] = {0};
  quat_left_xyz(qji_dC, dtheta_dqj);
  mat_block_set(J_pose_j, 6, 6, 8, 3, 5, dtheta_dqj);
  dot(factor->sqrt_info, 15, 15, J_pose_j, 15, 6, factor->jacs[3]);

  // -- Jacobian w.r.t. v_j
  real_t dv_dvj[3 * 3] = {0};
  mat_copy(C_it, 3, 3, dv_dvj);
  real_t J_vel_j[15 * 3] = {0};
  mat_block_set(J_vel_j, 3, 3, 5, 0, 2, dv_dvj);
  dot(factor->sqrt_info, 15, 15, J_vel_j, 15, 3, factor->jacs[4]);

  // -- Jacobian w.r.t. imu biases j
  zeros(factor->jacs[5], 15, 6);

  return 0;
}

////////////
// SOLVER //
////////////

/**
 * Return parameter type as a string
 */
void param_type_string(const int param_type, char *s) {
  switch (param_type) {
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
    default:
      FATAL("Invalid param type [%d]!\n", param_type);
      break;
  }

  return param_size;
}

/**
 * Setup Solver
 */
void solver_setup(solver_t *solver) {
  assert(solver);
  solver->max_iter = 10;
  solver->lambda = 1e4;
}

/**
 * Calculate cost with residual vector `r` of length `r_size`.
 */
real_t solver_cost(const real_t *r, const int r_size) {
  real_t r_sq[1] = {0};
  dot(r, 1, r_size, r, r_size, 1, r_sq);
  return 0.5 * r_sq[0];
}

/**
 * Fill Jacobian matrix
 */
void solver_fill_jacobian(param_order_t *hash,
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
    if (hmgets(hash, params[i]).fix) {
      continue;
    }

    // Get i-th parameter and corresponding Jacobian
    int idx_i = hmgets(hash, params[i]).idx;
    int size_i = param_local_size(hmgets(hash, params[i]).type);
    const real_t *J_i = jacs[i];

    // Fill in the Jacobian
    const int rs = J_row_idx;
    const int re = rs + r_size - 1;
    const int cs = idx_i;
    const int ce = idx_i + size_i - 1;
    mat_block_set(J, sv_size, rs, re, cs, ce, J_i);

    // Fill in the R.H.S of H dx = g, where g = -J_i' * r
    real_t *Jt_i = MALLOC(real_t, r_size * size_i);
    real_t *g_i = MALLOC(real_t, size_i);
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
void solver_fill_hessian(param_order_t *hash,
                         int num_params,
                         real_t **params,
                         real_t **jacs,
                         real_t *r,
                         int r_size,
                         int sv_size,
                         real_t *H,
                         real_t *g) {
  for (int i = 0; i < num_params; i++) {
    // Check if i-th parameter is fixed
    if (hmgets(hash, params[i]).fix) {
      continue;
    }

    // Get i-th parameter and corresponding Jacobian
    int idx_i = hmgets(hash, params[i]).idx;
    int size_i = param_local_size(hmgets(hash, params[i]).type);
    const real_t *J_i = jacs[i];
    real_t *Jt_i = MALLOC(real_t, r_size * size_i);
    mat_transpose(J_i, r_size, size_i, Jt_i);

    for (int j = i; j < num_params; j++) {
      // Check if j-th parameter is fixed
      if (hmgets(hash, params[j]).fix) {
        continue;
      }

      // Get j-th parameter and corresponding Jacobian
      int idx_j = hmgets(hash, params[j]).idx;
      int size_j = param_local_size(hmgets(hash, params[j]).type);
      const real_t *J_j = jacs[j];
      real_t *H_ij = MALLOC(real_t, size_i * size_j);
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
        real_t *H_ji = MALLOC(real_t, size_j * size_i);
        mat_transpose(H_ij, size_i, size_j, H_ji);
        mat_block_add(H, sv_size, rs, re, cs, ce, H_ij);
        mat_block_add(H, sv_size, cs, ce, rs, re, H_ji);
        free(H_ji);
      }

      // Clean up
      free(H_ij);
    }

    // Fill in the R.H.S of H dx = g, where g = -J_i' * r
    real_t *g_i = MALLOC(real_t, size_i);
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
real_t **solver_params_copy(const param_order_t *hash) {
  real_t **x = MALLOC(real_t *, hmlen(hash));

  for (int idx = 0; idx < hmlen(hash); idx++) {
    const int global_size = param_global_size(hash[idx].type);
    x[idx] = MALLOC(real_t, global_size);

    for (int i = 0; i < global_size; i++) {
      x[idx][i] = ((real_t *) hash[idx].key)[i];
    }
  }

  return x;
}

/**
 * Restore parameter values
 */
void solver_params_restore(param_order_t *hash, real_t **x) {
  for (int idx = 0; idx < hmlen(hash); idx++) {
    for (int i = 0; i < param_global_size(hash[idx].type); i++) {
      ((real_t *) hash[idx].key)[i] = x[idx][i];
    }
  }
}

/**
 * Free params
 */
void solver_params_free(const param_order_t *hash, real_t **x) {
  for (int idx = 0; idx < hmlen(hash); idx++) {
    free(x[idx]);
  }
  free(x);
}

/**
 * Update parameter
 */
void solver_update(param_order_t *hash, real_t *dx, int sv_size) {
  for (int i = 0; i < hmlen(hash); i++) {
    if (hash[i].fix) {
      continue;
    }

    real_t *data = hash[i].key;
    int idx = hash[i].idx;
    switch (hash[i].type) {
      case POSE_PARAM:
      case FIDUCIAL_PARAM:
      case EXTRINSIC_PARAM:
        pose_vector_update(data, dx + idx);
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
      default:
        FATAL("Invalid param type [%d]!\n", hash[i].type);
        break;
    }
  }
}

/**
 * Solve problem
 */
int solver_solve(solver_t *solver, void *data) {
  assert(solver != NULL);
  assert(solver->param_order_func != NULL);
  assert(solver->linearize_func != NULL);
  assert(data != NULL);

  // Determine parameter order
  int sv_size = 0;
  int r_size = 0;
  param_order_t *hash = solver->param_order_func(data, &sv_size, &r_size);
  assert(sv_size > 0);
  assert(r_size > 0);

  // Linearize
  real_t *H = CALLOC(real_t, sv_size * sv_size);
  real_t *g = CALLOC(real_t, sv_size);
  real_t *r = CALLOC(real_t, r_size);
  real_t *dx = CALLOC(real_t, sv_size);
  solver->linearize_func(data, sv_size, hash, H, g, r);
  real_t cost_km1 = solver_cost(r, r_size);
  printf("iter: 0, lambda_k: %.2e, cost: %.2e\n", solver->lambda, cost_km1);

  // Start cholmod workspace
#ifdef SOLVER_USE_SUITESPARSE
  cholmod_common common;
  cholmod_start(&common);
#endif

  // Solve
  int max_iter = solver->max_iter;
  real_t lambda_k = solver->lambda;
  real_t cost_k = 0.0;

  for (int iter = 0; iter < max_iter; iter++) {
    // Damp Hessian: H = H + lambda * I
    for (int i = 0; i < sv_size; i++) {
      H[(i * sv_size) + i] += lambda_k;
    }

    // Solve: H * dx = g
#ifdef SOLVER_USE_SUITESPARSE
    suitesparse_chol_solve(&common, H, sv_size, sv_size, g, sv_size, dx);
#else
    chol_solve(H, g, dx, sv_size);
#endif

    // Update parameters
    real_t **x_copy = solver_params_copy(hash);
    solver_update(hash, dx, sv_size);

    // Re-linearize
    {
      zeros(H, sv_size, sv_size);
      zeros(g, sv_size, 1);
      zeros(r, r_size, 1);
      solver->linearize_func(data, sv_size, hash, H, g, r);
      cost_k = solver_cost(r, r_size);
      printf("iter: %d, lambda_k: %.2e, cost: %.2e\n",
             iter + 1,
             lambda_k,
             cost_k);
    }

    // Accept or reject update*/
    if (cost_k < cost_km1) {
      // Accept update
      cost_km1 = cost_k;
      lambda_k /= 10.0;
    } else {
      // Reject update
      lambda_k *= 10.0;
      solver_params_restore(hash, x_copy);
    }
    solver_params_free(hash, x_copy);

    // Termination criteria
    // if (cost_k < 1e-10) {
    //   break;
    // }
  }

  // Clean up
#ifdef SOLVER_USE_SUITESPARSE
  cholmod_finish(&common);
#endif
  hmfree(hash);
  free(H);
  free(g);
  free(r);
  free(dx);

  return 0;
}

/////////////////
// CALIBRATION //
/////////////////

/**
 * Malloc a gimbal calibration view
 */
calib_gimbal_view_t *calib_gimbal_view_malloc(const timestamp_t ts,
                                              const int view_idx,
                                              const int cam_idx,
                                              const int *tag_ids,
                                              const int *corner_indices,
                                              const real_t *object_points,
                                              const real_t *keypoints,
                                              const int N,
                                              extrinsic_t *fiducial_ext,
                                              extrinsic_t *gimbal_ext,
                                              pose_t *pose,
                                              extrinsic_t *link0,
                                              extrinsic_t *link1,
                                              joint_t *joint0,
                                              joint_t *joint1,
                                              joint_t *joint2,
                                              extrinsic_t *cam_ext,
                                              camera_params_t *cam_params) {
  calib_gimbal_view_t *view = MALLOC(calib_gimbal_view_t, 1);
  calib_gimbal_view_setup(view);

  view->cam_idx = cam_idx;
  view->num_corners = N;
  if (N) {
    view->tag_ids = MALLOC(int, N);
    view->corner_indices = MALLOC(int, N);
    view->object_points = MALLOC(real_t, N * 3);
    view->keypoints = MALLOC(real_t, N * 2);
    view->calib_factors = MALLOC(calib_gimbal_factor_t, N);

    assert(view->tag_ids != NULL);
    assert(view->corner_indices != NULL);
    assert(view->object_points != NULL);
    assert(view->keypoints != NULL);

    for (int i = 0; i < N; i++) {
      view->tag_ids[i] = tag_ids[i];
      view->corner_indices[i] = corner_indices[i];
      view->object_points[i * 3] = object_points[i * 3];
      view->object_points[i * 3 + 1] = object_points[i * 3 + 1];
      view->object_points[i * 3 + 2] = object_points[i * 3 + 2];
      view->keypoints[i * 2] = keypoints[i * 2];
      view->keypoints[i * 2 + 1] = keypoints[i * 2 + 1];
    }

    const real_t var[2] = {1.0, 1.0};
    for (int i = 0; i < view->num_corners; i++) {
      const int tag_id = tag_ids[i];
      const int corner_idx = corner_indices[i];
      const real_t *p_FFi = &object_points[i * 3];
      const real_t *z = &keypoints[i * 2];

      view->tag_ids[i] = tag_id;
      view->corner_indices[i] = corner_idx;
      view->object_points[i * 3] = p_FFi[0];
      view->object_points[i * 3 + 1] = p_FFi[1];
      view->object_points[i * 3 + 2] = p_FFi[2];
      view->keypoints[i * 2] = z[0];
      view->keypoints[i * 2 + 1] = z[1];
      calib_gimbal_factor_setup(&view->calib_factors[i],
                                fiducial_ext,
                                gimbal_ext,
                                pose,
                                link0,
                                link1,
                                joint0,
                                joint1,
                                joint2,
                                cam_ext,
                                cam_params,
                                ts,
                                cam_idx,
                                tag_id,
                                corner_idx,
                                p_FFi,
                                z,
                                var);
    }
  }

  return view;
}

/**
 * Setup gimbal calibration view
 */
void calib_gimbal_view_setup(calib_gimbal_view_t *view) {
  view->ts = 0;
  view->cam_idx = 0;
  view->view_idx = 0;

  view->tag_ids = NULL;
  view->corner_indices = NULL;
  view->object_points = NULL;
  view->keypoints = NULL;
  view->num_corners = 0;
}

/**
 * Free gimbal calibration view
 */
void calib_gimbal_view_free(calib_gimbal_view_t *view) {
  if (view->num_corners) {
    free(view->tag_ids);
    free(view->corner_indices);
    free(view->object_points);
    free(view->keypoints);
    free(view->calib_factors);
  }
  free(view);
}

/**
 * Check whether `calib_gimbal_view_t` are equal
 */
int calib_gimbal_view_equals(const calib_gimbal_view_t *v0,
                             const calib_gimbal_view_t *v1) {
  CHECK(v0->ts == v1->ts);
  CHECK(v0->view_idx == v1->view_idx);
  CHECK(v0->cam_idx == v1->cam_idx);
  CHECK(v0->num_corners == v1->num_corners);

  if (v0->num_corners) {
    const size_t pts_len = v0->num_corners * 3;
    const size_t kps_len = v0->num_corners * 2;

    for (int i = 0; i < v0->num_corners; i++) {
      CHECK(v0->tag_ids[i] == v1->tag_ids[i]);
    }
    CHECK(vec_equals(v0->object_points, v1->object_points, pts_len));
    CHECK(vec_equals(v0->keypoints, v1->keypoints, kps_len));
    for (int i = 0; i < v0->num_corners; i++) {
      const calib_gimbal_factor_t *c0 = &v0->calib_factors[i];
      const calib_gimbal_factor_t *c1 = &v1->calib_factors[i];
      CHECK(calib_gimbal_factor_equals(c0, c1));
    }
  }

  return 1;
error:
  return 0;
}

/**
 * Print gimbal calibration view
 */
void calib_gimbal_view_print(calib_gimbal_view_t *view) {
  printf("ts: %ld\n", view->ts);
  printf("cam_idx: %d\n", view->cam_idx);
  printf("view_idx: %d\n", view->view_idx);
  printf("num_corners: %d\n", view->num_corners);
  printf("\n");

  printf("tag_id  corner_idx  object_point          keypoint\n");
  for (int i = 0; i < view->num_corners; i++) {
    printf("%d       ", view->tag_ids[i]);
    printf("%d           ", view->corner_indices[i]);
    printf("(%.2f, %.2f, %.2f)  ",
           view->object_points[i * 3],
           view->object_points[i * 3 + 1],
           view->object_points[i * 3 + 2]);
    printf("(%.2f, %.2f) ", view->keypoints[i * 2], view->keypoints[i * 2 + 1]);
    printf("\n");
  }
}

/**
 * Setup gimbal calibration data
 */
void calib_gimbal_setup(calib_gimbal_t *calib) {
  // Settings
  calib->fix_fiducial_exts = 1;
  calib->fix_gimbal_exts = 1;
  calib->fix_poses = 1;
  calib->fix_links = 0;
  calib->fix_joints = 0;
  calib->fix_cam_exts = 0;
  calib->fix_cam_params = 0;
  aprilgrid_setup(0, &calib->calib_target);

  // Counters
  calib->num_cams = 0;
  calib->num_views = 0;
  calib->num_poses = 0;
  calib->num_links = 0;
  calib->num_joints = 0;
  calib->num_calib_factors = 0;
  calib->num_joint_factors = 0;

  // Variables
  calib->timestamps = NULL;
  calib->cam_params = NULL;
  calib->cam_exts = NULL;
  calib->links = NULL;
  calib->joints = NULL;
  calib->poses = NULL;

  // Factors
  calib->views = NULL;
  calib->joint_factors = NULL;
}

/**
 * Malloc gimbal calibration
 */
calib_gimbal_t *calib_gimbal_malloc() {
  calib_gimbal_t *calib = MALLOC(calib_gimbal_t, 1);
  calib_gimbal_setup(calib);
  return calib;
}

/**
 * Free gimbal calibration data
 */
void calib_gimbal_free(calib_gimbal_t *calib) {
  assert(calib);

  free(calib->timestamps);
  free(calib->cam_params);
  free(calib->cam_exts);
  free(calib->links);

  if (calib->joints) {
    for (int view_idx = 0; view_idx < calib->num_views; view_idx++) {
      free(calib->joints[view_idx]);
      free(calib->joint_factors[view_idx]);
    }
    free(calib->joints);
    free(calib->joint_factors);
  }
  free(calib->poses);

  if (calib->views) {
    for (int view_idx = 0; view_idx < calib->num_views; view_idx++) {
      for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
        calib_gimbal_view_free(calib->views[view_idx][cam_idx]);
      }
      free(calib->views[view_idx]);
    }
    free(calib->views);
  }

  free(calib);
}

/**
 * Check if gimbal calibration problems are equal
 */
int calib_gimbal_equals(const calib_gimbal_t *calib0,
                        const calib_gimbal_t *calib1) {
  CHECK(calib0->fix_fiducial_exts == calib1->fix_fiducial_exts);
  CHECK(calib0->fix_gimbal_exts == calib1->fix_gimbal_exts);
  CHECK(calib0->fix_poses == calib1->fix_poses);
  CHECK(calib0->fix_links == calib1->fix_links);
  CHECK(calib0->fix_joints == calib1->fix_joints);
  CHECK(calib0->fix_cam_exts == calib1->fix_cam_exts);
  CHECK(calib0->fix_cam_params == calib1->fix_cam_params);
  CHECK(aprilgrid_equals(&calib0->calib_target, &calib1->calib_target) == 1);

  CHECK(calib0->fiducial_ext_ok == calib1->fiducial_ext_ok);
  CHECK(calib0->gimbal_ext_ok == calib1->gimbal_ext_ok);
  CHECK(calib0->poses_ok == calib1->poses_ok);
  CHECK(calib0->cams_ok == calib1->cams_ok);
  CHECK(calib0->links_ok == calib1->links_ok);
  CHECK(calib0->joints_ok == calib1->joints_ok);

  CHECK(calib0->num_cams == calib1->num_cams);
  CHECK(calib0->num_views == calib1->num_views);
  CHECK(calib0->num_poses == calib1->num_poses);
  CHECK(calib0->num_links == calib1->num_links);
  CHECK(calib0->num_joints == calib1->num_joints);
  CHECK(calib0->num_calib_factors == calib1->num_calib_factors);
  CHECK(calib0->num_joint_factors == calib1->num_joint_factors);

  CHECK(vec_equals(calib0->fiducial_exts.data, calib1->fiducial_exts.data, 7));
  CHECK(vec_equals(calib0->gimbal_exts.data, calib1->gimbal_exts.data, 7));
  for (int cam_idx = 0; cam_idx < calib0->num_cams; cam_idx++) {
    const real_t *p0 = calib0->cam_params[cam_idx].data;
    const real_t *p1 = calib1->cam_params[cam_idx].data;
    const real_t *e0 = calib0->cam_exts[cam_idx].data;
    const real_t *e1 = calib1->cam_exts[cam_idx].data;
    CHECK(vec_equals(p0, p1, 8));
    CHECK(vec_equals(e0, e1, 7));
  }
  for (int link_idx = 0; link_idx < calib0->num_links; link_idx++) {
    const real_t *p0 = calib0->links[link_idx].data;
    const real_t *p1 = calib1->links[link_idx].data;
    CHECK(vec_equals(p0, p1, 7));
  }
  for (int view_idx = 0; view_idx < calib0->num_views; view_idx++) {
    for (int joint_idx = 0; joint_idx < calib0->num_joints; joint_idx++) {
      const real_t *p0 = calib0->joints[view_idx][joint_idx].data;
      const real_t *p1 = calib1->joints[view_idx][joint_idx].data;
      CHECK(vec_equals(p0, p1, 1));
    }
  }
  for (int pose_idx = 0; pose_idx < calib0->num_poses; pose_idx++) {
    const real_t *p0 = calib0->poses[pose_idx].data;
    const real_t *p1 = calib1->poses[pose_idx].data;
    CHECK(vec_equals(p0, p1, 7));
  }

  // Compare factors
  // -- Compare Calib Factors
  for (int view_idx = 0; view_idx < calib0->num_views; view_idx++) {
    for (int cam_idx = 0; cam_idx < calib0->num_cams; cam_idx++) {
      const calib_gimbal_view_t *v0 = calib0->views[view_idx][cam_idx];
      const calib_gimbal_view_t *v1 = calib1->views[view_idx][cam_idx];
      CHECK(calib_gimbal_view_equals(v0, v1));
    }
  }
  // -- Compare Joint Factors
  for (int view_idx = 0; view_idx < calib0->num_views; view_idx++) {
    for (int joint_idx = 0; joint_idx < calib0->num_joints; joint_idx++) {
      const joint_factor_t *j0 = &calib0->joint_factors[view_idx][joint_idx];
      const joint_factor_t *j1 = &calib1->joint_factors[view_idx][joint_idx];
      CHECK(joint_factor_equals(j0, j1));
    }
  }

  return 1;
error:
  return 0;
}

/**
 * Make a copy of the gimbal calibration problem
 */
calib_gimbal_t *calib_gimbal_copy(const calib_gimbal_t *src) {
  calib_gimbal_t *dst = calib_gimbal_malloc();

  // Settings
  dst->fix_fiducial_exts = src->fix_fiducial_exts;
  dst->fix_gimbal_exts = src->fix_gimbal_exts;
  dst->fix_poses = src->fix_poses;
  dst->fix_cam_params = src->fix_cam_params;
  dst->fix_cam_exts = src->fix_cam_exts;
  dst->fix_links = src->fix_links;
  dst->fix_joints = src->fix_joints;
  aprilgrid_copy(&src->calib_target, &dst->calib_target);

  // Flags
  dst->fiducial_ext_ok = src->fiducial_ext_ok;
  dst->gimbal_ext_ok = src->gimbal_ext_ok;
  dst->poses_ok = src->poses_ok;
  dst->cams_ok = src->cams_ok;
  dst->links_ok = src->links_ok;
  dst->joints_ok = src->joints_ok;

  // Counters
  dst->num_cams = src->num_cams;
  dst->num_views = src->num_views;
  dst->num_poses = src->num_poses;
  dst->num_links = src->num_links;
  dst->num_joints = src->num_joints;
  dst->num_calib_factors = 0;
  dst->num_joint_factors = 0;

  // Variables
  // -- Timestamps
  dst->timestamps = MALLOC(timestamp_t, dst->num_views);
  for (size_t k = 0; k < dst->num_views; k++) {
    dst->timestamps[k] = src->timestamps[k];
  }
  // -- Fiducial
  dst->fiducial_exts = src->fiducial_exts;
  // -- Gimbal extrinsic T_BM0
  dst->gimbal_exts = src->gimbal_exts;
  // -- Cameras
  dst->cam_params = MALLOC(camera_params_t, src->num_cams);
  dst->cam_exts = MALLOC(extrinsic_t, src->num_cams);
  for (size_t cam_idx = 0; cam_idx < dst->num_cams; cam_idx++) {
    dst->cam_params[cam_idx] = src->cam_params[cam_idx];
    dst->cam_exts[cam_idx] = src->cam_exts[cam_idx];
  }
  // -- Links
  dst->links = MALLOC(extrinsic_t, src->num_links);
  for (size_t link_idx = 0; link_idx < dst->num_links; link_idx++) {
    dst->links[link_idx] = src->links[link_idx];
  }
  // -- Joints
  dst->joints = MALLOC(joint_t *, src->num_views);
  for (size_t view_idx = 0; view_idx < src->num_views; view_idx++) {
    dst->joints[view_idx] = MALLOC(joint_t, src->num_joints);
    for (size_t joint_idx = 0; joint_idx < src->num_joints; joint_idx++) {
      dst->joints[view_idx][joint_idx] = src->joints[view_idx][joint_idx];
    }
  }
  // -- Poses
  dst->poses = MALLOC(pose_t, src->num_poses);
  for (size_t pose_idx = 0; pose_idx < src->num_poses; pose_idx++) {
    dst->poses[pose_idx] = src->poses[pose_idx];
  }
  // Factors
  // -- View Factors
  dst->views = MALLOC(calib_gimbal_view_t **, dst->num_views);
  for (int view_idx = 0; view_idx < dst->num_views; view_idx++) {
    const int pose_idx = (dst->num_poses == 1) ? 0 : view_idx;
    dst->views[view_idx] = MALLOC(calib_gimbal_view_t *, dst->num_cams);

    for (int cam_idx = 0; cam_idx < dst->num_cams; cam_idx++) {
      calib_gimbal_view_t *src_view = src->views[view_idx][cam_idx];
      calib_gimbal_view_t *view =
          calib_gimbal_view_malloc(src_view->ts,
                                   src_view->view_idx,
                                   src_view->cam_idx,
                                   src_view->tag_ids,
                                   src_view->corner_indices,
                                   src_view->object_points,
                                   src_view->keypoints,
                                   src_view->num_corners,
                                   &dst->fiducial_exts,
                                   &dst->gimbal_exts,
                                   &dst->poses[pose_idx],
                                   &dst->links[0],
                                   &dst->links[1],
                                   &dst->joints[view_idx][0],
                                   &dst->joints[view_idx][1],
                                   &dst->joints[view_idx][2],
                                   &dst->cam_exts[cam_idx],
                                   &dst->cam_params[cam_idx]);
      dst->views[view_idx][cam_idx] = view;
      dst->num_calib_factors += view->num_corners;
    }
  }

  // -- Joint Factors
  const real_t joint_var = 0.1;
  dst->joint_factors = MALLOC(joint_factor_t *, src->num_views);
  for (size_t view_idx = 0; view_idx < src->num_views; view_idx++) {
    dst->joint_factors[view_idx] = MALLOC(joint_factor_t, src->num_joints);
    for (size_t joint_idx = 0; joint_idx < src->num_joints; joint_idx++) {
      joint_factor_setup(&dst->joint_factors[view_idx][joint_idx],
                         &dst->joints[view_idx][joint_idx],
                         dst->joints[view_idx][joint_idx].data[0],
                         joint_var);
      dst->num_joint_factors++;
    }
  }

  return dst;
}

/**
 * Print gimbal calibration data
 */
void calib_gimbal_print(const calib_gimbal_t *calib) {
  // Settings
  printf("fix_fiducial_exts: %d\n", calib->fix_fiducial_exts);
  printf("fix_gimbal_exts: %d\n", calib->fix_gimbal_exts);
  printf("fix_poses: %d\n", calib->fix_poses);
  printf("fix_links: %d\n", calib->fix_links);
  printf("fix_joints: %d\n", calib->fix_joints);
  printf("fix_cam_exts: %d\n", calib->fix_cam_exts);
  printf("fix_cam_params: %d\n", calib->fix_cam_params);
  printf("\n");

  // Configuration file
  for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
    camera_params_print(&calib->cam_params[cam_idx]);
    printf("\n");
  }
  for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
    char cam_str[20] = {0};
    sprintf(cam_str, "cam%d_exts", cam_idx);
    extrinsic_print(cam_str, &calib->cam_exts[cam_idx]);
  }
  for (int link_idx = 0; link_idx < calib->num_links; link_idx++) {
    char link_str[20] = {0};
    sprintf(link_str, "link%d_exts", link_idx);
    extrinsic_print(link_str, &calib->links[link_idx]);
  }
  extrinsic_print("gimbal_exts", &calib->gimbal_exts);
  extrinsic_print("fiducial_exts", &calib->fiducial_exts);
}

/**
 * Add fiducial to gimbal calibration problem
 */
void calib_gimbal_add_fiducial(calib_gimbal_t *calib,
                               const real_t fiducial_pose[7]) {
  assert(calib != NULL);
  assert(fiducial_pose);
  extrinsic_setup(&calib->fiducial_exts, fiducial_pose);
  calib->fiducial_ext_ok = 1;
}

/**
 * Add pose to gimbal calibration problem
 */
void calib_gimbal_add_pose(calib_gimbal_t *calib,
                           const timestamp_t ts,
                           const real_t pose[7]) {
  assert(calib != NULL);
  assert(pose);

  const int k = calib->num_poses + 1;
  calib->poses = REALLOC(calib->poses, pose_t, k);
  pose_setup(&calib->poses[k - 1], ts, pose);
  calib->num_poses++;
  calib->poses_ok = 1;
}

/**
 * Add gimbal extrinsic to gimbal calibration problem
 */
void calib_gimbal_add_gimbal_extrinsic(calib_gimbal_t *calib,
                                       const real_t gimbal_ext[7]) {
  assert(calib != NULL);
  assert(gimbal_ext);
  extrinsic_setup(&calib->gimbal_exts, gimbal_ext);
  calib->gimbal_ext_ok = 1;
}

/**
 * Add gimbal link to gimbal calibration problem
 */
void calib_gimbal_add_gimbal_link(calib_gimbal_t *calib,
                                  const int link_idx,
                                  const real_t gimbal_link[7]) {
  assert(calib != NULL);
  assert(gimbal_link);

  if (link_idx > (calib->num_links - 1)) {
    const int new_size = calib->num_links + 1;
    calib->links = REALLOC(calib->links, extrinsic_t, new_size);
    calib->num_links++;
  }

  extrinsic_setup(&calib->links[link_idx], gimbal_link);
  calib->links_ok = 1;
}

/**
 * Add camera to gimbal calibration problem
 */
void calib_gimbal_add_camera(calib_gimbal_t *calib,
                             const int cam_idx,
                             const int cam_res[2],
                             const char *proj_model,
                             const char *dist_model,
                             const real_t *cam_params,
                             const real_t *cam_ext) {
  assert(calib != NULL);
  assert(cam_idx <= calib->num_cams);
  assert(cam_res != NULL);
  assert(proj_model != NULL);
  assert(dist_model != NULL);
  assert(cam_params != NULL);
  assert(cam_ext != NULL);

  if (cam_idx > (calib->num_cams - 1)) {
    const int new_size = calib->num_cams + 1;
    calib->cam_params = REALLOC(calib->cam_params, camera_params_t, new_size);
    calib->cam_exts = REALLOC(calib->cam_exts, extrinsic_t, new_size);
  }

  camera_params_setup(&calib->cam_params[cam_idx],
                      cam_idx,
                      cam_res,
                      proj_model,
                      dist_model,
                      cam_params);
  extrinsic_setup(&calib->cam_exts[cam_idx], cam_ext);
  calib->num_cams++;
  calib->cams_ok = 1;
}

/**
 * Add view to gimbal calibration problem
 */
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
                           const int num_joints) {
  assert(calib != NULL);
  assert(&calib->poses[pose_idx] != NULL);
  assert((calib->num_joints == 0) ? 1 : (calib->num_joints == num_joints));

  // Allocate memory for joints and view
  const int num_cams = calib->num_cams;
  const real_t joint_var = 0.1;

  if (view_idx > (calib->num_views - 1)) {
    const int new_size = calib->num_views + 1;
    // Allocate memory for a new timestamp
    calib->timestamps = REALLOC(calib->timestamps, timestamp_t, new_size);
    calib->timestamps[view_idx] = ts;

    // Allocate memory for a new view
    calib->views = REALLOC(calib->views, calib_gimbal_view_t **, new_size);
    calib->views[view_idx] = MALLOC(calib_gimbal_view_t *, num_cams);

    // Allocate memory for gimbal joints
    calib->joints = REALLOC(calib->joints, joint_t *, new_size);
    calib->joints[view_idx] = MALLOC(joint_t, num_joints);
    calib->num_joints = num_joints;
    calib->joints_ok = 1;

    // Allocate memory for gimbal joint factors
    calib->joint_factors =
        REALLOC(calib->joint_factors, joint_factor_t *, new_size);
    calib->joint_factors[view_idx] = MALLOC(joint_factor_t, num_joints);

    for (int joint_idx = 0; joint_idx < num_joints; joint_idx++) {
      // Joint angle
      joint_setup(&calib->joints[view_idx][joint_idx],
                  ts,
                  joint_idx,
                  joints[joint_idx]);

      // Joint factor
      joint_factor_setup(&calib->joint_factors[view_idx][joint_idx],
                         &calib->joints[view_idx][joint_idx],
                         calib->joints[view_idx][joint_idx].data[0],
                         joint_var);
      calib->num_joint_factors++;
    }

    // Increment number of views
    calib->num_views++;
  }

  // Form a new calibration view
  calib_gimbal_view_t *view =
      calib_gimbal_view_malloc(ts,
                               view_idx,
                               cam_idx,
                               tag_ids,
                               corner_indices,
                               object_points,
                               keypoints,
                               num_corners,
                               &calib->fiducial_exts,
                               &calib->gimbal_exts,
                               &calib->poses[pose_idx],
                               &calib->links[0],
                               &calib->links[1],
                               &calib->joints[view_idx][0],
                               &calib->joints[view_idx][1],
                               &calib->joints[view_idx][2],
                               &calib->cam_exts[cam_idx],
                               &calib->cam_params[cam_idx]);
  calib->num_calib_factors += num_corners;

  // Update
  calib->views[view_idx][cam_idx] = view;
}

/**
 * Remove view from gimbal calibration problem.
 */
int calib_gimbal_remove_view(calib_gimbal_t *calib, const int view_idx) {
  assert(view_idx < calib->num_views);

  for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
    calib_gimbal_view_t *view = calib->views[view_idx][cam_idx];
    calib->num_calib_factors -= view->num_corners;
    calib_gimbal_view_free(view);
  }
  free(calib->views[view_idx]);

  free(calib->joints[view_idx]);
  free(calib->joint_factors[view_idx]);
  calib->num_joint_factors -= calib->num_joints;
  calib->num_views--;

  return 0;
}

/**
 * Skip line
 */
static void parse_skip_line(FILE *fp) {
  assert(fp != NULL);
  const size_t buf_len = 9046;
  char buf[9046] = {0};
  const char *read = fgets(buf, buf_len, fp);
  UNUSED(read);
}

/**
 * Parse integer vector from string line.
 * @returns `0` for success or `-1` for failure
 */
static int parse_vector_line(char *line, const char *type, void *data, int n) {
  assert(line != NULL);
  assert(data != NULL);
  char entry[MAX_LINE_LENGTH] = {0};
  int index = 0;

  for (size_t i = 0; i < strlen(line); i++) {
    char c = line[i];
    if (c == '[' || c == ' ') {
      continue;
    }

    if (c == ',' || c == ']' || c == '\n') {
      if (strcmp(type, "int") == 0) {
        ((int *) data)[index] = strtod(entry, NULL);
      } else if (strcmp(type, "double") == 0) {
        ((double *) data)[index] = strtod(entry, NULL);
      } else {
        FATAL("Invalid type [%s]\n", type);
      }
      index++;
      memset(entry, '\0', sizeof(char) * 100);
    } else {
      entry[strlen(entry)] = c;
    }
  }

  if (index != n) {
    return -1;
  }

  return 0;
}

/** Parse key-value pair from string line **/
static void parse_key_value(FILE *fp,
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
    FATAL("Failed to parse [%s]\n", key);
  }

  // Split key-value
  char delim[2] = ":";
  char *key_str = strtok(buf, delim);
  char *value_str = strtok(NULL, delim);
  if (key_str == NULL || value_str == NULL) {
    FATAL("Failed to parse [%s]\n", key);
  }
  key_str = string_strip(key_str);
  value_str = string_strip(value_str);

  // Check key matches
  if (strcmp(key_str, key) != 0) {
    FATAL("Failed to parse [%s]\n", key);
  }

  // Typecase value
  if (value_type == NULL) {
    FATAL("Value type not set!\n");
  }

  // Parse value
  if (strcmp(value_type, "int") == 0) {
    *(int *) value = atoi(value_str);
  } else if (strcmp(value_type, "double") == 0) {
    *(double *) value = atof(value_str);
  } else if (strcmp(value_type, "uint64_t") == 0) {
    *(uint64_t *) value = atol(value_str);
  } else if (strcmp(value_type, "string") == 0) {
    value_str = string_strip_char(value_str, '"');
    string_copy((char *) value, value_str);
  } else if (strcmp(value_type, "vec2i") == 0) {
    parse_vector_line(value_str, "int", value, 2);
  } else if (strcmp(value_type, "vec3i") == 0) {
    parse_vector_line(value_str, "int", value, 3);
  } else if (strcmp(value_type, "vec2d") == 0) {
    parse_vector_line(value_str, "double", value, 2);
  } else if (strcmp(value_type, "vec3d") == 0) {
    parse_vector_line(value_str, "double", value, 3);
  } else if (strcmp(value_type, "vec4d") == 0) {
    parse_vector_line(value_str, "double", value, 4);
  } else if (strcmp(value_type, "vec7d") == 0) {
    parse_vector_line(value_str, "double", value, 7);
  } else if (strcmp(value_type, "pose") == 0) {
    parse_vector_line(value_str, "double", value, 6);
  } else {
    FATAL("Invalid value type [%s]\n", value_type);
  }
}

static void calib_gimbal_load_config(calib_gimbal_t *calib,
                                     const char *data_path) {
  // Open config file
  char conf_path[100] = {0};
  string_cat(conf_path, data_path);
  string_cat(conf_path, "/calib.config");
  FILE *conf = fopen(conf_path, "r");
  if (conf == NULL) {
    FATAL("Failed to open [%s]!\n", conf_path);
  }

  // Parse general
  parse_key_value(conf, "num_cams", "int", &calib->num_cams);
  parse_key_value(conf, "num_links", "int", &calib->num_links);
  parse_skip_line(conf);

  // Parse camera parameters
  calib->cam_params = MALLOC(camera_params_t, calib->num_cams);
  for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
    real_t proj_params[4] = {0};
    real_t dist_params[4] = {0};

    camera_params_t *cam_params = &calib->cam_params[cam_idx];
    cam_params->cam_idx = cam_idx;

    parse_skip_line(conf);
    parse_key_value(conf, "resolution", "vec2i", cam_params->resolution);
    parse_key_value(conf, "proj_model", "string", cam_params->proj_model);
    parse_key_value(conf, "dist_model", "string", cam_params->dist_model);
    parse_key_value(conf, "proj_params", "vec4d", proj_params);
    parse_key_value(conf, "dist_params", "vec4d", dist_params);
    parse_skip_line(conf);

    cam_params->data[0] = proj_params[0];
    cam_params->data[1] = proj_params[1];
    cam_params->data[2] = proj_params[2];
    cam_params->data[3] = proj_params[3];

    cam_params->data[4] = dist_params[0];
    cam_params->data[5] = dist_params[1];
    cam_params->data[6] = dist_params[2];
    cam_params->data[7] = dist_params[3];
  }

  // Parse camera extrinsic
  calib->cam_exts = MALLOC(extrinsic_t, calib->num_cams);
  for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
    char key[30] = {0};
    real_t pose[7] = {0};
    sprintf(key, "cam%d_exts", cam_idx);
    parse_key_value(conf, key, "pose", pose);
    extrinsic_setup(&calib->cam_exts[cam_idx], pose);
  }
  calib->cams_ok = 1;

  // Parse links
  calib->links = MALLOC(extrinsic_t, calib->num_links);
  for (int link_idx = 0; link_idx < calib->num_links; link_idx++) {
    char key[30] = {0};
    real_t pose[7] = {0};
    sprintf(key, "link%d_exts", link_idx);
    parse_key_value(conf, key, "pose", pose);
    extrinsic_setup(&calib->links[link_idx], pose);
    calib->links_ok = 1;
  }

  // Parse gimbal extrinsic
  {
    real_t pose[7] = {0};
    parse_key_value(conf, "gimbal_exts", "pose", pose);
    extrinsic_setup(&calib->gimbal_exts, pose);
    calib->gimbal_ext_ok = 1;
  }

  // Parse fiducial
  {
    real_t pose[7] = {0};
    parse_key_value(conf, "fiducial_exts", "pose", pose);
    extrinsic_setup(&calib->fiducial_exts, pose);
    calib->fiducial_ext_ok = 1;
  }

  // Clean up
  fclose(conf);
}

static void calib_gimbal_load_joints(calib_gimbal_t *calib,
                                     const char *data_path) {
  // Open joint angles file
  char joints_path[100] = {0};
  string_cat(joints_path, data_path);
  string_cat(joints_path, "/joint_angles.sim");
  FILE *joints_file = fopen(joints_path, "r");
  if (joints_file == NULL) {
    FATAL("Failed to open [%s]!\n", joints_path);
  }

  // Parse
  parse_key_value(joints_file, "num_views", "int", &calib->num_views);
  parse_key_value(joints_file, "num_joints", "int", &calib->num_joints);
  parse_skip_line(joints_file);
  parse_skip_line(joints_file);

  calib->timestamps = MALLOC(timestamp_t, calib->num_views);
  calib->joints = MALLOC(joint_t *, calib->num_views);
  calib->joint_factors = MALLOC(joint_factor_t *, calib->num_views);

  int num_joints = calib->num_joints;
  for (int view_idx = 0; view_idx < calib->num_views; view_idx++) {
    // Get line
    const size_t buf_len = 1024;
    char buf[1024] = {0};
    if (fgets(buf, buf_len, joints_file) == NULL) {
      FATAL("Failed to view parse data!\n");
    }

    // Parse timestamp and joint angles
    size_t n = 0;
    char **s = string_split(buf, ',', &n);
    const timestamp_t ts = strtod(s[0], NULL);
    assert(n == calib->num_joints + 1);

    calib->timestamps[view_idx] = ts;
    calib->joints[view_idx] = MALLOC(joint_t, num_joints);
    calib->joint_factors[view_idx] = MALLOC(joint_factor_t, num_joints);

    const real_t joint_var = 0.1;
    for (size_t joint_idx = 0; joint_idx < num_joints; joint_idx++) {
      // Joint angles
      joint_t *joint = &calib->joints[view_idx][joint_idx];
      const real_t joint_angle = strtod(s[joint_idx + 1], NULL);
      joint_setup(joint, ts, joint_idx, joint_angle);
      free(s[joint_idx + 1]);
      calib->joints_ok = 1;

      // Joint factors
      joint_factor_setup(&calib->joint_factors[view_idx][joint_idx],
                         &calib->joints[view_idx][joint_idx],
                         calib->joints[view_idx][joint_idx].data[0],
                         joint_var);
      calib->num_joint_factors++;
    }
    free(s[0]);
    free(s);
  }

  // Clean up
  fclose(joints_file);
}

static void calib_gimbal_load_poses(calib_gimbal_t *calib,
                                    const char *data_path) {
  // Load poses
  char poses_path[100] = {0};
  string_cat(poses_path, data_path);
  string_cat(poses_path, "/poses.sim");
  FILE *poses_file = fopen(poses_path, "r");
  if (poses_file == NULL) {
    FATAL("Failed to open [%s]!\n", poses_path);
  }

  // Parse
  parse_key_value(poses_file, "num_poses", "int", &calib->num_poses);
  calib->poses = MALLOC(pose_t, calib->num_poses);
  parse_skip_line(poses_file);
  parse_skip_line(poses_file);
  for (int pose_idx = 0; pose_idx < calib->num_poses; pose_idx++) {
    // Get line
    const size_t buf_len = 1024;
    char buf[1024] = {0};
    if (fgets(buf, buf_len, poses_file) == NULL) {
      FATAL("Failed to view parse data!\n");
    }

    // Parse line
    real_t data[7] = {0};
    parse_vector_line(buf, "double", data, 7);
    calib->poses[pose_idx].ts = 0;
    vec_copy(data, 7, calib->poses[pose_idx].data);
    calib->poses_ok = 1;
  }
  fclose(poses_file);
}

static void calib_gimbal_load_views(calib_gimbal_t *calib,
                                    const char *data_path) {
  // Load views
  calib->views = MALLOC(calib_gimbal_view_t **, calib->num_views);
  for (int view_idx = 0; view_idx < calib->num_views; view_idx++) {
    calib->views[view_idx] = MALLOC(calib_gimbal_view_t *, calib->num_cams);

    for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
      // Form view file path
      char view_fpath[1000] = {0};
      char *vfile_fmt = "/cam%d/%ld.sim";
      const timestamp_t ts = calib->timestamps[view_idx];
      string_cat(view_fpath, data_path);
      sprintf(view_fpath + strlen(view_fpath), vfile_fmt, cam_idx, ts);

      // Load view data
      FILE *view_file = fopen(view_fpath, "r");
      if (view_file == NULL) {
        FATAL("Failed to open file [%s]\n", view_fpath);
      }

      int num_corners = 0;
      parse_key_value(view_file, "num_corners", "int", &num_corners);
      parse_skip_line(view_file);

      int *tag_ids = MALLOC(int, num_corners);
      int *corner_indices = MALLOC(int, num_corners);
      real_t *object_points = MALLOC(real_t, num_corners * 3);
      real_t *keypoints = MALLOC(real_t, num_corners * 2);

      if (num_corners) {
        parse_skip_line(view_file);
        for (int i = 0; i < num_corners; i++) {
          // Get line
          const size_t buf_len = 1024;
          char buf[1024] = {0};
          if (fgets(buf, buf_len, view_file) == NULL) {
            FATAL("Failed to view parse data!\n");
          }

          // Parse line
          real_t data[7] = {0};
          parse_vector_line(buf, "double", data, 7);

          // Add to view
          tag_ids[i] = (int) data[0];
          corner_indices[i] = (int) data[1];
          object_points[i * 3] = data[2];
          object_points[i * 3 + 1] = data[3];
          object_points[i * 3 + 2] = data[4];
          keypoints[i * 2] = data[5];
          keypoints[i * 2 + 1] = data[6];
        }
      }

      // Form view
      const int pose_idx = (calib->num_poses == 1) ? 0 : view_idx;
      calib_gimbal_view_t *view = NULL;
      view = calib_gimbal_view_malloc(ts,
                                      view_idx,
                                      cam_idx,
                                      tag_ids,
                                      corner_indices,
                                      object_points,
                                      keypoints,
                                      num_corners,
                                      &calib->fiducial_exts,
                                      &calib->gimbal_exts,
                                      &calib->poses[pose_idx],
                                      &calib->links[0],
                                      &calib->links[1],
                                      &calib->joints[view_idx][0],
                                      &calib->joints[view_idx][1],
                                      &calib->joints[view_idx][2],
                                      &calib->cam_exts[cam_idx],
                                      &calib->cam_params[cam_idx]);
      calib->views[view_idx][cam_idx] = view;
      calib->num_calib_factors += num_corners;
      free(tag_ids);
      free(corner_indices);
      free(object_points);
      free(keypoints);

      // Clean up
      fclose(view_file);
    }
  }
}

/**
 * Load gimbal calibration data
 */
calib_gimbal_t *calib_gimbal_load(const char *data_path) {
  calib_gimbal_t *calib = MALLOC(calib_gimbal_t, 1);
  calib_gimbal_setup(calib);
  calib_gimbal_load_config(calib, data_path);
  calib_gimbal_load_joints(calib, data_path);
  calib_gimbal_load_poses(calib, data_path);
  calib_gimbal_load_views(calib, data_path);
  return calib;
}

int calib_gimbal_validate(calib_gimbal_t *calib) {
  const int num_checks = 6;
  const char *checklist_names[] = {"calib->fiducial_ext_ok",
                                   "calib->gimbal_ext_ok",
                                   "calib->poses_ok",
                                   "calib->cams_ok",
                                   "calib->links_ok",
                                   "calib->joints_ok"};
  const int checklist_vals[] = {calib->fiducial_ext_ok,
                                calib->gimbal_ext_ok,
                                calib->poses_ok,
                                calib->cams_ok,
                                calib->links_ok,
                                calib->joints_ok};

  for (int i = 0; i < num_checks; i++) {
    if (checklist_vals[i] != 1) {
      LOG_ERROR("%s != 1\n", checklist_names[i]);
      return -1;
    }
  }

  return 0;
}

/**
 * Calculate the Shannon-Entropy of the current gimbal calibration problem.
 */
int calib_gimbal_shannon_entropy(calib_gimbal_t *calib, real_t *entropy) {
  // Determine parameter order
  int sv_size = 0;
  int r_size = 0;
  param_order_t *hash = calib_gimbal_param_order(calib, &sv_size, &r_size);

  // Form Hessian H
  real_t *H = CALLOC(real_t, sv_size * sv_size);
  real_t *g = CALLOC(real_t, sv_size);
  real_t *r = CALLOC(real_t, r_size);
  calib_gimbal_linearize_compact(calib, sv_size, hash, H, g, r);

  // Estimate covariance
  real_t *covar = CALLOC(real_t, sv_size * sv_size);
  pinv(H, sv_size, sv_size, covar);

  // Calculate shannon-entropy
  int status = 0;
  if (shannon_entropy(covar, sv_size, entropy) != 0) {
    status = -1;
  }

  // Clean up
  hmfree(hash);
  free(H);
  free(g);
  free(r);

  return status;
}

/**
 * Find the next best view for the gimbal calibration problem.
 */
void calib_gimbal_nbv(calib_gimbal_t *calib, real_t nbv_joints[3]) {
  // Sample settings
  const real_t parts_roll = 5;
  const real_t parts_pitch = 5;
  const real_t parts_yaw = 5;
  const real_t range_roll[2] = {deg2rad(-45.0), deg2rad(45.0)};
  const real_t range_pitch[2] = {deg2rad(-45.0), deg2rad(45.0)};
  const real_t range_yaw[2] = {deg2rad(-45.0), deg2rad(45.0)};

  // Find Next-Best-View
  const real_t droll = (range_roll[1] - range_roll[0]) / parts_roll;
  const real_t dpitch = (range_pitch[1] - range_pitch[0]) / parts_pitch;
  const real_t dyaw = (range_yaw[1] - range_yaw[0]) / parts_yaw;

  real_t entropy_init = 0.0;
  if (calib_gimbal_shannon_entropy(calib, &entropy_init) != 0) {
    return;
  }

  const int nbv_idx = calib->num_views;
  const timestamp_t ts = nbv_idx;
  const int pose_idx = 0;

  const int num_views = parts_roll * parts_pitch * parts_yaw;
  real_t *entropy_scores = MALLOC(real_t, num_views);
  real_t *entropy_joints = MALLOC(real_t, num_views * 3);

  {
    int idx = 0;
    for (real_t r = range_roll[0]; r < range_roll[1]; r += droll) {
      for (real_t p = range_pitch[0]; p < range_pitch[1]; p += dpitch) {
        for (real_t y = range_yaw[0]; y < range_yaw[1]; y += dyaw) {
          entropy_joints[idx + 0] = y;
          entropy_joints[idx + 1] = r;
          entropy_joints[idx + 2] = p;
          idx++;
        }
      }
    }
  }

#pragma omp parallel for
  for (int view_idx = 0; view_idx < num_views; view_idx++) {
    // Make a copy of the gimbal calibration problem
    calib_gimbal_t *calib_copy = calib_gimbal_copy(calib);

    // Form gimbal joint angles
    const int num_joints = 3;
    const real_t y = entropy_joints[view_idx + 0];
    const real_t r = entropy_joints[view_idx + 1];
    const real_t p = entropy_joints[view_idx + 2];
    const real_t joints[3] = {y, r, p};

    // Simulate gimbal view
    for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
      // Gimbal view
      sim_gimbal_view_t *view =
          sim_gimbal3_view(&calib_copy->calib_target,
                           ts,
                           nbv_idx,
                           calib_copy->fiducial_exts.data,
                           calib_copy->poses[pose_idx].data,
                           calib_copy->gimbal_exts.data,
                           calib_copy->links[0].data,
                           calib_copy->links[1].data,
                           joints[0],
                           joints[1],
                           joints[2],
                           cam_idx,
                           calib_copy->cam_params[cam_idx].resolution,
                           calib_copy->cam_params[cam_idx].data,
                           calib_copy->cam_exts[cam_idx].data);

      // Add view to calibration problem
      calib_gimbal_add_view(calib_copy,
                            pose_idx,
                            nbv_idx,
                            ts,
                            cam_idx,
                            view->num_measurements,
                            view->tag_ids,
                            view->corner_indices,
                            view->object_points,
                            view->keypoints,
                            joints,
                            num_joints);

      // Free view data
      sim_gimbal_view_free(view);
    }

    // Calculate shannon entropy
    real_t entropy_view = 0.0;
    if (calib_gimbal_shannon_entropy(calib_copy, &entropy_view) != 0) {
      continue;
    }
    entropy_scores[view_idx] = entropy_view;

    // // Remove view
    // calib_gimbal_remove_view(calib, nbv_idx);
    calib_gimbal_free(calib_copy);

    // printf("yaw: %.2f, ", y);
    // printf("roll: %.2f, ", r);
    // printf("pitch: %.2f, ", p);
    // printf("entropy_view: %.2f, ", entropy_view);
    // printf("entropy_best: %.2f\n", entropy_best);
    printf(".");
    fflush(stdout);
  }

  // Find best
  real_t entropy_best = 0.0;
  real_t joints_best[3] = {0.0, 0.0, 0.0};
  for (int view_idx = 0; view_idx < num_views; view_idx++) {
    if (entropy_scores[view_idx] < entropy_best) {
      entropy_best = entropy_scores[view_idx];
      joints_best[0] = entropy_joints[view_idx + 0];
      joints_best[1] = entropy_joints[view_idx + 1];
      joints_best[2] = entropy_joints[view_idx + 2];
    }
  }
  free(entropy_scores);
  free(entropy_joints);

  vec_copy(joints_best, 3, nbv_joints);
  printf("calib_entropy: %.2f, ", entropy_init);
  printf("nbv_entropy: %.2f, ", entropy_best);
  printf("nbv_joints: [");
  printf("%.2f, ", rad2deg(nbv_joints[0]));
  printf("%.2f, ", rad2deg(nbv_joints[1]));
  printf("%.2f]\n", rad2deg(nbv_joints[2]));
  // printf("\n");
}

param_order_t *calib_gimbal_param_order(const void *data,
                                        int *sv_size,
                                        int *r_size) {
  // Setup parameter order
  calib_gimbal_t *calib = (calib_gimbal_t *) data;
  assert(calib_gimbal_validate(calib) == 0);
  param_order_t *hash = NULL;
  int col_idx = 0;

  // -- Add body poses
  for (int pose_idx = 0; pose_idx < calib->num_poses; pose_idx++) {
    void *key = &calib->poses[pose_idx].data;
    param_order_t kv = {key, col_idx, POSE_PARAM, calib->fix_poses};
    hmputs(hash, kv);
    if (calib->fix_poses == 0) {
      col_idx += param_local_size(POSE_PARAM);
    }
  }
  // -- Add fiducial extrinsic
  {
    void *key = &calib->fiducial_exts.data;
    param_order_t kv = {key, col_idx, FIDUCIAL_PARAM, calib->fix_fiducial_exts};
    hmputs(hash, kv);
    if (calib->fix_fiducial_exts == 0) {
      col_idx += param_local_size(FIDUCIAL_PARAM);
    }
  }
  // -- Add gimbal extrinsic
  {
    void *key = &calib->gimbal_exts.data;
    param_order_t kv = {key, col_idx, EXTRINSIC_PARAM, calib->fix_gimbal_exts};
    hmputs(hash, kv);
    if (calib->fix_gimbal_exts == 0) {
      col_idx += param_local_size(EXTRINSIC_PARAM);
    }
  }
  // -- Add links
  for (int i = 0; i < calib->num_links; i++) {
    void *key = &calib->links[i].data;
    param_order_t kv = {key, col_idx, EXTRINSIC_PARAM, calib->fix_links};
    hmputs(hash, kv);
    if (calib->fix_links == 0) {
      col_idx += param_local_size(EXTRINSIC_PARAM);
    }
  }
  // -- Add camera extrinsic
  for (int i = 0; i < calib->num_cams; i++) {
    void *key = &calib->cam_exts[i].data;
    param_order_t kv = {key, col_idx, EXTRINSIC_PARAM, calib->fix_cam_exts};
    hmputs(hash, kv);
    if (calib->fix_cam_exts == 0) {
      col_idx += param_local_size(EXTRINSIC_PARAM);
    }
  }
  // -- Add camera parameters
  for (int i = 0; i < calib->num_cams; i++) {
    void *key = &calib->cam_params[i].data;
    param_order_t kv = {key, col_idx, CAMERA_PARAM, calib->fix_cam_params};
    hmputs(hash, kv);
    if (calib->fix_cam_params == 0) {
      col_idx += param_local_size(CAMERA_PARAM);
    }
  }
  // -- Add joints to hash
  for (int view_idx = 0; view_idx < calib->num_views; view_idx++) {
    joint_t *joints = calib->joints[view_idx];
    for (int i = 0; i < calib->num_joints; i++) {
      param_order_t val = {&joints[i].data,
                           col_idx,
                           JOINT_PARAM,
                           calib->fix_joints};
      hmputs(hash, val);
      if (calib->fix_joints == 0) {
        col_idx += param_local_size(JOINT_PARAM);
      }
    }
  }

  *sv_size = col_idx;
  *r_size = (calib->num_calib_factors * 2) + calib->num_joint_factors;
  return hash;
}

void calib_gimbal_linearize(const void *data,
                            const int J_rows,
                            const int J_cols,
                            param_order_t *hash,
                            real_t *J,
                            real_t *g,
                            real_t *r) {
  // Evaluate factors
  calib_gimbal_t *calib = (calib_gimbal_t *) data;
  assert(calib_gimbal_validate(calib) == 0);
  int J_row_idx = 0;

  for (int view_idx = 0; view_idx < calib->num_views; view_idx++) {
    for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
      calib_gimbal_view_t *view = calib->views[view_idx][cam_idx];
      for (int factor_idx = 0; factor_idx < view->num_corners; factor_idx++) {
        calib_gimbal_factor_t *factor = &view->calib_factors[factor_idx];
        calib_gimbal_factor_eval(factor);
        r[factor_idx * 2] = factor->r[0];
        r[factor_idx * 2 + 1] = factor->r[1];

        solver_fill_jacobian(hash,
                             factor->num_params,
                             factor->params,
                             factor->jacs,
                             factor->r,
                             factor->r_size,
                             J_cols,
                             J_row_idx,
                             J,
                             g);
        J_row_idx += 2;
      } // For each factor
    }   // For each cameras
  }     // For each views
}

void calib_gimbal_linearize_compact(const void *data,
                                    const int sv_size,
                                    param_order_t *hash,
                                    real_t *H,
                                    real_t *g,
                                    real_t *r) {
  // Evaluate factors
  calib_gimbal_t *calib = (calib_gimbal_t *) data;
  assert(calib_gimbal_validate(calib) == 0);

  int r_idx = 0;
  for (int view_idx = 0; view_idx < calib->num_views; view_idx++) {
    // Joint factors
    for (int i = 0; i < calib->num_joints; i++) {
      joint_factor_t *factor = &calib->joint_factors[view_idx][i];
      joint_factor_eval(factor);
      vec_copy(factor->r, factor->r_size, &r[r_idx]);

      solver_fill_hessian(hash,
                          factor->num_params,
                          factor->params,
                          factor->jacs,
                          factor->r,
                          factor->r_size,
                          sv_size,
                          H,
                          g);
      r_idx += factor->r_size;
    } // For each joint factor

    // Calib factors
    for (int cam_idx = 0; cam_idx < calib->num_cams; cam_idx++) {
      calib_gimbal_view_t *view = calib->views[view_idx][cam_idx];
      for (int factor_idx = 0; factor_idx < view->num_corners; factor_idx++) {
        calib_gimbal_factor_t *factor = &view->calib_factors[factor_idx];
        calib_gimbal_factor_eval(factor);
        vec_copy(factor->r, factor->r_size, &r[r_idx]);

        solver_fill_hessian(hash,
                            factor->num_params,
                            factor->params,
                            factor->jacs,
                            factor->r,
                            factor->r_size,
                            sv_size,
                            H,
                            g);
        r_idx += factor->r_size;
      } // For each calib factor
    }   // For each cameras
  }     // For each views

  // printf("rank: %d, sv_size: %d\n",
  //        eig_rank(H, sv_size, sv_size, 1e-4),
  //        sv_size);

  // real_t H_det = 0.0;
  // real_t *H_copy = MALLOC(real_t, sv_size * sv_size);
  // mat_copy(H, sv_size, sv_size, H_copy);
  // svd_det(H_copy, sv_size, sv_size, &H_det);
  // printf("det(H): %e\n", H_det);
  // free(H_copy);

  // mat_save("/tmp/H.csv", H, sv_size, sv_size);
  // exit(0);
}

///////////////////////
// INERTIAL ODOMETRY //
///////////////////////

void inertial_odometry_free(inertial_odometry_t *odom) {
  free(odom->factors);
  free(odom->poses);
  free(odom->vels);
  free(odom->biases);
  free(odom);
}

void inertial_odometry_save(const inertial_odometry_t *odom,
                            const char *save_path) {
  // Load file
  FILE *fp = fopen(save_path, "w");
  if (fp == NULL) {
    FATAL("Failed to open [%s]!\n", save_path);
  }

  // Write header
  fprintf(fp, "x,y,z,qw,qx,qy,qz,");
  fprintf(fp, "vx,vy,vz,");
  fprintf(fp, "ba_x,ba_y,ba_z,");
  fprintf(fp, "bg_x,bg_y,bg_z\n");

  // Write data
  for (int k = 0; k < (odom->num_factors + 1); k++) {
    const real_t *pos = odom->poses[k].data;
    const real_t *quat = odom->poses[k].data + 3;
    const real_t *vel = odom->vels[k].data;
    const real_t *ba = odom->biases[k].data;
    const real_t *bg = odom->biases[k].data + 3;
    fprintf(fp, "%f,%f,%f,", pos[0], pos[1], pos[2]);
    fprintf(fp, "%f,%f,%f,%f,", quat[0], quat[1], quat[2], quat[3]);
    fprintf(fp, "%f,%f,%f,", vel[0], vel[1], vel[2]);
    fprintf(fp, "%f,%f,%f,", ba[0], ba[1], ba[2]);
    fprintf(fp, "%f,%f,%f", bg[0], bg[1], bg[2]);
    fprintf(fp, "\n");
  }
}

param_order_t *inertial_odometry_param_order(const void *data,
                                             int *sv_size,
                                             int *r_size) {
  // Setup parameter order
  inertial_odometry_t *odom = (inertial_odometry_t *) data;
  param_order_t *hash = NULL;
  int col_idx = 0;

  for (int k = 0; k <= odom->num_factors; k++) {
    // Pose
    {
      pose_t *pose_k = &odom->poses[k];
      param_order_t val = {&pose_k->data, col_idx, POSE_PARAM, 0};
      hmputs(hash, val);
      col_idx += param_local_size(POSE_PARAM);
    }

    // Velocity
    {
      velocity_t *vel_k = &odom->vels[k];
      param_order_t val = {&vel_k->data, col_idx, VELOCITY_PARAM, 0};
      hmputs(hash, val);
      col_idx += param_local_size(VELOCITY_PARAM);
    }

    // Biases
    {
      imu_biases_t *biase_k = &odom->biases[k];
      param_order_t val = {&biase_k->data, col_idx, IMU_BIASES_PARAM, 0};
      hmputs(hash, val);
      col_idx += param_local_size(IMU_BIASES_PARAM);
    }
  }

  *sv_size = col_idx;
  *r_size = odom->num_factors * 15;
  return hash;
}

void inertial_odometry_linearize_compact(const void *data,
                                         const int sv_size,
                                         param_order_t *hash,
                                         real_t *H,
                                         real_t *g,
                                         real_t *r) {
  // Evaluate factors
  inertial_odometry_t *odom = (inertial_odometry_t *) data;

  for (int k = 0; k < odom->num_factors; k++) {
    imu_factor_t *factor = &odom->factors[k];
    imu_factor_eval(factor);
    vec_copy(factor->r, factor->r_size, &r[k * factor->r_size]);

    solver_fill_hessian(hash,
                        factor->num_params,
                        factor->params,
                        factor->jacs,
                        factor->r,
                        factor->r_size,
                        sv_size,
                        H,
                        g);
  }
}

/******************************************************************************
 * DATASET
 ******************************************************************************/

static int
parse_pose_data(const int i, const int j, const char *entry, pose_t *poses) {
  switch (j) {
    case 0:
      poses[i].ts = strtol(entry, NULL, 0);
      break;
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      poses[i].data[j - 1] = strtod(entry, NULL);
      break;
    default:
      return -1;
  }

  return 0;
}

/**
 * Load poses from file `fp`. The number of poses in file will be outputted to
 * `nb_poses`.
 */
pose_t *load_poses(const char *fp, int *nb_poses) {
  assert(fp != NULL);
  assert(nb_poses != NULL);

  // Obtain number of rows and columns in dsv data
  int nb_rows = dsv_rows(fp);
  int nb_cols = dsv_cols(fp, ',');
  if (nb_rows == -1 || nb_cols == -1) {
    return NULL;
  }

  // Initialize memory for pose data
  *nb_poses = nb_rows;
  pose_t *poses = MALLOC(pose_t, nb_rows);

  // Load file
  FILE *infile = fopen(fp, "r");
  if (infile == NULL) {
    fclose(infile);
    free(poses);
    return NULL;
  }

  // Loop through data
  char line[MAX_LINE_LENGTH] = {0};
  int row_idx = 0;
  int col_idx = 0;

  // Loop through data line by line
  while (fgets(line, MAX_LINE_LENGTH, infile) != NULL) {
    // Ignore if comment line
    if (line[0] == '#') {
      continue;
    }

    // Iterate through values in line separated by commas
    char entry[100] = {0};
    for (size_t i = 0; i < strlen(line); i++) {
      char c = line[i];
      if (c == ' ') {
        continue;
      }

      if (c == ',' || c == '\n') {
        if (parse_pose_data(row_idx, col_idx, entry, poses) != 0) {
          return NULL;
        }
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

  return poses;
}

/**
 * Associate pose data
 */
int **assoc_pose_data(pose_t *gnd_poses,
                      size_t nb_gnd_poses,
                      pose_t *est_poses,
                      size_t nb_est_poses,
                      double threshold,
                      size_t *nb_matches) {
  assert(gnd_poses != NULL);
  assert(est_poses != NULL);
  assert(nb_gnd_poses != 0);
  assert(nb_est_poses != 0);

  size_t gnd_idx = 0;
  size_t est_idx = 0;
  size_t k_end = (nb_gnd_poses > nb_est_poses) ? nb_est_poses : nb_gnd_poses;

  size_t match_idx = 0;
  int **matches = MALLOC(int *, k_end);

  while ((gnd_idx + 1) < nb_gnd_poses && (est_idx + 1) < nb_est_poses) {
    // Calculate time difference between ground truth and estimate
    double gnd_k_time = ts2sec(gnd_poses[gnd_idx].ts);
    double est_k_time = ts2sec(est_poses[est_idx].ts);
    double t_k_diff = fabs(gnd_k_time - est_k_time);

    // Check to see if next ground truth timestamp forms a smaller time diff
    double t_kp1_diff = threshold;
    if ((gnd_idx + 1) < nb_gnd_poses) {
      double gnd_kp1_time = ts2sec(gnd_poses[gnd_idx + 1].ts);
      t_kp1_diff = fabs(gnd_kp1_time - est_k_time);
    }

    // Conditions to call this pair (ground truth and estimate) a match
    int threshold_met = t_k_diff < threshold;
    int smallest_diff = t_k_diff < t_kp1_diff;

    // Mark pairs as a match or increment appropriate indicies
    if (threshold_met && smallest_diff) {
      matches[match_idx] = MALLOC(int, 2);
      matches[match_idx][0] = gnd_idx;
      matches[match_idx][1] = est_idx;
      match_idx++;

      gnd_idx++;
      est_idx++;

    } else if (gnd_k_time > est_k_time) {
      est_idx++;

    } else if (gnd_k_time < est_k_time) {
      gnd_idx++;
    }
  }

  // Clean up
  if (match_idx == 0) {
    free(matches);
    matches = NULL;
  }

  *nb_matches = match_idx;
  return matches;
}

/******************************************************************************
 * SIMULATION
 ******************************************************************************/

//////////////////
// SIM FEATURES //
//////////////////

/**
 * Load simulation feature data
 */
sim_features_t *sim_features_load(const char *csv_path) {
  sim_features_t *features_data = MALLOC(sim_features_t, 1);
  int nb_rows = 0;
  int nb_cols = 0;
  features_data->features = csv_data(csv_path, &nb_rows, &nb_cols);
  features_data->nb_features = nb_rows;
  return features_data;
}

/**
 * Free simulation feature data
 */
void sim_features_free(sim_features_t *feature_data) {
  // Pre-check
  if (feature_data == NULL) {
    return;
  }

  // Free data
  for (int i = 0; i < feature_data->nb_features; i++) {
    free(feature_data->features[i]);
  }
  free(feature_data->features);
  free(feature_data);
}

//////////////////
// SIM IMU DATA //
//////////////////

/**
 * Load simulation imu data
 */
sim_imu_data_t *sim_imu_data_load(const char *csv_path) {
  sim_imu_data_t *imu_data = MALLOC(sim_imu_data_t, 1);

  int nb_rows = 0;
  int nb_cols = 0;
  imu_data->data = csv_data(csv_path, &nb_rows, &nb_cols);
  imu_data->nb_measurements = nb_rows;

  return imu_data;
}

/**
 * Free simulation imu data
 */
void sim_imu_data_free(sim_imu_data_t *imu_data) {
  // Pre-check
  if (imu_data == NULL) {
    return;
  }

  // Free data
  for (int i = 0; i < imu_data->nb_measurements; i++) {
    free(imu_data->data[i]);
  }
  free(imu_data->data);
  free(imu_data);
}

/////////////////////
// SIM CAMERA DATA //
/////////////////////

/**
 * Extract timestamp from path
 */
static timestamp_t ts_from_path(const char *path) {
  char fname[128] = {0};
  char fext[128] = {0};
  path_file_name(path, fname);
  path_file_ext(path, fext);

  char ts_str[128] = {0};
  memcpy(ts_str, fname, strlen(fname) - strlen(fext) - 1);

  char *ptr;
  return strtol(ts_str, &ptr, 10);
}

/**
 * Load simulated camera frame
 */
sim_camera_frame_t *sim_camera_frame_load(const char *csv_path) {
  // Check if file exists
  if (file_exists(csv_path) == 0) {
    return NULL;
  }

  // Load csv data
  int nb_rows = 0;
  int nb_cols = 0;
  real_t **data = csv_data(csv_path, &nb_rows, &nb_cols);

  // Create sim_camera_frame_t
  sim_camera_frame_t *frame_data = MALLOC(sim_camera_frame_t, 1);
  frame_data->ts = ts_from_path(csv_path);
  frame_data->feature_ids = MALLOC(int, nb_rows);
  frame_data->keypoints = MALLOC(real_t *, nb_rows);
  frame_data->nb_measurements = nb_rows;
  for (int i = 0; i < nb_rows; i++) {
    frame_data->feature_ids[i] = (int) data[i][0];
    frame_data->keypoints[i] = MALLOC(real_t, 2);
    frame_data->keypoints[i][0] = data[i][1];
    frame_data->keypoints[i][1] = data[i][2];
  }

  // Clean up
  csv_free(data, nb_rows);

  return frame_data;
}

/**
 * Print camera frame
 */
void sim_camera_frame_print(sim_camera_frame_t *frame_data) {
  printf("ts: %ld\n", frame_data->ts);
  printf("nb_frames: %d\n", frame_data->nb_measurements);
  for (int i = 0; i < frame_data->nb_measurements; i++) {
    const int feature_id = frame_data->feature_ids[i];
    const real_t *kp = frame_data->keypoints[i];
    printf("- ");
    printf("feature_id: [%d], ", feature_id);
    printf("kp: [%.2f, %.2f]\n", kp[0], kp[1]);
  }
  printf("\n");
}

/**
 * Free simulated camera frame
 */
void sim_camera_frame_free(sim_camera_frame_t *frame_data) {
  // Pre-check
  if (frame_data == NULL) {
    return;
  }

  // Free data
  free(frame_data->feature_ids);
  for (int i = 0; i < frame_data->nb_measurements; i++) {
    free(frame_data->keypoints[i]);
  }
  free(frame_data->keypoints);
  free(frame_data);
}

/**
 * Load simulated camera data
 */
sim_camera_data_t *sim_camera_data_load(const char *dir_path) {
  assert(dir_path != NULL);

  // Form csv file path
  char *csv_path = path_join(dir_path, "/data.csv");
  if (file_exists(csv_path) == 0) {
    free(csv_path);
    return NULL;
  }

  // Check number of rows
  const int nb_rows = dsv_rows(csv_path);
  if (nb_rows == 0) {
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
  sim_camera_data_t *cam_data = MALLOC(sim_camera_data_t, 1);
  cam_data->frames = MALLOC(sim_camera_frame_t *, nb_rows);
  cam_data->nb_frames = nb_rows;
  cam_data->ts = MALLOC(timestamp_t, nb_rows);
  cam_data->poses = MALLOC(real_t *, nb_rows);

  int line_idx = 0;
  char line[MAX_LINE_LENGTH] = {0};
  while (fgets(line, MAX_LINE_LENGTH, csv_file) != NULL) {
    // Skip line if its a comment
    if (line[0] == '#') {
      continue;
    }

    // Parse line
    timestamp_t ts;
    double r[3] = {0};
    double q[4] = {0};
    sscanf(line,
           "%ld,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
           &ts,
           &r[0],
           &r[1],
           &r[2],
           &q[0],
           &q[1],
           &q[2],
           &q[3]);

    // Add camera frame to sim_camera_frame_t
    char fname[128] = {0};
    sprintf(fname, "/data/%ld.csv", ts);
    char *frame_csv = path_join(dir_path, fname);
    cam_data->frames[line_idx] = sim_camera_frame_load(frame_csv);
    free(frame_csv);

    // Add pose to sim_camera_frame_t
    cam_data->ts[line_idx] = ts;
    cam_data->poses[line_idx] = MALLOC(real_t, 7);
    cam_data->poses[line_idx][0] = r[0];
    cam_data->poses[line_idx][1] = r[1];
    cam_data->poses[line_idx][2] = r[2];
    cam_data->poses[line_idx][3] = q[0];
    cam_data->poses[line_idx][4] = q[1];
    cam_data->poses[line_idx][5] = q[2];
    cam_data->poses[line_idx][6] = q[3];

    // Update
    line_idx++;
  }

  // Clean up
  free(csv_path);
  fclose(csv_file);

  return cam_data;
}

/**
 * Free simulated camera data
 */
void sim_camera_data_free(sim_camera_data_t *cam_data) {
  // Pre-check
  if (cam_data == NULL) {
    return;
  }

  // Free data
  for (int k = 0; k < cam_data->nb_frames; k++) {
    sim_camera_frame_free(cam_data->frames[k]);
    free(cam_data->poses[k]);
  }
  free(cam_data->frames);
  free(cam_data->ts);
  free(cam_data->poses);
  free(cam_data);
}

/**
 * Simulate 3D features
 */
real_t **sim_create_features(const real_t origin[3],
                             const real_t dim[3],
                             const int nb_features) {
  assert(origin != NULL);
  assert(dim != NULL);
  assert(nb_features > 0);

  // Setup
  const real_t w = dim[0];
  const real_t l = dim[1];
  const real_t h = dim[2];
  const int features_per_side = nb_features / 4.0;
  int feature_idx = 0;
  real_t **features = MALLOC(real_t *, nb_features);

  // Features in the east side
  {
    const real_t x_bounds[2] = {origin[0] - w, origin[0] + w};
    const real_t y_bounds[2] = {origin[1] + l, origin[1] + l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx] = MALLOC(real_t, 3);
      features[feature_idx][0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx][1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx][2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }

  // Features in the north side
  {
    const real_t x_bounds[2] = {origin[0] + w, origin[0] + w};
    const real_t y_bounds[2] = {origin[1] - l, origin[1] + l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx] = MALLOC(real_t, 3);
      features[feature_idx][0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx][1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx][2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }

  // Features in the west side
  {
    const real_t x_bounds[2] = {origin[0] - w, origin[0] + w};
    const real_t y_bounds[2] = {origin[1] - l, origin[1] - l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx] = MALLOC(real_t, 3);
      features[feature_idx][0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx][1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx][2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }

  // Features in the south side
  {
    const real_t x_bounds[2] = {origin[0] - w, origin[0] - w};
    const real_t y_bounds[2] = {origin[1] - l, origin[1] + l};
    const real_t z_bounds[2] = {origin[2] - h, origin[2] + h};
    for (int i = 0; i < features_per_side; i++) {
      features[feature_idx] = MALLOC(real_t, 3);
      features[feature_idx][0] = randf(x_bounds[0], x_bounds[1]);
      features[feature_idx][1] = randf(y_bounds[0], y_bounds[1]);
      features[feature_idx][2] = randf(z_bounds[0], z_bounds[1]);
      feature_idx++;
    }
  }

  return features;
}

// void sim_circle_trajectory() {
//   real_t circle_r = 1.0;
//   real_t circle_v = 1.0;
//   real_t cam_rate = 10.0;
//   real_t imu_rate = 200.0;
//   int nb_features = 1000;

//   // Trajectory data
//   real_t g[3] = {0.0, 0.0, 9.81};
//   real_t circle_dist = 2.0 * M_PI * circle_r;
//   real_t time_taken = circle_dist / circle_v;
//   real_t w = -2.0 * M_PI * (1.0 / time_taken);
//   real_t theta_init = M_PI;
//   real_t yaw_init = M_PI / 2.0;
// }

/////////////////////
// SIM GIMBAL DATA //
/////////////////////

/**
 * Malloc a simulated gimbal view
 */
sim_gimbal_view_t *sim_gimbal_view_malloc(const int max_corners) {
  sim_gimbal_view_t *view = MALLOC(sim_gimbal_view_t, 1);

  view->num_measurements = 0;
  view->tag_ids = MALLOC(int, max_corners);
  view->corner_indices = MALLOC(int, max_corners);
  view->object_points = MALLOC(real_t, max_corners * 3);
  view->keypoints = MALLOC(real_t, max_corners * 2);

  assert(view->tag_ids != NULL);
  assert(view->corner_indices != NULL);
  assert(view->object_points != NULL);
  assert(view->keypoints != NULL);

  return view;
}

/**
 * Free simulated gimbal view
 */
void sim_gimbal_view_free(sim_gimbal_view_t *view) {
  free(view->tag_ids);
  free(view->corner_indices);
  free(view->object_points);
  free(view->keypoints);
  free(view);
}

/**
 * Print simulated gimbal view
 */
void sim_gimbal_view_print(const sim_gimbal_view_t *view) {
  printf("num_measurements: %d\n", view->num_measurements);
  for (size_t i = 0; i < view->num_measurements; i++) {
    printf("%d ", view->tag_ids[i]);
    printf("%d ", view->corner_indices[i]);
    printf("(%.2f, %.2f, %.2f) ",
           view->object_points[i + 0],
           view->object_points[i + 1],
           view->object_points[i + 2]);
    printf("(%.2f, %.2f) ", view->keypoints[i + 0], view->keypoints[i + 1]);
    printf("\n");
  }
}

/**
 * Malloc gimbal simulation
 */
sim_gimbal_t *sim_gimbal_malloc() {
  sim_gimbal_t *sim = MALLOC(sim_gimbal_t, 1);

  // Aprilgrid
  aprilgrid_setup(0, &sim->grid);

  // Fiducial pose
  const real_t ypr_WF[3] = {-M_PI / 2.0, 0.0, M_PI / 2.0};
  const real_t r_WF[3] = {0.5, 0.0, 0.0};
  POSE_ER(ypr_WF, r_WF, fiducial_ext);
  extrinsic_setup(&sim->fiducial_ext, fiducial_ext);

  // Gimbal pose
  real_t x = 0.0;
  real_t y = 0.0;
  aprilgrid_center(&sim->grid, &x, &y);
  const real_t r_WB[3] = {0, -y, 0};
  const real_t ypr_WB[3] = {0, 0, 0};
  POSE_ER(ypr_WB, r_WB, gimbal_pose);
  pose_setup(&sim->gimbal_pose, 0, gimbal_pose);

  // Gimbal extrinsic (body to gimbal)
  const real_t ypr_BM0[3] = {0.01, 0.01, 0.01};
  const real_t r_BM0[3] = {0.001, 0.001, 0.001};
  POSE_ER(ypr_BM0, r_BM0, gimbal_ext);
  extrinsic_setup(&sim->gimbal_ext, gimbal_ext);

  // Links
  sim->num_links = 2;
  sim->gimbal_links = MALLOC(extrinsic_t, sim->num_links);
  // -- Roll link
  const real_t ypr_L0M1[3] = {0.0, M_PI / 2, 0.0};
  const real_t r_L0M1[3] = {-0.1, 0.0, 0.15};
  POSE_ER(ypr_L0M1, r_L0M1, link_roll);
  extrinsic_setup(&sim->gimbal_links[0], link_roll);
  // -- Pitch link
  const real_t ypr_L1M2[3] = {0.0, 0.0, -M_PI / 2.0};
  const real_t r_L1M2[3] = {0.0, -0.05, 0.1};
  POSE_ER(ypr_L1M2, r_L1M2, link_pitch);
  extrinsic_setup(&sim->gimbal_links[1], link_pitch);

  // Joints
  sim->num_joints = 3;
  sim->gimbal_joints = MALLOC(joint_t, sim->num_joints);
  joint_setup(&sim->gimbal_joints[0], 0, 0, 0.0);
  joint_setup(&sim->gimbal_joints[1], 0, 1, 0.0);
  joint_setup(&sim->gimbal_joints[2], 0, 2, 0.0);

  // Setup cameras
  sim->num_cams = 2;
  // -- Camera extrinsic
  sim->cam_exts = MALLOC(extrinsic_t, sim->num_cams);
  // ---- cam0 extrinsic
  const real_t ypr_M2eC0[3] = {-M_PI / 2.0, M_PI / 2.0, 0.0};
  const real_t r_M2eC0[3] = {0.0, -0.05, 0.12};
  POSE_ER(ypr_M2eC0, r_M2eC0, cam0_exts);
  extrinsic_setup(&sim->cam_exts[0], cam0_exts);
  // ---- cam1 extrinsic
  const real_t ypr_M2eC1[3] = {-M_PI / 2.0, M_PI / 2.0, 0.0};
  const real_t r_M2eC1[3] = {0.0, -0.05, -0.12};
  POSE_ER(ypr_M2eC1, r_M2eC1, cam1_exts);
  extrinsic_setup(&sim->cam_exts[1], cam1_exts);
  // -- Camera parameters
  const int cam_res[2] = {640, 480};
  const real_t fov = 120.0;
  const real_t fx = pinhole_focal(cam_res[0], fov);
  const real_t fy = pinhole_focal(cam_res[0], fov);
  const real_t cx = cam_res[0] / 2.0;
  const real_t cy = cam_res[1] / 2.0;
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  const real_t data[8] = {fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0};
  sim->cam_params = MALLOC(camera_params_t, sim->num_cams);
  camera_params_setup(&sim->cam_params[0],
                      0,
                      cam_res,
                      proj_model,
                      dist_model,
                      data);
  camera_params_setup(&sim->cam_params[1],
                      1,
                      cam_res,
                      proj_model,
                      dist_model,
                      data);

  return sim;
}

/**
 * Free gimbal simulation
 */
void sim_gimbal_free(sim_gimbal_t *sim) {
  if (sim == NULL) {
    return;
  }

  FREE(sim->gimbal_links);
  FREE(sim->gimbal_joints);
  FREE(sim->cam_exts);
  FREE(sim->cam_params);
  FREE(sim);
}

/**
 * Print gimbal simulation
 */
void sim_gimbal_print(const sim_gimbal_t *sim) {
  // Configuration file
  for (int cam_idx = 0; cam_idx < sim->num_cams; cam_idx++) {
    camera_params_print(&sim->cam_params[cam_idx]);
    printf("\n");
  }
  for (int cam_idx = 0; cam_idx < sim->num_cams; cam_idx++) {
    char cam_str[20] = {0};
    sprintf(cam_str, "cam%d_ext", cam_idx);
    extrinsic_print(cam_str, &sim->cam_exts[cam_idx]);
  }
  for (int link_idx = 0; link_idx < sim->num_links; link_idx++) {
    char link_str[20] = {0};
    sprintf(link_str, "link%d_ext", link_idx);
    extrinsic_print(link_str, &sim->gimbal_links[link_idx]);
  }
  extrinsic_print("gimbal_ext", &sim->gimbal_ext);
  extrinsic_print("fiducial_ext", &sim->fiducial_ext);
}

/**
 * Set gimbal joint
 */
void sim_gimbal_set_joint(sim_gimbal_t *sim,
                          const int joint_idx,
                          const real_t angle) {
  sim->gimbal_joints[joint_idx].data[0] = angle;
}

/**
 * Get gimbal joint
 */
void sim_gimbal_get_joints(sim_gimbal_t *sim,
                           const int num_joints,
                           real_t *angles) {
  for (int i = 0; i < num_joints; i++) {
    angles[i] = sim->gimbal_joints[i].data[0];
  }
}

/**
 * Simulate 3-axis gimbal view
 */
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
                                    const real_t cam_ext[7]) {
  // Form: T_CiF
  TF(fiducial_pose, T_WF);
  TF(body_pose, T_WB);
  TF(gimbal_ext, T_BM0);
  TF(gimbal_link0, T_L0M1);
  TF(gimbal_link1, T_L1M2);
  GIMBAL_JOINT_TF(gimbal_joint0, T_M0L0);
  GIMBAL_JOINT_TF(gimbal_joint1, T_M1L1);
  GIMBAL_JOINT_TF(gimbal_joint2, T_M2L2);
  TF(cam_ext, T_L2Ci);
  TF_CHAIN(T_BCi, 7, T_BM0, T_M0L0, T_L0M1, T_M1L1, T_L1M2, T_M2L2, T_L2Ci);
  TF_INV(T_BCi, T_CiB);
  TF_INV(T_WB, T_BW);
  TF_CHAIN(T_CiF, 3, T_CiB, T_BW, T_WF);

  const int max_tags = grid->num_rows * grid->num_cols;
  const int max_corners = max_tags * 4;
  sim_gimbal_view_t *view = sim_gimbal_view_malloc(max_corners);

  for (int tag_id = 0; tag_id < max_tags; tag_id++) {
    for (int corner_idx = 0; corner_idx < 4; corner_idx++) {
      // Transform fiducial point to camera frame
      real_t p_FFi[3] = {0};
      aprilgrid_object_point(grid, tag_id, corner_idx, p_FFi);
      TF_POINT(T_CiF, p_FFi, p_CiFi);

      // Check point is infront of camera
      if (p_CiFi[2] < 0) {
        continue;
      }

      // Project image point to image plane
      real_t z[2] = {0};
      pinhole_radtan4_project(cam_params, p_CiFi, z);

      // Check projection
      const int x_ok = (z[0] < cam_res[0] && z[0] > 0);
      const int y_ok = (z[1] < cam_res[1] && z[1] > 0);
      if (x_ok == 0 || y_ok == 0) {
        continue;
      }

      // Add to measurements
      view->tag_ids[view->num_measurements] = tag_id;
      view->corner_indices[view->num_measurements] = corner_idx;
      view->object_points[view->num_measurements * 3] = p_FFi[0];
      view->object_points[view->num_measurements * 3 + 1] = p_FFi[1];
      view->object_points[view->num_measurements * 3 + 2] = p_FFi[2];
      view->keypoints[view->num_measurements * 2] = z[0];
      view->keypoints[view->num_measurements * 2 + 1] = z[1];
      view->num_measurements++;
    }
  }

  return view;
}

/**
 * Simulate 3-axis gimbal view
 */
sim_gimbal_view_t *sim_gimbal_view(const sim_gimbal_t *sim,
                                   const timestamp_t ts,
                                   const int view_idx,
                                   const int cam_idx,
                                   const real_t body_pose[7]) {

  return sim_gimbal3_view(&sim->grid,
                          ts,
                          view_idx,
                          sim->fiducial_ext.data,
                          body_pose,
                          sim->gimbal_ext.data,
                          sim->gimbal_links[0].data,
                          sim->gimbal_links[1].data,
                          sim->gimbal_joints[0].data[0],
                          sim->gimbal_joints[1].data[0],
                          sim->gimbal_joints[2].data[0],
                          cam_idx,
                          sim->cam_params->resolution,
                          sim->cam_params->data,
                          sim->cam_exts[cam_idx].data);
}
