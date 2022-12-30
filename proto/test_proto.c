#include "proto.h"
#include "munit.h"

/* TEST PARAMS */
#define TEST_DATA_PATH "./test_data/"
#define TEST_CSV TEST_DATA_PATH "test_csv.csv"
#define TEST_POSES_CSV TEST_DATA_PATH "poses.csv"
#define TEST_SIM_DATA TEST_DATA_PATH "sim_data"

/******************************************************************************
 * TEST MACROS
 ******************************************************************************/

int test_median_value() {
  real_t median = 0.0f;
  real_t buf[5] = {4.0, 1.0, 0.0, 3.0, 2.0};
  MEDIAN_VALUE(real_t, fltcmp2, buf, 5, median);
  MU_ASSERT(fltcmp(median, 2.0) == 0);

  return 0;
}

int test_mean_value() {
  real_t mean = 0.0f;
  real_t buf[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
  MEAN_VALUE(real_t, buf, 5, mean);
  MU_ASSERT(fltcmp(mean, 2.0) == 0);

  return 0;
}

/******************************************************************************
 * TEST FILESYSTEM
 ******************************************************************************/

int test_path_file_name() {
  const char *path = "/tmp/hello_world.csv";
  char fname[128] = {0};
  path_file_name(path, fname);
  MU_ASSERT(strcmp(fname, "hello_world.csv") == 0);

  return 0;
}

int test_path_file_ext() {
  const char *path = "/tmp/hello_world.csv";
  char fext[128] = {0};
  path_file_ext(path, fext);
  MU_ASSERT(strcmp(fext, "csv") == 0);

  return 0;
}

int test_path_dir_name() {
  const char *path = "/tmp/hello_world.csv";
  char dir_name[128] = {0};
  path_dir_name(path, dir_name);
  MU_ASSERT(strcmp(dir_name, "/tmp") == 0);

  return 0;
}

int test_path_join() {
  // Case A
  {
    const char *path_a = "/tmp/A";
    const char *path_b = "B.csv";
    char *c = path_join(path_a, path_b);
    MU_ASSERT(strcmp(c, "/tmp/A/B.csv") == 0);
    free(c);
  }

  // Case B
  {
    const char *path_a = "/tmp/A/";
    const char *path_b = "B.csv";
    char *c = path_join(path_a, path_b);
    MU_ASSERT(strcmp(c, "/tmp/A/B.csv") == 0);
    free(c);
  }

  return 0;
}

int test_list_files() {
  int nb_files = 0;
  char **files = list_files("/tmp", &nb_files);
  MU_ASSERT(files != NULL);
  MU_ASSERT(nb_files != 0);

  /* printf("nb_files: %d\n", nb_files); */
  for (int i = 0; i < nb_files; i++) {
    /* printf("file: %s\n", files[i]); */
    free(files[i]);
  }
  free(files);

  return 0;
}

int test_list_files_free() {
  int nb_files = 0;
  char **files = list_files("/tmp", &nb_files);
  list_files_free(files, nb_files);

  return 0;
}

int test_file_read() {
  char *text = file_read("test_data/poses.csv");
  /* printf("%s\n", text); */
  MU_ASSERT(text != NULL);
  free(text);

  return 0;
}

int test_skip_line() {
  FILE *fp = fopen("test_data/poses.csv", "r");
  skip_line(fp);
  fclose(fp);

  return 0;
}

int test_file_rows() {
  int nb_rows = file_rows("test_data/poses.csv");
  MU_ASSERT(nb_rows > 0);
  return 0;
}

int test_file_copy() {
  int retval = file_copy("test_data/poses.csv", "/tmp/poses.csv");
  char *text0 = file_read("test_data/poses.csv");
  char *text1 = file_read("/tmp/poses.csv");
  MU_ASSERT(retval == 0);
  MU_ASSERT(strcmp(text0, text1) == 0);
  free(text0);
  free(text1);

  return 0;
}

/******************************************************************************
 * TEST DATA
 ******************************************************************************/

int test_string_malloc() {
  char *s = string_malloc("hello world!");
  MU_ASSERT(strcmp(s, "hello world!") == 0);
  free(s);
  return 0;
}

int test_dsv_rows() {
  int nb_rows = dsv_rows(TEST_CSV);
  MU_ASSERT(nb_rows == 10);
  return 0;
}

int test_dsv_cols() {
  int nb_cols = dsv_cols(TEST_CSV, ',');
  MU_ASSERT(nb_cols == 10);
  return 0;
}

int test_dsv_fields() {
  int nb_fields = 0;
  char **fields = dsv_fields(TEST_CSV, ',', &nb_fields);
  const char *expected[10] = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"};
  if (fields == NULL) {
    printf("File not found [%s]\n", TEST_CSV);
    return -1;
  }

  MU_ASSERT(nb_fields == 10);
  for (int i = 0; i < nb_fields; i++) {
    MU_ASSERT(strcmp(fields[i], expected[i]) == 0);
    free(fields[i]);
  }
  free(fields);

  return 0;
}

int test_dsv_data() {
  int nb_rows = 0;
  int nb_cols = 0;
  real_t **data = dsv_data(TEST_CSV, ',', &nb_rows, &nb_cols);

  int index = 0;
  for (int i = 0; i < nb_rows; i++) {
    for (int j = 0; j < nb_rows; j++) {
      MU_ASSERT(fltcmp(data[i][j], index + 1) == 0);
      index++;
    }
  }
  dsv_free(data, nb_rows);

  return 0;
}

int test_dsv_free() {
  int nb_rows = 0;
  int nb_cols = 0;
  real_t **data = dsv_data(TEST_CSV, ',', &nb_rows, &nb_cols);
  dsv_free(data, nb_rows);

  return 0;
}

int test_csv_data() {
  int nb_rows = 0;
  int nb_cols = 0;
  real_t **data = csv_data(TEST_CSV, &nb_rows, &nb_cols);
  csv_free(data, nb_rows);

  return 0;
}

/******************************************************************************
 * TEST DATA-STRUCTURE
 ******************************************************************************/

// DARRAY ////////////////////////////////////////////////////////////////////

int test_darray_new_and_destroy(void) {
  darray_t *array = darray_new(sizeof(int), 100);

  MU_ASSERT(array != NULL);
  MU_ASSERT(array->contents != NULL);
  MU_ASSERT(array->end == 0);
  MU_ASSERT(array->element_size == sizeof(int));
  MU_ASSERT(array->max == 100);

  darray_destroy(array);
  return 0;
}

int test_darray_push_pop(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* test push */
  for (int i = 0; i < 1000; i++) {
    int *val = darray_new_element(test_darray);
    *val = i * 333;
    darray_push(test_darray, val);
  }
  MU_ASSERT(test_darray->max == 1300);

  /* test pop */
  for (int i = 999; i >= 0; i--) {
    int *val = darray_pop(test_darray);
    MU_ASSERT(val != NULL);
    MU_ASSERT(*val == i * 333);
    free(val);
  }

  darray_clear_destroy(test_darray);
  return 0;
}

int test_darray_contains(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* set element in array */
  int *val = darray_new_element(test_darray);
  *val = 99;
  darray_set(test_darray, 0, val);

  /* test contains */
  int res = darray_contains(test_darray, val, intcmp2);
  MU_ASSERT(res == 1);

  darray_clear_destroy(test_darray);
  return 0;
}

int test_darray_copy(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* set element in array */
  int *val = darray_new_element(test_darray);
  *val = 99;
  darray_set(test_darray, 0, val);

  /* test copy */
  darray_t *array_copy = darray_copy(test_darray);
  int *val_copy = darray_get(array_copy, 0);
  MU_ASSERT(val != val_copy);
  MU_ASSERT(intcmp2(val, val_copy) == 0);

  darray_clear_destroy(test_darray);
  darray_clear_destroy(array_copy);
  return 0;
}

int test_darray_new_element(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* test new */
  int *val1 = darray_new_element(test_darray);
  int *val2 = darray_new_element(test_darray);

  MU_ASSERT(val1 != NULL);
  MU_ASSERT(val2 != NULL);

  free(val1);
  free(val2);

  darray_clear_destroy(test_darray);
  return 0;
}

int test_darray_set_and_get(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* test set element */
  int *val1 = darray_new_element(test_darray);
  int *val2 = darray_new_element(test_darray);
  darray_set(test_darray, 0, val1);
  darray_set(test_darray, 1, val2);

  /* test get element */
  MU_ASSERT(darray_get(test_darray, 0) == val1);
  MU_ASSERT(darray_get(test_darray, 1) == val2);

  darray_clear_destroy(test_darray);
  return 0;
}

int test_darray_update(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* set element */
  int *new_val1 = darray_new_element(test_darray);
  int *new_val2 = darray_new_element(test_darray);
  *new_val1 = 123;
  *new_val2 = 987;

  /* update */
  darray_update(test_darray, 0, new_val1);
  darray_update(test_darray, 1, new_val2);

  /* assert */
  MU_ASSERT(darray_get(test_darray, 0) == new_val1);
  MU_ASSERT(darray_get(test_darray, 1) == new_val2);

  darray_clear_destroy(test_darray);
  return 0;
}

int test_darray_remove(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* set elements */
  int *val_1 = darray_new_element(test_darray);
  int *val_2 = darray_new_element(test_darray);
  *val_1 = 123;
  *val_2 = 987;
  darray_set(test_darray, 0, val_1);
  darray_set(test_darray, 1, val_2);

  /* remove element at index = 0 */
  int *result = darray_remove(test_darray, 0);
  MU_ASSERT(result != NULL);
  MU_ASSERT(*result == *val_1);
  MU_ASSERT(darray_get(test_darray, 0) == NULL);
  free(result);

  /* remove element at index = 1 */
  result = darray_remove(test_darray, 1);
  MU_ASSERT(result != NULL);
  MU_ASSERT(*result == *val_2);
  MU_ASSERT(darray_get(test_darray, 1) == NULL);
  free(result);

  darray_clear_destroy(test_darray);
  return 0;
}

int test_darray_expand_and_contract(void) {
  darray_t *test_darray = darray_new(sizeof(int), 100);

  /* test expand */
  size_t old_max = (unsigned int) test_darray->max;
  darray_expand(test_darray);
  MU_ASSERT((unsigned int) test_darray->max ==
            old_max + test_darray->expand_rate);

  /* test contract */
  darray_contract(test_darray);
  MU_ASSERT((unsigned int) test_darray->max == test_darray->expand_rate + 1);

  darray_clear_destroy(test_darray);
  return 0;
}

// LIST //////////////////////////////////////////////////////////////////////

int test_list_new_and_destroy(void) {
  list_t *list = list_new();
  MU_ASSERT(list != NULL);
  list_clear_destroy(list);
  return 0;
}

int test_list_push_pop(void) {
  /* Setup */
  list_t *list = list_new();
  char *t1 = string_malloc("test1 data");
  char *t2 = string_malloc("test2 data");
  char *t3 = string_malloc("test3 data");

  /* Push tests */
  list_push(list, t1);
  MU_ASSERT(strcmp(list->last->value, t1) == 0);

  list_push(list, t2);
  MU_ASSERT(strcmp(list->last->value, t2) == 0);

  list_push(list, t3);
  MU_ASSERT(strcmp(list->last->value, t3) == 0);
  MU_ASSERT(list->length == 3);

  /* Pop tests */
  char *val = list_pop(list);
  MU_ASSERT(val == t3);
  MU_ASSERT(list->first->value == t1);
  MU_ASSERT(list->last->value == t2);
  MU_ASSERT(list->length == 2);
  free(val);

  val = list_pop(list);
  MU_ASSERT(val == t2);
  MU_ASSERT(list->first->value == t1);
  MU_ASSERT(list->last->value == t1);
  MU_ASSERT(list->length == 1);
  free(val);

  val = list_pop(list);
  MU_ASSERT(val == t1);
  MU_ASSERT(list->first == NULL);
  MU_ASSERT(list->last == NULL);
  MU_ASSERT(list->length == 0);
  free(val);

  list_clear_destroy(list);
  return 0;
}

int test_list_shift(void) {
  /* Setup */
  list_t *list = list_new();
  char *t1 = string_malloc("test1 data");
  char *t2 = string_malloc("test2 data");

  /* Push elements */
  list_push(list, t1);
  list_push(list, t2);

  /* Shift */
  char *val = list_shift(list);
  MU_ASSERT(val == t1);
  MU_ASSERT(list->length == 1);
  free(val);

  val = list_shift(list);
  MU_ASSERT(val == t2);
  MU_ASSERT(list->length == 0);
  free(val);

  list_clear_destroy(list);
  return 0;
}

int test_list_unshift(void) {
  /* Setup */
  list_t *list = list_new();
  char *t1 = string_malloc("test1 data");
  char *t2 = string_malloc("test2 data");
  char *t3 = string_malloc("test3 data");

  /* Unshift */
  list_unshift(list, t1);
  MU_ASSERT(strcmp(list->first->value, t1) == 0);
  MU_ASSERT(strcmp(list->first->value, t1) == 0);
  MU_ASSERT(list->length == 1);

  list_unshift(list, t2);
  MU_ASSERT(strcmp(list->first->value, t2) == 0);
  MU_ASSERT(strcmp(list->first->value, t2) == 0);
  MU_ASSERT(list->length == 2);

  list_unshift(list, t3);
  MU_ASSERT(strcmp(list->first->value, t3) == 0);
  MU_ASSERT(strcmp(list->first->value, t3) == 0);
  MU_ASSERT(list->length == 3);
  list_clear_destroy(list);

  return 0;
}

int test_list_remove(void) {
  /* Push elements */
  list_t *list = list_new();
  char *t1 = string_malloc("test1 data");
  char *t2 = string_malloc("test2 data");
  char *t3 = string_malloc("test3 data");
  list_push(list, t1);
  list_push(list, t2);
  list_push(list, t3);

  /* Remove 2nd value */
  void *value = list_remove(list, t2, strcmp2);
  free(value);

  /* Assert */
  MU_ASSERT(list->length == 2);
  MU_ASSERT(strcmp(list->first->next->value, t3) == 0);
  MU_ASSERT(strcmp(list->first->value, t1) == 0);

  /* Remove 2nd value */
  value = list_remove(list, t3, strcmp2);
  free(value);

  /* Assert */
  MU_ASSERT(list->length == 1);
  MU_ASSERT(list->first->next == NULL);
  MU_ASSERT(strcmp(list->first->value, t1) == 0);
  list_clear_destroy(list);

  return 0;
}

int test_list_remove_destroy(void) {
  /* Setup */
  list_t *list = list_new();
  char *t1 = string_malloc("test1 data");
  char *t2 = string_malloc("test2 data");
  char *t3 = string_malloc("test3 data");

  /* Push elements */
  list_push(list, t1);
  list_push(list, t2);
  list_push(list, t3);

  /* Remove 2nd value */
  int result = list_remove_destroy(list, t2, strcmp2, free);

  /* Assert */
  MU_ASSERT(result == 0);
  MU_ASSERT(list->length == 2);
  MU_ASSERT(strcmp(list->first->next->value, t3) == 0);
  MU_ASSERT(strcmp(list->first->value, t1) == 0);

  /* Remove 2nd value */
  result = list_remove_destroy(list, t3, strcmp2, free);

  /* Assert */
  MU_ASSERT(result == 0);
  MU_ASSERT(list->length == 1);
  MU_ASSERT(list->first->next == NULL);
  MU_ASSERT(strcmp(list->first->value, t1) == 0);
  list_clear_destroy(list);

  return 0;
}

// STACK /////////////////////////////////////////////////////////////////////

int test_mstack_new_and_destroy(void) {
  mstack_t *s = stack_new();

  MU_ASSERT(s->size == 0);
  MU_ASSERT(s->root == NULL);
  MU_ASSERT(s->end == NULL);

  mstack_destroy(s);
  return 0;
}

int test_mstack_push(void) {
  mstack_t *s = stack_new();
  float f1 = 2.0;
  float f2 = 4.0;
  float f3 = 8.0;

  /* push float 1 */
  mstack_push(s, &f1);
  MU_ASSERT(fltcmp(*(float *) s->end->value, *(float *) &f1) == 0);
  MU_ASSERT(s->size == 1);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(s->end->prev == NULL);

  /* push float 2 */
  mstack_push(s, &f2);
  MU_ASSERT(fltcmp(*(float *) s->end->value, *(float *) &f2) == 0);
  MU_ASSERT(s->size == 2);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(s->end->prev->value == &f1);
  MU_ASSERT(fltcmp(*(float *) s->end->prev->value, *(float *) &f1) == 0);

  /* push float 3 */
  mstack_push(s, &f3);
  MU_ASSERT(fltcmp(*(float *) s->end->value, *(float *) &f3) == 0);
  MU_ASSERT(s->size == 3);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(s->end->prev->value == &f2);
  MU_ASSERT(fltcmp(*(float *) s->end->prev->value, *(float *) &f2) == 0);

  mstack_destroy(s);
  return 0;
}

int test_mstack_pop(void) {
  mstack_t *s = stack_new();
  float f1 = 2.0;
  float f2 = 4.0;
  float f3 = 8.0;

  /* push float 1 */
  mstack_push(s, &f1);
  MU_ASSERT(fltcmp(*(float *) s->end->value, *(float *) &f1) == 0);
  MU_ASSERT(s->size == 1);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(s->end->prev == NULL);

  /* push float 2 */
  mstack_push(s, &f2);
  MU_ASSERT(fltcmp(*(float *) s->end->value, *(float *) &f2) == 0);
  MU_ASSERT(s->size == 2);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(s->end->prev->value == &f1);
  MU_ASSERT(fltcmp(*(float *) s->end->prev->value, *(float *) &f1) == 0);

  /* push float 3 */
  mstack_push(s, &f3);
  MU_ASSERT(fltcmp(*(float *) s->end->value, *(float *) &f3) == 0);
  MU_ASSERT(s->size == 3);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(s->end->prev->value == &f2);
  MU_ASSERT(fltcmp(*(float *) s->end->prev->value, *(float *) &f2) == 0);

  /* pop float 3 */
  float *flt_ptr = mstack_pop(s);
  MU_ASSERT(fltcmp(*(float *) flt_ptr, *(float *) &f3) == 0);
  MU_ASSERT(s->size == 2);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(fltcmp(*(float *) s->root->value, *(float *) &f1) == 0);

  /* pop float 2 */
  flt_ptr = mstack_pop(s);
  MU_ASSERT(fltcmp(*(float *) flt_ptr, *(float *) &f2) == 0);
  MU_ASSERT(s->size == 1);
  MU_ASSERT(s->root->value == &f1);
  MU_ASSERT(fltcmp(*(float *) s->root->value, *(float *) &f1) == 0);

  /* pop float 1 */
  flt_ptr = mstack_pop(s);
  MU_ASSERT(fltcmp(*(float *) flt_ptr, *(float *) &f1) == 0);
  MU_ASSERT(s->size == 0);
  MU_ASSERT(s->root == NULL);
  MU_ASSERT(s->end == NULL);

  mstack_destroy(s);
  return 0;
}

// QUEUE /////////////////////////////////////////////////////////////////////

int test_queue_new_and_destroy(void) {
  queue_t *q = queue_new();
  MU_ASSERT(q != NULL);
  MU_ASSERT(q->count == 0);
  queue_destroy(q);

  return 0;
}

int test_queue_enqueue_dequeue(void) {
  queue_t *q = queue_new();
  char *t1 = "test1 data";
  char *t2 = "test2 data";
  char *t3 = "test3 data";

  /* Enqueue tests */
  queue_enqueue(q, t1);
  MU_ASSERT(queue_first(q) == t1);
  MU_ASSERT(queue_last(q) == t1);
  MU_ASSERT(q->count == 1);

  queue_enqueue(q, t2);
  MU_ASSERT(queue_first(q) == t1);
  MU_ASSERT(queue_last(q) == t2);
  MU_ASSERT(q->count == 2);

  queue_enqueue(q, t3);
  MU_ASSERT(queue_first(q) == t1);
  MU_ASSERT(queue_last(q) == t3);
  MU_ASSERT(q->count == 3);

  /* Dequeue tests */
  char *val = queue_dequeue(q);
  MU_ASSERT(val == t1);
  MU_ASSERT(queue_first(q) == t2);
  MU_ASSERT(queue_last(q) == t3);
  MU_ASSERT(q->count == 2);

  val = queue_dequeue(q);
  MU_ASSERT(val == t2);
  MU_ASSERT(queue_first(q) == t3);
  MU_ASSERT(queue_last(q) == t3);
  MU_ASSERT(q->count == 1);

  val = queue_dequeue(q);
  MU_ASSERT(val == t3);
  MU_ASSERT(queue_first(q) == NULL);
  MU_ASSERT(queue_last(q) == NULL);
  MU_ASSERT(q->count == 0);

  // Clean up
  queue_destroy(q);

  return 0;
}

// HASHMAP ///////////////////////////////////////////////////////////////////

static int traverse_called;

static int traverse_good_cb(hashmap_node_t *node) {
  UNUSED(node);
  traverse_called++;
  return 0;
}

static int traverse_fail_cb(hashmap_node_t *node) {
  UNUSED(node);
  traverse_called++;
  if (traverse_called == 2) {
    return 1;
  } else {
    return 0;
  }
}

hashmap_t *hashmap_test_setup(void) {
  hashmap_t *map;
  char *test1;
  char *test2;
  char *test3;
  char *expect1;
  char *expect2;
  char *expect3;

  /* setup */
  map = hashmap_new();

  /* key and values */
  test1 = "test data 1";
  test2 = "test data 2";
  test3 = "xest data 3";
  expect1 = "THE VALUE 1";
  expect2 = "THE VALUE 2";
  expect3 = "THE VALUE 3";

  /* set */
  hashmap_set(map, test1, expect1);
  hashmap_set(map, test2, expect2);
  hashmap_set(map, test3, expect3);

  return map;
}

void hashmap_test_teardown(hashmap_t *map) {
  hashmap_destroy(map);
}

int test_hashmap_new_destroy(void) {
  hashmap_t *map;

  map = hashmap_new();
  MU_ASSERT(map != NULL);
  hashmap_destroy(map);

  return 0;
}

int test_hashmap_clear_destroy(void) {
  hashmap_t *map;

  map = hashmap_new();
  hashmap_set(map, "test", "hello");
  hashmap_clear_destroy(map);

  return 0;
}

int test_hashmap_get_set(void) {
  /* Setup */
  int rc;
  char *result;

  hashmap_t *map = hashmap_test_setup();
  char *test1 = "test data 1";
  char *test2 = "test data 2";
  char *test3 = "xest data 3";
  char *expect1 = "THE VALUE 1";
  char *expect2 = "THE VALUE 2";
  char *expect3 = "THE VALUE 3";

  /* Set and get test1 */
  rc = hashmap_set(map, test1, expect1);
  MU_ASSERT(rc == 0);
  result = hashmap_get(map, test1);
  MU_ASSERT(strcmp(result, expect1) == 0);

  /* Set and get test2 */
  rc = hashmap_set(map, test2, expect2);
  MU_ASSERT(rc == 0);
  result = hashmap_get(map, test2);
  MU_ASSERT(strcmp(result, expect2) == 0);

  /* Set and get test3 */
  rc = hashmap_set(map, test3, expect3);
  MU_ASSERT(rc == 0);
  result = hashmap_get(map, test3);
  MU_ASSERT(strcmp(result, expect3) == 0);

  /* Clean up */
  hashmap_test_teardown(map);

  return 0;
}

int test_hashmap_delete(void) {
  /* Setup */
  char *deleted = NULL;
  char *result = NULL;

  hashmap_t *map = hashmap_test_setup();
  char *test1 = "test data 1";
  char *test2 = "test data 2";
  char *test3 = "xest data 3";
  char *expect1 = "THE VALUE 1";
  char *expect2 = "THE VALUE 2";
  char *expect3 = "THE VALUE 3";

  /* Delete test1 */
  deleted = hashmap_delete(map, test1);
  MU_ASSERT(deleted != NULL);
  MU_ASSERT(strcmp(deleted, expect1) == 0);
  free(deleted);

  result = hashmap_get(map, test1);
  MU_ASSERT(result == NULL);

  /* Delete test2 */
  deleted = hashmap_delete(map, test2);
  MU_ASSERT(deleted != NULL);
  MU_ASSERT(strcmp(deleted, expect2) == 0);
  free(deleted);

  result = hashmap_get(map, test2);
  MU_ASSERT(result == NULL);

  /* Delete test3 */
  deleted = hashmap_delete(map, test3);
  MU_ASSERT(deleted != NULL);
  MU_ASSERT(strcmp(deleted, expect3) == 0);
  free(deleted);

  result = hashmap_get(map, test3);
  MU_ASSERT(result == NULL);

  /* Clean up */
  hashmap_test_teardown(map);

  return 0;
}

int test_hashmap_traverse(void) {
  int retval;
  hashmap_t *map;

  /* setup */
  map = hashmap_test_setup();

  /* traverse good cb */
  traverse_called = 0;
  retval = hashmap_traverse(map, traverse_good_cb);
  MU_ASSERT(retval == 0);
  MU_ASSERT(traverse_called == 3);

  /* traverse good bad */
  traverse_called = 0;
  retval = hashmap_traverse(map, traverse_fail_cb);
  MU_ASSERT(retval == 1);
  MU_ASSERT(traverse_called == 2);

  /* clean up */
  hashmap_test_teardown(map);

  return 0;
}

/******************************************************************************
 * TEST TIME
 ******************************************************************************/

int test_tic_toc() {
  struct timespec t_start = tic();
  sleep(1.0);
  MU_ASSERT(fabs(toc(&t_start) - 1.0) < 1e-2);
  return 0;
}

int test_mtoc() {
  struct timespec t_start = tic();
  sleep(1.0);
  MU_ASSERT(fabs(mtoc(&t_start) - 1000) < 1);
  return 0;
}

int test_time_now() {
  timestamp_t t_now = time_now();
  // printf("t_now: %ld\n", t_now);
  MU_ASSERT(t_now > 0);
  return 0;
}

/******************************************************************************
 * TEST NETWORK
 ******************************************************************************/

int test_tcp_server_setup() {
  tcp_server_t server;
  const int port = 8080;
  tcp_server_setup(&server, port);
  return 0;
}

/******************************************************************************
 * TEST MATHS
 ******************************************************************************/

int test_min() {
  MU_ASSERT(PMIN(1, 2) == 1);
  MU_ASSERT(PMIN(2, 1) == 1);
  return 0;
}

int test_max() {
  MU_ASSERT(PMAX(1, 2) == 2);
  MU_ASSERT(PMAX(2, 1) == 2);
  return 0;
}

int test_randf() {
  const real_t val = randf(0.0, 10.0);
  MU_ASSERT(val < 10.0);
  MU_ASSERT(val > 0.0);
  return 0;
}

int test_deg2rad() {
  MU_ASSERT(fltcmp(deg2rad(180.0f), M_PI) == 0);
  return 0;
}

int test_rad2deg() {
  MU_ASSERT(fltcmp(rad2deg(M_PI), 180.0f) == 0);
  return 0;
}

int test_fltcmp() {
  MU_ASSERT(fltcmp(1.0, 1.0) == 0);
  MU_ASSERT(fltcmp(1.0, 1.01) != 0);
  return 0;
}

int test_fltcmp2() {
  const real_t x = 1.0f;
  const real_t y = 1.0f;
  const real_t z = 1.01f;
  MU_ASSERT(fltcmp2(&x, &y) == 0);
  MU_ASSERT(fltcmp2(&x, &z) != 0);
  return 0;
}

int test_pythag() {
  MU_ASSERT(fltcmp(pythag(3.0, 4.0), 5.0) == 0);
  return 0;
}

int test_lerp() {
  MU_ASSERT(fltcmp(lerp(0.0, 1.0, 0.5), 0.5) == 0);
  MU_ASSERT(fltcmp(lerp(0.0, 10.0, 0.8), 8.0) == 0);
  return 0;
}

int test_lerp3() {
  real_t a[3] = {0.0, 1.0, 2.0};
  real_t b[3] = {1.0, 2.0, 3.0};
  real_t c[3] = {0.0, 0.0, 0.0};
  real_t t = 0.5;

  lerp3(a, b, t, c);
  MU_ASSERT(fltcmp(c[0], 0.5) == 0);
  MU_ASSERT(fltcmp(c[1], 1.5) == 0);
  MU_ASSERT(fltcmp(c[2], 2.5) == 0);

  return 0;
}

int test_sinc() {
  return 0;
}

int test_mean() {
  real_t vals[4] = {1.0, 2.0, 3.0, 4.0};
  MU_ASSERT(fltcmp(mean(vals, 4), 2.5) == 0);

  return 0;
}

int test_median() {
  {
    const real_t vals[5] = {2.0, 3.0, 1.0, 4.0, 5.0};
    const real_t retval = median(vals, 5);
    MU_ASSERT(fltcmp(retval, 3.0) == 0);
  }

  {
    const real_t vals2[6] = {2.0, 3.0, 1.0, 4.0, 5.0, 6.0};
    const real_t retval = median(vals2, 6);
    MU_ASSERT(fltcmp(retval, 3.5) == 0);
  }

  return 0;
}

int test_var() {
  real_t vals[4] = {1.0, 2.0, 3.0, 4.0};
  MU_ASSERT(fltcmp(var(vals, 4), 1.666666667) == 0);

  return 0;
}

int test_stddev() {
  real_t vals[4] = {1.0, 2.0, 3.0, 4.0};
  MU_ASSERT(fltcmp(stddev(vals, 4), sqrt(1.666666667)) == 0);

  return 0;
}

/******************************************************************************
 * TEST LINEAR ALGEBRA
 ******************************************************************************/

int test_eye() {
  real_t A[25] = {0.0};
  eye(A, 5, 5);

  /* print_matrix("I", A, 5, 5); */
  size_t idx = 0;
  size_t rows = 5;
  size_t cols = 5;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      real_t expected = (i == j) ? 1.0 : 0.0;
      MU_ASSERT(fltcmp(A[idx], expected) == 0);
      idx++;
    }
  }

  return 0;
}

int test_ones() {
  real_t A[25] = {0.0};
  ones(A, 5, 5);

  /* print_matrix("A", A, 5, 5); */
  size_t idx = 0;
  size_t rows = 5;
  size_t cols = 5;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      MU_ASSERT((fabs(A[idx] - 1.0) < 1e-5));
      idx++;
    }
  }

  return 0;
}

int test_zeros() {
  real_t A[25] = {0.0};
  zeros(A, 5, 5);

  /* print_matrix("A", A, 5, 5); */
  size_t idx = 0;
  size_t rows = 5;
  size_t cols = 5;
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      MU_ASSERT((fabs(A[idx] - 0.0) < 1e-5));
      idx++;
    }
  }

  return 0;
}

int test_mat_set() {
  real_t A[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  mat_set(A, 3, 0, 0, 1.0);
  mat_set(A, 3, 1, 1, 1.0);
  mat_set(A, 3, 2, 2, 1.0);

  /* print_matrix("A", A, 3, 3); */
  MU_ASSERT(fltcmp(mat_val(A, 3, 0, 0), 1.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 3, 1, 1), 1.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 3, 2, 2), 1.0) == 0);

  return 0;
}

int test_mat_val() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

  /* print_matrix("A", A, 3, 3); */
  MU_ASSERT(fltcmp(mat_val(A, 3, 0, 0), 1.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 3, 0, 1), 2.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 3, 0, 2), 3.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 3, 1, 0), 4.0) == 0);

  return 0;
}

int test_mat_copy() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[9] = {0};

  mat_copy(A, 3, 3, B);
  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(B[i], i + 1.0) == 0);
  }

  return 0;
}

int test_mat_row_set() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[3] = {0.0, 0.0, 0.0};

  /* Set first row zeros */
  mat_row_set(A, 3, 0, B);
  for (int i = 0; i < 3; i++) {
    MU_ASSERT(fltcmp(A[i], 0.0) == 0);
  }

  /* Set second row zeros */
  mat_row_set(A, 3, 1, B);
  for (int i = 0; i < 6; i++) {
    MU_ASSERT(fltcmp(A[i], 0.0) == 0);
  }

  /* Set third row zeros */
  mat_row_set(A, 3, 1, B);
  for (int i = 0; i < 6; i++) {
    MU_ASSERT(fltcmp(A[i], 0.0) == 0);
  }

  return 0;
}

int test_mat_col_set() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[3] = {0.0, 0.0, 0.0};

  /* Set first column zeros */
  mat_col_set(A, 3, 3, 0, B);
  for (int i = 0; i < 3; i++) {
    MU_ASSERT(fltcmp(A[i * 3], 0.0) == 0);
  }

  /* Set second column zeros */
  mat_col_set(A, 3, 3, 1, B);
  for (int i = 0; i < 3; i++) {
    MU_ASSERT(fltcmp(A[(i * 3) + 1], 0.0) == 0);
  }

  /* Set third column zeros */
  mat_col_set(A, 3, 3, 2, B);
  for (int i = 0; i < 3; i++) {
    MU_ASSERT(fltcmp(A[(i * 3) + 2], 0.0) == 0);
  }

  /* Check whether full matrix is zeros */
  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(A[i], 0.0) == 0);
  }

  return 0;
}

int test_mat_block_get() {
  // clang-format off
  real_t A[9] = {0.0, 1.0, 2.0,
                 3.0, 4.0, 5.0,
                 6.0, 7.0, 8.0};
  real_t B[4] = {0.0};
  real_t C[4] = {0.0};
  // clang-format on
  mat_block_get(A, 3, 1, 2, 1, 2, B);
  mat_block_get(A, 3, 0, 1, 1, 2, C);

  // print_matrix("A", A, 3, 3);
  // print_matrix("B", B, 2, 2);
  // print_matrix("C", C, 2, 2);

  MU_ASSERT(fltcmp(mat_val(B, 2, 0, 0), 4.0) == 0);
  MU_ASSERT(fltcmp(mat_val(B, 2, 0, 1), 5.0) == 0);
  MU_ASSERT(fltcmp(mat_val(B, 2, 1, 0), 7.0) == 0);
  MU_ASSERT(fltcmp(mat_val(B, 2, 1, 1), 8.0) == 0);

  MU_ASSERT(fltcmp(mat_val(C, 2, 0, 0), 1.0) == 0);
  MU_ASSERT(fltcmp(mat_val(C, 2, 0, 1), 2.0) == 0);
  MU_ASSERT(fltcmp(mat_val(C, 2, 1, 0), 4.0) == 0);
  MU_ASSERT(fltcmp(mat_val(C, 2, 1, 1), 5.0) == 0);

  return 0;
}

int test_mat_block_set() {
  // clang-format off
  real_t A[4 * 4] = {0.0, 1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0, 7.0,
                     8.0, 9.0, 10.0, 11.0,
                     12.0, 13.0, 14.0, 15.0};
  real_t B[2 * 2] = {0.0, 0.0,
                     0.0, 0.0};
  // clang-format on

  // print_matrix("A", A, 3, 3);
  // print_matrix("B", B, 2, 2);
  mat_block_set(A, 4, 1, 2, 1, 2, B);
  // print_matrix("A", A, 4, 4);
  // print_matrix("B", B, 2, 2);

  MU_ASSERT(fltcmp(mat_val(A, 4, 1, 1), 0.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 4, 1, 2), 0.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 4, 2, 1), 0.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 4, 2, 2), 0.0) == 0);

  return 0;
}

int test_mat_diag_get() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t d[3] = {0.0, 0.0, 0.0};
  mat_diag_get(A, 3, 3, d);

  // print_matrix("A", A, 3, 3);
  // print_vector("d", d, 3);
  MU_ASSERT(fltcmp(d[0], 1.0) == 0);
  MU_ASSERT(fltcmp(d[1], 5.0) == 0);
  MU_ASSERT(fltcmp(d[2], 9.0) == 0);

  return 0;
}

int test_mat_diag_set() {
  real_t A[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  real_t d[4] = {1.0, 2.0, 3.0};
  mat_diag_set(A, 3, 3, d);

  // print_matrix("A", A, 3, 3);
  MU_ASSERT(fltcmp(mat_val(A, 3, 0, 0), 1.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 3, 1, 1), 2.0) == 0);
  MU_ASSERT(fltcmp(mat_val(A, 3, 2, 2), 3.0) == 0);

  return 0;
}

int test_mat_triu() {
  // clang-format off
  real_t A[16] = {1.0, 2.0, 3.0, 4.0,
                  5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0,
                  13.0, 14.0, 15.0, 16.0};
  real_t U[16] = {0};
  // clang-format on
  mat_triu(A, 4, U);
  // print_matrix("U", U, 4, 4);

  return 0;
}

int test_mat_tril() {
  // clang-format off
  real_t A[16] = {1.0, 2.0, 3.0, 4.0,
                  5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0,
                  13.0, 14.0, 15.0, 16.0};
  real_t L[16] = {0};
  // clang-format on
  mat_tril(A, 4, L);
  // print_matrix("L", L, 4, 4);

  return 0;
}

int test_mat_trace() {
  // clang-format off
  real_t A[16] = {1.0, 2.0, 3.0, 4.0,
                  5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0,
                  13.0, 14.0, 15.0, 16.0};
  // clang-format on
  const real_t tr = mat_trace(A, 4, 4);
  MU_ASSERT(fltcmp(tr, 1.0 + 6.0 + 11.0 + 16.0) == 0.0);

  return 0;
}

int test_mat_transpose() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t At[9] = {0.0};
  real_t At_expected[9] = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
  mat_transpose(A, 3, 3, At);
  MU_ASSERT(mat_equals(At, At_expected, 3, 3, 1e-8));

  real_t B[2 * 3] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  real_t Bt[3 * 2] = {0};
  real_t Bt_expected[3 * 2] = {1.0, 4.0, 2.0, 5.0, 3.0, 6.0};
  mat_transpose(B, 2, 3, Bt);
  for (int i = 0; i < 6; i++) {
    MU_ASSERT(fltcmp(Bt[i], Bt_expected[i]) == 0);
  }

  return 0;
}

int test_mat_add() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[9] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  real_t C[9] = {0.0};
  mat_add(A, B, C, 3, 3);
  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(C[i], 10.0) == 0);
  }

  return 0;
}

int test_mat_sub() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t C[9] = {0.0};
  mat_sub(A, B, C, 3, 3);
  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(C[i], 0.0) == 0);
  }

  return 0;
}

int test_mat_scale() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  mat_scale(A, 3, 3, 2.0);
  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(A[i], 2 * (i + 1)) == 0);
  }

  return 0;
}

int test_vec_add() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[9] = {9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0};
  real_t C[9] = {0.0};
  vec_add(A, B, C, 9);
  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(C[i], 10.0) == 0);
  }

  return 0;
}

int test_vec_sub() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t C[9] = {0.0};
  vec_sub(A, B, C, 9);
  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(C[i], 0.0) == 0);
  }

  return 0;
}

int test_dot() {
  real_t A[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
  real_t B[3] = {1.0, 2.0, 3.0};
  real_t C[3] = {0.0};

  /* Multiply matrix A and B */
  dot(A, 3, 3, B, 3, 1, C);

  MU_ASSERT(fltcmp(C[0], 14.0) == 0);
  MU_ASSERT(fltcmp(C[1], 32.0) == 0);
  MU_ASSERT(fltcmp(C[2], 50.0) == 0);

  return 0;
}

int test_hat() {
  real_t x[3] = {1.0, 2.0, 3.0};
  real_t S[3 * 3] = {0};

  hat(x, S);

  MU_ASSERT(fltcmp(S[0], 0.0) == 0);
  MU_ASSERT(fltcmp(S[1], -3.0) == 0);
  MU_ASSERT(fltcmp(S[2], 2.0) == 0);

  MU_ASSERT(fltcmp(S[3], 3.0) == 0);
  MU_ASSERT(fltcmp(S[4], 0.0) == 0);
  MU_ASSERT(fltcmp(S[5], -1.0) == 0);

  MU_ASSERT(fltcmp(S[6], -2.0) == 0);
  MU_ASSERT(fltcmp(S[7], 1.0) == 0);
  MU_ASSERT(fltcmp(S[8], 0.0) == 0);

  return 0;
}

int test_check_jacobian() {
  const size_t m = 2;
  const size_t n = 3;
  const real_t threshold = 1e-6;
  const int print = 0;

  // Positive test
  {
    const real_t fdiff[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const real_t jac[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    int retval = check_jacobian("test_check_jacobian",
                                fdiff,
                                jac,
                                m,
                                n,
                                threshold,
                                print);
    MU_ASSERT(retval == 0);
  }

  // Negative test
  {
    const real_t fdiff[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
    const real_t jac[6] = {0.0, 1.0, 2.0, 3.1, 4.0, 5.0};
    int retval = check_jacobian("test_check_jacobian",
                                fdiff,
                                jac,
                                m,
                                n,
                                threshold,
                                print);
    MU_ASSERT(retval == -1);
  }

  return 0;
}

int test_svd() {
  // Matrix A
  // clang-format off
  real_t A[6 * 4] = {
    7.52, -1.10, -7.95,  1.08,
    -0.76,  0.62,  9.34, -7.10,
     5.13,  6.62, -5.66,  0.87,
    -4.75,  8.52,  5.75,  5.30,
     1.33,  4.91, -5.49, -3.52,
    -2.40, -6.77,  2.34,  3.95
  };
  // clang-format on

  // Decompose A with SVD
  // struct timespec t = tic();
  real_t U[6 * 4] = {0};
  real_t s[4] = {0};
  real_t V[4 * 4] = {0};
  svd(A, 6, 4, U, s, V);
  // printf("time taken: [%fs]\n", toc(&t));

  // Multiply the output to see if it can form matrix A again
  // U * S * Vt
  real_t S[4 * 4] = {0};
  real_t Vt[4 * 4] = {0};
  real_t US[6 * 4] = {0};
  real_t USVt[6 * 4] = {0};
  mat_diag_set(S, 4, 4, s);
  mat_transpose(V, 4, 4, Vt);
  dot(U, 6, 4, S, 4, 4, US);
  dot(US, 6, 4, Vt, 4, 4, USVt);

  // print_matrix("U", U, 6, 4);
  // print_matrix("S", S, 4, 4);
  // print_matrix("V", V, 4, 4);
  // print_matrix("USVt", USVt, 6, 4);
  // print_matrix("A", A, 6, 4);
  MU_ASSERT(mat_equals(USVt, A, 6, 4, 1e-5));

  return 0;
}

int test_pinv() {
  // clang-format off
  const int m = 4;
  const int n = 4;
  real_t A[4 * 4] = {
     7.52, -1.10, -7.95,  1.08,
    -0.76,  0.62,  9.34, -7.10,
     5.13,  6.62, -5.66,  0.87,
    -4.75,  8.52,  5.75,  5.30,
  };
  // clang-format on

  // Invert matrix A using SVD
  // struct timespec t = tic();
  real_t A_inv[4 * 4] = {0};
  pinv(A, m, n, A_inv);
  // printf("time taken: [%fs]\n", toc(&t));

  // Inverse check: A * A_inv = eye
  MU_ASSERT(check_inv(A, A_inv, 4) == 0);

  return 0;
}

int test_svd_det() {
  // clang-format off
  const int m = 4;
  const int n = 4;
  real_t A[4 * 4] = {
     7.52, -1.10, -7.95,  1.08,
    -0.76,  0.62,  9.34, -7.10,
     5.13,  6.62, -5.66,  0.87,
    -4.75,  8.52,  5.75,  5.30,
  };
  // clang-format on

  real_t det = 0.0;
  MU_ASSERT(svd_det(A, m, n, &det) == 0);

  return 0;
}

int test_chol() {
  // clang-format off
  const int n = 3;
  real_t A[9] = {
    4.0, 12.0, -16.0,
    12.0, 37.0, -43.0,
    -16.0, -43.0, 98.0
  };
  // clang-format on

  // struct timespec t = tic();
  real_t L[9] = {0};
  chol(A, n, L);
  // printf("time taken: [%fs]\n", toc(&t));
  // mat_save("/tmp/L.csv", L, 3, 3);

  real_t Lt[9] = {0};
  real_t LLt[9] = {0};
  mat_transpose(L, n, n, Lt);
  dot(L, n, n, Lt, n, n, LLt);

  int debug = 0;
  if (debug) {
    print_matrix("L", L, n, n);
    printf("\n");
    print_matrix("Lt", Lt, n, n);
    printf("\n");
    print_matrix("LLt", LLt, n, n);
    printf("\n");
    print_matrix("A", A, n, n);
  }
  MU_ASSERT(mat_equals(A, LLt, n, n, 1e-5));

  return 0;
}

int test_chol_solve() {
  // clang-format off
  const int n = 3;
  real_t A[9] = {
    2.0, -1.0, 0.0,
    -1.0, 2.0, -1.0,
    0.0, -1.0, 1.0
  };
  real_t b[3] = {1.0, 0.0, 0.0};
  real_t x[3] = {0.0, 0.0, 0.0};
  // clang-format on

  // struct timespec t = tic();
  chol_solve(A, b, x, n);
  // printf("time taken: [%fs]\n", toc(&t));
  // print_vector("x", x, n);

  MU_ASSERT(fltcmp(x[0], 1.0) == 0);
  MU_ASSERT(fltcmp(x[1], 1.0) == 0);
  MU_ASSERT(fltcmp(x[2], 1.0) == 0);

  return 0;
}

int test_qr() {
  // clang-format off
  const int m = 3;
  const int n = 3;
  real_t A[3 * 3] = {
    12, -51,   4,
    6,  167, -68,
    -4,  24, -41
  };
  // clang-format on

  real_t R[3 * 3] = {0};
  qr(A, m, n, R);
  // print_matrix("A", A, 3, 3);
  // print_matrix("R", R, 3, 3);

  return 0;
}

int test_eig_sym() {
  // clang-format off
  const int m = 5;
  const int n = 5;
  real_t A[5 * 5] = {
     1.96, -6.49, -0.47, -7.20, -0.65,
    -6.49,  3.80, -6.39,  1.50, -6.34,
    -0.47, -6.39,  4.17, -1.51,  2.67,
    -7.20,  1.50, -1.51,  5.70,  1.80,
    -0.65, -6.34,  2.67,  1.80, -7.10
  };
  // clang-format on

  // Eigen-decomposition
  real_t V[5 * 5] = {0};
  real_t w[5] = {0};
  int retval = eig_sym(A, m, n, V, w);
  MU_ASSERT(retval == 0);

  // Assert
  //
  //   A * V == lambda * V
  //
  // where:
  //
  //   A: original matrix
  //   V: Eigen-vectors
  //   lambda: Eigen-values
  //
  DOT(A, 5, 5, V, 5, 5, AV);

  for (int j = 0; j < n; j++) {
    real_t lv[5] = {0};
    mat_col_get(V, m, n, j, lv);
    vec_scale(lv, 5, w[j]);

    real_t av[5] = {0};
    mat_col_get(A, m, n, j, av);

    MU_ASSERT(vec_equals(av, lv, 5) == 0);
  }

  // print_matrix("AV", AV, 5, 5);
  // print_matrix("A", A, 5, 5);
  // print_matrix("V", V, 5, 5);
  // print_vector("w", w, 5);

  return 0;
}

int test_eig_inv() {
  // clang-format off
  const int m = 5;
  const int n = 5;
  real_t A[5 * 5] = {
     1.96, -6.49, -0.47, -7.20, -0.65,
    -6.49,  3.80, -6.39,  1.50, -6.34,
    -0.47, -6.39,  4.17, -1.51,  2.67,
    -7.20,  1.50, -1.51,  5.70,  1.80,
    -0.65, -6.34,  2.67,  1.80, -7.10
  };
  // clang-format on

  // Invert matrix A using SVD
  // struct timespec t = tic();
  real_t A_inv[5 * 5] = {0};
  eig_inv(A, m, n, 1, A_inv);

  // DOT(A, 5, 5, A_inv, 5, 5, check);
  // print_matrix("check", check, 5, 5);
  // printf("time taken: [%fs]\n", toc(&t));

  // Inverse check: A * A_inv = eye
  MU_ASSERT(check_inv(A, A_inv, 5) == 0);

  return 0;
}

int test_suitesparse_chol_solve() {
  // clang-format off
  const int n = 3;
  real_t A[9] = {
    2.0, -1.0, 0.0,
    -1.0, 2.0, -1.0,
    0.0, -1.0, 1.0
  };
  real_t b[3] = {1.0, 0.0, 0.0};
  real_t x[3] = {0.0, 0.0, 0.0};
  // clang-format on

  // struct timespec t = tic();
  cholmod_common common;
  cholmod_start(&common);
  suitesparse_chol_solve(&common, A, n, n, b, n, x);
  cholmod_finish(&common);
  // printf("time taken: [%fs]\n", toc(&t));
  // print_vector("x", x, n);

  MU_ASSERT(fltcmp(x[0], 1.0) == 0);
  MU_ASSERT(fltcmp(x[1], 1.0) == 0);
  MU_ASSERT(fltcmp(x[2], 1.0) == 0);

  return 0;
}

/******************************************************************************
 * TEST TRANSFORMS
 ******************************************************************************/

int test_tf_rot_set() {
  real_t C[9];
  for (int i = 0; i < 9; i++) {
    C[i] = 1.0;
  }

  real_t T[16] = {0.0};
  tf_rot_set(T, C);
  /* print_matrix("T", T, 4, 4); */

  MU_ASSERT(fltcmp(T[0], 1.0) == 0);
  MU_ASSERT(fltcmp(T[1], 1.0) == 0);
  MU_ASSERT(fltcmp(T[2], 1.0) == 0);
  MU_ASSERT(fltcmp(T[3], 0.0) == 0);

  MU_ASSERT(fltcmp(T[4], 1.0) == 0);
  MU_ASSERT(fltcmp(T[5], 1.0) == 0);
  MU_ASSERT(fltcmp(T[6], 1.0) == 0);
  MU_ASSERT(fltcmp(T[7], 0.0) == 0);

  MU_ASSERT(fltcmp(T[8], 1.0) == 0);
  MU_ASSERT(fltcmp(T[9], 1.0) == 0);
  MU_ASSERT(fltcmp(T[10], 1.0) == 0);
  MU_ASSERT(fltcmp(T[11], 0.0) == 0);

  MU_ASSERT(fltcmp(T[12], 0.0) == 0);
  MU_ASSERT(fltcmp(T[13], 0.0) == 0);
  MU_ASSERT(fltcmp(T[14], 0.0) == 0);
  MU_ASSERT(fltcmp(T[15], 0.0) == 0);

  return 0;
}

int test_tf_trans_set() {
  real_t r[3] = {1.0, 2.0, 3.0};

  real_t T[16] = {0.0};
  tf_trans_set(T, r);
  /* print_matrix("T", T, 4, 4); */

  MU_ASSERT(fltcmp(T[0], 0.0) == 0);
  MU_ASSERT(fltcmp(T[1], 0.0) == 0);
  MU_ASSERT(fltcmp(T[2], 0.0) == 0);
  MU_ASSERT(fltcmp(T[3], 1.0) == 0);

  MU_ASSERT(fltcmp(T[4], 0.0) == 0);
  MU_ASSERT(fltcmp(T[5], 0.0) == 0);
  MU_ASSERT(fltcmp(T[6], 0.0) == 0);
  MU_ASSERT(fltcmp(T[7], 2.0) == 0);

  MU_ASSERT(fltcmp(T[8], 0.0) == 0);
  MU_ASSERT(fltcmp(T[9], 0.0) == 0);
  MU_ASSERT(fltcmp(T[10], 0.0) == 0);
  MU_ASSERT(fltcmp(T[11], 3.0) == 0);

  MU_ASSERT(fltcmp(T[12], 0.0) == 0);
  MU_ASSERT(fltcmp(T[13], 0.0) == 0);
  MU_ASSERT(fltcmp(T[14], 0.0) == 0);
  MU_ASSERT(fltcmp(T[15], 0.0) == 0);

  return 0;
}

int test_tf_trans_get() {
  // clang-format off
  real_t T[16] = {1.0, 2.0, 3.0, 4.0,
                  5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0,
                  13.0, 14.0, 15.0, 16.0};
  // clang-format on

  /* Get translation vector */
  real_t r[3];
  tf_trans_get(T, r);
  MU_ASSERT(fltcmp(r[0], 4.0) == 0);
  MU_ASSERT(fltcmp(r[1], 8.0) == 0);
  MU_ASSERT(fltcmp(r[2], 12.0) == 0);

  return 0;
}

int test_tf_rot_get() {
  /* Transform */
  // clang-format off
  real_t T[16] = {1.0, 2.0, 3.0, 4.0,
                  5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0,
                  13.0, 14.0, 15.0, 16.0};
  // clang-format on

  /* Get rotation matrix */
  real_t C[9];
  tf_rot_get(T, C);

  MU_ASSERT(fltcmp(C[0], 1.0) == 0);
  MU_ASSERT(fltcmp(C[1], 2.0) == 0);
  MU_ASSERT(fltcmp(C[2], 3.0) == 0);

  MU_ASSERT(fltcmp(C[3], 5.0) == 0);
  MU_ASSERT(fltcmp(C[4], 6.0) == 0);
  MU_ASSERT(fltcmp(C[5], 7.0) == 0);

  MU_ASSERT(fltcmp(C[6], 9.0) == 0);
  MU_ASSERT(fltcmp(C[7], 10.0) == 0);
  MU_ASSERT(fltcmp(C[8], 11.0) == 0);

  return 0;
}

int test_tf_quat_get() {
  /* Transform */
  // clang-format off
  real_t T[16] = {1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0};
  // clang-format on

  /* Create rotation matrix */
  const real_t ypr_in[3] = {deg2rad(10.0), deg2rad(20.0), deg2rad(30.0)};
  real_t C[9] = {0};
  euler321(ypr_in, C);
  tf_rot_set(T, C);

  /* Extract quaternion from transform */
  real_t q[4] = {0};
  tf_quat_get(T, q);

  /* Convert quaternion back to euler angles */
  real_t ypr_out[3] = {0};
  quat2euler(q, ypr_out);

  MU_ASSERT(fltcmp(rad2deg(ypr_out[0]), 10.0) == 0);
  MU_ASSERT(fltcmp(rad2deg(ypr_out[1]), 20.0) == 0);
  MU_ASSERT(fltcmp(rad2deg(ypr_out[2]), 30.0) == 0);

  return 0;
}

int test_tf_inv() {
  /* Create Transform */
  // clang-format off
  real_t T[16] = {1.0, 0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0, 0.0,
                  0.0, 0.0, 1.0, 0.0,
                  0.0, 0.0, 0.0, 1.0};
  // clang-format on
  /* -- Set rotation component */
  const real_t euler[3] = {deg2rad(10.0), deg2rad(20.0), deg2rad(30.0)};
  real_t C[9] = {0};
  euler321(euler, C);
  tf_rot_set(T, C);
  /* -- Set translation component */
  real_t r[3] = {1.0, 2.0, 3.0};
  tf_trans_set(T, r);

  /* Invert transform */
  real_t T_inv[16] = {0};
  tf_inv(T, T_inv);

  /* real_t Invert transform */
  real_t T_inv_inv[16] = {0};
  tf_inv(T_inv, T_inv_inv);

  /* Assert */
  int idx = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      MU_ASSERT(fltcmp(T_inv_inv[idx], T[idx]) == 0);
    }
  }

  return 0;
}

int test_tf_point() {
  /* Transform */
  // clang-format off
  real_t T[16] = {1.0, 0.0, 0.0, 1.0,
                  0.0, 1.0, 0.0, 2.0,
                  0.0, 0.0, 1.0, 3.0,
                  0.0, 0.0, 0.0, 1.0};
  // clang-format on

  /* Point */
  real_t p[3] = {1.0, 2.0, 3.0};

  /* Transform point */
  real_t result[3] = {0};
  tf_point(T, p, result);

  return 0;
}

int test_tf_hpoint() {
  /* Transform */
  // clang-format off
  real_t T[16] = {1.0, 0.0, 0.0, 1.0,
                  0.0, 1.0, 0.0, 2.0,
                  0.0, 0.0, 1.0, 3.0,
                  0.0, 0.0, 0.0, 1.0};
  // clang-format on

  /* Homogeneous point */
  real_t hp[4] = {1.0, 2.0, 3.0, 1.0};

  /* Transform homogeneous point */
  real_t result[4] = {0};
  tf_hpoint(T, hp, result);

  return 0;
}

int test_tf_perturb_rot() {
  /* Transform */
  // clang-format off
  real_t T[4 * 4] = {1.0, 0.0, 0.0, 1.0,
                     0.0, 1.0, 0.0, 2.0,
                     0.0, 0.0, 1.0, 3.0,
                     0.0, 0.0, 0.0, 1.0};
  // clang-format on

  /* Perturb rotation */
  const real_t step_size = 1e-2;
  tf_perturb_rot(T, step_size, 0);

  /* Assert */
  MU_ASSERT(fltcmp(T[0], 1.0) == 0);
  MU_ASSERT(fltcmp(T[5], 1.0) != 0);
  MU_ASSERT(fltcmp(T[10], 1.0) != 0);

  return 0;
}

int test_tf_perturb_trans() {
  /* Transform */
  // clang-format off
  real_t T[4 * 4] = {1.0, 0.0, 0.0, 1.0,
                     0.0, 1.0, 0.0, 2.0,
                     0.0, 0.0, 1.0, 3.0,
                     0.0, 0.0, 0.0, 1.0};
  // clang-format on

  /* Perturb translation */
  const real_t step_size = 1e-2;
  tf_perturb_trans(T, step_size, 0);

  /* Assert */
  MU_ASSERT(fltcmp(T[3], 1.01) == 0);
  MU_ASSERT(fltcmp(T[7], 2.0) == 0);
  MU_ASSERT(fltcmp(T[11], 3.0) == 0);

  return 0;
}

int test_tf_chain() {
  /* First transform */
  const real_t r0[3] = {0.0, 0.0, 0.1};
  const real_t euler0[3] = {deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)};
  real_t T0[4 * 4] = {0};
  real_t C0[9] = {0};

  euler321(euler0, C0);
  tf_rot_set(T0, C0);
  tf_trans_set(T0, r0);
  T0[15] = 1.0;

  /* Second transform */
  const real_t r1[3] = {0.0, 0.0, 0.1};
  const real_t euler1[3] = {deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)};
  real_t T1[4 * 4] = {0};
  real_t C1[9] = {0};

  euler321(euler1, C1);
  tf_rot_set(T1, C1);
  tf_trans_set(T1, r1);
  T1[15] = 1.0;

  /* Third transform */
  const real_t r2[3] = {0.0, 0.0, 0.1};
  const real_t euler2[3] = {deg2rad(0.0), deg2rad(0.0), deg2rad(0.0)};
  real_t T2[4 * 4] = {0};
  real_t C2[9] = {0};

  euler321(euler2, C2);
  tf_rot_set(T2, C2);
  tf_trans_set(T2, r2);
  T2[15] = 1.0;

  /* Chain transforms */
  const real_t *tfs[3] = {T0, T1, T2};
  const int N = 3;
  real_t T_out[4 * 4] = {0};
  tf_chain(tfs, N, T_out);

  return 0;
}

int test_euler321() {
  /* Euler to rotation matrix */
  const real_t euler[3] = {deg2rad(10.0), deg2rad(20.0), deg2rad(30.0)};
  real_t C[9] = {0};
  euler321(euler, C);

  /* Rotation matrix to quaternion */
  real_t q[4] = {0};
  rot2quat(C, q);

  /* Quaternion to Euler angles*/
  real_t euler2[3] = {0};
  quat2euler(q, euler2);

  MU_ASSERT(fltcmp(euler2[0], euler[0]) == 0);
  MU_ASSERT(fltcmp(euler2[1], euler[1]) == 0);
  MU_ASSERT(fltcmp(euler2[2], euler[2]) == 0);

  return 0;
}

int test_rot2quat() {
  /* Rotation matrix to quaternion */
  const real_t C[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  real_t q[4] = {0.0};
  rot2quat(C, q);

  MU_ASSERT(fltcmp(q[0], 1.0) == 0);
  MU_ASSERT(fltcmp(q[1], 0.0) == 0);
  MU_ASSERT(fltcmp(q[2], 0.0) == 0);
  MU_ASSERT(fltcmp(q[3], 0.0) == 0);

  return 0;
}

int test_quat2euler() {
  const real_t C[9] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};

  /* Rotation matrix to quaternion */
  real_t q[4] = {0.0};
  rot2quat(C, q);

  /* Quaternion to Euler angles */
  real_t ypr[3] = {0.0};
  quat2euler(q, ypr);

  MU_ASSERT(fltcmp(ypr[0], 0.0) == 0);
  MU_ASSERT(fltcmp(ypr[1], 0.0) == 0);
  MU_ASSERT(fltcmp(ypr[2], 0.0) == 0);

  return 0;
}

int test_quat2rot() {
  /* Euler to rotation matrix */
  const real_t euler[3] = {deg2rad(10.0), deg2rad(20.0), deg2rad(30.0)};
  real_t C[9] = {0};
  euler321(euler, C);

  /* Rotation matrix to quaternion */
  real_t q[4] = {0.0};
  rot2quat(C, q);
  /* print_vector("q", q, 4); */

  /* Quaternion to rotation matrix */
  real_t rot[9] = {0.0};
  quat2rot(q, rot);

  for (int i = 0; i < 9; i++) {
    MU_ASSERT(fltcmp(C[i], rot[i]) == 0);
  }

  return 0;
}

/******************************************************************************
 * TEST LIE
 ******************************************************************************/

int test_lie_Exp_Log() {
  const real_t phi[3] = {0.1, 0.2, 0.3};
  real_t C[3 * 3] = {0};
  lie_Exp(phi, C);

  real_t rvec[3] = {0};
  lie_Log(C, rvec);

  // print_vector("phi", phi, 3);
  // printf("\n");
  // print_matrix("C", C, 3, 3);
  // print_vector("rvec", rvec, 3);

  MU_ASSERT(fltcmp(phi[0], rvec[0]) == 0);
  MU_ASSERT(fltcmp(phi[1], rvec[1]) == 0);
  MU_ASSERT(fltcmp(phi[2], rvec[2]) == 0);

  return 0;
}

/******************************************************************************
 * TEST CV
 ******************************************************************************/

int test_image_setup() {
  return 0;
}

int test_image_load() {
  return 0;
}

int test_image_print_properties() {
  return 0;
}

int test_image_free() {
  return 0;
}

int test_linear_triangulation() {
  /* Setup camera */
  const int image_width = 640;
  const int image_height = 480;
  const real_t fov = 120.0;
  const real_t fx = pinhole_focal(image_width, fov);
  const real_t fy = pinhole_focal(image_width, fov);
  const real_t cx = image_width / 2;
  const real_t cy = image_height / 2;
  const real_t proj_params[4] = {fx, fy, cx, cy};
  real_t K[3 * 3];
  pinhole_K(proj_params, K);

  /* Setup camera pose T_WC0 */
  const real_t ypr_WC0[3] = {-M_PI / 2.0, 0, -M_PI / 2.0};
  const real_t r_WC0[3] = {0.0, 0.0, 0.0};
  real_t T_WC0[4 * 4] = {0};
  tf_euler_set(T_WC0, ypr_WC0);
  tf_trans_set(T_WC0, r_WC0);

  /* Setup camera pose T_WC1 */
  const real_t euler_WC1[3] = {-M_PI / 2.0, 0, -M_PI / 2.0};
  const real_t r_WC1[3] = {0.1, 0.1, 0.0};
  real_t T_WC1[4 * 4] = {0};
  tf_euler_set(T_WC1, euler_WC1);
  tf_trans_set(T_WC1, r_WC1);

  /* Setup projection matrices */
  real_t P0[3 * 4] = {0};
  real_t P1[3 * 4] = {0};
  pinhole_projection_matrix(proj_params, T_WC0, P0);
  pinhole_projection_matrix(proj_params, T_WC1, P1);

  /* Setup 3D and 2D correspondance points */
  int nb_tests = 100;
  for (int i = 0; i < nb_tests; i++) {
    const real_t p_W[3] = {5.0, randf(-1.0, 1.0), randf(-1.0, 1.0)};

    real_t T_C0W[4 * 4] = {0};
    real_t T_C1W[4 * 4] = {0};
    tf_inv(T_WC0, T_C0W);
    tf_inv(T_WC1, T_C1W);

    real_t p_C0[3] = {0};
    real_t p_C1[3] = {0};
    tf_point(T_C0W, p_W, p_C0);
    tf_point(T_C1W, p_W, p_C1);

    real_t z0[2] = {0};
    real_t z1[2] = {0};
    pinhole_project(proj_params, p_C0, z0);
    pinhole_project(proj_params, p_C1, z1);

    /* Test */
    real_t p_W_est[3] = {0};
    linear_triangulation(P0, P1, z0, z1, p_W_est);

    /* Assert */
    real_t diff[3] = {0};
    vec_sub(p_W, p_W_est, diff, 3);
    const real_t norm = vec_norm(diff, 3);
    /* print_vector("p_W [gnd]", p_W, 3); */
    /* print_vector("p_W [est]", p_W_est, 3); */
    MU_ASSERT(norm < 1e-4);
  }

  return 0;
}

int test_radtan4_distort() {
  const real_t params[4] = {0.01, 0.001, 0.001, 0.001};
  const real_t p[2] = {0.1, 0.2};
  real_t p_d[2] = {0};
  radtan4_distort(params, p, p_d);

  // print_vector("p", p, 2);
  // print_vector("p_d", p_d, 2);

  return 0;
}

int test_radtan4_undistort() {
  const real_t params[4] = {0.01, 0.001, 0.001, 0.001};
  const real_t p[2] = {0.1, 0.2};

  real_t p_d[2] = {0};
  real_t p_out[2] = {0};
  radtan4_distort(params, p, p_d);
  radtan4_undistort(params, p_d, p_out);

  // print_vector("p", p, 2);
  // print_vector("p_d", p_d, 2);
  // print_vector("p_out", p_out, 2);
  // printf("dp[0]: %f\n", p[0] - p_out[0]);
  // printf("dp[1]: %f\n", p[1] - p_out[1]);

  MU_ASSERT(fltcmp(p[0], p_out[0]) == 0);
  MU_ASSERT(fltcmp(p[1], p_out[1]) == 0);

  return 0;
}

int test_radtan4_point_jacobian() {
  const real_t params[4] = {0.01, 0.001, 0.001, 0.001};
  const real_t p[2] = {0.1, 0.2};
  real_t J_point[2 * 2] = {0};
  radtan4_point_jacobian(params, p, J_point);

  /* Calculate numerical diff */
  const real_t step = 1e-4;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 2] = {0};
  {
    real_t p_d[2] = {0};
    radtan4_distort(params, p, p_d);

    for (int i = 0; i < 2; i++) {
      real_t p_diff[2] = {p[0], p[1]};
      p_diff[i] = p[i] + step;

      real_t p_d_prime[2] = {0};
      radtan4_distort(params, p_diff, p_d_prime);

      J_numdiff[i] = (p_d_prime[0] - p_d[0]) / step;
      J_numdiff[i + 2] = (p_d_prime[1] - p_d[1]) / step;
    }
  }

  /* Check jacobian */
  // print_vector("p", p, 2);
  // print_matrix("J_point", J_point, 2, 2);
  // print_matrix("J_numdiff", J_numdiff, 2, 2);
  MU_ASSERT(check_jacobian("J", J_numdiff, J_point, 2, 2, tol, 0) == 0);

  return 0;
}

int test_radtan4_params_jacobian() {
  const real_t params[4] = {0.01, 0.001, 0.001, 0.001};
  const real_t p[2] = {0.1, 0.2};
  real_t J_param[2 * 4] = {0};
  radtan4_params_jacobian(params, p, J_param);

  /* Calculate numerical diff */
  const real_t step = 1e-4;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 4] = {0};
  {
    real_t p_d[2] = {0};
    radtan4_distort(params, p, p_d);

    for (int i = 0; i < 4; i++) {
      real_t params_diff[4] = {params[0], params[1], params[2], params[3]};
      params_diff[i] = params[i] + step;

      real_t p_d_prime[2] = {0};
      radtan4_distort(params_diff, p, p_d_prime);

      J_numdiff[i] = (p_d_prime[0] - p_d[0]) / step;
      J_numdiff[i + 4] = (p_d_prime[1] - p_d[1]) / step;
    }
  }

  /* Check jacobian */
  // print_vector("p", p, 2);
  // print_matrix("J_param", J_param, 2, 4);
  // print_matrix("J_numdiff", J_numdiff, 2, 4);
  MU_ASSERT(check_jacobian("J", J_numdiff, J_param, 2, 4, tol, 0) == 0);

  return 0;
}

int test_equi4_distort() {
  const real_t params[4] = {0.01, 0.001, 0.001, 0.001};
  const real_t p[2] = {0.1, 0.2};
  real_t p_d[2] = {0};
  equi4_distort(params, p, p_d);

  // print_vector("p", p, 2);
  // print_vector("p_d", p_d, 2);

  return 0;
}

int test_equi4_undistort() {
  const real_t params[4] = {0.01, 0.001, 0.001, 0.001};
  const real_t p[2] = {0.1, 0.2};
  real_t p_d[2] = {0};
  real_t p_out[2] = {0};
  equi4_distort(params, p, p_d);
  equi4_undistort(params, p_d, p_out);

  // print_vector("p", p, 2);
  // print_vector("p_d", p_d, 2);
  // print_vector("p_out", p_out, 2);
  // printf("dp[0]: %f\n", p[0] - p_out[0]);
  // printf("dp[1]: %f\n", p[1] - p_out[1]);

  MU_ASSERT(fltcmp(p[0], p_out[0]) == 0);
  MU_ASSERT(fltcmp(p[1], p_out[1]) == 0);
  return 0;
}

int test_equi4_point_jacobian() {
  const real_t params[4] = {0.01, 0.001, 0.001, 0.001};
  const real_t p[2] = {0.1, 0.2};
  real_t J_point[2 * 2] = {0};
  equi4_point_jacobian(params, p, J_point);

  /* Calculate numerical diff */
  const real_t step = 1e-4;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 2] = {0};
  {
    real_t p_d[2] = {0};
    equi4_distort(params, p, p_d);

    for (int i = 0; i < 2; i++) {
      real_t p_diff[2] = {p[0], p[1]};
      p_diff[i] = p[i] + step;

      real_t p_d_prime[2] = {0};
      equi4_distort(params, p_diff, p_d_prime);

      J_numdiff[i] = (p_d_prime[0] - p_d[0]) / step;
      J_numdiff[i + 2] = (p_d_prime[1] - p_d[1]) / step;
    }
  }

  /* Check jacobian */
  // print_vector("p", p, 2);
  // print_matrix("J_point", J_point, 2, 2);
  // print_matrix("J_numdiff", J_numdiff, 2, 2);
  MU_ASSERT(check_jacobian("J", J_numdiff, J_point, 2, 2, tol, 0) == 0);

  return 0;
}

int test_equi4_params_jacobian() {
  const real_t params[4] = {0.01, 0.01, 0.01, 0.01};
  const real_t p[2] = {0.1, 0.2};
  real_t J_param[2 * 4] = {0};
  equi4_params_jacobian(params, p, J_param);

  /* Calculate numerical diff */
  const real_t step = 1e-8;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 4] = {0};
  {
    real_t p_d[2] = {0};
    equi4_distort(params, p, p_d);

    for (int i = 0; i < 4; i++) {
      real_t params_diff[4] = {params[0], params[1], params[2], params[3]};
      params_diff[i] = params[i] + step;

      real_t p_d_prime[2] = {0};
      equi4_distort(params_diff, p, p_d_prime);

      J_numdiff[i] = (p_d_prime[0] - p_d[0]) / step;
      J_numdiff[i + 4] = (p_d_prime[1] - p_d[1]) / step;
    }
  }

  /* Check jacobian */
  // print_vector("p", p, 2);
  // print_matrix("J_param", J_param, 2, 4);
  // print_matrix("J_numdiff", J_numdiff, 2, 4);
  MU_ASSERT(check_jacobian("J", J_numdiff, J_param, 2, 4, tol, 0) == 0);

  return 0;
}

int test_pinhole_focal() {
  const real_t focal = pinhole_focal(640, 90.0);
  MU_ASSERT(fltcmp(focal, 320.0) == 0);
  return 0;
}

int test_pinhole_K() {
  const real_t params[4] = {1.0, 2.0, 3.0, 4.0};
  real_t K[3 * 3] = {0};
  pinhole_K(params, K);

  MU_ASSERT(fltcmp(K[0], 1.0) == 0);
  MU_ASSERT(fltcmp(K[1], 0.0) == 0);
  MU_ASSERT(fltcmp(K[2], 3.0) == 0);

  MU_ASSERT(fltcmp(K[3], 0.0) == 0);
  MU_ASSERT(fltcmp(K[4], 2.0) == 0);
  MU_ASSERT(fltcmp(K[5], 4.0) == 0);

  MU_ASSERT(fltcmp(K[6], 0.0) == 0);
  MU_ASSERT(fltcmp(K[7], 0.0) == 0);
  MU_ASSERT(fltcmp(K[8], 1.0) == 0);

  return 0;
}

int test_pinhole_projection_matrix() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t params[4] = {fx, fy, cx, cy};

  /* Camera pose */
  const real_t ypr_WC0[3] = {-M_PI / 2.0, 0, -M_PI / 2.0};
  const real_t r_WC0[3] = {0.0, 0.0, 0.0};
  real_t T_WC0[4 * 4] = {0};
  tf_euler_set(T_WC0, ypr_WC0);
  tf_trans_set(T_WC0, r_WC0);

  /* Camera projection matrix */
  real_t P[3 * 4] = {0};
  pinhole_projection_matrix(params, T_WC0, P);

  /* Project point using projection matrix */
  const real_t p_W[3] = {1.0, 0.1, 0.2};
  const real_t hp_W[4] = {p_W[0], p_W[1], p_W[2], 1.0};
  real_t hp[3] = {0};
  dot(P, 3, 4, hp_W, 4, 1, hp);
  real_t z[2] = {hp[0], hp[1]};

  /* Project point by inverting T_WC0 and projecting the point */
  real_t p_C[3] = {0};
  real_t T_C0W[4 * 4] = {0};
  real_t z_gnd[2] = {0};
  tf_inv(T_WC0, T_C0W);
  tf_point(T_C0W, p_W, p_C);
  pinhole_project(params, p_C, z_gnd);

  /* Assert */
  MU_ASSERT(fltcmp(z_gnd[0], z[0]) == 0);
  MU_ASSERT(fltcmp(z_gnd[1], z[1]) == 0);

  return 0;
}

int test_pinhole_project() {
  const real_t img_w = 640;
  const real_t img_h = 480;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t params[4] = {fx, fy, cx, cy};
  const real_t p_C[3] = {0.0, 0.0, 1.0};
  real_t z[2] = {0.0, 0.0};
  pinhole_project(params, p_C, z);

  /* print_vector("p_C", p_C, 3); */
  /* print_vector("z", z, 2); */
  MU_ASSERT(fltcmp(z[0], 320.0) == 0);
  MU_ASSERT(fltcmp(z[1], 240.0) == 0);

  return 0;
}

int test_pinhole_point_jacobian() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t params[4] = {fx, fy, cx, cy};

  /* Calculate analytical jacobian */
  real_t J_point[2 * 2] = {0};
  pinhole_point_jacobian(params, J_point);

  /* Numerical differentiation */
  const real_t p_C[3] = {0.1, 0.2, 1.0};
  real_t z[2] = {0};
  pinhole_project(params, p_C, z);

  const real_t h = 1e-8;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 2] = {0};

  for (size_t i = 0; i < 2; i++) {
    real_t z_fd[2] = {0};
    real_t p_C_fd[3] = {p_C[0], p_C[1], p_C[2]};
    p_C_fd[i] += h;
    pinhole_project(params, p_C_fd, z_fd);
    J_numdiff[i] = (z_fd[0] - z[0]) / h;
    J_numdiff[i + 2] = (z_fd[1] - z[1]) / h;
  }

  /* Assert */
  MU_ASSERT(check_jacobian("J_point", J_numdiff, J_point, 2, 2, tol, 0) == 0);

  return 0;
}

int test_pinhole_params_jacobian() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t params[4] = {fx, fy, cx, cy};

  /* Calculate analytical jacobian */
  const real_t p_C[3] = {0.1, 0.2, 1.0};
  const real_t x[2] = {p_C[0] / p_C[2], p_C[1] / p_C[2]};
  real_t J_params[2 * 4] = {0};
  pinhole_params_jacobian(params, x, J_params);

  /* Numerical differentiation */
  real_t z[2] = {0};
  pinhole_project(params, p_C, z);

  const real_t h = 1e-8;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 4] = {0};

  for (size_t i = 0; i < 4; i++) {
    real_t z_fd[2] = {0};
    real_t params_fd[4] = {params[0], params[1], params[2], params[3]};
    params_fd[i] += h;
    pinhole_project(params_fd, p_C, z_fd);

    J_numdiff[i + 0] = (z_fd[0] - z[0]) / h;
    J_numdiff[i + 4] = (z_fd[1] - z[1]) / h;
  }

  /* Assert */
  MU_ASSERT(check_jacobian("J_params", J_numdiff, J_params, 2, 4, tol, 0) == 0);

  return 0;
}

int test_pinhole_radtan4_project() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t k1 = 0.3;
  const real_t k2 = 0.01;
  const real_t p1 = 0.01;
  const real_t p2 = 0.01;
  const real_t params[8] = {fx, fy, cx, cy, k1, k2, p1, p2};

  const real_t p_C[3] = {0.1, 0.2, 10.0};
  real_t x[2] = {0};
  pinhole_radtan4_project(params, p_C, x);

  /* print_vector("x", x, 2); */
  MU_ASSERT(fltcmp(x[0], 323.204000) == 0);
  MU_ASSERT(fltcmp(x[1], 166.406400) == 0);

  return 0;
}

int test_pinhole_radtan4_project_jacobian() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t k1 = 0.3;
  const real_t k2 = 0.01;
  const real_t p1 = 0.01;
  const real_t p2 = 0.01;
  const real_t params[8] = {fx, fy, cx, cy, k1, k2, p1, p2};

  /* Calculate analytical jacobian */
  const real_t p_C[3] = {0.1, 0.2, 10.0};
  real_t J[2 * 3] = {0};
  pinhole_radtan4_project_jacobian(params, p_C, J);

  /* Numerical differentiation */
  real_t z[2] = {0};
  pinhole_radtan4_project(params, p_C, z);

  const real_t h = 1e-8;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 3] = {0};

  for (size_t i = 0; i < 3; i++) {
    real_t z_fd[2] = {0};
    real_t p_C_fd[3] = {p_C[0], p_C[1], p_C[2]};
    p_C_fd[i] += h;

    pinhole_radtan4_project(params, p_C_fd, z_fd);
    J_numdiff[i] = (z_fd[0] - z[0]) / h;
    J_numdiff[i + 3] = (z_fd[1] - z[1]) / h;
  }

  /* Assert */
  /* print_matrix("J_numdiff", J_numdiff, 2, 3); */
  /* print_matrix("J", J, 2, 3); */
  MU_ASSERT(check_jacobian("J", J_numdiff, J, 2, 3, tol, 0) == 0);

  return 0;
}

int test_pinhole_radtan4_params_jacobian() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t k1 = 0.3;
  const real_t k2 = 0.01;
  const real_t p1 = 0.01;
  const real_t p2 = 0.01;
  const real_t params[8] = {fx, fy, cx, cy, k1, k2, p1, p2};

  /* Calculate analytical jacobian */
  const real_t p_C[3] = {0.1, 0.2, 10.0};
  real_t J_params[2 * 8] = {0};
  pinhole_radtan4_params_jacobian(params, p_C, J_params);

  /* Numerical differentiation */
  real_t z[2] = {0};
  pinhole_radtan4_project(params, p_C, z);

  const real_t h = 1e-8;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 8] = {0};

  for (size_t i = 0; i < 8; i++) {
    real_t z_fd[2] = {0};

    real_t params_fd[8] = {0};
    memcpy(params_fd, params, sizeof(real_t) * 8);
    params_fd[i] += h;

    pinhole_radtan4_project(params_fd, p_C, z_fd);
    J_numdiff[i] = (z_fd[0] - z[0]) / h;
    J_numdiff[i + 8] = (z_fd[1] - z[1]) / h;
  }

  /* Assert */
  /* print_matrix("J_numdiff", J_numdiff, 2, 8); */
  /* print_matrix("J_params", J_params, 2, 8); */
  MU_ASSERT(check_jacobian("J_params", J_numdiff, J_params, 2, 8, tol, 0) == 0);

  return 0;
}

int test_pinhole_equi4_project() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t k1 = 0.1;
  const real_t k2 = 0.01;
  const real_t k3 = 0.01;
  const real_t k4 = 0.01;
  const real_t params[8] = {fx, fy, cx, cy, k1, k2, k3, k4};

  const real_t p_C[3] = {0.1, 0.2, 10.0};
  real_t x[2] = {0};
  pinhole_equi4_project(params, p_C, x);

  /* print_vector("x", x, 2); */
  MU_ASSERT(fltcmp(x[0], 323.199627) == 0);
  MU_ASSERT(fltcmp(x[1], 166.399254) == 0);

  return 0;
}

int test_pinhole_equi4_project_jacobian() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t k1 = 0.1;
  const real_t k2 = 0.01;
  const real_t k3 = 0.01;
  const real_t k4 = 0.01;
  const real_t params[8] = {fx, fy, cx, cy, k1, k2, k3, k4};

  /* Calculate analytical jacobian */
  const real_t p_C[3] = {0.1, 0.2, 10.0};
  real_t J[2 * 3] = {0};
  pinhole_equi4_project_jacobian(params, p_C, J);

  /* Numerical differentiation */
  real_t z[2] = {0};
  pinhole_equi4_project(params, p_C, z);

  const real_t h = 1e-8;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 3] = {0};

  for (size_t i = 0; i < 3; i++) {
    real_t z_fd[2] = {0};
    real_t p_C_fd[3] = {p_C[0], p_C[1], p_C[2]};
    p_C_fd[i] += h;

    pinhole_equi4_project(params, p_C_fd, z_fd);
    J_numdiff[i] = (z_fd[0] - z[0]) / h;
    J_numdiff[i + 3] = (z_fd[1] - z[1]) / h;
  }

  /* Assert */
  /* print_matrix("J_numdiff", J_numdiff, 2, 3); */
  /* print_matrix("J", J, 2, 3); */
  MU_ASSERT(check_jacobian("J", J_numdiff, J, 2, 3, tol, 0) == 0);

  return 0;
}

int test_pinhole_equi4_params_jacobian() {
  /* Camera parameters */
  const int img_w = 640;
  const int img_h = 320;
  const real_t fx = pinhole_focal(img_w, 90.0);
  const real_t fy = pinhole_focal(img_w, 90.0);
  const real_t cx = img_w / 2.0;
  const real_t cy = img_h / 2.0;
  const real_t k1 = 0.1;
  const real_t k2 = 0.01;
  const real_t k3 = 0.01;
  const real_t k4 = 0.01;
  const real_t params[8] = {fx, fy, cx, cy, k1, k2, k3, k4};

  /* Calculate analytical jacobian */
  const real_t p_C[3] = {0.1, 0.2, 10.0};
  real_t J_params[2 * 8] = {0};
  pinhole_equi4_params_jacobian(params, p_C, J_params);

  /* Numerical differentiation */
  real_t z[2] = {0};
  pinhole_equi4_project(params, p_C, z);

  const real_t h = 1e-8;
  const real_t tol = 1e-4;
  real_t J_numdiff[2 * 8] = {0};

  for (size_t i = 0; i < 8; i++) {
    real_t z_fd[2] = {0};

    real_t params_fd[8] = {0};
    memcpy(params_fd, params, sizeof(real_t) * 8);
    params_fd[i] += h;

    pinhole_equi4_project(params_fd, p_C, z_fd);
    J_numdiff[i] = (z_fd[0] - z[0]) / h;
    J_numdiff[i + 8] = (z_fd[1] - z[1]) / h;
  }

  /* Assert */
  /* print_matrix("J_numdiff", J_numdiff, 2, 8); */
  /* print_matrix("J_params", J_params, 2, 8); */
  MU_ASSERT(check_jacobian("J_params", J_numdiff, J_params, 2, 8, tol, 0) == 0);

  return 0;
}

/******************************************************************************
 * TEST SENSOR FUSION
 ******************************************************************************/

int test_pose() {
  timestamp_t ts = 1;
  pose_t pose;

  real_t data[7] = {0.1, 0.2, 0.3, 1.0, 1.1, 2.2, 3.3};
  pose_setup(&pose, ts, data);

  MU_ASSERT(pose.ts == 1);

  MU_ASSERT(fltcmp(pose.data[0], 0.1) == 0.0);
  MU_ASSERT(fltcmp(pose.data[1], 0.2) == 0.0);
  MU_ASSERT(fltcmp(pose.data[2], 0.3) == 0.0);
  MU_ASSERT(fltcmp(pose.data[3], 1.0) == 0.0);
  MU_ASSERT(fltcmp(pose.data[4], 1.1) == 0.0);
  MU_ASSERT(fltcmp(pose.data[5], 2.2) == 0.0);
  MU_ASSERT(fltcmp(pose.data[6], 3.3) == 0.0);

  return 0;
}

int test_extrinsics() {
  extrinsic_t extrinsic;

  real_t data[7] = {1.0, 2.0, 3.0, 1.0, 0.1, 0.2, 0.3};
  extrinsic_setup(&extrinsic, data);

  MU_ASSERT(fltcmp(extrinsic.data[0], 1.0) == 0.0);
  MU_ASSERT(fltcmp(extrinsic.data[1], 2.0) == 0.0);
  MU_ASSERT(fltcmp(extrinsic.data[2], 3.0) == 0.0);
  MU_ASSERT(fltcmp(extrinsic.data[3], 1.0) == 0.0);
  MU_ASSERT(fltcmp(extrinsic.data[4], 0.1) == 0.0);
  MU_ASSERT(fltcmp(extrinsic.data[5], 0.2) == 0.0);
  MU_ASSERT(fltcmp(extrinsic.data[6], 0.3) == 0.0);

  return 0;
}

int test_imu_biases() {
  timestamp_t ts = 1;
  imu_biases_t biases;

  real_t ba[3] = {1.0, 2.0, 3.0};
  real_t bg[3] = {4.0, 5.0, 6.0};
  imu_biases_setup(&biases, ts, ba, bg);

  MU_ASSERT(biases.ts == 1);

  MU_ASSERT(fltcmp(biases.data[0], 1.0) == 0.0);
  MU_ASSERT(fltcmp(biases.data[1], 2.0) == 0.0);
  MU_ASSERT(fltcmp(biases.data[2], 3.0) == 0.0);

  MU_ASSERT(fltcmp(biases.data[3], 4.0) == 0.0);
  MU_ASSERT(fltcmp(biases.data[4], 5.0) == 0.0);
  MU_ASSERT(fltcmp(biases.data[5], 6.0) == 0.0);

  return 0;
}

int test_feature() {
  feature_t feature;

  real_t data[3] = {0.1, 0.2, 0.3};
  feature_setup(&feature, data);

  MU_ASSERT(fltcmp(feature.data[0], 0.1) == 0.0);
  MU_ASSERT(fltcmp(feature.data[1], 0.2) == 0.0);
  MU_ASSERT(fltcmp(feature.data[2], 0.3) == 0.0);

  return 0;
}

int test_idfs() {
  const real_t zero3[3] = {0.0, 0.0, 0.0};
  idfs_t idfs;

  // Setup
  idfs_setup(&idfs);
  MU_ASSERT(idfs.num_features == 0);
  for (size_t i = 0; i < IDFS_MAX_NUM; i++) {
    MU_ASSERT(idfs.status[i] == 0);
    MU_ASSERT(idfs.feature_ids[i] == 0);
    MU_ASSERT(vec_equals(idfs.data + i * 3, zero3, 3));
  }

  // Add features
  for (size_t i = 0; i < 10; i++) {
    const size_t feature_id = i;
    const real_t feature_data[3] = {i, i, i};
    idfs_add(&idfs, feature_id, feature_data);
  }
  // idfs_print(&idfs);

  // Make feature as lost
  idfs_mark_lost(&idfs, 2);
  idfs_mark_lost(&idfs, 4);
  idfs_mark_lost(&idfs, 6);
  MU_ASSERT(idfs.status[2] == 0);
  MU_ASSERT(idfs.status[4] == 0);
  MU_ASSERT(idfs.status[6] == 0);
  // idfs_print(&idfs);

  return 0;
}

int test_camera_params_setup() {
  camera_params_t camera;
  const int cam_idx = 0;
  const int cam_res[2] = {752, 480};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  const real_t data[8] = {640, 480, 320, 240, 0.0, 0.0, 0.0, 0.0};
  camera_params_setup(&camera, cam_idx, cam_res, proj_model, dist_model, data);
  camera_params_print(&camera);

  return 0;
}

int test_pose_factor() {
  /* Pose */
  timestamp_t ts = 1;
  pose_t pose;
  real_t data[7] = {0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0};
  pose_setup(&pose, ts, data);

  /* Setup pose factor */
  pose_factor_t factor;
  real_t var[6] = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
  pose_factor_setup(&factor, &pose, var);

  /* Check Jacobians */
  const real_t step_size = 1e-8;
  const real_t tol = 1e-4;
  CHECK_FACTOR_J(0, factor, pose_factor_eval, step_size, tol, 0);

  return 0;
}

int test_ba_factor() {
  // Timestamp
  timestamp_t ts = 0;

  // Camera pose
  const real_t pose_data[7] = {0.01, 0.01, 0.0, 0.5, -0.5, 0.5, -0.5};
  pose_t pose;
  pose_setup(&pose, ts, pose_data);

  // Feature
  const real_t p_W[3] = {1.0, 0.1, 0.2};
  feature_t feature;
  feature_setup(&feature, p_W);

  // Camera parameters
  const int cam_idx = 0;
  const int cam_res[2] = {640, 480};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  const real_t cam_data[8] = {320, 240, 320, 240, 0.03, 0.01, 0.001, 0.001};
  camera_params_t cam;
  camera_params_setup(&cam, cam_idx, cam_res, proj_model, dist_model, cam_data);

  // Project point from world to image plane
  real_t T_WC[4 * 4] = {0};
  real_t T_CW[4 * 4] = {0};
  real_t p_C[3] = {0.0};
  real_t z[2] = {0.0};
  tf(pose_data, T_WC);
  tf_inv(T_WC, T_CW);
  tf_point(T_CW, p_W, p_C);
  pinhole_radtan4_project(cam_data, p_C, z);

  // Bundle adjustment factor
  ba_factor_t factor;
  real_t var[2] = {1.0, 1.0};
  ba_factor_setup(&factor, &pose, &feature, &cam, z, var);

  // Check Jacobians
  const real_t step_size = 1e-8;
  const real_t tol = 1e-4;
  CHECK_FACTOR_J(0, factor, ba_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(1, factor, ba_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(2, factor, ba_factor_eval, step_size, tol, 0);

  return 0;
}

int test_vision_factor() {
  // Timestamp
  timestamp_t ts = 0;

  // Body pose
  pose_t pose;
  const real_t pose_data[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  pose_setup(&pose, ts, pose_data);

  // extrinsic
  extrinsic_t cam_exts;
  const real_t exts_data[7] = {0.01, 0.02, 0.03, 0.5, 0.5, -0.5, -0.5};
  extrinsic_setup(&cam_exts, exts_data);

  // Feature
  feature_t feature;
  const real_t p_W[3] = {1.0, 0.0, 0.0};
  feature_setup(&feature, p_W);

  // Camera parameters
  camera_params_t cam;
  const int cam_idx = 0;
  const int cam_res[2] = {640, 480};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  const real_t cam_data[8] = {320, 240, 320, 240, 0.0, 0.0, 0.0, 0.0};
  camera_params_setup(&cam, cam_idx, cam_res, proj_model, dist_model, cam_data);

  // Project point from world to image plane
  real_t T_WB[4 * 4] = {0};
  real_t T_BW[4 * 4] = {0};
  real_t T_BCi[4 * 4] = {0};
  real_t T_CiB[4 * 4] = {0};
  real_t T_CiW[4 * 4] = {0};
  real_t p_Ci[3] = {0};
  real_t z[2];
  tf(pose_data, T_WB);
  tf(exts_data, T_BCi);
  tf_inv(T_WB, T_BW);
  tf_inv(T_BCi, T_CiB);
  dot(T_CiB, 4, 4, T_BW, 4, 4, T_CiW);
  tf_point(T_CiW, p_W, p_Ci);
  pinhole_radtan4_project(cam_data, p_Ci, z);

  // Setup camera factor
  vision_factor_t factor;
  real_t var[2] = {1.0, 1.0};
  vision_factor_setup(&factor, &pose, &cam_exts, &feature, &cam, z, var);

  // Check Jacobians
  const real_t step_size = 1e-8;
  const real_t tol = 1e-4;
  CHECK_FACTOR_J(0, factor, vision_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(1, factor, vision_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(2, factor, vision_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(3, factor, vision_factor_eval, step_size, tol, 0);

  return 0;
}

int test_joint_factor() {
  // Joint angle
  const timestamp_t ts = 0;
  const int joint_idx = 0;
  const real_t z = 0.01;
  joint_t joint;
  joint_setup(&joint, ts, joint_idx, z);

  // Joint angle factor
  joint_factor_t factor;
  const real_t var = 0.1;
  joint_factor_setup(&factor, &joint, z, var);

  // Evaluate
  joint_factor_eval(&factor);

  // Check Jacobians
  const double tol = 1e-4;
  const double step_size = 1e-8;
  CHECK_FACTOR_J(0, factor, joint_factor_eval, step_size, tol, 0);

  return 0;
}

typedef struct test_calib_camera_data_t {
  real_t T_WF[4 * 4];
  real_t T_WB[4 * 4];
  real_t T_BF[4 * 4];
  real_t T_BCi[4 * 4];

  pose_t fiducial;     // T_WF
  pose_t pose;         // T_WB
  pose_t rel_pose;     // T_BF
  extrinsic_t cam_ext; // T_BCi
  camera_params_t cam_params;

  int cam_idx;
  int tag_id;
  int corner_idx;
  real_t p_FFi[3];
  real_t z[2];
} test_calib_camera_data_t;

void test_calib_camera_data_setup(test_calib_camera_data_t *data) {
  // Calibration target pose T_WF
  real_t fiducial_data[7] = {0};
  real_t ypr_WF[3] = {-M_PI / 2.0, 0.0, M_PI / 2.0};
  real_t r_WF[3] = {0.01, 0.01, 0.01};
  tf_er(ypr_WF, r_WF, data->T_WF);
  tf_vector(data->T_WF, fiducial_data);
  pose_setup(&data->fiducial, 0, fiducial_data);

  // Body pose T_WB
  real_t pose_data[7] = {0};
  real_t ypr_WB[3] = {-M_PI / 2.0, 0.0, -M_PI / 2.0};
  real_t r_WB[3] = {-10.0, 0.001, 0.001};
  tf_er(ypr_WB, r_WB, data->T_WB);
  tf_vector(data->T_BF, pose_data);
  pose_setup(&data->pose, 0, pose_data);

  // Relative pose T_BF
  real_t rel_pose_data[7] = {0};
  TF_INV(data->T_WB, T_BW);
  tf_chain2(2, T_BW, data->T_WF, data->T_BF);
  tf_vector(data->T_BF, rel_pose_data);
  pose_setup(&data->rel_pose, 0, rel_pose_data);

  // Camera extrinsics T_BCi
  real_t cam_ext_data[7] = {0};
  real_t ypr_BCi[3] = {0.01, 0.01, 0.0};
  real_t r_BCi[3] = {0.001, 0.001, 0.001};
  tf_er(ypr_BCi, r_BCi, data->T_BCi);
  tf_vector(data->T_BCi, cam_ext_data);
  extrinsic_setup(&data->cam_ext, cam_ext_data);

  // Camera
  data->cam_idx = 0;
  const int cam_res[2] = {640, 480};
  const real_t fov = 90.0;
  const real_t fx = pinhole_focal(cam_res[0], fov);
  const real_t fy = pinhole_focal(cam_res[0], fov);
  const real_t cx = cam_res[0] / 2.0;
  const real_t cy = cam_res[1] / 2.0;
  const real_t cam_data[8] = {fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  camera_params_setup(&data->cam_params,
                      data->cam_idx,
                      cam_res,
                      proj_model,
                      dist_model,
                      cam_data);

  // Project to image plane
  aprilgrid_t grid;
  aprilgrid_setup(0, &grid);

  data->tag_id = 1;
  data->corner_idx = 2;
  aprilgrid_object_point(&grid, data->tag_id, data->corner_idx, data->p_FFi);

  TF_INV(data->T_BCi, T_CiB);
  TF_CHAIN(T_CiF, 2, T_CiB, data->T_BF);
  TF_POINT(T_CiF, data->p_FFi, p_CiFi);
  pinhole_radtan4_project(cam_data, p_CiFi, data->z);
}

int test_calib_camera_factor() {
  // Setup
  test_calib_camera_data_t calib_data;
  test_calib_camera_data_setup(&calib_data);

  calib_camera_factor_t factor;
  const real_t var[2] = {1.0, 1.0};
  calib_camera_factor_setup(&factor,
                            &calib_data.rel_pose,
                            &calib_data.cam_ext,
                            &calib_data.cam_params,
                            calib_data.cam_idx,
                            calib_data.tag_id,
                            calib_data.corner_idx,
                            calib_data.p_FFi,
                            calib_data.z,
                            var);

  // Evaluate
  calib_camera_factor_eval(&factor);

  // Check Jacobians
  const double tol = 1e-4;
  const double step_size = 1e-8;
  CHECK_FACTOR_J(0, factor, calib_camera_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(1, factor, calib_camera_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(2, factor, calib_camera_factor_eval, step_size, tol, 0);

  return 0;
}

typedef struct test_calib_imucam_data_t {
  real_t T_WF[4 * 4];
  real_t T_WS[4 * 4];
  real_t T_SC0[4 * 4];
  real_t T_C0Ci[4 * 4];

  pose_t fiducial;     // T_WF
  pose_t imu_pose;     // T_WB
  extrinsic_t imu_ext; // T_SC0
  extrinsic_t cam_ext; // T_C0Ci
  camera_params_t cam_params;
  time_delay_t time_delay;

  int cam_idx;
  int tag_id;
  int corner_idx;
  real_t p_FFi[3];
  real_t z[2];
} test_calib_imucam_data_t;

void test_calib_imucam_data_setup(test_calib_imucam_data_t *data) {
  // Calibration target pose T_WF
  real_t fiducial_data[7] = {0};
  real_t ypr_WF[3] = {-M_PI / 2.0, 0.0, M_PI / 2.0};
  real_t r_WF[3] = {0.01, 0.01, 0.01};
  tf_er(ypr_WF, r_WF, data->T_WF);
  tf_vector(data->T_WF, fiducial_data);
  pose_setup(&data->fiducial, 0, fiducial_data);

  // IMU pose T_WS
  real_t imu_pose_data[7] = {0};
  real_t ypr_WS[3] = {-M_PI / 2.0, 0.0, -M_PI / 2.0};
  real_t r_WS[3] = {-10.0, 0.001, 0.001};
  tf_er(ypr_WS, r_WS, data->T_WS);
  tf_vector(data->T_WS, imu_pose_data);
  pose_setup(&data->imu_pose, 0, imu_pose_data);

  // IMU extrinsics T_SC0
  real_t imu_ext_data[7] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
  tf(imu_ext_data, data->T_SC0);
  extrinsic_setup(&data->imu_ext, imu_ext_data);

  // Camera extrinsics T_C0Ci
  real_t cam_ext_data[7] = {0};
  real_t ypr_C0Ci[3] = {0.01, 0.01, 0.0};
  real_t r_C0Ci[3] = {0.001, 0.001, 0.001};
  tf_er(ypr_C0Ci, r_C0Ci, data->T_C0Ci);
  tf_vector(data->T_C0Ci, cam_ext_data);
  extrinsic_setup(&data->cam_ext, cam_ext_data);

  // Camera
  data->cam_idx = 0;
  const int cam_res[2] = {640, 480};
  const real_t fov = 90.0;
  const real_t fx = pinhole_focal(cam_res[0], fov);
  const real_t fy = pinhole_focal(cam_res[0], fov);
  const real_t cx = cam_res[0] / 2.0;
  const real_t cy = cam_res[1] / 2.0;
  const real_t cam_data[8] = {fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  camera_params_setup(&data->cam_params,
                      data->cam_idx,
                      cam_res,
                      proj_model,
                      dist_model,
                      cam_data);

  // Time delay
  time_delay_setup(&data->time_delay, 0.0);

  // Project to image plane
  aprilgrid_t grid;
  aprilgrid_setup(0, &grid);

  data->tag_id = 1;
  data->corner_idx = 2;
  aprilgrid_object_point(&grid, data->tag_id, data->corner_idx, data->p_FFi);

  TF_INV(data->T_WS, T_SW);
  TF_INV(data->T_SC0, T_C0S);
  TF_INV(data->T_C0Ci, T_CiC0);
  TF_CHAIN(T_CiF, 4, T_CiC0, T_C0S, T_SW, data->T_WF);
  TF_POINT(T_CiF, data->p_FFi, p_CiFi);
  pinhole_radtan4_project(cam_data, p_CiFi, data->z);
}

int test_calib_imucam_factor() {
  // Setup
  test_calib_imucam_data_t calib_data;
  test_calib_imucam_data_setup(&calib_data);

  calib_imucam_factor_t factor;
  const real_t var[2] = {1.0, 1.0};
  const real_t v[2] = {0.01, 0.02};
  calib_imucam_factor_setup(&factor,
                            &calib_data.fiducial,
                            &calib_data.imu_pose,
                            &calib_data.imu_ext,
                            &calib_data.cam_ext,
                            &calib_data.cam_params,
                            &calib_data.time_delay,
                            calib_data.cam_idx,
                            calib_data.tag_id,
                            calib_data.corner_idx,
                            calib_data.p_FFi,
                            calib_data.z,
                            v,
                            var);

  // Evaluate
  calib_imucam_factor_eval(&factor);

  // Check Jacobians
  const double tol = 1e-2;
  const double step_size = 1e-8;
  const int debug = 0;
  CHECK_FACTOR_J(0, factor, calib_imucam_factor_eval, step_size, tol, debug);
  CHECK_FACTOR_J(1, factor, calib_imucam_factor_eval, step_size, tol, debug);
  CHECK_FACTOR_J(2, factor, calib_imucam_factor_eval, step_size, tol, debug);
  CHECK_FACTOR_J(3, factor, calib_imucam_factor_eval, step_size, tol, debug);
  CHECK_FACTOR_J(4, factor, calib_imucam_factor_eval, step_size, tol, debug);
  CHECK_FACTOR_J(5, factor, calib_imucam_factor_eval, step_size, tol, debug);

  return 0;
}

static void setup_calib_gimbal_factor(calib_gimbal_factor_t *factor,
                                      extrinsic_t *fiducial_exts,
                                      extrinsic_t *gimbal_exts,
                                      pose_t *pose,
                                      extrinsic_t *link0,
                                      extrinsic_t *link1,
                                      joint_t *joint0,
                                      joint_t *joint1,
                                      joint_t *joint2,
                                      extrinsic_t *cam_exts,
                                      camera_params_t *cam) {
  // Body pose T_WB
  real_t ypr_WB[3] = {0.0, 0.0, 0.0};
  real_t r_WB[3] = {0.0, 0.0, 0.0};
  real_t T_WB[4 * 4] = {0};
  tf_er(ypr_WB, r_WB, T_WB);

  real_t x_WB[7] = {0};
  tf_vector(T_WB, x_WB);
  pose_setup(pose, 0, x_WB);

  // Fiducial pose T_WF
  real_t ypr_WF[3] = {-M_PI / 2.0, 0.0, M_PI / 2.0};
  real_t r_WF[3] = {0.5, 0.0, 0.0};
  real_t T_WF[4 * 4] = {0};
  tf_er(ypr_WF, r_WF, T_WF);

  real_t x_WF[7] = {0};
  tf_vector(T_WF, x_WF);
  extrinsic_setup(fiducial_exts, x_WF);

  // Relative fiducial pose T_BF
  real_t T_BF[4 * 4] = {0};
  TF_INV(T_WB, T_BW);
  dot(T_BW, 4, 4, T_WF, 4, 4, T_BF);

  // Gimbal extrinsic
  real_t ypr_BM0[3] = {0.01, 0.01, 0.01};
  real_t r_BM0[3] = {0.0, 0.0, 0.0};
  real_t T_BM0[4 * 4] = {0};
  gimbal_setup_extrinsic(ypr_BM0, r_BM0, T_BM0, gimbal_exts);

  // Roll link
  real_t ypr_L0M1[3] = {0.0, M_PI / 2, 0.0};
  real_t r_L0M1[3] = {-0.1, 0.0, 0.15};
  real_t T_L0M1[4 * 4] = {0};
  gimbal_setup_extrinsic(ypr_L0M1, r_L0M1, T_L0M1, link0);

  // Pitch link
  real_t ypr_L1M2[3] = {0.0, 0.0, -M_PI / 2.0};
  real_t r_L1M2[3] = {0.0, -0.05, 0.1};
  real_t T_L1M2[4 * 4] = {0};
  gimbal_setup_extrinsic(ypr_L1M2, r_L1M2, T_L1M2, link1);

  // Joint0
  const real_t th0 = 0.01;
  real_t T_M0L0[4 * 4] = {0};
  gimbal_setup_joint(0, 0, th0, T_M0L0, joint0);

  // Joint1
  const real_t th1 = 0.02;
  real_t T_M1L1[4 * 4] = {0};
  gimbal_setup_joint(0, 1, th1, T_M1L1, joint1);

  // Joint2
  const real_t th2 = 0.03;
  real_t T_M2L2[4 * 4] = {0};
  gimbal_setup_joint(0, 2, th2, T_M2L2, joint2);

  // Camera extrinsic
  const real_t ypr_L2C0[3] = {-M_PI / 2, M_PI / 2, 0.0};
  const real_t r_L2C0[3] = {0.0, -0.05, 0.12};
  real_t T_L2C0[4 * 4] = {0};
  gimbal_setup_extrinsic(ypr_L2C0, r_L2C0, T_L2C0, cam_exts);

  // Camera parameters K
  const int cam_idx = 0;
  const int cam_res[2] = {640, 480};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  const real_t fov = 120.0;
  const real_t fx = pinhole_focal(cam_res[0], fov);
  const real_t fy = pinhole_focal(cam_res[0], fov);
  const real_t cx = cam_res[0] / 2.0;
  const real_t cy = cam_res[1] / 2.0;
  const real_t cam_params[8] = {fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0};
  camera_params_setup(cam,
                      cam_idx,
                      cam_res,
                      proj_model,
                      dist_model,
                      cam_params);

  // Form T_C0F
  real_t T_BC0[4 * 4] = {0};
  real_t T_C0F[4 * 4] = {0};
  tf_chain2(7, T_BM0, T_M0L0, T_L0M1, T_M1L1, T_L1M2, T_M2L2, T_L2C0, T_BC0);
  TF_INV(T_BC0, T_C0B);
  tf_chain2(2, T_C0B, T_BF, T_C0F);

  // Project point to image plane
  const real_t p_FFi[3] = {0.0, 0.0, 0.0};
  real_t p_C0Fi[3] = {0};
  real_t z[2] = {0};
  tf_point(T_C0F, p_FFi, p_C0Fi);
  pinhole_radtan4_project(cam_params, p_C0Fi, z);

  // Setup factor
  const timestamp_t ts = 0;
  const int tag_id = 0;
  const int corner_idx = 0;
  const real_t var[2] = {1.0, 1.0};
  calib_gimbal_factor_setup(factor,
                            fiducial_exts,
                            gimbal_exts,
                            pose,
                            link0,
                            link1,
                            joint0,
                            joint1,
                            joint2,
                            cam_exts,
                            cam,
                            ts,
                            cam_idx,
                            tag_id,
                            corner_idx,
                            p_FFi,
                            z,
                            var);
}

int test_calib_gimbal_factor() {
  calib_gimbal_factor_t factor;
  extrinsic_t fiducial_exts;
  extrinsic_t gimbal_exts;
  pose_t pose;
  extrinsic_t link0;
  extrinsic_t link1;
  joint_t joint0;
  joint_t joint1;
  joint_t joint2;
  extrinsic_t cam_exts;
  camera_params_t cam;
  setup_calib_gimbal_factor(&factor,
                            &fiducial_exts,
                            &gimbal_exts,
                            &pose,
                            &link0,
                            &link1,
                            &joint0,
                            &joint1,
                            &joint2,
                            &cam_exts,
                            &cam);

  // Evaluate
  calib_gimbal_factor_eval(&factor);

  // Check Jacobians
  const double tol = 1e-4;
  const double step_size = 1e-8;
  CHECK_FACTOR_J(0, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(1, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(2, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(3, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(4, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(5, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(6, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(7, factor, calib_gimbal_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(8, factor, calib_gimbal_factor_eval, step_size, tol, 0);

  return 0;
}

int test_imu_buf_setup() {
  imu_buf_t imu_buf;
  imu_buf_setup(&imu_buf);

  return 0;
}

int test_imu_buf_add() {
  imu_buf_t imu_buf;
  imu_buf_setup(&imu_buf);

  timestamp_t ts = 0;
  real_t acc[3] = {1.0, 2.0, 3.0};
  real_t gyr[3] = {1.0, 2.0, 3.0};
  imu_buf_add(&imu_buf, ts, acc, gyr);

  MU_ASSERT(imu_buf.size == 1);
  MU_ASSERT(imu_buf.ts[0] == ts);
  MU_ASSERT(fltcmp(imu_buf.acc[0][0], 1.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.acc[0][1], 2.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.acc[0][2], 3.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.gyr[0][0], 1.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.gyr[0][1], 2.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.gyr[0][2], 3.0) == 0);

  return 0;
}

int test_imu_buf_clear() {
  imu_buf_t imu_buf;
  imu_buf_setup(&imu_buf);

  timestamp_t ts = 0;
  real_t acc[3] = {1.0, 2.0, 3.0};
  real_t gyr[3] = {1.0, 2.0, 3.0};
  imu_buf_add(&imu_buf, ts, acc, gyr);
  imu_buf_clear(&imu_buf);

  MU_ASSERT(imu_buf.size == 0);
  MU_ASSERT(imu_buf.ts[0] == 0);
  MU_ASSERT(fltcmp(imu_buf.acc[0][0], 0.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.acc[0][1], 0.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.acc[0][2], 0.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.gyr[0][0], 0.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.gyr[0][1], 0.0) == 0);
  MU_ASSERT(fltcmp(imu_buf.gyr[0][2], 0.0) == 0);

  return 0;
}

int test_imu_buf_copy() {
  imu_buf_t imu_buf;
  imu_buf_setup(&imu_buf);

  timestamp_t ts = 0;
  real_t acc[3] = {1.0, 2.0, 3.0};
  real_t gyr[3] = {1.0, 2.0, 3.0};
  imu_buf_add(&imu_buf, ts, acc, gyr);

  imu_buf_t imu_buf2;
  imu_buf_setup(&imu_buf2);
  imu_buf_copy(&imu_buf, &imu_buf2);

  MU_ASSERT(imu_buf2.size == 1);
  MU_ASSERT(imu_buf2.ts[0] == ts);
  MU_ASSERT(fltcmp(imu_buf2.acc[0][0], 1.0) == 0);
  MU_ASSERT(fltcmp(imu_buf2.acc[0][1], 2.0) == 0);
  MU_ASSERT(fltcmp(imu_buf2.acc[0][2], 3.0) == 0);
  MU_ASSERT(fltcmp(imu_buf2.gyr[0][0], 1.0) == 0);
  MU_ASSERT(fltcmp(imu_buf2.gyr[0][1], 2.0) == 0);
  MU_ASSERT(fltcmp(imu_buf2.gyr[0][2], 3.0) == 0);

  return 0;
}

typedef struct imu_test_data_t {
  size_t nb_measurements;
  real_t *timestamps;
  real_t **poses;
  real_t **velocities;
  real_t **imu_acc;
  real_t **imu_gyr;
} imu_test_data_t;

static int setup_imu_test_data(imu_test_data_t *test_data) {
  // Circle trajectory configurations
  const real_t imu_rate = 200.0;
  const real_t circle_r = 5.0;
  const real_t circle_v = 1.0;
  const real_t circle_dist = 2.0 * M_PI * circle_r;
  const real_t time_taken = circle_dist / circle_v;
  const real_t w = -2.0 * M_PI * (1.0 / time_taken);
  const real_t theta_init = M_PI;
  const real_t yaw_init = M_PI / 2.0;

  // Allocate memory for test data
  test_data->nb_measurements = time_taken * imu_rate;
  test_data->timestamps = CALLOC(real_t, test_data->nb_measurements);
  test_data->poses = CALLOC(real_t *, test_data->nb_measurements);
  test_data->velocities = CALLOC(real_t *, test_data->nb_measurements);
  test_data->imu_acc = CALLOC(real_t *, test_data->nb_measurements);
  test_data->imu_gyr = CALLOC(real_t *, test_data->nb_measurements);

  // Simulate IMU poses
  const real_t dt = 1.0 / imu_rate;
  timestamp_t ts = 0.0;
  real_t theta = theta_init;
  real_t yaw = yaw_init;

  for (size_t k = 0; k < test_data->nb_measurements; k++) {
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
    // print_vector("pose", pose, 7);

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
    test_data->timestamps[k] = ts;
    test_data->poses[k] = vector_malloc(pose, 7);
    test_data->velocities[k] = vector_malloc(v_WS, 3);
    test_data->imu_acc[k] = vector_malloc(acc, 3);
    test_data->imu_gyr[k] = vector_malloc(gyr, 3);

    theta += w * dt;
    yaw += w * dt;
    ts += sec2ts(dt);
  }

  return 0;
}

static void free_imu_test_data(imu_test_data_t *test_data) {
  for (size_t k = 0; k < test_data->nb_measurements; k++) {
    free(test_data->poses[k]);
    free(test_data->velocities[k]);
    free(test_data->imu_acc[k]);
    free(test_data->imu_gyr[k]);
  }

  free(test_data->timestamps);
  free(test_data->poses);
  free(test_data->velocities);
  free(test_data->imu_acc);
  free(test_data->imu_gyr);
}

int test_imu_factor_propagate_step() {
  // Setup test data
  imu_test_data_t test_data;
  setup_imu_test_data(&test_data);

  // Setup IMU buffer
  const int n = 9;
  imu_buf_t imu_buf;
  imu_buf_setup(&imu_buf);
  for (int k = 0; k < n; k++) {
    const timestamp_t ts = test_data.timestamps[k];
    const real_t *acc = test_data.imu_acc[k];
    const real_t *gyr = test_data.imu_gyr[k];
    imu_buf_add(&imu_buf, ts, acc, gyr);
  }

  // Setup state
  real_t r[3] = {0.0, 0.0, 0.0};
  real_t v[3] = {0.0, 0.0, 0.0};
  real_t q[4] = {1.0, 0.0, 0.0, 0.0};
  real_t ba[3] = {0};
  real_t bg[3] = {0};

  // Integrate imu measuremenets
  real_t Dt = 0.0;
  real_t dt = 0.0;
  for (int k = 0; k < imu_buf.size; k++) {
    if (k + 1 < imu_buf.size) {
      const timestamp_t ts_i = imu_buf.ts[k];
      const timestamp_t ts_j = imu_buf.ts[k + 1];
      dt = ts2sec(ts_j) - ts2sec(ts_i);
    }
    const real_t *a = imu_buf.acc[k];
    const real_t *w = imu_buf.gyr[k];
    imu_factor_propagate_step(r, v, q, ba, bg, a, w, dt);
    Dt += dt;
  }

  TF(test_data.poses[0], T_WS_i_gnd);
  TF(test_data.poses[n], T_WS_j_gnd);
  TF_QR(q, r, dT);
  TF_CHAIN(T_WS_j_est, 2, T_WS_i_gnd, dT);

  real_t dr[3] = {0};
  real_t dtheta = 0.0;
  TF_VECTOR(T_WS_j_est, pose_j_est);
  TF_VECTOR(T_WS_j_gnd, pose_j_gnd);
  pose_diff2(pose_j_gnd, pose_j_est, dr, &dtheta);
  MU_ASSERT(fltcmp(dtheta, 0.0) == 0);

  // Clean up
  free_imu_test_data(&test_data);

  return 0;
}

// int test_imu_factor_setup() {
//   // Setup test data
//   imu_test_data_t test_data;
//   setup_imu_test_data(&test_data);

//   // Setup IMU buffer
//   imu_buf_t imu_buf;
//   imu_buf_setup(&imu_buf);
//   for (int k = 0; k < 10; k++) {
//     const timestamp_t ts = test_data.timestamps[k];
//     const real_t *acc = test_data.imu_acc[k];
//     const real_t *gyr = test_data.imu_gyr[k];
//     imu_buf_add(&imu_buf, ts, acc, gyr);
//   }

//   // Setup IMU factor
//   const int idx_i = 0;
//   const int idx_j = 10 - 1;
//   const timestamp_t ts_i = test_data.timestamps[idx_i];
//   const timestamp_t ts_j = test_data.timestamps[idx_j];
//   const real_t *v_i = test_data.velocities[idx_i];
//   const real_t ba_i[3] = {0, 0, 0};
//   const real_t bg_i[3] = {0, 0, 0};
//   const real_t *v_j = test_data.velocities[idx_j];
//   const real_t ba_j[3] = {0, 0, 0};
//   const real_t bg_j[3] = {0, 0, 0};
//   pose_t pose_i;
//   pose_t pose_j;
//   velocity_t vel_i;
//   velocity_t vel_j;
//   imu_biases_t biases_i;
//   imu_biases_t biases_j;
//   pose_setup(&pose_i, ts_i, test_data.poses[idx_i]);
//   pose_setup(&pose_j, ts_j, test_data.poses[idx_j]);
//   velocity_setup(&vel_i, ts_i, v_i);
//   velocity_setup(&vel_j, ts_j, v_j);
//   imu_biases_setup(&biases_i, ts_i, ba_i, bg_i);
//   imu_biases_setup(&biases_j, ts_j, ba_j, bg_j);

//   imu_params_t imu_params;
//   imu_params.imu_idx = 0;
//   imu_params.rate = 200.0;
//   imu_params.sigma_a = 0.08;
//   imu_params.sigma_g = 0.004;
//   imu_params.sigma_aw = 0.00004;
//   imu_params.sigma_gw = 2.0e-6;
//   imu_params.g = 9.81;

//   imu_factor_t imu_factor;
//   imu_factor_setup(&imu_factor,
//                    &imu_params,
//                    &imu_buf,
//                    &pose_i,
//                    &vel_i,
//                    &biases_i,
//                    &pose_j,
//                    &vel_j,
//                    &biases_j);

//   MU_ASSERT(imu_factor.pose_i == &pose_i);
//   MU_ASSERT(imu_factor.vel_i == &vel_i);
//   MU_ASSERT(imu_factor.biases_i == &biases_i);
//   MU_ASSERT(imu_factor.pose_i == &pose_i);
//   MU_ASSERT(imu_factor.vel_j == &vel_j);
//   MU_ASSERT(imu_factor.biases_j == &biases_j);

//   // Clean up
//   free_imu_test_data(&test_data);

//   return 0;
// }

int test_imu_factor_eval() {
  // Setup test data
  imu_test_data_t test_data;
  setup_imu_test_data(&test_data);

  // Setup IMU buffer
  imu_buf_t imu_buf;
  imu_buf_setup(&imu_buf);
  for (int k = 0; k < 20; k++) {
    const timestamp_t ts = test_data.timestamps[k];
    const real_t *acc = test_data.imu_acc[k];
    const real_t *gyr = test_data.imu_gyr[k];
    imu_buf_add(&imu_buf, ts, acc, gyr);
  }

  // Setup IMU factor
  const int idx_i = 0;
  const int idx_j = 20 - 1;
  const timestamp_t ts_i = test_data.timestamps[idx_i];
  const timestamp_t ts_j = test_data.timestamps[idx_j];
  const real_t *v_i = test_data.velocities[idx_i];
  const real_t ba_i[3] = {0, 0, 0};
  const real_t bg_i[3] = {0, 0, 0};
  const real_t *v_j = test_data.velocities[idx_j];
  const real_t ba_j[3] = {0, 0, 0};
  const real_t bg_j[3] = {0, 0, 0};
  pose_t pose_i;
  pose_t pose_j;
  velocity_t vel_i;
  velocity_t vel_j;
  imu_biases_t biases_i;
  imu_biases_t biases_j;
  pose_setup(&pose_i, ts_i, test_data.poses[idx_i]);
  pose_setup(&pose_j, ts_j, test_data.poses[idx_j]);
  velocity_setup(&vel_i, ts_i, v_i);
  velocity_setup(&vel_j, ts_j, v_j);
  imu_biases_setup(&biases_i, ts_i, ba_i, bg_i);
  imu_biases_setup(&biases_j, ts_j, ba_j, bg_j);

  imu_params_t imu_params;
  imu_params.imu_idx = 0;
  imu_params.rate = 200.0;
  imu_params.sigma_a = 0.08;
  imu_params.sigma_g = 0.004;
  imu_params.sigma_aw = 0.00004;
  imu_params.sigma_gw = 2.0e-6;
  imu_params.g = 9.81;

  imu_factor_t factor;
  imu_factor_setup(&factor,
                   &imu_params,
                   &imu_buf,
                   &pose_i,
                   &vel_i,
                   &biases_i,
                   &pose_j,
                   &vel_j,
                   &biases_j);

  MU_ASSERT(factor.pose_i == &pose_i);
  MU_ASSERT(factor.vel_i == &vel_i);
  MU_ASSERT(factor.biases_i == &biases_i);
  MU_ASSERT(factor.pose_i == &pose_i);
  MU_ASSERT(factor.vel_j == &vel_j);
  MU_ASSERT(factor.biases_j == &biases_j);

  // Evaluate IMU factor
  imu_factor_eval(&factor);

  // Check Jacobians
  const double tol = 1e-4;
  const double step_size = 1e-8;
  eye(factor.sqrt_info, 15, 15);
  CHECK_FACTOR_J(0, factor, imu_factor_eval, step_size, 1e-3, 0);
  CHECK_FACTOR_J(1, factor, imu_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(2, factor, imu_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(3, factor, imu_factor_eval, step_size, 1e-3, 0);
  CHECK_FACTOR_J(4, factor, imu_factor_eval, step_size, tol, 0);
  CHECK_FACTOR_J(5, factor, imu_factor_eval, step_size, tol, 0);

  // Clean up
  free_imu_test_data(&test_data);

  return 0;
}

int test_inertial_odometry() {
  // Setup test data
  imu_test_data_t test_data;
  setup_imu_test_data(&test_data);

  // Inertial Odometry
  const int num_partitions = test_data.nb_measurements / 20.0;
  const size_t N = test_data.nb_measurements / (real_t) num_partitions;
  inertial_odometry_t *odom = MALLOC(inertial_odometry_t, 1);
  // -- IMU params
  odom->imu_params.imu_idx = 0;
  odom->imu_params.rate = 200.0;
  odom->imu_params.sigma_a = 0.08;
  odom->imu_params.sigma_g = 0.004;
  odom->imu_params.sigma_aw = 0.00004;
  odom->imu_params.sigma_gw = 2.0e-6;
  odom->imu_params.g = 9.81;
  // -- Variables
  odom->num_factors = 0;
  odom->factors = MALLOC(imu_factor_t, num_partitions);
  odom->poses = MALLOC(pose_t, num_partitions + 1);
  odom->vels = MALLOC(velocity_t, num_partitions + 1);
  odom->biases = MALLOC(imu_biases_t, num_partitions + 1);

  const timestamp_t ts_i = test_data.timestamps[0];
  const real_t *v_i = test_data.velocities[0];
  const real_t ba_i[3] = {0, 0, 0};
  const real_t bg_i[3] = {0, 0, 0};
  pose_setup(&odom->poses[0], ts_i, test_data.poses[0]);
  velocity_setup(&odom->vels[0], ts_i, v_i);
  imu_biases_setup(&odom->biases[0], ts_i, ba_i, bg_i);

  for (int i = 1; i < num_partitions; i++) {
    const int ks = i * N;
    const int ke = PMIN((i + 1) * N - 1, test_data.nb_measurements - 1);

    // Setup imu buffer
    imu_buf_t imu_buf;
    imu_buf_setup(&imu_buf);
    for (size_t k = 0; k < N; k++) {
      const timestamp_t ts = test_data.timestamps[ks + k];
      const real_t *acc = test_data.imu_acc[ks + k];
      const real_t *gyr = test_data.imu_gyr[ks + k];
      imu_buf_add(&imu_buf, ts, acc, gyr);
    }

    // Setup parameters
    const timestamp_t ts_j = test_data.timestamps[ke];
    const real_t *v_j = test_data.velocities[ke];
    const real_t ba_j[3] = {0, 0, 0};
    const real_t bg_j[3] = {0, 0, 0};
    pose_setup(&odom->poses[i], ts_j, test_data.poses[ke]);
    velocity_setup(&odom->vels[i], ts_j, v_j);
    imu_biases_setup(&odom->biases[i], ts_j, ba_j, bg_j);

    // Setup IMU factor
    imu_factor_setup(&odom->factors[i - 1],
                     &odom->imu_params,
                     &imu_buf,
                     &odom->poses[i - 1],
                     &odom->vels[i - 1],
                     &odom->biases[i - 1],
                     &odom->poses[i],
                     &odom->vels[i],
                     &odom->biases[i]);
    odom->num_factors++;
  }

  // Save ground truth
  inertial_odometry_save(odom, "/tmp/imu_odom-gnd.csv");

  // Perturb ground truth
  for (int k = 0; k <= odom->num_factors; k++) {
    odom->poses[k].data[0] += randf(-1.0, 1.0);
    odom->poses[k].data[1] += randf(-1.0, 1.0);
    odom->poses[k].data[2] += randf(-1.0, 1.0);

    odom->vels[k].data[0] += randf(-1.0, 1.0);
    odom->vels[k].data[1] += randf(-1.0, 1.0);
    odom->vels[k].data[2] += randf(-1.0, 1.0);
  }
  inertial_odometry_save(odom, "/tmp/imu_odom-init.csv");

  // Solve
  solver_t solver;
  solver_setup(&solver);
  solver.param_order_func = &inertial_odometry_param_order;
  solver.linearize_func = &inertial_odometry_linearize_compact;

  // printf("num_measurements: %ld\n", test_data.nb_measurements);
  // printf("num_factors: %d\n", odom->num_factors);
  solver_solve(&solver, odom);
  inertial_odometry_save(odom, "/tmp/imu_odom-est.csv");

  // Clean up
  inertial_odometry_free(odom);
  free_imu_test_data(&test_data);

  return 0;
}

#ifdef USE_CERES

/**
 * This is the equivalent of a use-defined CostFunction in the C++ Ceres API.
 * This is passed as a callback to the Ceres C API, which internally converts
 * the callback into a CostFunction.
 */
static int ceres_exp_residual(void *user_data,
                              double **parameters,
                              double *residuals,
                              double **jacobians) {
  double *measurement = (double *) user_data;
  double x = measurement[0];
  double y = measurement[1];
  double m = parameters[0][0];
  double c = parameters[1][0];
  residuals[0] = y - exp(m * x + c);
  if (jacobians == NULL) {
    return 1;
  }
  if (jacobians[0] != NULL) {
    jacobians[0][0] = -x * exp(m * x + c); /* dr/dm */
  }
  if (jacobians[1] != NULL) {
    jacobians[1][0] = -exp(m * x + c); /* dr/dc */
  }
  return 1;
}

int test_ceres_example() {
  int num_observations = 67;
  double data[] = {
      0.000000e+00, 1.133898e+00, 7.500000e-02, 1.334902e+00, 1.500000e-01,
      1.213546e+00, 2.250000e-01, 1.252016e+00, 3.000000e-01, 1.392265e+00,
      3.750000e-01, 1.314458e+00, 4.500000e-01, 1.472541e+00, 5.250000e-01,
      1.536218e+00, 6.000000e-01, 1.355679e+00, 6.750000e-01, 1.463566e+00,
      7.500000e-01, 1.490201e+00, 8.250000e-01, 1.658699e+00, 9.000000e-01,
      1.067574e+00, 9.750000e-01, 1.464629e+00, 1.050000e+00, 1.402653e+00,
      1.125000e+00, 1.713141e+00, 1.200000e+00, 1.527021e+00, 1.275000e+00,
      1.702632e+00, 1.350000e+00, 1.423899e+00, 1.425000e+00, 1.543078e+00,
      1.500000e+00, 1.664015e+00, 1.575000e+00, 1.732484e+00, 1.650000e+00,
      1.543296e+00, 1.725000e+00, 1.959523e+00, 1.800000e+00, 1.685132e+00,
      1.875000e+00, 1.951791e+00, 1.950000e+00, 2.095346e+00, 2.025000e+00,
      2.361460e+00, 2.100000e+00, 2.169119e+00, 2.175000e+00, 2.061745e+00,
      2.250000e+00, 2.178641e+00, 2.325000e+00, 2.104346e+00, 2.400000e+00,
      2.584470e+00, 2.475000e+00, 1.914158e+00, 2.550000e+00, 2.368375e+00,
      2.625000e+00, 2.686125e+00, 2.700000e+00, 2.712395e+00, 2.775000e+00,
      2.499511e+00, 2.850000e+00, 2.558897e+00, 2.925000e+00, 2.309154e+00,
      3.000000e+00, 2.869503e+00, 3.075000e+00, 3.116645e+00, 3.150000e+00,
      3.094907e+00, 3.225000e+00, 2.471759e+00, 3.300000e+00, 3.017131e+00,
      3.375000e+00, 3.232381e+00, 3.450000e+00, 2.944596e+00, 3.525000e+00,
      3.385343e+00, 3.600000e+00, 3.199826e+00, 3.675000e+00, 3.423039e+00,
      3.750000e+00, 3.621552e+00, 3.825000e+00, 3.559255e+00, 3.900000e+00,
      3.530713e+00, 3.975000e+00, 3.561766e+00, 4.050000e+00, 3.544574e+00,
      4.125000e+00, 3.867945e+00, 4.200000e+00, 4.049776e+00, 4.275000e+00,
      3.885601e+00, 4.350000e+00, 4.110505e+00, 4.425000e+00, 4.345320e+00,
      4.500000e+00, 4.161241e+00, 4.575000e+00, 4.363407e+00, 4.650000e+00,
      4.161576e+00, 4.725000e+00, 4.619728e+00, 4.800000e+00, 4.737410e+00,
      4.875000e+00, 4.727863e+00, 4.950000e+00, 4.669206e+00,
  };

  /* Note: Typically it is better to compact m and c into one block,
   * but in this case use separate blocks to illustrate the use of
   * multiple parameter blocks. */
  double m = 0.0;
  double c = 0.0;
  double *parameter_pointers[] = {&m, &c};
  int parameter_sizes[2] = {1, 1};
  ceres_problem_t *problem;
  ceres_init();
  problem = ceres_create_problem();

  /* Add all the residuals. */
  for (int i = 0; i < num_observations; ++i) {
    ceres_problem_add_residual_block(problem,
                                     ceres_exp_residual,
                                     &data[2 * i],
                                     NULL,
                                     NULL,
                                     1,
                                     2,
                                     parameter_sizes,
                                     parameter_pointers);
  }

  ceres_solve(problem, 10);
  ceres_free_problem(problem);
  // printf("Initial m: 0.0, c: 0.0\n");
  // printf("Final m: %g, c: %g\n", m, c);

  return 0;
}

#endif // USE_CERES

int test_solver_setup() {
  solver_t solver;
  solver_setup(&solver);
  return 0;
}

typedef struct cam_view_t {
  pose_t pose;
  ba_factor_t factors[1000];
  int nb_factors;
  camera_params_t *cam_params;
} cam_view_t;

int test_solver_eval() {
  /* Load test data */
  const char *dir_path = TEST_SIM_DATA "/cam0";
  sim_camera_data_t *cam_data = sim_camera_data_load(dir_path);

  /* Camera parameters */
  camera_params_t cam;
  const int cam_idx = 0;
  const int cam_res[2] = {640, 480};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  const real_t params[8] = {640, 480, 320, 240, 0.0, 0.0, 0.0, 0.0};
  camera_params_setup(&cam, cam_idx, cam_res, proj_model, dist_model, params);

  /* Features container */
  features_t features;
  features_setup(&features);

  /* Loop over simulated camera frames */
  const real_t var[2] = {1.0, 1.0};
  cam_view_t *cam_views = MALLOC(cam_view_t, cam_data->nb_frames);
  for (int k = 0; k < cam_data->nb_frames; k++) {
    /* Camera frame */
    const sim_camera_frame_t *frame = cam_data->frames[k];

    /* Pose */
    pose_t *pose = &cam_views[k].pose;
    pose_setup(pose, frame->ts, cam_data->poses[k]);

    /* Add factors */
    cam_views[k].nb_factors = frame->nb_measurements;
    for (int i = 0; i < frame->nb_measurements; i++) {
      const int feature_id = frame->feature_ids[i];
      const real_t *z = frame->keypoints[i];

      /* Feature */
      feature_t *feature = NULL;
      if (features_exists(&features, feature_id)) {
        feature = features_get(&features, feature_id);
      } else {
        const real_t param[3] = {0};
        feature = features_add(&features, feature_id, param);
      }

      /* Factor */
      ba_factor_t *factor = &cam_views[k].factors[i];
      ba_factor_setup(factor, pose, feature, &cam, z, var);
    }
  }

  /* solver_t solver; */
  /* solver_setup(&solver); */

  /* Clean up */
  free(cam_views);
  sim_camera_data_free(cam_data);

  return 0;
}

int test_calib_gimbal_copy() {
  const char *data_path = "/tmp/sim_gimbal";
  calib_gimbal_t *src = calib_gimbal_load(data_path);
  calib_gimbal_t *dst = calib_gimbal_copy(src);

  MU_ASSERT(src != NULL);
  MU_ASSERT(dst != NULL);
  MU_ASSERT(calib_gimbal_equals(src, dst));

  calib_gimbal_print(src);
  calib_gimbal_free(src);
  calib_gimbal_free(dst);

  return 0;
}

int test_calib_gimbal_add_fiducial() {
  calib_gimbal_t *calib = calib_gimbal_malloc();

  real_t fiducial_pose[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  calib_gimbal_add_fiducial(calib, fiducial_pose);
  MU_ASSERT(vec_equals(calib->fiducial_exts.data, fiducial_pose, 7));

  calib_gimbal_free(calib);

  return 0;
}

int test_calib_gimbal_add_pose() {
  calib_gimbal_t *calib = calib_gimbal_malloc();

  real_t pose[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  calib_gimbal_add_pose(calib, 0, pose);
  MU_ASSERT(vec_equals(calib->poses[0].data, pose, 7));
  MU_ASSERT(calib->num_poses == 1);

  calib_gimbal_free(calib);

  return 0;
}

int test_calib_gimbal_add_gimbal_extrinsic() {
  calib_gimbal_t *calib = calib_gimbal_malloc();

  real_t gimbal_ext[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  calib_gimbal_add_gimbal_extrinsic(calib, gimbal_ext);
  MU_ASSERT(vec_equals(gimbal_ext, calib->gimbal_exts.data, 7));

  calib_gimbal_free(calib);
  return 0;
}

int test_calib_gimbal_add_gimbal_link() {
  calib_gimbal_t *calib = calib_gimbal_malloc();

  real_t link0[7] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};
  calib_gimbal_add_gimbal_link(calib, 0, link0);
  MU_ASSERT(vec_equals(link0, calib->links[0].data, 7));
  MU_ASSERT(calib->num_links == 1);

  real_t link1[7] = {8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};
  calib_gimbal_add_gimbal_link(calib, 1, link1);
  MU_ASSERT(vec_equals(link1, calib->links[1].data, 7));
  MU_ASSERT(calib->num_links == 2);

  calib_gimbal_free(calib);
  return 0;
}

int test_calib_gimbal_add_camera() {
  calib_gimbal_t *calib = calib_gimbal_malloc();

  const int cam_res[2] = {640, 480};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  real_t cam0_params[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  real_t cam0_ext[7] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  real_t cam1_params[8] = {2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0};
  real_t cam1_ext[7] = {3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0};

  calib_gimbal_add_camera(calib,
                          0,
                          cam_res,
                          proj_model,
                          dist_model,
                          cam0_params,
                          cam0_ext);
  MU_ASSERT(vec_equals(cam0_params, calib->cam_params[0].data, 8));
  MU_ASSERT(vec_equals(cam0_ext, calib->cam_exts[0].data, 7));
  MU_ASSERT(calib->num_cams == 1);

  calib_gimbal_add_camera(calib,
                          1,
                          cam_res,
                          proj_model,
                          dist_model,
                          cam1_params,
                          cam1_ext);
  MU_ASSERT(vec_equals(cam1_params, calib->cam_params[1].data, 8));
  MU_ASSERT(vec_equals(cam1_ext, calib->cam_exts[1].data, 7));
  MU_ASSERT(calib->num_cams == 2);

  calib_gimbal_free(calib);
  return 0;
}

int test_calib_gimbal_add_remove_view() {
  // Setup gimbal calibration
  calib_gimbal_t *calib = calib_gimbal_malloc();

  // -- Add fiducial
  real_t fiducial[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  calib_gimbal_add_fiducial(calib, fiducial);

  // -- Add pose
  const timestamp_t ts = 0;
  real_t pose[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  calib_gimbal_add_pose(calib, ts, pose);

  // -- Add gimbal links
  real_t link0[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  real_t link1[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  calib_gimbal_add_gimbal_link(calib, 0, link0);
  calib_gimbal_add_gimbal_link(calib, 1, link1);

  // -- Add camera
  const int cam_res[2] = {640, 480};
  const char *proj_model = "pinhole";
  const char *dist_model = "radtan4";
  real_t cam0_params[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  real_t cam0_ext[7] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  calib_gimbal_add_camera(calib,
                          0,
                          cam_res,
                          proj_model,
                          dist_model,
                          cam0_params,
                          cam0_ext);

  // -- Add view
  const int pose_idx = 0;
  const int view_idx = 0;
  const int cam_idx = 0;
  const int num_corners = 4;
  const int tag_ids[4] = {0, 0, 0, 0};
  const int corner_indices[4] = {0, 1, 2, 3};
  const real_t object_points[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  const real_t keypoints[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  const real_t joints[3] = {0.0, 0.0, 0.0};
  const int num_joints = 3;
  calib_gimbal_add_view(calib,
                        pose_idx,
                        view_idx,
                        ts,
                        cam_idx,
                        num_corners,
                        tag_ids,
                        corner_indices,
                        object_points,
                        keypoints,
                        joints,
                        num_joints);
  MU_ASSERT(calib->num_cams == 1);
  MU_ASSERT(calib->num_views == 1);
  MU_ASSERT(calib->num_poses == 1);
  MU_ASSERT(calib->num_links == 2);
  MU_ASSERT(calib->num_joints == 3);
  MU_ASSERT(calib->num_calib_factors == 4);
  MU_ASSERT(calib->num_joint_factors == 3);
  MU_ASSERT(calib->num_joints == 3);

  // -- Remove view
  calib_gimbal_remove_view(calib, view_idx);
  MU_ASSERT(calib->num_cams == 1);
  MU_ASSERT(calib->num_views == 0);
  MU_ASSERT(calib->num_poses == 1);
  MU_ASSERT(calib->num_links == 2);
  MU_ASSERT(calib->num_joints == 3);
  MU_ASSERT(calib->num_calib_factors == 0);
  MU_ASSERT(calib->num_joint_factors == 0);
  MU_ASSERT(calib->num_joints == 3);

  // Clean up
  calib_gimbal_free(calib);

  return 0;
}

int test_calib_gimbal_load() {
  const char *data_path = "/tmp/sim_gimbal";
  calib_gimbal_t *calib = calib_gimbal_load(data_path);
  MU_ASSERT(calib != NULL);
  // calib_gimbal_print(calib);
  calib_gimbal_free(calib);

  return 0;
}

static void compare_gimbal_calib(const calib_gimbal_t *gnd,
                                 const calib_gimbal_t *est) {
  assert(gnd->num_views == est->num_views);
  assert(gnd->num_cams == est->num_cams);
  assert(gnd->num_calib_factors == est->num_calib_factors);
  assert(gnd->num_joint_factors == est->num_joint_factors);

  // Compare estimated vs ground-truth
  printf("\n");
  {
    printf("num_views: %d\n", gnd->num_views);
    printf("num_cams: %d\n", gnd->num_cams);
    printf("num_poses: %d\n", gnd->num_poses);
    printf("num_links: %d\n", gnd->num_links);
    printf("num_joints: %d\n", gnd->num_joints);
    printf("num_calib_factors: %d\n", gnd->num_calib_factors);
    printf("num_joint_factors: %d\n", gnd->num_joint_factors);

    // Fiducial
    {
      real_t dr[3] = {0};
      real_t dtheta = 0.0;
      pose_diff2(gnd->fiducial_exts.data, est->fiducial_exts.data, dr, &dtheta);
      printf("fiducial ");
      printf("dr: [%.4f, %.4f, %.4f], ", dr[0], dr[1], dr[2]);
      printf("dtheta: %f [deg]\n", rad2deg(dtheta));
    }

    // Links
    for (int link_idx = 0; link_idx < est->num_links; link_idx++) {
      real_t dr[3] = {0};
      real_t dtheta = 0.0;
      pose_diff2(gnd->links[link_idx].data,
                 est->links[link_idx].data,
                 dr,
                 &dtheta);
      printf("link_exts[%d] ", link_idx);
      printf("dr: [%.4f, %.4f, %.4f], ", dr[0], dr[1], dr[2]);
      printf("dtheta: %f [deg]\n", rad2deg(dtheta));
    }

    // Joints
    real_t joints[3] = {0};
    for (int view_idx = 0; view_idx < gnd->num_views; view_idx++) {
      for (int joint_idx = 0; joint_idx < gnd->num_joints; joint_idx++) {
        const real_t gnd_angle = gnd->joints[view_idx][joint_idx].data[0];
        const real_t est_angle = est->joints[view_idx][joint_idx].data[0];
        joints[joint_idx] += rad2deg(fabs(gnd_angle - est_angle));
      }
    }
    for (int joint_idx = 0; joint_idx < gnd->num_joints; joint_idx++) {
      printf("joint[%d] total diff: %f [deg]\n", joint_idx, joints[joint_idx]);
    }

    // Camera extrinsic
    for (int cam_idx = 0; cam_idx < est->num_cams; cam_idx++) {
      real_t dr[3] = {0};
      real_t dtheta = 0.0;
      pose_diff2(gnd->cam_exts[cam_idx].data,
                 est->cam_exts[cam_idx].data,
                 dr,
                 &dtheta);
      printf("cam_exts[%d] ", cam_idx);
      printf("dr: [%.4f, %.4f, %.4f], ", dr[0], dr[1], dr[2]);
      printf("dtheta: %f [deg]\n", rad2deg(dtheta));
    }

    // Camera parameters
    for (int cam_idx = 0; cam_idx < est->num_cams; cam_idx++) {
      real_t *cam_gnd = gnd->cam_params[cam_idx].data;
      real_t *cam_est = est->cam_params[cam_idx].data;
      real_t diff[8] = {0};
      vec_sub(cam_gnd, cam_est, diff, 8);

      printf("cam_params[%d] ", cam_idx);
      print_vector("diff", diff, 8);
    }
  }
  printf("\n");
}

int test_calib_gimbal_solve() {
  // Setup
  const int debug = 0;
  const char *data_path = "/tmp/sim_gimbal";
  calib_gimbal_t *calib_gnd = calib_gimbal_load(data_path);
  calib_gimbal_t *calib_est = calib_gimbal_load(data_path);
  MU_ASSERT(calib_gnd != NULL);
  MU_ASSERT(calib_est != NULL);

  // Perturb parameters
  {
    // printf("Ground Truth:\n");
    // calib_gimbal_print(calib_gnd);
    // printf("\n");

    // Perturb
    real_t dx[6] = {0.01, 0.01, 0.01, 0.05, 0.05, 0.05};
    // pose_vector_update(calib_est->fiducial_exts.data, dx);
    // pose_vector_update(calib_est->cam_exts[0].data, dx);
    // pose_vector_update(calib_est->cam_exts[1].data, dx);
    for (int link_idx = 0; link_idx < calib_est->num_links; link_idx++) {
      pose_vector_update(calib_est->links[link_idx].data, dx);
    }
    for (int view_idx = 0; view_idx < calib_est->num_views; view_idx++) {
      for (int joint_idx = 0; joint_idx < calib_est->num_joints; joint_idx++) {
        calib_est->joints[view_idx][joint_idx].data[0] += randf(-0.1, 0.1);
      }
    }
    // printf("\n");

    //     printf("Initial:\n");
    //     calib_gimbal_print(calib_est);
    //     printf("\n");
  }
  if (debug) {
    compare_gimbal_calib(calib_gnd, calib_est);
  }

  calib_est->cam_exts[0].data[0] = 0.0;
  calib_est->cam_exts[0].data[1] = 0.0;
  calib_est->cam_exts[0].data[2] = 0.0;
  // calib_est->cam_exts[0].data[3] = 1.0;
  // calib_est->cam_exts[0].data[4] = 0.0;
  // calib_est->cam_exts[0].data[5] = 0.0;
  // calib_est->cam_exts[0].data[6] = 0.0;

  // Solve
  solver_t solver;
  solver_setup(&solver);
  solver.verbose = debug;
  solver.max_iter = 20;
  solver.param_order_func = &calib_gimbal_param_order;
  solver.linearize_func = &calib_gimbal_linearize_compact;
  solver_solve(&solver, calib_est);
  if (debug) {
    compare_gimbal_calib(calib_gnd, calib_est);
  }

  // printf("Estimated:\n");
  // calib_gimbal_print(calib);
  // printf("\n");

  // Clean up
  calib_gimbal_free(calib_gnd);
  calib_gimbal_free(calib_est);

  return 0;
}

#ifdef USE_CERES
int test_calib_gimbal_ceres_solve() {
  // Setup simulation data
  const char *data_path = "/tmp/sim_gimbal";
  calib_gimbal_t *calib_gnd = calib_gimbal_load(data_path);
  calib_gimbal_t *calib_est = calib_gimbal_load(data_path);
  MU_ASSERT(calib_gnd != NULL);
  MU_ASSERT(calib_est != NULL);

  // Perturb parameters
  {
    // printf("Ground Truth:\n");
    // calib_gimbal_print(calib_gnd);
    // printf("\n");

    // Perturb
    // real_t dx[6] = {0.01, 0.01, 0.01, 0.1, 0.1, 0.1};
    // pose_vector_update(calib_est->fiducial_exts.data, dx);
    // pose_vector_update(calib_est->cam_exts[0].data, dx);
    // pose_vector_update(calib_est->cam_exts[1].data, dx);
    // for (int link_idx = 0; link_idx < 3; link_idx++) {
    //   pose_vector_update(calib_est->links[link_idx].data, dx);
    // }
    // for (int view_idx = 0; view_idx < calib_est->num_views; view_idx++) {
    //   for (int joint_idx = 0; joint_idx < 3; joint_idx++) {
    //     calib_est->joints[view_idx][joint_idx].data[0] += 0.1;
    //   }
    // }
    // printf("\n");

    //     printf("Initial:\n");
    //     calib_gimbal_print(calib_est);
    //     printf("\n");
  }

  // Setup ceres problem
  ceres_init();
  ceres_problem_t *problem = ceres_create_problem();
  ceres_local_parameterization_t *pose_pm =
      ceres_create_pose_local_parameterization();

  const int num_residuals = 2;
  const int num_params = 10;
  for (int view_idx = 0; view_idx < calib_est->num_views; view_idx++) {
    for (int cam_idx = 0; cam_idx < calib_est->num_cams; cam_idx++) {
      calib_gimbal_view_t *view = calib_est->views[view_idx][cam_idx];
      for (int factor_idx = 0; factor_idx < view->num_factors; factor_idx++) {
        real_t *param_ptrs[] = {calib_est->fiducial_exts.data,
                                calib_est->gimbal_exts.data,
                                calib_est->poses[view_idx].data,
                                calib_est->links[0].data,
                                calib_est->links[1].data,
                                calib_est->joints[view_idx][0].data,
                                calib_est->joints[view_idx][1].data,
                                calib_est->joints[view_idx][2].data,
                                calib_est->cam_exts[cam_idx].data,
                                calib_est->cam_params[cam_idx].data};
        int param_sizes[10] = {
            7, // Fiducial extrinsic
            7, // Gimbal extrinscis
            7, // Pose
            7, // Link0
            7, // Link1
            1, // Joint0
            1, // Joint1
            1, // Joint2
            7, // Camera extrinsic
            8, // Camera Parameters
        };
        ceres_problem_add_residual_block(problem,
                                         &calib_gimbal_factor_ceres_eval,
                                         &view->factors[factor_idx],
                                         NULL,
                                         NULL,
                                         num_residuals,
                                         num_params,
                                         param_sizes,
                                         param_ptrs);
      } // For each corners
    }   // For each cameras
  }     // For each views

  // for (int view_idx = 0; view_idx < calib_est->num_views; view_idx++) {
  //   ceres_set_parameter_constant(problem,
  //   calib_est->joints[view_idx][0].data);
  //   ceres_set_parameter_constant(problem,
  //   calib_est->joints[view_idx][1].data);
  //   ceres_set_parameter_constant(problem,
  //   calib_est->joints[view_idx][2].data);
  // }

  // ceres_set_parameter_constant(problem, calib_est->fiducial_exts.data);
  ceres_set_parameter_constant(problem, calib_est->gimbal_exts.data);
  ceres_set_parameter_constant(problem, calib_est->links[0].data);
  ceres_set_parameter_constant(problem, calib_est->links[1].data);
  ceres_set_parameter_constant(problem, calib_est->cam_exts[0].data);
  ceres_set_parameter_constant(problem, calib_est->cam_exts[1].data);
  ceres_set_parameter_constant(problem, calib_est->cam_params[0].data);
  ceres_set_parameter_constant(problem, calib_est->cam_params[1].data);

  for (int view_idx = 0; view_idx < calib_est->num_poses; view_idx++) {
    ceres_set_parameter_constant(problem, calib_est->poses[view_idx].data);
    ceres_set_parameterization(problem,
                               calib_est->poses[view_idx].data,
                               pose_pm);
  }
  ceres_set_parameterization(problem, calib_est->fiducial_exts.data, pose_pm);
  ceres_set_parameterization(problem, calib_est->gimbal_exts.data, pose_pm);
  ceres_set_parameterization(problem, calib_est->links[0].data, pose_pm);
  ceres_set_parameterization(problem, calib_est->links[1].data, pose_pm);
  ceres_set_parameterization(problem, calib_est->cam_exts[0].data, pose_pm);
  ceres_set_parameterization(problem, calib_est->cam_exts[1].data, pose_pm);
  ceres_solve(problem, 10);

  // Compare ground-truth vs estimates
  compare_gimbal_calib(calib_gnd, calib_est);

  // Clean up
  ceres_free_problem(problem);
  calib_gimbal_free(calib_gnd);
  calib_gimbal_free(calib_est);

  return 0;
}
#endif // USE_CERES

/******************************************************************************
 * TEST DATASET
 ******************************************************************************/

int test_assoc_pose_data() {
  const double threshold = 0.01;
  const char *matches_fpath = "./gnd_est_matches.csv";
  const char *gnd_data_path = "test_data/euroc/MH01_groundtruth.csv";
  const char *est_data_path = "test_data/euroc/MH01_estimate.csv";

  /* Load ground-truth poses */
  int nb_gnd_poses = 0;
  pose_t *gnd_poses = load_poses(gnd_data_path, &nb_gnd_poses);

  /* Load estimate poses */
  int nb_est_poses = 0;
  pose_t *est_poses = load_poses(est_data_path, &nb_est_poses);

  /* Associate data */
  size_t nb_matches = 0;
  int **matches = assoc_pose_data(gnd_poses,
                                  nb_gnd_poses,
                                  est_poses,
                                  nb_est_poses,
                                  threshold,
                                  &nb_matches);
  printf("Time Associated:\n");
  printf(" - [%s]\n", gnd_data_path);
  printf(" - [%s]\n", est_data_path);
  printf("threshold:  %.4f [s]\n", threshold);
  printf("nb_matches: %ld\n", nb_matches);

  /* Save matches to file */
  FILE *matches_csv = fopen(matches_fpath, "w");
  fprintf(matches_csv, "#gnd_idx,est_idx\n");
  for (size_t i = 0; i < nb_matches; i++) {
    uint64_t gnd_ts = gnd_poses[matches[i][0]].ts;
    uint64_t est_ts = est_poses[matches[i][1]].ts;
    double t_diff = fabs(ts2sec(gnd_ts - est_ts));
    if (t_diff > threshold) {
      printf("ERROR! Time difference > threshold!\n");
      printf("ground_truth_index: %d\n", matches[i][0]);
      printf("estimate_index: %d\n", matches[i][1]);
      break;
    }
    fprintf(matches_csv, "%d,%d\n", matches[i][0], matches[i][1]);
  }
  fclose(matches_csv);

  /* Clean up */
  for (size_t i = 0; i < nb_matches; i++) {
    free(matches[i]);
  }
  free(matches);
  free(gnd_poses);
  free(est_poses);

  return 0;
}

/******************************************************************************
 * TEST SIM
 ******************************************************************************/

// SIM FEATURES //////////////////////////////////////////////////////////////

int test_sim_features_load() {
  const char *csv_file = TEST_SIM_DATA "/features.csv";
  sim_features_t *features_data = sim_features_load(csv_file);
  MU_ASSERT(features_data->nb_features > 0);
  sim_features_free(features_data);
  return 0;
}

// SIM IMU DATA //////////////////////////////////////////////////////////////

int test_sim_imu_data_load() {
  const char *csv_file = TEST_SIM_DATA "/imu0/data.csv";
  sim_imu_data_t *imu_data = sim_imu_data_load(csv_file);
  sim_imu_data_free(imu_data);
  return 0;
}

// SIM CAMERA DATA ///////////////////////////////////////////////////////////

int test_sim_camera_frame_load() {
  const char *frame_csv = TEST_SIM_DATA "/cam0/data/100000000.csv";
  sim_camera_frame_t *frame_data = sim_camera_frame_load(frame_csv);

  MU_ASSERT(frame_data != NULL);
  MU_ASSERT(frame_data->ts == 100000000);
  MU_ASSERT(frame_data->feature_ids[0] == 1);

  sim_camera_frame_free(frame_data);

  return 0;
}

int test_sim_camera_data_load() {
  const char *dir_path = TEST_SIM_DATA "/cam0";
  sim_camera_data_t *cam_data = sim_camera_data_load(dir_path);
  sim_camera_data_free(cam_data);
  return 0;
}

int test_sim_gimbal_malloc_free() {
  sim_gimbal_t *sim = sim_gimbal_malloc();
  sim_gimbal_free(sim);
  return 0;
}

int test_sim_gimbal_view() {
  sim_gimbal_t *sim = sim_gimbal_malloc();

  const timestamp_t ts = 0;
  const int view_idx = 0;
  const int cam_idx = 0;
  real_t pose[7] = {0, 0, 0, 1, 0, 0, 0};

  sim_gimbal_view_t *view = sim_gimbal_view(sim, ts, view_idx, cam_idx, pose);
  sim_gimbal_view_free(view);

  sim_gimbal_free(sim);
  return 0;
}

int test_sim_gimbal_solve() {
  // Setup gimbal simulator
  sim_gimbal_t *sim = sim_gimbal_malloc();

  // Setup gimbal calibrator
  calib_gimbal_t *calib = calib_gimbal_malloc();
  const timestamp_t ts = 0;
  calib_gimbal_add_fiducial(calib, sim->fiducial_ext.data);
  calib_gimbal_add_pose(calib, ts, sim->gimbal_pose.data);
  calib_gimbal_add_gimbal_extrinsic(calib, sim->gimbal_ext.data);
  calib_gimbal_add_gimbal_link(calib, 0, sim->gimbal_links[0].data);
  calib_gimbal_add_gimbal_link(calib, 1, sim->gimbal_links[1].data);
  for (int cam_idx = 0; cam_idx < sim->num_cams; cam_idx++) {
    calib_gimbal_add_camera(calib,
                            cam_idx,
                            sim->cam_params[cam_idx].resolution,
                            sim->cam_params[cam_idx].proj_model,
                            sim->cam_params[cam_idx].dist_model,
                            sim->cam_params[cam_idx].data,
                            sim->cam_exts[cam_idx].data);
  }

  // Setup solver
  solver_t solver;
  solver_setup(&solver);
  solver.param_order_func = &calib_gimbal_param_order;
  solver.linearize_func = &calib_gimbal_linearize_compact;

  // Simulate gimbal views
  int num_views = 100;
  int num_cams = 2;
  const int pose_idx = 0;
  sim_gimbal_view_t *view = NULL;

  for (int view_idx = 0; view_idx < num_views; view_idx++) {
    // Add gimbal view
    for (int cam_idx = 0; cam_idx < num_cams; cam_idx++) {
      // Simulate single gimbal view
      const timestamp_t ts = view_idx;
      view = sim_gimbal_view(sim, ts, view_idx, cam_idx, sim->gimbal_pose.data);

      // Add view to calibration problem
      real_t joints[3] = {0};
      sim_gimbal_get_joints(sim, 3, joints);
      calib_gimbal_add_view(calib,
                            pose_idx,
                            view_idx,
                            view_idx,
                            cam_idx,
                            view->num_measurements,
                            view->tag_ids,
                            view->corner_indices,
                            view->object_points,
                            view->keypoints,
                            joints,
                            sim->num_joints);
      sim_gimbal_view_free(view);
    }

    // Find gimbal NBV
    real_t nbv_joints[3] = {0};
    calib_gimbal_nbv(calib, nbv_joints);
    sim_gimbal_set_joint(sim, 0, nbv_joints[0]);
    sim_gimbal_set_joint(sim, 1, nbv_joints[1]);
    sim_gimbal_set_joint(sim, 2, nbv_joints[2]);
  }

  // Solve
  solver_solve(&solver, calib);

  // Clean up
  calib_gimbal_free(calib);
  sim_gimbal_free(sim);

  return 0;
}

void test_suite() {
  // MACROS
  MU_ADD_TEST(test_median_value);
  MU_ADD_TEST(test_mean_value);

  // FILE SYSTEM
  MU_ADD_TEST(test_path_file_name);
  MU_ADD_TEST(test_path_file_ext);
  MU_ADD_TEST(test_path_dir_name);
  MU_ADD_TEST(test_path_join);
  MU_ADD_TEST(test_list_files);
  MU_ADD_TEST(test_list_files_free);
  MU_ADD_TEST(test_file_read);
  MU_ADD_TEST(test_skip_line);
  MU_ADD_TEST(test_file_rows);
  MU_ADD_TEST(test_file_copy);

  // DATA
  MU_ADD_TEST(test_string_malloc);
  MU_ADD_TEST(test_dsv_rows);
  MU_ADD_TEST(test_dsv_cols);
  MU_ADD_TEST(test_dsv_fields);
  MU_ADD_TEST(test_dsv_data);
  MU_ADD_TEST(test_dsv_free);

  // DATA-STRUCTURE
  MU_ADD_TEST(test_darray_new_and_destroy);
  MU_ADD_TEST(test_darray_push_pop);
  MU_ADD_TEST(test_darray_contains);
  MU_ADD_TEST(test_darray_copy);
  MU_ADD_TEST(test_darray_new_element);
  MU_ADD_TEST(test_darray_set_and_get);
  MU_ADD_TEST(test_darray_update);
  MU_ADD_TEST(test_darray_remove);
  MU_ADD_TEST(test_darray_expand_and_contract);
  MU_ADD_TEST(test_list_new_and_destroy);
  // MU_ADD_TEST(test_list_push_pop);
  MU_ADD_TEST(test_list_shift);
  MU_ADD_TEST(test_list_unshift);
  MU_ADD_TEST(test_list_remove);
  MU_ADD_TEST(test_list_remove_destroy);
  MU_ADD_TEST(test_mstack_new_and_destroy);
  MU_ADD_TEST(test_mstack_push);
  MU_ADD_TEST(test_mstack_pop);
  MU_ADD_TEST(test_queue_new_and_destroy);
  MU_ADD_TEST(test_queue_enqueue_dequeue);
  MU_ADD_TEST(test_hashmap_new_destroy);
  MU_ADD_TEST(test_hashmap_clear_destroy);
  MU_ADD_TEST(test_hashmap_get_set);
  MU_ADD_TEST(test_hashmap_delete);
  MU_ADD_TEST(test_hashmap_traverse);

  // TIME
  MU_ADD_TEST(test_tic_toc);
  MU_ADD_TEST(test_mtoc);
  MU_ADD_TEST(test_time_now);

  // NETWORK
  MU_ADD_TEST(test_tcp_server_setup);

  // MATHS
  MU_ADD_TEST(test_min);
  MU_ADD_TEST(test_max);
  MU_ADD_TEST(test_randf);
  MU_ADD_TEST(test_deg2rad);
  MU_ADD_TEST(test_rad2deg);
  MU_ADD_TEST(test_fltcmp);
  MU_ADD_TEST(test_fltcmp2);
  MU_ADD_TEST(test_pythag);
  MU_ADD_TEST(test_lerp);
  MU_ADD_TEST(test_lerp3);
  MU_ADD_TEST(test_sinc);
  MU_ADD_TEST(test_mean);
  MU_ADD_TEST(test_median);
  MU_ADD_TEST(test_var);
  MU_ADD_TEST(test_stddev);

  // LINEAR ALGEBRA
  MU_ADD_TEST(test_eye);
  MU_ADD_TEST(test_ones);
  MU_ADD_TEST(test_zeros);
  MU_ADD_TEST(test_mat_set);
  MU_ADD_TEST(test_mat_val);
  MU_ADD_TEST(test_mat_copy);
  MU_ADD_TEST(test_mat_row_set);
  MU_ADD_TEST(test_mat_col_set);
  MU_ADD_TEST(test_mat_block_get);
  MU_ADD_TEST(test_mat_block_set);
  MU_ADD_TEST(test_mat_diag_get);
  MU_ADD_TEST(test_mat_diag_set);
  MU_ADD_TEST(test_mat_triu);
  MU_ADD_TEST(test_mat_tril);
  MU_ADD_TEST(test_mat_trace);
  MU_ADD_TEST(test_mat_transpose);
  MU_ADD_TEST(test_mat_add);
  MU_ADD_TEST(test_mat_sub);
  MU_ADD_TEST(test_mat_scale);
  MU_ADD_TEST(test_vec_add);
  MU_ADD_TEST(test_vec_sub);
  MU_ADD_TEST(test_dot);
  MU_ADD_TEST(test_hat);
  MU_ADD_TEST(test_check_jacobian);
  MU_ADD_TEST(test_svd);
  MU_ADD_TEST(test_pinv);
  MU_ADD_TEST(test_svd_det);
  MU_ADD_TEST(test_chol);
  MU_ADD_TEST(test_chol_solve);
  MU_ADD_TEST(test_qr);
  MU_ADD_TEST(test_eig_sym);
  MU_ADD_TEST(test_eig_inv);

  // SUITE-SPARSE
  MU_ADD_TEST(test_suitesparse_chol_solve);

  // TRANSFORMS
  MU_ADD_TEST(test_tf_rot_set);
  MU_ADD_TEST(test_tf_trans_set);
  MU_ADD_TEST(test_tf_trans_get);
  MU_ADD_TEST(test_tf_rot_get);
  MU_ADD_TEST(test_tf_quat_get);
  MU_ADD_TEST(test_tf_inv);
  MU_ADD_TEST(test_tf_point);
  MU_ADD_TEST(test_tf_hpoint);
  MU_ADD_TEST(test_tf_perturb_rot);
  MU_ADD_TEST(test_tf_perturb_trans);
  MU_ADD_TEST(test_tf_chain);
  MU_ADD_TEST(test_euler321);
  MU_ADD_TEST(test_rot2quat);
  MU_ADD_TEST(test_quat2euler);
  MU_ADD_TEST(test_quat2rot);

  // LIE
  MU_ADD_TEST(test_lie_Exp_Log);

  // CV
  MU_ADD_TEST(test_image_setup);
  MU_ADD_TEST(test_image_load);
  MU_ADD_TEST(test_image_print_properties);
  MU_ADD_TEST(test_image_free);
  MU_ADD_TEST(test_linear_triangulation);
  MU_ADD_TEST(test_radtan4_distort);
  MU_ADD_TEST(test_radtan4_undistort);
  MU_ADD_TEST(test_radtan4_point_jacobian);
  MU_ADD_TEST(test_radtan4_params_jacobian);
  MU_ADD_TEST(test_equi4_distort);
  MU_ADD_TEST(test_equi4_undistort);
  MU_ADD_TEST(test_equi4_point_jacobian);
  MU_ADD_TEST(test_equi4_params_jacobian);
  MU_ADD_TEST(test_pinhole_focal);
  MU_ADD_TEST(test_pinhole_K);
  MU_ADD_TEST(test_pinhole_projection_matrix);
  MU_ADD_TEST(test_pinhole_project);
  MU_ADD_TEST(test_pinhole_point_jacobian);
  MU_ADD_TEST(test_pinhole_params_jacobian);
  MU_ADD_TEST(test_pinhole_radtan4_project);
  MU_ADD_TEST(test_pinhole_radtan4_project_jacobian);
  MU_ADD_TEST(test_pinhole_radtan4_params_jacobian);
  MU_ADD_TEST(test_pinhole_equi4_project);
  MU_ADD_TEST(test_pinhole_equi4_project_jacobian);
  MU_ADD_TEST(test_pinhole_equi4_params_jacobian);

  // SENSOR FUSION
  MU_ADD_TEST(test_pose);
  MU_ADD_TEST(test_extrinsics);
  MU_ADD_TEST(test_imu_biases);
  MU_ADD_TEST(test_feature);
  MU_ADD_TEST(test_idfs);
  MU_ADD_TEST(test_pose_factor);
  MU_ADD_TEST(test_ba_factor);
  MU_ADD_TEST(test_vision_factor);
  MU_ADD_TEST(test_joint_factor);
  MU_ADD_TEST(test_calib_camera_factor);
  MU_ADD_TEST(test_calib_imucam_factor);
  MU_ADD_TEST(test_calib_gimbal_factor);
  MU_ADD_TEST(test_imu_buf_setup);
  MU_ADD_TEST(test_imu_buf_add);
  MU_ADD_TEST(test_imu_buf_clear);
  MU_ADD_TEST(test_imu_buf_copy);
  MU_ADD_TEST(test_imu_factor_propagate_step);
  MU_ADD_TEST(test_imu_factor_eval);
  MU_ADD_TEST(test_inertial_odometry);
#ifdef USE_CERES
  MU_ADD_TEST(test_ceres_example);
#endif // USE_CERES
  MU_ADD_TEST(test_solver_setup);
  // MU_ADD_TEST(test_solver_eval);
  // MU_ADD_TEST(test_calib_gimbal_copy);
  MU_ADD_TEST(test_calib_gimbal_add_fiducial);
  MU_ADD_TEST(test_calib_gimbal_add_pose);
  MU_ADD_TEST(test_calib_gimbal_add_gimbal_extrinsic);
  MU_ADD_TEST(test_calib_gimbal_add_gimbal_link);
  MU_ADD_TEST(test_calib_gimbal_add_camera);
  MU_ADD_TEST(test_calib_gimbal_add_remove_view);
  MU_ADD_TEST(test_calib_gimbal_load);
  MU_ADD_TEST(test_calib_gimbal_solve);
#ifdef USE_CERES
  MU_ADD_TEST(test_calib_gimbal_ceres_solve);
#endif // USE_CERES

  // DATASET
  // MU_ADD_TEST(test_assoc_pose_data);

  // SIM
  MU_ADD_TEST(test_sim_features_load);
  MU_ADD_TEST(test_sim_imu_data_load);
  MU_ADD_TEST(test_sim_camera_frame_load);
  MU_ADD_TEST(test_sim_camera_data_load);
  MU_ADD_TEST(test_sim_gimbal_malloc_free);
  MU_ADD_TEST(test_sim_gimbal_view);
  // MU_ADD_TEST(test_sim_gimbal_solve);
}

MU_RUN_TESTS(test_suite)
