#pragma once

#include <ceres/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

// ceres::Problem
// struct ceres_problem_s;
// typedef struct ceres_problem_s ceres_problem_t;

// ceres::ResidualBlock
struct ceres_residual_block_id_s;
typedef struct ceres_residual_block_id_s ceres_residual_block_id_t;

// ceres::LocalParameterization
struct ceres_local_parameterization_s;
typedef struct ceres_local_parameterization_s ceres_local_parameterization_t;

// Cost function pointer
typedef int (*ceres_cost_function_t)(void *user_data,
                                     double **parameters,
                                     double *residuals,
                                     double **jacobians);

// Loss function pointer
typedef void (*ceres_loss_function_t)(void *user_data,
                                      double squared_norm,
                                      double out[3]);

/**
 * Initialize ceres
 */
void ceres_init(void);

/* Create and destroy ceres::Problem */
ceres_problem_t *ceres_create_problem(void);
void ceres_free_problem(ceres_problem_t *problem);

/* Create and destroy ceres::LocalParameterization */
ceres_local_parameterization_t *ceres_create_pose_local_parameterization(void);
void ceres_free_local_parameterization(ceres_local_parameterization_t *p);

// /**
//  * Add residual block to ceres::Problem
//  */
// ceres_residual_block_id_t *
// ceres_problem_add_residual_block(ceres_problem_t *problem,
//                                  ceres_cost_function_t cost_function,
//                                  void *cost_function_data,
//                                  ceres_loss_function_t loss_function,
//                                  void *loss_function_data,
//                                  int num_residuals,
//                                  int num_parameter_blocks,
//                                  int *parameter_block_sizes,
//                                  double **parameters);

/**
 * Set local parameterization for a particular parameter
 */
void ceres_set_parameterization(ceres_problem_t *c_problem,
                                double *values,
                                ceres_local_parameterization_t *c_local);

/**
 * Set parameter constant
 */
void ceres_set_parameter_constant(ceres_problem_t *c_problem, double *values);

/**
 * Solve ceres::Problem
 */
// void ceres_solve(ceres_problem_t *problem, const int max_iter, const int verbose);

#ifdef __cplusplus
} // extern "C"
#endif
