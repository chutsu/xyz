#include "xyz_ceres.h"

#include <iostream>
#include <ceres/c_api.h>
#include <ceres/cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/local_parameterization.h>
#include <ceres/problem.h>
#include <ceres/solver.h>
#include <ceres/types.h>

// GLOBAL VARIABLES
static int ceres_initialized = 0;

/**
 * Wrapper for ceres::CostFunction
 */
class CostFunctionWrapper final : public ceres::CostFunction {
public:
  CostFunctionWrapper(ceres_cost_function_t cost_function,
                      void *user_data,
                      int num_residuals,
                      int num_parameter_blocks,
                      int *parameter_block_sizes)
      : cost_function_(cost_function), user_data_(user_data) {
    set_num_residuals(num_residuals);
    for (int i = 0; i < num_parameter_blocks; ++i) {
      mutable_parameter_block_sizes()->push_back(parameter_block_sizes[i]);
    }
  }

  bool Evaluate(double const *const *parameters,
                double *residuals,
                double **jacobians) const final {
    return (*cost_function_)(user_data_,
                             const_cast<double **>(parameters),
                             residuals,
                             jacobians);
  }

private:
  ceres_cost_function_t cost_function_;
  void *user_data_;
};

/**
 * Wrapper for ceres::LossFunction
 */
class LossFunctionWrapper final : public ceres::LossFunction {
public:
  explicit LossFunctionWrapper(ceres_loss_function_t loss_function,
                               void *user_data)
      : loss_function_(loss_function), user_data_(user_data) {
  }
  void Evaluate(double sq_norm, double *rho) const final {
    (*loss_function_)(user_data_, sq_norm, rho);
  }

private:
  ceres_loss_function_t loss_function_;
  void *user_data_;
};

/**
 * Calculate vector norm of `x` of length `n`.
 * @returns Norm of vector x
 */
static double vec_norm(const double *x, const size_t n) {
  assert(x != NULL);
  assert(n > 0);

  double sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += x[i] * x[i];
  }
  return sqrt(sum);
}

/**
 * Sinc.
 * @return Result of sinc
 */
static double sinc(const double x) {
  if (fabs(x) > 1e-6) {
    return sin(x) / x;
  } else {
    const double c2 = 1.0 / 6.0;
    const double c4 = 1.0 / 120.0;
    const double c6 = 1.0 / 5040.0;
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double x6 = x2 * x2 * x2;
    return 1.0 - c2 * x2 + c4 * x4 - c6 * x6;
  }
}

/**
 * Form delta quaternion `dq` from a small rotation vector `dalpha`.
 */
static void quat_delta(const double dalpha[3], double dq[4]) {
  assert(dalpha != NULL);
  assert(dq != NULL);

  const double half_norm = 0.5 * vec_norm(dalpha, 3);
  const double k = sinc(half_norm) * 0.5;
  const double vector[3] = {k * dalpha[0], k * dalpha[1], k * dalpha[2]};
  double scalar = cos(half_norm);

  dq[0] = scalar;
  dq[1] = vector[0];
  dq[2] = vector[1];
  dq[3] = vector[2];
}

/**
 * Quaternion left-multiply `p` with `q`, results are outputted to `r`.
 */
static void quat_lmul(const double p[4], const double q[4], double r[4]) {
  assert(p != NULL);
  assert(q != NULL);
  assert(r != NULL);

  const double pw = p[0];
  const double px = p[1];
  const double py = p[2];
  const double pz = p[3];

  r[0] = pw * q[0] - px * q[1] - py * q[2] - pz * q[3];
  r[1] = px * q[0] + pw * q[1] - pz * q[2] + py * q[3];
  r[2] = py * q[0] + pz * q[1] + pw * q[2] - px * q[3];
  r[3] = pz * q[0] - py * q[1] + px * q[2] + pw * q[3];
}

/**
 * Quaternion multiply `p` with `q`, results are outputted to `r`.
 */
static void quat_mul(const double p[4], const double q[4], double r[4]) {
  assert(p != NULL);
  assert(q != NULL);
  assert(r != NULL);
  quat_lmul(p, q, r);
}

/**
 * Return Quaternion norm
 */
static double quat_norm(const double q[4]) {
  return sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
}

/**
 * Normalize Quaternion
 */
static void quat_normalize(double q[4]) {
  const double n = quat_norm(q);
  q[0] = q[0] / n;
  q[1] = q[1] / n;
  q[2] = q[2] / n;
  q[3] = q[3] / n;
}

/**
 * Pose local parameterization
 */
class PoseLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x,
                    const double *dx,
                    double *x_plus_dx) const {
    x_plus_dx[0] = x[0] + dx[0];
    x_plus_dx[1] = x[1] + dx[1];
    x_plus_dx[2] = x[2] + dx[2];

    double dq[4] = {0};
    quat_delta(dx + 3, dq);
    quat_mul(x + 3, dq, x_plus_dx + 3);
    quat_normalize(x_plus_dx + 3);

    return true;
  }

  virtual bool ComputeJacobian(const double *x, double *J) const {
    // clang-format off
    J[0]  = 1; J[1]  = 0; J[2]  = 0;  J[3] = 0; J[4]  = 0; J[5]  = 0;
    J[6]  = 0; J[7]  = 1; J[8]  = 0;  J[9] = 0; J[10] = 0; J[11] = 0;
    J[12] = 0; J[13] = 0; J[14] = 1; J[15] = 0; J[16] = 0; J[17] = 0;
    J[18] = 0; J[19] = 0; J[20] = 0; J[21] = 1; J[22] = 0; J[23] = 0;
    J[24] = 0; J[25] = 0; J[26] = 0; J[27] = 0; J[28] = 1; J[29] = 0;
    J[30] = 0; J[31] = 0; J[32] = 0; J[33] = 0; J[34] = 0; J[35] = 1;
    J[36] = 0; J[37] = 0; J[38] = 0; J[39] = 0; J[40] = 0; J[41] = 0;
    // clang-format on
    return true;
  }

  virtual int GlobalSize() const {
    return 7;
  }

  virtual int LocalSize() const {
    return 6;
  }
};

void ceres_init() {
  if (ceres_initialized) {
    return;
  }

  // This is not ideal, but it's not clear what to do if there is no gflags and
  // no access to command line arguments.
  char message[] = "<unknown>";
  google::InitGoogleLogging(message);
  ceres_initialized = 1;
}

ceres_problem_t *ceres_create_problem() {
  return reinterpret_cast<ceres_problem_t *>(new ceres::Problem);
}

void ceres_free_problem(ceres_problem_t *problem) {
  delete reinterpret_cast<ceres::Problem *>(problem);
}

// ceres_residual_block_id_t *
// ceres_problem_add_residual_block(ceres_problem_t *problem,
//                                  ceres_cost_function_t cost_function,
//                                  void *cost_function_data,
//                                  ceres_loss_function_t loss_function,
//                                  void *loss_function_data,
//                                  int num_residuals,
//                                  int num_parameter_blocks,
//                                  int *parameter_block_sizes,
//                                  double **parameters) {
//   auto *ceres_problem = reinterpret_cast<ceres::Problem *>(problem);

//   auto callback_cost_function =
//       std::make_unique<CostFunctionWrapper>(cost_function,
//                                             cost_function_data,
//                                             num_residuals,
//                                             num_parameter_blocks,
//                                             parameter_block_sizes);

//   std::unique_ptr<ceres::LossFunction> callback_loss_function;
//   if (loss_function != nullptr) {
//     callback_loss_function =
//         std::make_unique<LossFunctionWrapper>(loss_function,
//                                               loss_function_data);
//   }

//   std::vector<double *> parameter_blocks(parameters,
//                                          parameters + num_parameter_blocks);
//   return reinterpret_cast<ceres_residual_block_id_t *>(
//       ceres_problem->AddResidualBlock(callback_cost_function.release(),
//                                       callback_loss_function.release(),
//                                       parameter_blocks));
// }

ceres_local_parameterization_t *ceres_create_pose_local_parameterization() {
  return reinterpret_cast<ceres_local_parameterization_t *>(
      new PoseLocalParameterization);
}

void ceres_free_local_parameterization(ceres_local_parameterization_t *p) {
  delete reinterpret_cast<ceres::LocalParameterization *>(p);
}

void ceres_set_parameterization(ceres_problem_t *c_problem,
                                double *values,
                                ceres_local_parameterization_t *c_local) {
  auto *problem = reinterpret_cast<ceres::Problem *>(c_problem);
  auto *local = reinterpret_cast<ceres::LocalParameterization *>(c_local);
  problem->SetParameterization(values, local);
}

void ceres_set_parameter_constant(ceres_problem_t *c_problem, double *values) {
  auto *problem = reinterpret_cast<ceres::Problem *>(c_problem);
  problem->SetParameterBlockConstant(values);
}

// void ceres_solve(ceres_problem_t *c_problem,
//                  const int max_iter,
//                  const int verbose) {
//   auto *problem = reinterpret_cast<ceres::Problem *>(c_problem);
//   ceres::Solver::Options options;
//   options.max_num_iterations = max_iter;
//   options.minimizer_progress_to_stdout = verbose;

//   ceres::Solver::Summary summary;
//   ceres::Solve(options, problem, &summary);

//   if (verbose) {
//     std::cout << summary.FullReport() << std::endl;
//   }
// }
