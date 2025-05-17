#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "basisSplines/basis.h"
#include <Eigen/Core>
#include <memory>

namespace BasisSplines {
/**
 * @brief Spline coefficient transform for the given spline basis.
 *
 * Performs transformations of spline coefficients and provides transformation
 * matrices for coefficients.
 *
 */
class Transform {
public:
  Transform(const std::shared_ptr<Basis> basis) : m_basis{basis} {}

  /**
   * @brief Transforms the given spline coefficients to derivative coefficients.
   *
   * Performs less calculations than the transformation matrix.
   *
   * @param coeffs coefficients to transform.
   * @param order derivative order.
   * @return Eigen::ArrayXd derivative coefficients.
   */
  Eigen::ArrayXd derivative(const Eigen::ArrayXd &coeffs, int order = 1) const {
    return derivative(*m_basis.get(), coeffs, order);
  }

  /**
   * @brief Dertermines a matrix A to transform the spline coefficients c to
   * derivative coefficients dc.
   *
   * dc = A * c
   *
   * @param order derivative order.
   * @return Eigen::MatrixXd transformation matrix.
   */
  Eigen::MatrixXd derivative(int order = 1) const {
    return derivative(*m_basis.get(), order);
  };

  /**
   * @brief Transforms the given spline coefficients to integral coefficients.
   *
   * @param coeffs coefficients to transform.
   * @param order integral order.
   * @return Eigen::ArrayXd integral coefficients.
   */
  Eigen::ArrayXd integral(const Eigen::ArrayXd &coeffs, int order = 1) const {
    return integral(*m_basis.get(), coeffs, order);
  }

  /**
   * @brief Dertermines a matrix A to transform the spline coefficients c to
   * integral coefficients ic.
   *
   * ic = A * c
   *
   * @param order integral order.
   * @return Eigen::MatrixXd transformation matrix.
   */
  Eigen::MatrixXd integral(int order = 1) const {
    return integral(*m_basis.get(), order);
  }

  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> sum(const Basis &basis) const;

  Eigen::ArrayXXd product(const Basis &basis);

  /**
   * @brief Transforms the given spline coefficients to derivative coefficients.
   *
   * Performs less calculations than the transformation matrix.
   *
   * @param basis basis spline.
   * @param coeffs coefficients to transform.
   * @param order derivative order.
   * @return Eigen::ArrayXd derivative coefficients.
   */
  static Eigen::ArrayXd
  derivative(const Basis &basis, const Eigen::ArrayXd &coeffs, int order = 1) {
    // coefficients of derivative spline coeffs = o * (c_i+1 - c_i) / (k_i+o -
    // k_i+1)
    Eigen::ArrayXd coeffsRes(basis.dim() - 1);
    for (int idx{}; idx < coeffsRes.size(); ++idx)
      coeffsRes(idx) =
          (basis.order() - 1) * (coeffs(idx + 1) - coeffs(idx)) /
          (basis.knots()(idx + basis.order()) - basis.knots()(idx + 1));

    // base case order 1 derivative
    if (order == 1)
      return coeffsRes;

    // recursion higher order derivative
    return derivative(basis.orderDecrease(), coeffsRes, order - 1);
  }

  /**
   * @brief Dertermines a matrix A to transform the spline coefficients c to
   * derivative coefficients dc.
   *
   * dc = A * c
   *
   * @param basis basis spline.
   * @param order derivative order.
   * @return Eigen::ArrayXd derivative coefficients.
   */
  static Eigen::MatrixXd derivative(const Basis &basis, int order = 1) {
    // determine transformation matrix
    Eigen::MatrixXd transform(
        Eigen::MatrixXd::Zero(basis.dim() - 1, basis.dim()));
    for (int cRow{}; cRow < transform.rows(); ++cRow) {
      transform(cRow, cRow) =
          (basis.order() - 1) /
          (basis.knots()(cRow + 1) - basis.knots()(basis.order() + cRow));
      transform(cRow, cRow + 1) = -transform(cRow, cRow);
    }

    // base case order 1 derivative
    if (order == 1)
      return transform;

    // recursion higher order derivative
    return derivative(basis.orderDecrease(), order - 1) * transform;
  }

  /**
   * @brief Transforms the given spline coefficients to integral coefficients.
   *
   * @param basis basis spline.
   * @param coeffs coefficients to transform.
   * @param order integral order.
   * @return Eigen::ArrayXd integral coefficients.
   */
  static Eigen::ArrayXd integral(const Basis &basis,
                                 const Eigen::ArrayXd &coeffs, int order = 1) {
    // coefficients of derivative spline coeffs_i+1 = c_i * (k_i+o -
    // k_i) / o + coeffs_i
    Eigen::ArrayXd coeffsRes(basis.dim() + 1);
    for (int idx{}; idx < coeffsRes.size() - 1; ++idx)
      coeffsRes(idx + 1) =
          coeffs(idx) *
              (basis.knots()(idx + basis.order()) - basis.knots()(idx)) /
              basis.order() +
          coeffsRes(idx);

    // base case order 1 derivative
    if (order == 1)
      return coeffsRes;

    // recursion higher order derivative
    return integral(basis.orderIncrease(), coeffsRes, order - 1);
  }

  /**
   * @brief Dertermines a matrix A to transform the spline coefficients c to
   * integral coefficients ic.
   *
   * ic = A * c
   *
   * @param basis basis spline.
   * @param order integral order.
   * @return Eigen::MatrixXd transformation matrix.
   */
  static Eigen::MatrixXd integral(const Basis &basis, int order = 1) {
    // initialize transformation matrix with zeros
    Eigen::MatrixXd transform(
        Eigen::MatrixXd::Zero(basis.dim() + 1, basis.dim()));

    // fill transformation matrix
    int cCol{};
    for (auto col : transform.colwise()) {
      for (int cRow{cCol + 1}; cRow < transform.rows(); ++cRow) {
        col(cRow) =
            (basis.knots()(basis.order() + cCol) - basis.knots()(cCol)) /
            basis.order();
      }
      ++cCol;
    }

    // base case order 1 integral
    if (order == 1)
      return transform;

    // recursion higher order integral
    return integral(basis.orderIncrease(), order - 1) * transform;
  }

private:
  std::shared_ptr<Basis> m_basis{};
};
}; // namespace BasisSplines

#endif