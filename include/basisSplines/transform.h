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
    // coefficients of derivative spline coeffs = o * (c_i+1 - c_i) / (k_i+o -
    // k_i+1)
    Eigen::ArrayXd coeffsRes(m_basis->dim() - 1);
    for (int idx{}; idx < coeffsRes.size(); ++idx)
      coeffsRes(idx) = (m_basis->order() - 1) *
                       (coeffs(idx + 1) - coeffs(idx)) /
                       (m_basis->knots()(idx + m_basis->order()) -
                        m_basis->knots()(idx + 1));

    // base case order 1 derivative
    if (order == 1)
      return coeffsRes;

    // recursion higher order derivative
    return Transform{std::make_shared<Basis>(m_basis->derivative(order - 1))}
        .derivative(coeffsRes, order - 1);
  }

  Eigen::ArrayXXd derivative(int order = 1) const {
    Eigen::ArrayXXd transform(m_basis->dim() - 1, m_basis->dim());
    for (int cRow{}; cRow < transform.rows(); ++cRow) {
      transform(cRow, cRow) = 1.0 / (m_basis->knots()(cRow + 1) -
                                     m_basis->knots()(m_basis->order() + cRow));
      transform(cRow, cRow + 1) = -transform(cRow, cRow);
    }
    return transform;
  };

  Eigen::ArrayXXd integral(int order = 1) const;

  std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> sum(const Basis &basis) const;

  Eigen::ArrayXXd product(const Basis &basis);

private:
  std::shared_ptr<Basis> m_basis{};
};
}; // namespace BasisSplines

#endif