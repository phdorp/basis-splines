#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <memory>

#include "basisSplines/basis.h"

namespace BasisSplines {
/**
 * @brief Determines the spline coefficients for a basis to approximate the
 * given observations.
 *
 */
class Interpolate {
public:
  /**
   * @brief Construct a new Interpolate for the given Basis.
   *
   * @param basis spline basis.
   */
  Interpolate(const std::shared_ptr<Basis> basis) : m_basis{basis} {};

  /**
   * @brief Determine coefficients that fit a spline function at the given
   * "points" to the given "observations".
   *
   * @param observations values to fit the spline function.
   * @param points evaluation points corresponding to the "observations".
   * @return Eigen::ArrayXd spline coefficients fitting the observations.
   */
  Eigen::ArrayXd fit(const Eigen::ArrayXd &observations,
                     const Eigen::ArrayXd &points) const {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> basis{
        m_basis->operator()(points).matrix()};
    return basis.solve(observations.matrix()).array();
  }

  /**
   * @brief Determine coefficients that fit a spline function at the given
   * "points" to the "observations".
   *
   * @param observations observed derivatives to fit the spline function.
   * @param points evaluation points corresponding to the "observations".
   * @return Eigen::ArrayXd spline coefficients fitting the observations.
   */
  Eigen::ArrayXd fit(const Eigen::ArrayXXd &observations,
                     const Eigen::ArrayXd &points) const;

private:
  std::shared_ptr<Basis> m_basis{}; /**<< spline basis */
};
}; // namespace BasisSplines
#endif