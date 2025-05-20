#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
#include <memory>

#include "basisSplines/internal/basisBase.h"

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
  Interpolate(const std::shared_ptr<BasisBase> basis) : m_basis{basis} {};

  /**
   * @brief Determine coefficients that fit a spline function at the given
   * "points" to the given "observations".
   *
   * @param observations values to fit the spline function.
   * @param points evaluation points corresponding to the "observations".
   * @return Eigen::MatrixXd spline coefficients fitting the observations.
   */
  Eigen::MatrixXd fit(const Eigen::MatrixXd &observations,
                      const Eigen::VectorXd &points) const {
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> basis{
        m_basis->operator()(points)};
    return basis.solve(observations);
  }

  /**
   * @brief Determine coefficients that fit a spline function to the given
   * process.
   *
   * @param process function representation of the process.
   * @return Eigen::MatrixXd spline coefficients fitting the process.
   */
  Eigen::MatrixXd
  fit(std::function<Eigen::MatrixXd(Eigen::VectorXd)> process) const {
    return fit(process(m_basis->greville()), m_basis->greville());
  }

private:
  std::shared_ptr<BasisBase> m_basis{}; /**<< spline basis */
};
}; // namespace BasisSplines
#endif