#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <functional>
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
   * @tparam DecompositionType type of Eigen matrix decomposition
   * https://eigen.tuxfamily.org/dox/group__DenseDecompositionBenchmark.html
   * @param observations values to fit the spline function.
   * @param points evaluation points corresponding to the "observations".
   * @return Eigen::MatrixXd spline coefficients fitting the observations.
   */
  template <
      typename DecompositionType = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>
  Eigen::MatrixXd fit(const Eigen::MatrixXd &observations,
                      const Eigen::VectorXd &points) const {
    return DecompositionType{m_basis->operator()(points)}.solve(observations);
  }

  /**
   * @brief Determine coefficients that fit a spline function at the given
   * "points" and the given "observations". The observations consist of an n
   x
   * m-array with n observations and derivatives until order m - 1.
   *
   * @param observations values and derivatives to fit the spline function.
   * @param points evaluation points corresponding to the "observations".
   * @return Eigen::ArrayXd spline coefficients fitting the observations.
   */
  template <
      typename DecompositionType = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>>
  Eigen::ArrayXd fit(const std::vector<Eigen::VectorXd> &observations,
                     const std::vector<Eigen::VectorXi> &derivOrders,
                     const Eigen::ArrayXd &points) const {

    // store transformation matrices and spline bases
    std::vector<Eigen::MatrixXd> transforms(m_basis->order() - 1);
    std::vector<Basis> bases(m_basis->order() - 1);

    // TODO: iterative application of derivative based on previous result
    int cOrder{};
    for (Eigen::MatrixXd &transform : transforms) {
      transform = m_basis->derivative(bases[cOrder], cOrder);
      ++cOrder;
    }

    // evaluate basis functions at "points" and transform according to given
    // derivartive
    Eigen::MatrixXd basisValues(m_basis->dim(), m_basis->dim());
    int cObs{};
    int cRow{};
    for (const Eigen::VectorXd &observation : observations) {
      int cValue{};
      for (double value : observation) {
        basisValues(cRow++, Eigen::all) =
            bases[derivOrders[cObs](cValue)](points(cObs)) *
            transforms[derivOrders[cObs](cValue)];
        ++cValue;
      }
      ++cObs;
    }

    // arrange observation in array
    Eigen::VectorXd splineValues(m_basis->dim());
    int cElem{};
    for (const auto &observation : observations)
      for (double value : observation)
        splineValues(cElem++) = value;

    // solve for spline coefficients
    return DecompositionType{basisValues}.solve(splineValues);
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
  std::shared_ptr<Basis> m_basis; /**<< spline basis */
};
}; // namespace BasisSplines
#endif