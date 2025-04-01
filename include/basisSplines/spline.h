#ifndef SPLINE_H
#define SPLINE_H

#include <Eigen/Core>
#include <memory>

#include "basisSplines/basis.h"

namespace BasisSplines {
/**
 * @brief Polynomial spline in basis form.
 *
 * Represents a spline function for a given basis. The spline implements the
 * derivative, integral, product, and sum of two splines. Splines are
 * represented äquivalently by splines of higher order or with additional knots.
 *
 */
class Spline {
public:
  /**
   * @brief Construct a new Spline in basis form.
   *
   * @param basis spline basis.
   * @param coefficients spline coefficients.
   */
  Spline(const std::shared_ptr<Basis> basis, const Eigen::ArrayXd &coefficients)
      : m_basis{basis}, m_coefficients{coefficients} {}

  /**
   * @brief Returns the spline coefficients.
   *
   * @return const Eigen::ArrayXd& spline coefficients.
   */
  const Eigen::ArrayXd &coefficients() const { return m_coefficients; }

  /**
   * @brief Returns the spline basis.
   *
   * @return const std::shared_ptr<Basis> spline basis.
   */
  const std::shared_ptr<Basis> basis() const { return m_basis; }

  /**
   * @brief Evaluate spline at given points.
   *
   * @param points evaluation points.
   * @return Eigen::ArrayXd spline function values at "points".
   */
  Eigen::ArrayXd operator()(const Eigen::ArrayXd &points) const {
    return (m_basis->operator()(points).matrix() * m_coefficients.matrix())
        .array();
  }

  /**
   * @brief Create new spline with negative spline coefficients.
   *
   * @return Spline spline with negative spline coefficients.
   */
  Spline operator-() const { return {m_basis, -m_coefficients}; }

  /**
   * @brief Create new spline as sum of two splines.
   *
   * @param splineL left spline in sum.
   * @param splineR right spline in sum.
   * @return Spline representation of spline sum.
   */
  friend Spline operator+(const Spline &splineL, const Spline &splineR);

  /**
   * @brief Create new spline as difference between two splines.
   *
   * @param splineL left spline in difference.
   * @param splineR right spline in difference.
   * @return Spline representation of spline difference.
   */
  friend Spline operator-(const Spline &splineL, const Spline &splineR);

  /**
   * @brief Create new spline as product of two splines.
   *
   * @param splineL left spline in product.
   * @param splineR right spline in product.
   * @return Spline representation of spline product.
   */
  friend Spline operator*(const Spline &splineL, const Spline &splineR);

private:
  std::shared_ptr<Basis> m_basis{}; /**<< spline basis */
  Eigen::ArrayXd m_coefficients{};  /**<< spline coefficients */
};
}; // namespace BasisSplines

#endif
