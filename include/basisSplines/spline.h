#ifndef SPLINE_H
#define SPLINE_H

#include <Eigen/Core>
#include <memory>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"

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
   * @brief Create new spline as derivative of this spline.
   *
   * @param order derivative order.
   * @return Spline as derivative of "order".
   */
  Spline derivative(int order = 1) const {
    // basis for derivative spline of order - 1
    std::shared_ptr<Basis> basis{std::make_shared<Basis>(
        m_basis->knots()(Eigen::seqN(1, m_basis->knots().size() - 2)),
        m_basis->order() - 1)};

    // coefficients of derivative spline coeffs = o * (c_i+1 - c_i) / (k_i+o - k_i+1)
    Eigen::ArrayXd coeffs(basis->dim());
    for (int idx{}; idx < coeffs.size(); ++idx)
      coeffs(idx) = (m_basis->order() - 1) *
                    (m_coefficients(idx + 1) - m_coefficients(idx)) /
                    (m_basis->knots()(idx + m_basis->order()) -
                     m_basis->knots()(idx + 1));
    return {basis, coeffs};
  }

  /**
   * @brief Create new spline as integral of this spline.
   *
   * @param order integral order.
   * @return Spline as integral of "order".
   */
  Spline integral(int order = 1) const;

  /**
   * @brief Create new spline as sum of this and another spline.
   *
   * @tparam Interp type of interpolation.
   * @param spline function to add with.
   * @return Spline representation of spline sum.
   */
  template <typename Interp = Interpolate>
  Spline add(const Spline &spline) const {
    const std::shared_ptr<Basis> basis{std::make_shared<Basis>(
        m_basis->combine(*spline.basis().get(),
                         std::max(m_basis->order(), spline.basis()->order())))};
    const Interp interp{basis};
    return {basis, interp.fit([&](const Eigen::ArrayXd &points) {
              Eigen::ArrayXd procSum{(*this)(points) + spline(points)};
              return procSum;
            })};
  }

  /**
   * @brief Create new spline as product of this and another spline.
   *
   * @tparam Interp type of interpolation.
   * @param spline function to multiply with.
   * @return Spline representation of spline product.
   */
  template <typename Interp = Interpolate>
  Spline prod(const Spline &spline) const {
    const std::shared_ptr<Basis> basis{std::make_shared<Basis>(
        m_basis->combine(*spline.basis().get(),
                         m_basis->order() + spline.basis()->order() - 1))};
    const Interp interp{basis};
    return {basis, interp.fit([&](const Eigen::ArrayXd &points) {
              Eigen::ArrayXd procProd{(*this)(points)*spline(points)};
              return procProd;
            })};
  }

private:
  std::shared_ptr<Basis> m_basis{}; /**<< spline basis */
  Eigen::ArrayXd m_coefficients{};  /**<< spline coefficients */
};
}; // namespace BasisSplines

#endif
