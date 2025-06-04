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
  // MARK: public methods
  Spline() = default;

  /**
   * @brief Construct a new Spline in basis form.
   *
   * @param basis spline basis.
   * @param coefficients spline coefficients.
   */
  Spline(const std::shared_ptr<Basis> basis,
         const Eigen::VectorXd &coefficients)
      : m_basis{basis}, m_coefficients{coefficients} {}

  /**
   * @brief Returns the spline coefficients.
   *
   * @return const Eigen::ArrayXd& spline coefficients.
   */
  const Eigen::VectorXd &coefficients() const { return m_coefficients; }

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
   * @brief Evaluate spline at given point.
   *
   * @param point evaluation point.
   * @return double spline fucntion value at "point".
   */
  double operator()(double point) const { return (*this)({{point}})(0); }

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
    // create derivative basis and determine coefficients
    Basis basisNew {};
    Eigen::VectorXd coeffsNew(m_basis->derivative(basisNew, m_coefficients, order));

    // return derivative spline
    return {std::make_shared<Basis>(basisNew), coeffsNew};
  }

  /**
   * @brief Create new spline as integral of this spline.
   *
   * @param order integral order.
   * @return Spline as integral of "order".
   */
  Spline integral(int order = 1) const {
    // create derivative basis and determine coefficients
    Basis basisNew {};
    Eigen::VectorXd coeffsNew(m_basis->integral(basisNew, m_coefficients, order));

    // return derivative spline
    return {std::make_shared<Basis>(basisNew), coeffsNew};
  }

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
              Eigen::VectorXd procSum{(*this)(points) + spline(points)};
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
              Eigen::VectorXd procProd{(*this)(points)*spline(points)};
              return procProd;
            })};
  }

  /**
   * @brief Create new spline including the given and this splines' knots.
   * The new spline coincides with this spline.
   * The distance between coefficients and spline is decreased.
   * The knot multiplicity must remain smaller than the basis order.
   *
   * @tparam Interp type of interpolation.
   * @param knots knots to insert to this basis' knots.
   * @return Spline new spline including the given knots.
   */
  template <typename Interp = Interpolate>
  Spline insertKnots(const Eigen::ArrayXd &knots) const {
    // create new basis with inserted knot
    const std::shared_ptr<Basis> basis{
        std::make_shared<Basis>(m_basis->insertKnots(knots))};
    // determine new coefficients via interpolation
    const Interp interp{basis};
    return {basis, interp.fit([&](const Eigen::ArrayXd &points) {
              Eigen::VectorXd procInsert{(*this)(points)};
              return procInsert;
            })};
  }

  /**
   * @brief Determine a spline representing the "first" and the "last" segment
   * of "this" spline.
   *
   * @param first index of the first segment.
   * @param last index of the last segment.
   * @return Spline segment spline.
   */
  Spline getSegment(int first, int last) const {
    // determine basis representation of segments
    const std::shared_ptr<Basis> basisSeg{
        std::make_shared<Basis>(m_basis->getSegment(first, last))};

    // points to fit segment on this spline
    // TODO: segment wise linear spacing
    const auto breakpoints{m_basis->getBreakpoints()};
    const Eigen::ArrayXd points{
        Eigen::ArrayXd::LinSpaced(basisSeg->dim(), breakpoints.first(first),
                                  breakpoints.first(last + 1))};

    // determine new coefficients by fitting
    return {basisSeg, Interpolate{basisSeg}.fit((*this)(points), points)};
  }

private:
  // MARK: private properties
  std::shared_ptr<Basis> m_basis{}; /**<< spline basis */
  Eigen::VectorXd m_coefficients{}; /**<< spline coefficients */
};
}; // namespace BasisSplines

#endif
