#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Core>

namespace BasisSplines {
/**
 * @brief Basis of piecewise polynomial functions represented by truncated
 * powers.
 *
 * The basis is defined by its order and an increasing sequence of knots.
 * It provides properties that are derived from the knots and degree.
 * Allows the combination of two splines bases.
 */
class Basis {
public:
  /**
   * @brief Construct a new Basis for the given knots and order.
   *
   * @param knots locations of the Basis knots.
   * @param order basis order.
   */
  Basis(const Eigen::ArrayXd &knots, double order)
      : m_knots{knots}, m_order{order} {}

  /**
   * @brief Determine basis dimensionality.
   *
   * @return int basis dimensionality.
   */
  int dim();

  /**
   * @brief Determine basis order.
   *
   * @return int basis order.
   */
  int order();

  /**
   * @brief Determine basis knots.
   *
   * @return const Eigen::ArrayXd& basis knots.
   */
  const Eigen::ArrayXd& knots();

  /**
   * @brief Determine basis breakpoints and continuities at breakpoints.
   *
   * @return std::pair<Eigen::ArrayXd, Eigen::ArrayXi> breakpoints and
   * continuities.
   */
  std::pair<Eigen::ArrayXd, Eigen::ArrayXi> breakpoints();

  /**
   * @brief Evaluate the truncated power basis at the given points.
   *
   * @param points evaluation points.
   * @return Eigen::ArrayXd values of truncated powers.
   */
  Eigen::ArrayXd operator()(const Eigen::ArrayXd points);

  /**
   * @brief Determine the Greville sites representing the knot averages.
   *
   * @return Eigen::ArrayXd greville sites.
   */
  Eigen::ArrayXd greville();

  /**
   * @brief Combine the knots of this and another Basis.
   *
   * @param basis other basis to combine with.
   * @return Eigen::ArrayXd knots as the combination of both bases.
   */
  Eigen::ArrayXd combine(const Basis &basis);

private:
  Eigen::ArrayXd m_knots{}; /**<< basis knots */
  double m_order = 0 /**<< basis order */;
};
}; // namespace BasisSplines

#endif