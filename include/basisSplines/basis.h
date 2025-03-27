#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Core>
#include <numeric>

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
  Basis(const Eigen::ArrayXd &knots, int order)
      : m_knots{knots}, m_order{order} {}

  /**
   * @brief Determine basis dimensionality.
   *
   * @return int basis dimensionality.
   */
  int dim() const { return m_knots.size() - m_order; }

  /**
   * @brief Determine basis order.
   *
   * @return int basis order.
   */
  int order() const { return m_order; }

  /**
   * @brief Determine basis knots.
   *
   * @return const Eigen::ArrayXd& basis knots.
   */
  const Eigen::ArrayXd &knots() const { return m_knots; }

  /**
   * @brief Determine basis breakpoints and continuities at breakpoints.
   *
   * @return std::pair<Eigen::ArrayXd, Eigen::ArrayXi> breakpoints and
   * continuities.
   */
  std::pair<Eigen::ArrayXd, Eigen::ArrayXi> breakpoints() const {
    int idxBps{};
    Eigen::ArrayXd breakpoints(m_knots.size());
    breakpoints(0) = m_knots(0);
    Eigen::ArrayXi continuities{Eigen::ArrayXi::Zero(m_knots.size()) + m_order};
    --continuities(0);

    auto curKnot{m_knots.begin() + 1};
    for (; curKnot != m_knots.end(); ++curKnot) {
      if (*curKnot > breakpoints(idxBps))
        breakpoints(++idxBps) = *curKnot;
      --continuities(idxBps);
    }

    return {breakpoints(Eigen::seqN(0, idxBps + 1)),
            continuities(Eigen::seqN(0, idxBps + 1))};
  }

  /**
   * @brief Evaluate the truncated power basis at the given points.
   *
   * @param points evaluation points.
   * @return Eigen::ArrayXd values of truncated powers.
   */
  Eigen::ArrayXXd operator()(const Eigen::ArrayXd &points) const {
    Eigen::ArrayXXd values{Eigen::ArrayXXd::Zero(points.size(), dim())};

    int cPoint{};
    for (double point : points) {
      Eigen::ArrayXXd valuesTmp{
          Eigen::ArrayXXd::Zero(m_order, m_knots.size() - 1)};

      for (int cKnot{}; cKnot < m_knots.size() - 1; ++cKnot) {
        valuesTmp(0, cKnot) =
            inKnotSeg(m_knots(cKnot), m_knots(cKnot + 1), point) ? 1 : 0;
      }

      for (int cOrder{2}; cOrder <= m_order; ++cOrder) {
        for (int cKnot{}; cKnot < m_knots.size() - cOrder - 1; ++cKnot) {
          const double denumL{m_knots(cKnot + cOrder - 1) - m_knots(cKnot)};
          const double weightL{
              std::abs(denumL) > 1e-10 ? (point - m_knots(cKnot)) / denumL : 0};
          const double denumR{m_knots(cKnot + cOrder) - m_knots(cKnot + 1)};
          const double weightR{std::abs(denumR) > 1e-10
                                   ? (m_knots(cKnot + cOrder) - point) / denumR
                                   : 0};
          valuesTmp(cOrder - 1, cKnot) =
              weightL * valuesTmp(cOrder - 2, cKnot) +
              weightR * valuesTmp(cOrder - 2, cKnot + 1);
        }
      }

      values(cPoint++, Eigen::seqN(0, dim())) =
          valuesTmp(m_order - 1, Eigen::seqN(0, dim()));
    }

    return values;
  }

  /**
   * @brief Determine the Greville sites representing the knot averages.
   *
   * @return Eigen::ArrayXd greville sites.
   */
  Eigen::ArrayXd greville() const {
    Eigen::ArrayXd grevilleSites(dim());
    for (int cKnot{}; cKnot < dim(); ++cKnot) {
      auto begin{m_knots.begin() + cKnot + 1};
      auto end{begin + m_order - 1};
      grevilleSites(cKnot) = std::accumulate(begin, end, 0.0) / (m_order - 1);
    }

    return grevilleSites;
  }

  /**
   * @brief Combine the knots of this and another Basis.
   *
   * @param basis other basis to combine with.
   * @return Eigen::ArrayXd knots as the combination of both bases.
   */
  Eigen::ArrayXd combine(const Basis &basis);

private:
  Eigen::ArrayXd m_knots; /**<< basis knots */
  int m_order{};          /**<< basis order */

  bool inKnotSeg(double knotL, double knotR, double point) const {
    if (knotR == m_knots(0))
      return point >= knotL && point <= knotR;
    return point > knotL && point <= knotR;
  }
};
}; // namespace BasisSplines

#endif