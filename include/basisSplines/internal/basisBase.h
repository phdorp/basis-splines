#ifndef BASIS_BASE_H
#define BASIS_BASE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <numeric>

namespace BasisSplines {
/**
 * @brief BasisBase of piecewise polynomial functions represented by truncated
 * powers.
 *
 * The basis is defined by its order and an increasing sequence of knots.
 * It provides properties that are derived from the knots and degree.
 * Allows the combination of two splines bases.
 */
class BasisBase {
public:
  BasisBase() = default;

  /**
   * @brief Construct a new BasisBase for the given knots and order.
   *
   * @param knots locations of the BasisBase knots.
   * @param order basis order.
   */
  BasisBase(const Eigen::ArrayXd &knots, int order)
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
  std::pair<Eigen::ArrayXd, Eigen::ArrayXi>
  getBreakpoints(double accuracy = 1e-6) const {
    return toBreakpoints(m_knots, m_order, accuracy);
  }

  /**
   * @brief Evaluate the truncated power basis at the given points.
   *
   * @param points evaluation points.
   * @param accDenum accuracy basis denominator.
   * @param accDomain point accuracy at domain limits.
   * @return Eigen::ArrayXd values of truncated powers with "points.size()" rows
   * and "self->dim()" columns.
   */
  Eigen::MatrixXd operator()(const Eigen::ArrayXd &points,
                             double accDenum = 1e-6,
                             double accDomain = 1e-6) const {
    Eigen::MatrixXd values{Eigen::MatrixXd::Zero(points.size(), dim())};

    int cPoint{};
    for (double point : points) {
      std::vector<Eigen::VectorXd> valuesTmp(m_order);

      valuesTmp[0].resize(m_knots.size() - 1);
      for (int cKnot{}; cKnot < m_knots.size() - 1; ++cKnot) {
        valuesTmp[0](cKnot) =
            inKnotSeg(m_knots(cKnot), m_knots(cKnot + 1), point, accDomain) ? 1
                                                                            : 0;
      }

      for (int cOrder{2}; cOrder <= m_order; ++cOrder) {
        for (int cKnot{}; cKnot < m_knots.size() - cOrder; ++cKnot) {
          const double denumL{m_knots(cKnot + cOrder - 1) - m_knots(cKnot)};
          const double weightL{std::abs(denumL) > accDenum
                                   ? (point - m_knots(cKnot)) / denumL
                                   : 0};
          const double denumR{m_knots(cKnot + cOrder) - m_knots(cKnot + 1)};
          const double weightR{std::abs(denumR) > accDenum
                                   ? (m_knots(cKnot + cOrder) - point) / denumR
                                   : 0};
          valuesTmp[cOrder - 1].resize(m_knots.size() - cOrder);
          valuesTmp[cOrder - 1](cKnot) =
              weightL * valuesTmp[cOrder - 2](cKnot) +
              weightR * valuesTmp[cOrder - 2](cKnot + 1);
        }
      }

      values(cPoint++, Eigen::seqN(0, dim())) =
          valuesTmp[m_order - 1](Eigen::seqN(0, dim()));
    }

    return values;
  }

  /**
   * @brief Determine the Greville sites representing the knot averages.
   *
   * @return Eigen::ArrayXd greville sites.
   */
  Eigen::ArrayXd greville() const {
    // basis order 1 greville abs. coincide with knots
    if (m_order == 1)
      return m_knots;

    // higher order basis knot averages
    Eigen::ArrayXd grevilleSites(dim());

    for (int cKnot{}; cKnot < dim(); ++cKnot) {
      auto begin{m_knots.begin() + cKnot + 1};
      auto end{begin + m_order - 1};
      grevilleSites(cKnot) = std::accumulate(begin, end, 0.0) / (m_order - 1);
    }

    return grevilleSites;
  }

  /**
   * @brief Convert breakpoints to knots.
   *
   * @param bps breakpoints for conversion.
   * @param conts continuity at the breakpoints.
   * @param order basis order.
   * @return Eigen::ArrayXd knot representation of given breakpoints.
   */
  static Eigen::ArrayXd toKnots(const Eigen::ArrayXd &bps,
                                const Eigen::ArrayXi &conts, int order) {
    Eigen::ArrayXi mults{order - conts};
    Eigen::ArrayXd knots(bps.size() * order - conts.sum());
    auto mult{mults.begin()};
    auto bp{bps.begin()};
    for (double &knot : knots) {
      knot = *bp;
      --(*mult);
      if (*mult == 0) {
        ++mult;
        ++bp;
      }
    }

    return knots;
  }

  static Eigen::ArrayXd
  toKnots(const std::pair<Eigen::ArrayXd, Eigen::ArrayXi> &bps, int order) {
    return toKnots(bps.first, bps.second, order);
  }

  static std::pair<Eigen::ArrayXd, Eigen::ArrayXi>
  toBreakpoints(const Eigen::ArrayXd &knots, int order,
                double accuracy = 1e-6) {
    int idxBps{};
    Eigen::ArrayXd breakpoints(knots.size());
    breakpoints(0) = knots(0);
    Eigen::ArrayXi continuities{Eigen::ArrayXi::Zero(knots.size()) + order};
    --continuities(0);

    auto curKnot{knots.begin() + 1};
    for (; curKnot != knots.end(); ++curKnot) {
      if (*curKnot > breakpoints(idxBps) + accuracy)
        breakpoints(++idxBps) = *curKnot;
      --continuities(idxBps);
    }

    return {breakpoints(Eigen::seqN(0, idxBps + 1)),
            continuities(Eigen::seqN(0, idxBps + 1))};
  }

private:
  Eigen::ArrayXd m_knots; /**<< basis knots */
  int m_order{};          /**<< basis order */

  bool inKnotSeg(double knotL, double knotR, double point,
                 double accuracy = 1e-6) const {
    if (knotL == m_knots(0))
      return point >= knotL - accuracy && point <= knotR;
    else if (knotR == m_knots(m_knots.size() - 1))
      return point > knotL && point <= knotR + accuracy;
    return point > knotL && point <= knotR;
  }
};
}; // namespace BasisSplines

#endif