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
  Basis() = default;

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
   * @brief Create new basis including the given and this basis' knots.
   *
   * @param knots knots to insert to this basis' knots.
   * @return Basis new basis including the given knots.
   */
  Basis insertKnots(const Eigen::ArrayXd &knots) const {
    // concatenate knots with this basis knots
    Eigen::ArrayXd knotsNew(knots.size() + m_knots.size());
    knotsNew << knots, m_knots;
    // sort for increasing knot sequence
    std::sort(knotsNew.begin(), knotsNew.end());
    return {knotsNew, m_order};
  }

  /**
   * @brief Determine basis breakpoints and continuities at breakpoints.
   *
   * @return std::pair<Eigen::ArrayXd, Eigen::ArrayXi> breakpoints and
   * continuities.
   */
  std::pair<Eigen::ArrayXd, Eigen::ArrayXi>
  breakpoints(double accuracy = 1e-6) const {
    int idxBps{};
    Eigen::ArrayXd breakpoints(m_knots.size());
    breakpoints(0) = m_knots(0);
    Eigen::ArrayXi continuities{Eigen::ArrayXi::Zero(m_knots.size()) + m_order};
    --continuities(0);

    auto curKnot{m_knots.begin() + 1};
    for (; curKnot != m_knots.end(); ++curKnot) {
      if (*curKnot > breakpoints(idxBps) + accuracy)
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
  /**
   * @brief Evaluate the truncated power basis at the given points.
   *
   * @param points evaluation points.
   * @param accDenum accuracy basis denominator.
   * @param accDomain point accuracy at domain limits.
   * @return Eigen::ArrayXd values of truncated powers with "points.size()" rows
   * and "self->dim()" columns.
   */
  Eigen::ArrayXXd operator()(const Eigen::ArrayXd &points,
                             double accDenum = 1e-6,
                             double accDomain = 1e-6) const {
    Eigen::ArrayXXd values{Eigen::ArrayXXd::Zero(points.size(), dim())};

    int cPoint{};
    for (double point : points) {
      std::vector<Eigen::ArrayXd> valuesTmp(m_order);

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
  Basis combine(const Basis &basis, int order, double accuracy = 1e-6) const {
    Eigen::ArrayXd knotsThis{toKnots(breakpoints(), order)};
    Eigen::ArrayXd knotsOther{toKnots(basis.breakpoints(), order)};

    // create combined knots worst case length
    Eigen::ArrayXd knotsComb(knotsThis.size() + knotsOther.size());

    // iterate over both knot arrays
    // compare knots and place smaller one in "knotsComb"
    // auto knotComb {knotsComb.begin()};
    auto knotThis{knotsThis.begin()};
    auto knotOther{knotsOther.begin()};
    int numKnotsComb{};
    for (auto &knotComb : knotsComb) {
      bool atThisEnd{knotThis == (knotsThis.end())};
      bool atOtherEnd{knotOther == (knotsOther.end())};

      if (atThisEnd && atOtherEnd)
        break;

      // assign this knot if smaller or other end is reached
      if (*knotThis < *knotOther - accuracy && !atThisEnd || atOtherEnd)
        knotComb = *(knotThis++);
      // assign other knot if smaller or other end is reached
      else if (*knotOther < *knotThis - accuracy && !atOtherEnd || atThisEnd)
        knotComb = *(knotOther++);
      // asign this and other knot, which are equal
      else {
        knotComb = *knotOther;
        if (!atOtherEnd)
          ++knotOther;
        if (!atThisEnd)
          ++knotThis;
      }

      ++numKnotsComb;
    }

    return {knotsComb(Eigen::seqN(0, numKnotsComb)), order};
  }

  /**
   * @brief Determine basis derivative.
   * Reduces order by 1.
   *
   * @param order derivative order.
   * @return Basis derivative basis.
   */
  Basis derivative(int order = 1) const {
    Basis basis{m_knots(Eigen::seqN(1, m_knots.size() - 2)), m_order - 1};
    if (order == 1)
      return basis;
    return basis.derivative(order - 1);
  }

  /**
   * @brief Determine basis derivative.
   * Reduces order by 1.
   *
   * @param order derivative order.
   * @return Basis derivative basis.
   */
  Basis integral(int order = 1) const {
    Eigen::ArrayXd knots(m_knots.size() + 2);
    knots << m_knots(0), m_knots, *(m_knots.end());
    Basis basis{knots, m_order + 1};
    if (order == 1)
      return basis;
    return basis.integral(order - 1);
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