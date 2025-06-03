#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <numeric>

#include "basisSplines/math.h"

namespace BasisSplines {

class Interpolate;

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
  // MARK: public methods
  Basis() = default;

  /**
   * @brief Construct a new BasisBase for the given knots and order.
   *
   * @param knots locations of the BasisBase knots.
   * @param order basis order.
   */
  Basis(const Eigen::ArrayXd &knots, int order)
      : m_knots{knots}, m_order{order} {}

  /**
   * @brief Create new basis including the given and this basis' knots.
   *
   * @param knotsIn knots to insert to this basis' knots.
   * @return Basis new basis including the given knots.
   */
  Basis insertKnots(const Eigen::ArrayXd &knotsIn) const {
    // concatenate knots with this basis knots
    Eigen::ArrayXd knotsNew(knotsIn.size() + knots().size());
    knotsNew << knotsIn, knots();
    // sort for increasing knot sequence
    std::sort(knotsNew.begin(), knotsNew.end());
    return {knotsNew, order()};
  }

  /**
   * @brief Combine the knots of this and another Basis.
   *
   * @param basis other basis to combine with.
   * @return Eigen::ArrayXd knots as the combination of both bases.
   */
  Basis combine(const Basis &basis, int order, double accuracy = 1e-6) const {
    Eigen::ArrayXd knotsThis{toKnots(getBreakpoints(), order)};
    Eigen::ArrayXd knotsOther{toKnots(basis.getBreakpoints(), order)};

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
   * @brief Determine new basis with decreased order.
   *
   * @param orderDec order to decrease.
   * @return Basis basis with reduced order.
   */
  Basis orderDecrease(int orderDec = 1) const {
    assert(orderDec >= 0 && "Order decrease must be positive.");

    // base case: no order decrease, create new instance of current basis
    if (orderDec == 0)
      return Basis{*this};

    // create basis of lower order
    return {knots()(Eigen::seqN(orderDec, knots().size() - 2 * orderDec)),
            order() - orderDec};
  }

  /**
   * @brief Determine new basis with increased order.
   *
   * @param orderInc order to increase.
   * @return Basis basis with increased order.
   */
  Basis orderIncrease(int orderInc = 1) const {
    assert(orderInc >= 0 && "Order increase must be positive.");

    // base case: no order increase, create new instance of current basis
    if (orderInc == 0)
      return Basis{*this};

    // create new basis of lower order and additional breakpoints
    Eigen::ArrayXd knotsNew(knots().size() + 2 * orderInc);
    knotsNew << Eigen::ArrayXd::Zero(orderInc) + knots()(0), knots(), Eigen::ArrayXd::Zero(orderInc) + *(knots().end());
    Basis basis{knotsNew, order() + orderInc};

    // create basis of higher order
    return {knotsNew, order() + orderInc};
  }

  /**
   * @brief Dertermines a matrix A to transform the spline coefficients c to
   * derivative coefficients dc.
   *
   * dc = A * c
   *
   * @param basis basis of reduced order.
   * @param orderDer derivative order.
   * @return Eigen::MatrixXd transformation matrix.
   */
  Eigen::MatrixXd derivative(Basis &basis, int orderDer = 1) const {
    if (orderDer == 0) {
      basis = *this;
      return Eigen::MatrixXd::Identity(dim(), dim());
    }

    // determine transformation matrix
    Eigen::MatrixXd transform(Eigen::MatrixXd::Zero(dim() - 1, dim()));
    for (int cRow{}; cRow < transform.rows(); ++cRow) {
      transform(cRow, cRow) =
          (order() - 1) / (knots()(cRow + 1) - knots()(order() + cRow));
      transform(cRow, cRow + 1) = -transform(cRow, cRow);
    }

    // provide basis derivative basis with decreased order
    Basis basisDeriv{orderDecrease()};

    // base case order 1 derivative
    if (orderDer == 1) {
      basis = basisDeriv;
      return transform;
    }

    // recursion higher order derivative
    return basisDeriv.derivative(basis, orderDer - 1) * transform;
  };

  /**
   * @brief Transforms the given values, which are basis spline values or
   * coefficients, to the derivative of this basis.
   *
   * @param basis basis of reduced order.
   * @param values basis values or spline coefficients.
   * @param orderDer derivative order.
   * @return Eigen::VectorXd derivative values.
   */
  Eigen::VectorXd derivative(Basis &basis, const Eigen::VectorXd &values,
                             int orderDer = 1) const {
    if (orderDer == 0) {
      basis = *this;
      return values;
    }

    // provide basis derivative basis with decreased order
    Basis basisDeriv{orderDecrease()};

    // values transformed to derivative valuesNew = o * (values_i+1 - values_i)
    // / (k_i+o - k_i+1)
    Eigen::VectorXd valuesNew(basisDeriv.dim());
    for (int idx{}; idx < valuesNew.size(); ++idx)
      valuesNew(idx) = (order() - 1) * (values(idx + 1) - values(idx)) /
                       (knots()(idx + order()) - knots()(idx + 1));

    // base case order 1 derivative
    if (orderDer == 1) {
      basis = basisDeriv;
      return valuesNew;
    }

    // recursion higher order derivative
    return basisDeriv.derivative(basis, valuesNew, orderDer - 1);
  };

  /**
   * @brief Dertermines a matrix A to transform the spline coefficients c to
   * integral coefficients ic.
   *
   * ic = A * c
   *
   * @param basis basis spline.
   * @param orderInt integral order.
   * @return Eigen::MatrixXd transformation matrix.
   */
  Eigen::MatrixXd integral(Basis &basis, int orderInt = 1) const {

    if (orderInt == 0) {
      basis = *this;
      return Eigen::MatrixXd::Identity(dim(), dim());
    }

    // initialize transformation matrix with zeros
    Eigen::MatrixXd transform(Eigen::MatrixXd::Zero(dim() + 1, dim()));

    // fill transformation matrix
    int cCol{};
    for (auto col : transform.colwise()) {
      for (int cRow{cCol + 1}; cRow < transform.rows(); ++cRow) {
        col(cRow) = (knots()(order() + cCol) - knots()(cCol)) / order();
      }
      ++cCol;
    }

    // provide basis integral basis with increased order
    Basis basisDeriv{orderIncrease()};

    // base case order 1 integral
    if (orderInt == 1) {
      basis = basisDeriv;
      return transform;
    }

    // recursion higher order integral
    return basisDeriv.integral(basis, orderInt - 1) * transform;
  }

  /**
   * @brief Transforms the given values, which are basis spline values or
   * coefficients, to the integral of this basis.
   *
   * @param basis basis of increased order.
   * @param values basis values or spline coefficients.
   * @param orderInt integral order.
   * @return Eigen::VectorXd integral values.
   */
  Eigen::VectorXd integral(Basis &basis, const Eigen::VectorXd &values,
                           int orderInt = 1) const {
    if (orderInt == 0) {
      basis = *this;
      return values;
    }

    // provide basis integral basis with decreased order
    Basis basisInt{orderIncrease()};

    // values transformed to integral valuesNew_i+1 = values_i * (k_i+o -
    // k_i) / o + valuesNew_i
    Eigen::VectorXd valuesNew(basisInt.dim());
    for (int idx{}; idx < valuesNew.size() - 1; ++idx)
      valuesNew(idx + 1) =
          values(idx) * (knots()(idx + order()) - knots()(idx)) / order() +
          valuesNew(idx);

    // base case order 1 integral
    if (orderInt == 1) {
      basis = basisInt;
      return valuesNew;
    }

    // recursion higher order integral
    return basisInt.integral(basis, valuesNew, orderInt - 1);
  };

  /**
   * @brief Determine transformation matrices Tl and Tr for left and right
   * operand coefficients cl and cr to get sum coefficients cs.
   *
   * cs = Tl * cl + Tr * cr
   *
   * @param basis right operand basis.
   * @param basisOut sum basis.
   * @return std::pair<Eigen::MatrixXd, Eigen::MatrixXd> transformation matrices
   * Tl and Tr.
   */
  template <typename Interp = Interpolate>
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> add(const Basis &basis,
                                                  Basis &basisOut) const {
    // combine this and other basis to sum basis
    basisOut = combine(basis, std::max(order(), basis.order()));

    // instantiate interpolate with sum basis
    const Interp interp{std::make_shared<Basis>(basisOut)};
    // determine transform for this basis by interpolating the sum basis
    const Eigen::MatrixXd transformThis{interp.fit([&](Eigen::ArrayXd points) {
      Eigen::ArrayXXd values{(*this)(points)};
      return values;
    })};
    // determine transform for other basis by interpolating the sum basis
    const Eigen::MatrixXd transformOther{interp.fit([&](Eigen::ArrayXd points) {
      Eigen::ArrayXXd values{basis(points)};
      return values;
    })};

    return {transformThis, transformOther};
  }

  /**
   * @brief Determine transformation matrixT for coefficients cl and cr to get
   * product coefficients cs.
   *
   * cs = T (cl \kron cr)
   *
   * @param basis right operand basis.
   * @param basisOut product basis.
   * @return Eigen::MatrixXd transformation matrix T.
   */
  template <typename Interp = Interpolate>
  Eigen::MatrixXd prod(const Basis &basis, Basis &basisOut) const {
    // combine this and other basis to sum basis
    basisOut = combine(basis, order() + basis.order());

    // instantiate interpolate with sum basis
    const Interp interp{std::make_shared<Basis>(basisOut)};
    // determine transform for this basis by interpolating the sum basis
    const Eigen::MatrixXd transformThis{interp.fit([&](Eigen::ArrayXd points) {
      Eigen::ArrayXXd values{khatriRao((*this)(points), basis(points))};
      return values;
    })};

    return transformThis;
  }

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
   * @brief Set breakpoint continuities at given breakpoint indices.
   *
   * @param continuityNew new continuities.
   * @param idcs breakpoint indices to set.
   */
  void setContinuities(const Eigen::ArrayXi &continuityNew,
                       const Eigen::ArrayXi &idcs) {
    auto [breakpoints, conts] = getBreakpoints();

    if ((continuityNew < 0).any())
      throw std::invalid_argument("Breakpoint continuity must be positive.");

    if ((continuityNew >= m_order).any())
      throw std::invalid_argument(
          "Breakpoint continuity must not exceed basis order.");

    conts(idcs) = continuityNew;

    m_knots = toKnots({breakpoints, conts}, m_order);
  }

  /**
   * @brief Set breakpoints at given breakpoint indices.
   *
   * @param breakpointsNew new breakpoints.
   * @param idcs breakpoint indices to set.
   */
  void setBreakpoints(const Eigen::ArrayXd &breakpointsNew,
                      const Eigen::ArrayXi &idcs) {
    auto [breakpoints, conts] = getBreakpoints();

    breakpoints(idcs) = breakpointsNew;

    if (!checkIncreasing(breakpoints))
      throw std::invalid_argument(
          "Breakpoints not aranged in strictly increasing order.");

    m_knots = toKnots({breakpoints, conts}, m_order);
  }

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

  Eigen::MatrixXd operator()(double point, double accDenum = 1e-6,
                             double accDomain = 1e-6) const {
    return (*this)({{point}});
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
   * @brief Determine a basis representing the "first" to the "last" segment of
   * "this" basis.
   *
   * @param first index of the first segment.
   * @param last index of the last segment.
   * @return Basis segment basis.
   */
  Basis getSegment(int first, int last) const {

    const auto breakpoints{getBreakpoints()};

    // find first and last knots of the given segments
    auto end{
        std::find(m_knots.begin(), m_knots.end(), breakpoints.first(last + 1)) +
        m_order};
    auto begin{(std::find(std::make_reverse_iterator(m_knots.end()),
                          std::make_reverse_iterator(m_knots.begin()),
                          breakpoints.first(first)))
                   .base() -
               m_order};

    // copy knots to new variable
    Eigen::ArrayXd knots(end - begin);
    int cElem{};
    for (; begin < end; ++begin)
      knots(cElem++) = *begin;

    return {knots, m_order};
  }

  // MARK: public statics

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
  // MARK: private properties
  Eigen::ArrayXd m_knots; /**<< basis knots */
  int m_order{};          /**<< basis order */

  // MARK: private methods
  bool inKnotSeg(double knotL, double knotR, double point,
                 double accuracy = 1e-6) const {
    if (knotL == m_knots(0))
      return point >= knotL - accuracy && point <= knotR;
    else if (knotR == m_knots(m_knots.size() - 1))
      return point > knotL && point <= knotR + accuracy;
    return point > knotL && point <= knotR;
  }

  // MARK: private statics
  static bool checkIncreasing(const Eigen::ArrayXd &sequence) {
    for (auto elemPtr{sequence.begin() + 1}; elemPtr < sequence.end();
         ++elemPtr)
      if (*elemPtr - *(elemPtr - 1) < 0)
        return false;
    return true;
  }
};
}; // namespace BasisSplines

#endif