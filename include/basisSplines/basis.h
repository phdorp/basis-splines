#ifndef BASIS_H
#define BASIS_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <numeric>

#include "basisSplines/internal/basisBase.h"
#include "basisSplines/interpolate.h"

namespace BasisSplines {
/**
 * @brief Basis of piecewise polynomial functions represented by truncated
 * powers.
 *
 * The basis is defined by its order and an increasing sequence of knots.
 * It provides properties that are derived from the knots and degree.
 * Allows the combination of two splines bases.
 */
class Basis : public BasisBase {
public:
  Basis() = default;

  /**
   * @brief Construct a new BasisBase for the given knots and order.
   *
   * @param knots locations of the BasisBase knots.
   * @param order basis order.
   */
  Basis(const Eigen::ArrayXd &knots, int order)
      : BasisBase{knots, order} {}

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
   * @brief Determine new basis with decreased order.
   *
   * @param orderDec order to decrease.
   * @return Basis basis with reduced order.
   */
  Basis orderDecrease(int orderDec = 1) const {
    Basis basis{knots()(Eigen::seqN(1, knots().size() - 2)), order() - 1};
    if (orderDec == 1)
      return basis;
    return basis.orderDecrease(orderDec - 1);
  }

  /**
   * @brief Determine new basis with increased order.
   *
   * @param orderInc order to increase.
   * @return Basis basis with increased order.
   */
  Basis orderIncrease(int orderInc = 1) const {
    Eigen::ArrayXd knotsNew(knots().size() + 2);
    knotsNew << knots()(0), knots(), *(knots().end());
    Basis basis{knotsNew, order() + 1};
    if (orderInc == 1)
      return basis;
    return basis.orderIncrease(orderInc - 1);
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
};
}; // namespace BasisSplines

#endif