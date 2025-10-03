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
 * Represents a multidimensional spline S(t) determined by its coefficients C
 * for a given basis B(t).
 *
 * S(t) = C^T B(t)
 *
 */
class Spline {
public:
  // MARK: public methods
  Spline() = default;

  /**
   * @brief Construct a new spline in basis form from a "basis" spline and the
   * "coefficients". The number of "coefficients" rows must correspond with the
   * "basis" dimensionality.
   *
   * @param basis spline basis.
   * @param coefficients spline coefficients.
   */
  Spline(const std::shared_ptr<Basis> basis,
         const Eigen::MatrixXd &coefficients)
      : m_basis{basis}, m_coefficients{coefficients} {
    assert(coefficients.rows() == basis->dim() &&
           "Coefficients must have same rows as basis dimensionality.");
  }

  /**
   * @brief Get the spline coefficients.
   * The number of rows corresponds with the basis spline dimensionality.
   * The number of columns corresponds with the spline output dimensionality.
   *
   * @return const Eigen::ArrayXd& spline coefficients.
   */
  const Eigen::MatrixXd &getCoefficients() const { return m_coefficients; }

  /**
   * @brief Set the spline coefficients.
   * The coefficients' size must equal the spline's coefficients' size.
   *
   * @param coefficients new spline coefficients.
   */
  void setCoefficients(const Eigen::MatrixXd coefficients) {
    assert(coefficients.rows() == m_coefficients.rows() &&
           "Coefficients must have same rows as spline coefficients.");
    assert(coefficients.cols() == m_coefficients.cols() &&
           "Coefficients must have same columns as spline coefficients.");
    m_coefficients = coefficients;
  }

  /**
   * @brief Get the spline basis.
   *
   * @return const std::shared_ptr<Basis> spline basis.
   */
  const std::shared_ptr<Basis> basis() const { return m_basis; }

  /**
   * @brief Get the spline output dimensionality.
   *
   * @return int spline output dimensionality.
   */
  int dim() const { return m_coefficients.cols(); }

  /**
   * @brief Evaluate spline at given "points".
   * The number of output rows corresponds with the number of "points".
   * The number of output columns corresponds with the spline output
   * dimensionality.
   * [Boo01, def. (51)]
   *
   * @param points evaluation points.
   * @return Eigen::ArrayXd spline function values at "points".
   */
  Eigen::ArrayXXd operator()(const Eigen::ArrayXd &points) const {
    return (m_basis->operator()(points) * m_coefficients);
  }

  /**
   * @brief Create new spline with negated spline coefficients.
   *
   * @return Spline spline with negated spline coefficients.
   */
  Spline operator-() const { return {m_basis, -m_coefficients}; }

  /**
   * @brief Create new spline as derivative of this spline.
   *
   * @param orderDer derivative order.
   * @return Spline as derivative of "orderDer".
   */
  Spline derivative(int orderDer = 1) const {
    assert(orderDer >= 0 && "Derivative order must be positive.");

    // create derivative basis and determine coefficients
    Basis basisNew{};
    Eigen::MatrixXd coeffsNew(
        m_basis->derivative(basisNew, m_coefficients, orderDer));

    // return derivative spline
    return {std::make_shared<Basis>(basisNew), coeffsNew};
  }

  /**
   * @brief Create new spline as integral of this spline.
   *
   * @param orderInt integral order.
   * @return Spline as integral of "orderInt".
   */
  Spline integral(int orderInt = 1) const {
    assert(orderInt >= 0 && "Derivative order must be positive.");

    // create derivative basis and determine coefficients
    Basis basisNew{};
    Eigen::MatrixXd coeffsNew(
        m_basis->integral(basisNew, m_coefficients, orderInt));

    // return derivative spline
    return {std::make_shared<Basis>(basisNew), coeffsNew};
  }

  /**
   * @brief Create new spline as sum of "this" and "other" spline.
   * Combine basis of "this" and "other" splines to create the sum basis.
   * Determine sum coefficients by interpolating the sum of this and other
   * spline.
   *
   * @tparam Interp type of interpolation.
   * @param other right spline summand.
   * @param accScale accepted difference between "this" and "other" splines'
   * basis scaling.
   * @param accBps tolerance for assigning knots to breakpoint.
   * @return Spline representation of spline sum.
   */
  template <typename Interp = Interpolate>
  Spline add(const Spline &other, double accScale = 1e-6,
             double accBps = 1e-6) const {
    // combine this and other basis to new basis
    // new basis order is maximum of this and other basis order
    const std::shared_ptr<Basis> newBasis{std::make_shared<Basis>(
        m_basis->combine(*other.basis().get(),
                         std::max(m_basis->order(), other.basis()->order()),
                         accScale, accBps))};

    // determine coefficients by interpolating sum of this and other spline
    return {newBasis, Interp{newBasis}.fit([&](const Eigen::ArrayXd &points) {
              return Eigen::MatrixXd{(*this)(points) + other(points)};
            })};
  }

  /**
   * @brief Create new spline as product of "this" and "other" spline.
   * Combine basis of "this" and "other" splines to create the product basis.
   * Determine product coefficients by interpolating the product of "this" and
   * "other" spline.
   *
   * @tparam Interp type of interpolation.
   * @param other right product spline.
   * @param accScale accepted difference between "this" and "other" splines'
   * basis scaling.
   * @param accBps tolerance for assigning knots to breakpoint.
   * @return Spline representation of spline product.
   */
  template <typename Interp = Interpolate>
  Spline prod(const Spline &other, double accScale = 1e-6,
              double accBps = 1e-6) const {
    // combine this and other basis to new basis
    // new basis order is sum of this and other basis order - 1
    const std::shared_ptr<Basis> newBasis{std::make_shared<Basis>(
        m_basis->combine(*other.basis().get(),
                         m_basis->order() + other.basis()->order() - 1,
                         accScale, accBps))};

    // determine coefficients by interpolating product of this and other spline
    return {newBasis, Interp{newBasis}.fit([&](const Eigen::ArrayXd &points) {
              return Eigen::MatrixXd{(*this)(points)*other(points)};
            })};
  }

  /**
   * @brief Inserts multiple knots into the spline.
   *
   * Creates a copy of the current spline and sequentially inserts each knot
   * from the provided array into the copy using the insertKnot method. The
   * resulting spline with all knots inserted is returned.
   *
   * @param knots Knot values to be inserted.
   * @return Spline New spline with the inserted knots.
   */
  Spline insertKnots(const Eigen::ArrayXd &knots) const {
    Spline spline{*this};

    for (double knot : knots)
      spline = spline.insertKnot(knot);

    return spline;
  }

  /**
   * @brief Create equivalent spline with inserted knot.
   *
   * Creates a new basis by inserting the specified knot into the
   * current basis. The coefficients are interpolated to create a new equivalent
   * spline.
   *
   * @param knot The knot value to insert into the spline.
   * @return Spline A new Spline object with the inserted knot and interpolated
   * coefficients.
   */
  Spline insertKnot(double knot) const {
    // create new basis with inserted knot
    const std::shared_ptr<Basis> basis{
        std::make_shared<Basis>(m_basis->insertKnots({{knot}}))};

    return {basis, interpolateCoefficients(knot)};
  }

  /**
   * @brief Create new spline with order increased by "change".
   * The new spline coincides with "this" spline.
   * The distance between coefficients and spline is decreased.
   *
   * @tparam Interp type of interpolation.
   * @param change positive order change.
   * @return Spline new spline with increased order.
   */
  template <typename Interp = Interpolate>
  Spline orderElevation(int change) const {
    assert(change >= 0 && "Order change must be positive.");

    // create new basis with increased order
    const std::shared_ptr<Basis> basis{
        std::make_shared<Basis>(m_basis->orderElevation(change))};

    // determine new coefficients via interpolation
    return {basis, Interp{basis}.fit([&](const Eigen::ArrayXd &points) {
              return Eigen::MatrixXd{(*this)(points)};
            })};
  }

  /**
   * @brief Get the roots along all output dimensions.
   *
   * The roots are estimated with algorithm from [1] until the maximum number of
   iterations or the desired accuracy.
   *
   * [1] K. Mørken and M. Reimers, “An unconditionally convergent method for
   computing zeros of splines and polynomials,” Math. Comp., vol. 76, no. 258,
   pp. 845–865, Jan. 2007, doi: 10.1090/S0025-5718-07-01923-0.
   *
   * @param maxIter Maximum number of iterations.
   * @param accAbs Tolerance for the spline output to be considered zero.
   * @return Eigen::ArrayXd Roots along all output dimensions.
   */
  std::vector<Eigen::ArrayXd> getRoots(int maxIter = 10,
                                       double accAbs = 1e-6) const {
    std::vector<Eigen::ArrayXd> zeros(m_coefficients.cols());

    for (int dim{}; dim < m_coefficients.cols(); ++dim)
      zeros[dim] = getRoots(dim, maxIter, accAbs);

    return zeros;
  }

  /**
   * @brief Get the roots along the specified output dimension.
   *
   * The roots are estimated with algorithm from [1] until the maximum number of
   iterations or the desired accuracy.
   *
   * [1] K. Mørken and M. Reimers, “An unconditionally convergent method for
   computing zeros of splines and polynomials,” Math. Comp., vol. 76, no. 258,
   pp. 845–865, Jan. 2007, doi: 10.1090/S0025-5718-07-01923-0.
   *
   * @param dim Output dimension to evaluate.
   * @param maxIter Maximum number of iterations.
   * @param accAbs Tolerance for the spline output to be considered zero.
   * @return Eigen::ArrayXd Roots along the given output dimension.
   */
  Eigen::ArrayXd getRoots(int dim, int maxIter = 10,
                          double accAbs = 1e-6) const {
    auto [rootIdcs, roots] = getRootIdcs(dim, accAbs);

    int cntRoot{};
    for (int rootIdx : rootIdcs) {
      if (rootIdx >= 0)
        roots(cntRoot++) = estimateRoot(rootIdx, dim, maxIter, accAbs);
    }

    return roots;
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
    // determine "begin" and "end" knot iterators of segment
    auto [begin, end] = m_basis->getSegmentKnots(first, last);

    // determine basis representation of segments
    const std::shared_ptr<Basis> basisSeg{
        std::make_shared<Basis>(m_basis->getSegment(begin, end))};

    // determine indices of coefficients of semgnet
    int firstCoeff{static_cast<int>(begin - m_basis->knots().begin())};
    int lastCoeff{static_cast<int>(end - m_basis->knots().begin()) -
                  m_basis->order() - 1};

    // new spline
    return {basisSeg,
            m_coefficients(Eigen::seq(firstCoeff, lastCoeff), Eigen::all)};
  }

  /**
   * @brief Determine spline with knots clamped to spline segment.
   *
   * @tparam Interp type of interpolation.
   * @return Spline clamped spline.
   */
  template <typename Interp = Interpolate> Spline getClamped() const {
    // determine clamped basis
    const std::shared_ptr<Basis> basisClamped{
        std::make_shared<Basis>(m_basis->getClamped())};

    // determine clamped spline coefficients by fitting clamped basis to this
    // spline
    return {basisClamped,
            Interp{basisClamped}.fit([&](const Eigen::MatrixXd &points) {
              return Eigen::MatrixXd{(*this)(points)};
            })};
  }

private:
  // MARK: private properties

  std::shared_ptr<Basis> m_basis{}; /**<< spline basis */
  Eigen::MatrixXd m_coefficients{}; /**<< spline coefficients */

  // MARK: private methods
  /**
   * @brief Interpolates coefficients along each dimension when inserting a new
   * knot.
   *
   * @param knot The position of the knot to be inserted.
   * @return Eigen::MatrixXd The matrix of interpolated coefficients for all
   * dimensions.
   */
  Eigen::MatrixXd interpolateCoefficients(double knot) const {

    Eigen::MatrixXd coeffs(m_coefficients.rows() + 1, m_coefficients.cols());

    for (int cDim{}; cDim < dim(); ++cDim)
      coeffs(Eigen::all, cDim) = interpolateCoefficients(knot, cDim);

    return coeffs;
  }

  /**
   * @brief Interpolates coefficients when inserting a new knot.
   *
   * This function computes the new set of spline coefficients after inserting a
   * knot at the given position. The coefficients are updated according to
   * [Boehm 1980].
   *
   * @param knotInsert The position of the knot to be inserted.
   * @param dim The dimension (column) of the coefficients to be updated.
   * @return Eigen::VectorXd The updated coefficients vector with one additional
   * element.
   */
  Eigen::VectorXd interpolateCoefficients(double knotInsert, int dim) const {
    Eigen::VectorXd coeffsNew(m_coefficients.rows() + 1);
    const Eigen::VectorXd coeffs{m_coefficients(Eigen::all, dim)};
    const Eigen::ArrayXd knots{m_basis->knots()};
    const Eigen::Index order{m_basis->order()};

    Eigen::Index knotIdx{};

    // case 1: copy coefficients
    while (knotInsert >= knots(knotIdx + order - 1)) {
      coeffsNew(knotIdx) = coeffs(knotIdx);
      ++knotIdx;
    }

    // case 2: interpolate coefficients
    while (knots(knotIdx) < knotInsert &&
           knotInsert < knots(knotIdx + order - 1)) {
      double weight{(knotInsert - knots(knotIdx)) /
                    (knots(knotIdx + order - 1) - knots(knotIdx))};
      coeffsNew(knotIdx) =
          (1 - weight) * coeffs(knotIdx - 1) + weight * coeffs(knotIdx);
      ++knotIdx;
    }

    // case 3: shift coefficients
    for (; knotIdx < coeffsNew.size(); ++knotIdx)
      coeffsNew(knotIdx) = coeffs(knotIdx - 1);

    return coeffsNew;
  }

  /**
   * @brief Iterative algorithm from [1] to estimate the root between two
   sign-changing coefficients corresponding to "rootIdx".
   *
   * In each iteration, a knot is inserted at the current root candidate. The
   current root candidate is determined as the center between the current left
   and right greville sites.
   * The algorithm terminates after a maximum number of iterations or if a root
   is found with the desired accuracy.
   *
   * [1] K. Mørken and M. Reimers, “An unconditionally convergent method for
   computing zeros of splines and polynomials,” Math. Comp., vol. 76, no. 258,
   pp. 845–865, Jan. 2007, doi: 10.1090/S0025-5718-07-01923-0.
   *
   * @param rootIdx Index corresponding to a pair of coefficients containing a
   root.
   * @param dim Output dimension to evaluate.
   * @param maxIter Maximum number of iterations.
   * @param accAbs Tolerance for the spline output to be considered zero.
   * @return double Root estimates along the given output dimension.
   */
  double estimateRoot(int rootIdx, int dim, int maxIter, double accAbs) const {
    Spline inserted{*this};
    Eigen::VectorXd coeffs{inserted.getCoefficients()(Eigen::all, dim)};
    double rootGuess{inserted.basis()->greville(rootIdx)};

    for (int iter = 0; iter < maxIter && !isRoot(rootGuess, dim, accAbs);
         iter++) {
      double leftCoeff{coeffs(rootIdx)};
      double leftGrev{inserted.basis()->greville(rootIdx)};
      bool leftAlmostZero{abs(leftCoeff) <= accAbs};

      // left greville is root
      if (leftAlmostZero)
        rootGuess = leftGrev;

      // guess root between left and right greville
      else {
        double rightCoeff{coeffs(rootIdx + 1)};
        double rightGrev{inserted.basis()->greville(rootIdx + 1)};
        bool rightAlmostZero{abs(rightCoeff) <= accAbs};

        double param{-leftCoeff / (rightCoeff - leftCoeff)};
        rootGuess = (1 - param) * leftGrev + param * rightGrev;

        inserted = inserted.insertKnots(Eigen::ArrayXd{{rootGuess}});
        coeffs = inserted.getCoefficients()(Eigen::all, dim);

        // after insertion, check if zero moved between new greville and former
        // right greville
        if (coeffs(rootIdx + 1) * coeffs(rootIdx + 2) <= 0)
          rootIdx += 1;
      }
    }

    return rootGuess;
  };

  /**
   * @brief Get indices of coefficient pairs that contain a root and get trivial
   * roots.
   *
   * Each negative index corresponds to a trivial root and must not be
   * processed further. Positive indices correspond to a non-trivial root. The
   * root must be estimated.
   *
   * @param dim Dimension of coefficients to evaluate.
   * @param absTol Absolute tolerance defining coefficients close to zero.
   * @return std::pair<Eigen::ArrayXi, Eigen::ArrayXd> Indices of coefficient
   * pairs containing a root and trivial roots.
   */
  std::pair<Eigen::ArrayXi, Eigen::ArrayXd> getRootIdcs(int dim,
                                                        double absTol) const {
    const Eigen::VectorXd coeffs{m_coefficients(Eigen::all, dim)};
    const int maxRoots = coeffs.size();

    Eigen::ArrayXi rootIdcs = Eigen::ArrayXi::Constant(maxRoots, -1);
    Eigen::ArrayXd roots(maxRoots);
    int cntRoots{};

    for (Eigen::Index cntCoeff{}; cntCoeff < maxRoots; ++cntCoeff) {
      double coeff{coeffs(cntCoeff)};
      bool coeffAlmostZero{abs(coeff) <= absTol};

      // root candidate at coefficient
      if (coeffAlmostZero) {
        double greville = m_basis->greville(cntCoeff);

        if (isRoot(greville, dim, absTol))
          roots(cntRoots++) = greville;
      }

      // root between coefficients
      else if (cntCoeff < maxRoots - 1) {
        double nextCoeff{coeffs(cntCoeff + 1)};
        bool nextNotZero{abs(nextCoeff) > absTol};
        bool containsRoot{coeff * nextCoeff <= 0};

        if (nextNotZero && containsRoot)
          rootIdcs(cntRoots++) = cntCoeff;
      }
    }

    return {rootIdcs.head(cntRoots), roots.head(cntRoots)};
  }

  /**
   * @brief Determines if a "value" is considered a root for the given absolute
   * tolerance "absTol".
   *
   * @param value Test value to check for root.
   * @param dim Output dimension to check against.
   * @param absTol Absolute tolerance for considering the "value" a root.
   * @return true The "value" is a root.
   * @return false The "value" is not a root.
   */
  bool isRoot(double value, int dim, double absTol) const {
    return abs((*this)({{value}})(0, dim)) <= absTol;
  }
};
}; // namespace BasisSplines

#endif
