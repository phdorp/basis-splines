#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"
#include "basisSplines/math.h"
#include "basisSplines/spline.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class BasisTest : public TestBase {
protected:
  static Eigen::ArrayXd polyO3(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{points.pow(2)};
  }

  static Eigen::ArrayXd polyO3Der(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{2 * points};
  }

  static Eigen::ArrayXd polyO3Dder(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd::Zero(points.size()) + 2;
  }

  static Eigen::ArrayXd polyO3Int(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{points.pow(3) / 3};
  }

  static Eigen::ArrayXd polyO3Iint(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{points.pow(4) / 12};
  }

  // create basis of order 3
  const Eigen::ArrayXd m_knotsO3{
      {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}; /**<< knots of order 3 basis */
  std::shared_ptr<Basis> m_basisO3{
      std::make_shared<Basis>(m_knotsO3, 3)}; /**<< order 3 basis */

  // create order 3 interatpolation
  const Interpolate m_interpolateO3{
      m_basisO3}; /**<< interpolation for order 3 basis */

  // create order 3 spline interpolating order 3 polynomial
  const Spline m_splineO3{
      m_basisO3, m_interpolateO3.fit(&polyO3)}; /**<< spline of order 3 */

  // create order 3 derivative spline
  std::shared_ptr<Basis> m_basisO3Der{std::make_shared<Basis>(
      m_basisO3->orderDecrease())}; /**<< order 3 derivative basis */
  const Interpolate m_interpolateO3Der{
      m_basisO3Der}; /**<< order 3 derivative interpolation */
  Spline m_splineO3Der{
      m_basisO3Der,
      m_interpolateO3Der.fit(&polyO3Der)}; /** order 3 derivative spline */

  // create order 3 second derivative spline
  std::shared_ptr<Basis> m_basisO3Dder{std::make_shared<Basis>(
      m_basisO3Der->orderDecrease())}; /**<< order 3 second derivative basis */
  const Interpolate m_interpolateO3Dder{
      m_basisO3Dder}; /**<< order 3 second derivative interpolation */
  Spline m_splineO3Dder{
      m_basisO3Dder, m_interpolateO3Dder.fit(
                         &polyO3Dder)}; /** order 3 second derivative spline */

  // create order 3 integral spline
  std::shared_ptr<Basis> m_basisO3Int{std::make_shared<Basis>(
      m_basisO3->orderIncrease())}; /**<< order 3 integral basis */
  const Interpolate m_interpolateO3Int{
      m_basisO3Int}; /**<< order 3 integral interpolation */
  Spline m_splineO3Int{
      m_basisO3Int,
      m_interpolateO3Int.fit(&polyO3Int)}; /** order 3 integral spline */

  // create order 3 second integral spline
  std::shared_ptr<Basis> m_basisO3Iint{std::make_shared<Basis>(
      m_basisO3Int->orderIncrease())}; /**<< order 3 integral basis */
  const Interpolate m_interpolateO3Iint{
      m_basisO3Iint}; /**<< order 3 integral interpolation */
  Spline m_splineO3Iint{
      m_basisO3Iint,
      m_interpolateO3Iint.fit(&polyO3Iint)}; /** order 3 integral spline */

  const Eigen::ArrayXd m_points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};
};

/**
 * @brief Test the determintation of greville sites for basis functions of
 * order 3.
 *
 */
TEST_F(BasisTest, GrevilleO3) {
  const Eigen::ArrayXd valuesEst{m_basisO3->greville()};
  const Eigen::ArrayXd valuesGtr{{0.0, 0.25, 0.75, 1.0}};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}

/**
 * @brief Test the determination of breakpoints for basis functions of order 3.
 *
 */
TEST_F(BasisTest, BreakpointsO3) {
  const std::pair<Eigen::ArrayXd, Eigen::ArrayXi> valuesEst{
      m_basisO3->getBreakpoints()};
  const std::pair<Eigen::ArrayXd, Eigen::ArrayXi> valuesGtr{{{0.0, 0.5, 1.0}},
                                                            {{0, 2, 0}}};

  expectAllClose(valuesEst.first, valuesGtr.first, 1e-10);
  expectAllClose(valuesEst.second, valuesGtr.second, 1e-10);
}

/**
 * @brief Combine a basis of order 3 with a basis of order 2.
 *
 */
TEST_F(BasisTest, CombineO3O2) {
  const Eigen::ArrayXd knotsO2{{0.0, 0.0, 0.2, 0.5, 0.6, 1.0, 1.0}};
  const Basis basisO2{knotsO2, 2};

  const Basis estimate{m_basisO3->combine(basisO2, m_basisO3->order())};

  const Basis groundTruth{
      {{0.0, 0.0, 0.0, 0.2, 0.2, 0.5, 0.5, 0.6, 0.6, 1.0, 1.0, 1.0}},
      m_basisO3->order()};

  expectAllClose(estimate.knots(), groundTruth.knots(), 1e-6);
  EXPECT_EQ(estimate.order(), groundTruth.order());
}

/**
 * @brief Test conversion from breakpoints to knots for spline of order 3.
 *
 */
TEST_F(BasisTest, ToKnotsO3) {
  const auto [bps, conts] = m_basisO3->getBreakpoints();

  const Eigen::ArrayXd valuesEst{
      Basis::toKnots(bps, conts, m_basisO3->order())};
  const Eigen::ArrayXd valuesGtr{m_knotsO3};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}

/**
 * @brief Test generation of derivative transformation matrix.
 *
 */
TEST_F(BasisTest, DerivMatO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Der.coefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXd valuesEst{m_basisO3->derivative(basisEst, 1) *
                                 m_splineO3.coefficients().matrix()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Der.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of second derivative transformation matrix.
 *
 */
TEST_F(BasisTest, DderivMatO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Dder.coefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXd valuesEst{m_basisO3->derivative(basisEst, 2) *
                                 m_splineO3.coefficients().matrix()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Dder.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of integral transformation matrix.
 *
 */
TEST_F(BasisTest, IntMatO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXd valuesGtr{m_splineO3Int.coefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXd valuesEst{m_basisO3->integral(basisEst, 1) *
                                 m_splineO3.coefficients().matrix()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Int.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of second integral transformation matrix.
 *
 */
TEST_F(BasisTest, IintMatO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXd valuesGtr{m_splineO3Iint.coefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXd valuesEst{m_basisO3->integral(basisEst, 2) *
                                 m_splineO3.coefficients().matrix()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Iint.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test summing two splines of order 3.
 *
 */
TEST_F(BasisTest, SumMatO3) {
  // instatiate left operand spline of order 3
  const Eigen::VectorXd coeffsL{Eigen::VectorXd::Random(m_basisO3->dim())};

  // instantiate right operand spline of order 3
  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  const Basis basisR{knotsR, 3};
  const Eigen::VectorXd coeffsR{Eigen::VectorXd::Random(basisR.dim())};

  // get gt from basis evaluations
  const Eigen::ArrayXd valuesGtr{(*m_basisO3)(m_points)*coeffsL +
                                 basisR(m_points) * coeffsR};

  // determine sum transformations
  Basis basisEst{};
  const auto [transformL, transformR] = m_basisO3->add(basisR, basisEst);

  // get estimate by applying sum transformations
  const Eigen::ArrayXd valuesEst{basisEst(m_points) *
                                 (transformL * coeffsL + transformR * coeffsR)};

  // test if evaluations are alomst equal
  expectAllClose(valuesGtr, valuesEst, 1e-10);

  // ground truth basis
  const Basis basisGtr{
      m_basisO3->combine(basisR, std::max(m_basisO3->order(), basisR.order()))};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test product two splines of order 3.
 *
 */
TEST_F(BasisTest, ProdMatO3) {
  // instatiate left operand spline of order 3
  const Eigen::VectorXd coeffsL{Eigen::VectorXd::Random(m_basisO3->dim())};

  // instantiate right operand spline of order 3
  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  const Basis basisR{knotsR, 3};
  const Eigen::VectorXd coeffsR{Eigen::VectorXd::Random(basisR.dim())};

  // get gt from basis evaluations
  const Eigen::ArrayXd valuesGtr{((*m_basisO3)(m_points)*coeffsL).array() *
                                 (basisR(m_points) * coeffsR).array()};

  // determine product transformations
  Basis basisEst{};
  const Eigen::MatrixXd transform{m_basisO3->prod(basisR, basisEst)};

  // get estimate by applying product transformations
  const Eigen::ArrayXd coeffsProd{transform *
                                  kron(coeffsL.matrix(), coeffsR.matrix())};
  const Eigen::ArrayXd valuesEst{basisEst(m_points).matrix() *
                                 coeffsProd.matrix()};

  // test if evaluations are alomst equal
  expectAllClose(valuesGtr, valuesEst, 1e-10);

  // ground truth basis
  const Basis basisGtr{
      m_basisO3->combine(basisR, m_basisO3->order() + basisR.order())};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
