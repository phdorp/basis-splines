#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"
#include "basisSplines/spline.h"
#include "basisTest.h"

namespace BasisSplines {
namespace Internal {
class SplineTest : public BasisTest {

protected:
  const Eigen::ArrayXd m_knotsO2{{0.0, 0.0, 0.5, 1.0, 1.0}};
  std::shared_ptr<Basis> m_basisO2{std::make_shared<Basis>(m_knotsO2, 2)};

  const Spline m_splineO3Seg3{m_basisO3Seg3,
                              Eigen::VectorXd::Random(m_basisO3Seg3->dim())};
};

/**
 * @brief Test piecewise linear spline function.
 *
 */
TEST_F(SplineTest, SplineEvalO1) {
  // setup spline function of linear segments
  const Eigen::ArrayXd coeffs{{0.0, 1.0, 0.25}};
  const Spline spline{m_basisO2, coeffs};

  // evaluate spline functions
  const Eigen::ArrayXd points{{0.0, 0.25, 0.5, 1.0}};
  const Eigen::ArrayXd valuesEst{spline(points)};

  // ground truth assumes picewise linear function between coefficients
  const Eigen::ArrayXd valuesGtr{{0.0, 0.5, 1.0, 0.25}};

  expectAllClose(valuesEst, valuesGtr, 1e-6);
}

/**
 * @brief Test summing two splines of order 3.
 *
 */
TEST_F(SplineTest, SplineSumO3) {
  // instatiate left operand spline of order 3
  const Eigen::ArrayXd coeffsL{Eigen::ArrayXd::Random(m_basisO3->dim())};
  const Spline splineL{m_basisO3, coeffsL};

  // instantiate right operand spline of order 3
  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  std::shared_ptr<Basis> basisR{std::make_shared<Basis>(knotsR, 3)};
  const Eigen::ArrayXd coeffsR{Eigen::ArrayXd::Random(basisR->dim())};
  const Spline splineR{basisR, coeffsR};

  // get gt from spline sum
  const Eigen::ArrayXd valuesGtr{splineL(m_points) + splineR(m_points)};

  // get estimate from sum spline
  const Spline spline{splineL.add(splineR)};
  const Eigen::ArrayXd valuesEst{spline(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-10);
}

/**
 * @brief Test multiplying two splines of order 3.
 *
 */
TEST_F(SplineTest, SplineProdO3) {
  // instatiate left operand spline of order 3
  const Eigen::ArrayXd coeffsL{Eigen::ArrayXd::Random(m_basisO3->dim())};
  const Spline splineL{m_basisO3, coeffsL};

  // instantiate right operand spline of order 3
  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  std::shared_ptr<Basis> basisR{std::make_shared<Basis>(knotsR, 3)};
  const Eigen::ArrayXd coeffsR{Eigen::ArrayXd::Random(basisR->dim())};
  const Spline splineR{basisR, coeffsR};

  // get gt from spline product
  const Eigen::ArrayXd valuesGtr{splineL(m_points) * splineR(m_points)};

  // get estimate from product spline
  const Spline spline{splineL.prod(splineR)};
  const Eigen::ArrayXd valuesEst{spline(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-10);
}

/**
 * @brief Test multiplying splines of order 4 and 3.
 *
 */
TEST_F(SplineTest, SplineProdO3O4) {
  // instatiate left operand spline of order 3
  const Eigen::ArrayXd coeffsL{Eigen::ArrayXd::Random(m_basisO3->dim())};
  const Spline splineL{m_basisO3, coeffsL};

  // instantiate right operand spline of order 4
  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  std::shared_ptr<Basis> basisR{std::make_shared<Basis>(knotsR, 4)};
  const Eigen::ArrayXd coeffsR{Eigen::ArrayXd::Random(basisR->dim())};
  const Spline splineR{basisR, coeffsR};

  // get gt from spline product
  const Eigen::ArrayXd valuesGtr{splineL(m_points) * splineR(m_points)};

  // get estimate from product spline
  const Spline spline{splineL.prod(splineR)};
  const Eigen::ArrayXd valuesEst{spline(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-10);
}

/**
 * @brief Test derivative spline of order 3.
 *
 */
TEST_F(SplineTest, SplineDerivO3) {
  const Eigen::ArrayXd valuesGtr{polyO3Der(m_points)};
  const Eigen::ArrayXd valuesEst{m_splineO3.derivative()(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test second order derivative spline of order 3.
 *
 */
TEST_F(SplineTest, SplineDderivO3) {
  const Eigen::ArrayXd valuesGtr{polyO3Dder(m_points)};
  const Eigen::ArrayXd valuesEst{m_splineO3.derivative(2)(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test integral spline of order 3.
 *
 */
TEST_F(SplineTest, SplineIntO3) {
  const Eigen::ArrayXd valuesGtr{polyO3Int(m_points)};
  const Eigen::ArrayXd valuesEst{m_splineO3.integral()(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test second order integral spline of order 3.
 *
 */
TEST_F(SplineTest, SplineIintO3) {
  const Eigen::ArrayXd valuesGtr{polyO3Iint(m_points)};
  const Eigen::ArrayXd valuesEst{m_splineO3.integral(2)(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

TEST_F(SplineTest, InsertKnots) {
  // instantiate spline of order 3
  const Eigen::ArrayXd coeffs{Eigen::ArrayXd::Random(m_basisO3->dim())};
  const Spline spline{m_basisO3, coeffs};

  // insert knots
  const Eigen::ArrayXd knotsInsert{{0.4, 0.5, 0.6}};
  const Spline splineInsert{spline.insertKnots(knotsInsert)};

  // get ground truth from initial spline
  const Eigen::ArrayXd valuesGtr{spline(m_points)};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{splineInsert(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-6);
}

/**
 * @brief Test retrieving first 2 segments from spline function of order 3.
 *
 */
TEST_F(SplineTest, GetSegment01O3) {
  const Spline splineSeg{m_splineO3Seg3.getSegment(0, 1)};

  const auto breakpoints{splineSeg.basis()->getBreakpoints()};

  const Eigen::ArrayXd pointsSubset{
      getPointsSubset(breakpoints.first(0), breakpoints.first(2))};

  const Eigen::ArrayXd valuesEst{splineSeg(pointsSubset)};
  const Eigen::ArrayXd valuesGtr{m_splineO3Seg3(pointsSubset)};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}

/**
 * @brief Test retrieving last 2 segments from spline of order 3.
 *
 */
TEST_F(SplineTest, GetSegment12O3) {
  const Spline splineSeg{m_splineO3Seg3.getSegment(1, 2)};

  const auto breakpoints{splineSeg.basis()->getBreakpoints()};

  const Eigen::ArrayXd pointsSubset{
      getPointsSubset(breakpoints.first(1), breakpoints.first(3))};

  const Eigen::ArrayXd valuesEst{splineSeg(pointsSubset)};
  const Eigen::ArrayXd valuesGtr{m_splineO3Seg3(pointsSubset)};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}

}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  std::srand(0);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
