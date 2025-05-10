#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class SplineTest : public TestBase {
protected:
  void SetUp() {

  }

  const Eigen::ArrayXd m_knotsO2{{0.0, 0.0, 0.5, 1.0, 1.0}};
  std::shared_ptr<Basis> m_basisO2{std::make_shared<Basis>(m_knotsO2, 2)};

  const Eigen::ArrayXd m_knotsO3{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}};
  std::shared_ptr<Basis> m_basisO3{std::make_shared<Basis>(m_knotsO3, 3)};

  const Eigen::ArrayXd m_points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};
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
  // instantiate spline of order 3
  const Eigen::ArrayXd coeffs{Eigen::ArrayXd::Random(m_basisO3->dim())};
  const Spline spline{m_basisO3, coeffs};

  // get gt first derivative with finite differences
  const double step{1e-8};
  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0 - step)};
  const Eigen::ArrayXd valuesGtr{(spline(points + step) - spline(points)) /
                                 step};

  // get estimate from spline derivative
  const Eigen::ArrayXd valuesEst{spline.derivative()(points)};

  expectAllClose(valuesGtr, valuesEst, 1e-6);
}

/**
 * @brief Test integral spline of order 2.
 *
 */
TEST_F(SplineTest, SplineIntO2) {
  // instantiate a spline of order 2
  const int order{2};
  const Eigen::ArrayXd knots{{0.0, 0.0, 1.0, 1.0}};
  std::shared_ptr<Basis> basis{std::make_shared<Basis>(knots, order)};
  const Eigen::ArrayXd coeffs{{0, 1}};
  const Spline spline{basis, coeffs};

  // spline integral ground truth
  Eigen::ArrayXd valuesGtr {spline(m_points).pow(2) / 2};
  const Eigen::ArrayXd valuesEst{spline.integral()(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-6);
}

/**
 * @brief Test integral spline of order 3.
 *
 */
TEST_F(SplineTest, SplineIntO3) {
  // instantiate spline of order 3
  const Eigen::ArrayXd coeffs{Eigen::ArrayXd::Random(m_basisO3->dim())};
  const Spline spline{m_basisO3, coeffs};

  // get ground truth with numeric approximation
  const double step{m_points(m_points.size() - 1) / (m_points.size() - 1)};
  Eigen::ArrayXd valuesGtr(m_points.size());
  const Eigen::ArrayXd splineValues{spline(m_points)};
  std::partial_sum(splineValues.begin(), splineValues.end(), valuesGtr.begin());
  valuesGtr *= step;

  // get estimate from spline integral
  const Eigen::ArrayXd valuesEst{spline.integral()(m_points)};

  expectAllClose(valuesGtr, valuesEst, 1e-2);
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

}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  std::srand(0);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
