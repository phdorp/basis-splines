#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class SplineTest : public TestBase {
protected:
  void SetUp() {}
};

/**
 * @brief Test piecewise linear spline function.
 *
 */
TEST_F(SplineTest, SplineEvalOrder1) {
  // setup basis of order 2 with continuity 0, 1, 0
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.5, 1.0, 1.0}};
  const int order{2};
  std::shared_ptr<Basis> basis{std::make_shared<Basis>(knots, order)};

  // setup spline function of linear segments
  const Eigen::ArrayXd coeffs{{0.0, 1.0, 0.25}};
  const Spline spline{basis, coeffs};

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
TEST_F(SplineTest, SplineSumOrder3) {
  const int order{3};
  const Eigen::ArrayXd knotsL{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0}};
  std::shared_ptr<Basis> basisL{std::make_shared<Basis>(knotsL, order)};
  const Eigen::ArrayXd coeffsL{Eigen::ArrayXd::Random(knotsL.size() - order)};
  const Spline splineL{basisL, coeffsL};

  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  std::shared_ptr<Basis> basisR{std::make_shared<Basis>(knotsR, order)};
  const Eigen::ArrayXd coeffsR{Eigen::ArrayXd::Random(knotsR.size() - order)};
  const Spline splineR{basisR, coeffsR};

  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};
  const Eigen::ArrayXd valuesGtr{splineL(points) + splineR(points)};

  const Spline spline{splineL.add(splineR)};
  const Eigen::ArrayXd valuesEst{spline(points)};

  expectAllClose(valuesGtr, valuesEst, 1e-10);
}

/**
 * @brief Test multiplying two splines of order 3.
 *
 */
TEST_F(SplineTest, SplineProdOrder3) {
  const int order{3};
  const Eigen::ArrayXd knotsL{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0}};
  std::shared_ptr<Basis> basisL{std::make_shared<Basis>(knotsL, order)};
  const Eigen::ArrayXd coeffsL{Eigen::ArrayXd::Random(knotsL.size() - order)};
  const Spline splineL{basisL, coeffsL};

  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  std::shared_ptr<Basis> basisR{std::make_shared<Basis>(knotsR, order)};
  const Eigen::ArrayXd coeffsR{Eigen::ArrayXd::Random(knotsR.size() - order)};
  const Spline splineR{basisR, coeffsR};

  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};
  const Eigen::ArrayXd valuesGtr{splineL(points) * splineR(points)};

  const Spline spline{splineL.prod(splineR)};
  const Eigen::ArrayXd valuesEst{spline(points)};

  expectAllClose(valuesGtr, valuesEst, 1e-10);
}

/**
 * @brief Test multiplying splines of order 4 and 3.
 *
 */
TEST_F(SplineTest, SplineProdOrderL3R4) {
  const int orderL{3};
  const Eigen::ArrayXd knotsL{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0}};
  std::shared_ptr<Basis> basisL{std::make_shared<Basis>(knotsL, orderL)};
  const Eigen::ArrayXd coeffsL{Eigen::ArrayXd::Random(knotsL.size() - orderL)};
  const Spline splineL{basisL, coeffsL};

  const int orderR{4};
  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  std::shared_ptr<Basis> basisR{std::make_shared<Basis>(knotsR, orderR)};
  const Eigen::ArrayXd coeffsR{Eigen::ArrayXd::Random(knotsR.size() - orderR)};
  const Spline splineR{basisR, coeffsR};

  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};
  const Eigen::ArrayXd valuesGtr{splineL(points) * splineR(points)};

  const Spline spline{splineL.prod(splineR)};
  const Eigen::ArrayXd valuesEst{spline(points)};

  expectAllClose(valuesGtr, valuesEst, 1e-10);
}

/**
 * @brief Test derivative spline of order 3.
 *
 */
TEST_F(SplineTest, SplineDerivOrder3) {
  const int order{3};
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}};
  std::shared_ptr<Basis> basis{std::make_shared<Basis>(knots, order)};
  const Eigen::ArrayXd coeffs{Eigen::ArrayXd::Random(knots.size() - order)};
  const Spline spline{basis, coeffs};

  const double step{1e-8};
  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0 - step)};
  const Eigen::ArrayXd valuesGtr{(spline(points + step) - spline(points)) /
                                 step};

  const Eigen::ArrayXd valuesEst{spline.derivative()(points)};

  expectAllClose(valuesGtr, valuesEst, 1e-6);
}

/**
 * @brief Test integral spline of order 2.
 *
 */
TEST_F(SplineTest, SplineIntOrder2) {
  const int order{2};
  const Eigen::ArrayXd knots{{0.0, 0.0, 1.0, 1.0}};
  std::shared_ptr<Basis> basis{std::make_shared<Basis>(knots, order)};
  const Eigen::ArrayXd coeffs{{0, 1}};
  const Spline spline{basis, coeffs};

  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};
  Eigen::ArrayXd valuesGtr {spline(points).pow(2) / 2};
  const Eigen::ArrayXd valuesEst{spline.integral()(points)};

  expectAllClose(valuesGtr, valuesEst, 1e-6);
}

/**
 * @brief Test integral spline of order 3.
 *
 */
TEST_F(SplineTest, SplineIntOrder3) {
  const int order{3};
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}};
  std::shared_ptr<Basis> basis{std::make_shared<Basis>(knots, order)};
  const Eigen::ArrayXd coeffs{Eigen::ArrayXd::Random(knots.size() - order)};
  const Spline spline{basis, coeffs};

  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};
  const double step{points(points.size() - 1) / (points.size() - 1)};
  Eigen::ArrayXd valuesGtr(points.size());
  const Eigen::ArrayXd splineValues{spline(points)};
  std::partial_sum(splineValues.begin(), splineValues.end(), valuesGtr.begin());
  valuesGtr *= step;
  const Eigen::ArrayXd valuesEst{spline.integral()(points)};

  expectAllClose(valuesGtr, valuesEst, 1e-2);
}

}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  std::srand(0);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
