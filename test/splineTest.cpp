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

}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
