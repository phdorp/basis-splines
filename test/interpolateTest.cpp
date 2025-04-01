#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"
#include "basisSplines/spline.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class InterpolateTest : public TestBase {
protected:
  void SetUp() {}
};

/**
 * @brief Test interpolation of a piecewise linear spline function.
 * In the special case of linear segments the transformation matrix is a unit
 * matrix.
 *
 */
TEST_F(InterpolateTest, InterpolateSplineOrder2) {
  // setup basis of order 2 with continuity 0, 1, 0
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.5, 1.0, 1.0}};
  const int order{2};
  std::shared_ptr<Basis> basis{std::make_shared<Basis>(knots, order)};

  const Eigen::ArrayXd coeffsGtr{{0.0, 1.0, 0.25}};
  const Spline spline{basis, coeffsGtr};

  const Interpolate interp{basis};

  const Eigen::ArrayXd points{basis->greville()};
  const Eigen::ArrayXd coeffsEst{interp.fit(spline(points), points)};

  expectAllClose(coeffsGtr, coeffsEst, 1e-6);
}

/**
 * @brief Test interpolation of a piecewise quadratic spline function.
 *
 */
TEST_F(InterpolateTest, InterpolateSplineOrder3) {
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.0, 0.5, 0.5, 0.75, 1.0, 1.0}};
  const int order{3};
  std::shared_ptr<Basis> basis{std::make_shared<Basis>(knots, order)};

  const Eigen::ArrayXd coeffsGtr{Eigen::ArrayXd::Random(knots.size() - order)};
  const Spline spline{basis, coeffsGtr};

  const Interpolate interp{basis};

  const Eigen::ArrayXd points{basis->greville()};
  const Eigen::ArrayXd coeffsEst{interp.fit(spline(points), points)};

  expectAllClose(coeffsGtr, coeffsEst, 1e-6);
}
}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  std::srand(0);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
