#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class BasisTest : public TestBase {
protected:
  void SetUp() {}
};

/**
 * @brief Test the evaluation of basis functions of order 1.
 *
 */
TEST_F(BasisTest, BasisEvalOrder1) {
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}};
  const int order{1};
  const Eigen::ArrayXd points{{0.0, 0.75}};

  const Basis basis{knots, order};
  const Eigen::ArrayXXd valuesEst{basis(points)};

  const Eigen::ArrayXXd valuesGtr{{1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
                                  {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}

/**
 * @brief Test the determination of breakpoints for basis functions of order 3.
 *
 */
TEST_F(BasisTest, BreakpointsOrder2) {
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.5, 1.0, 1.0}};
  const int order{2};

  const Basis basis{knots, order};
  const std::pair<Eigen::ArrayXd, Eigen::ArrayXi> valuesEst{
      basis.breakpoints()};
  const std::pair<Eigen::ArrayXd, Eigen::ArrayXi> valuesGtr{{{0.0, 0.5, 1.0}},
                                                            {{0, 1, 0}}};

  expectAllClose(valuesEst.first, valuesGtr.first, 1e-10);
  expectAllClose(valuesEst.second, valuesGtr.second, 1e-10);
}

/**
 * @brief Combine two spline bases of order 2.
 *
 */
TEST_F(BasisTest, CombineOrder2) {
  const int order{2};
  const Eigen::ArrayXd knotsA{{0.0, 0.0, 0.2, 0.2, 0.5, 1.0, 1.0}};
  const Basis basisA{knotsA, order};
  const Eigen::ArrayXd knotsB {{0.0, 0.0, 0.5, 0.6, 1.0, 1.0}};
  const Basis basisB{knotsB, order};

  const Basis estimate {basisA.combine(basisB, order)};

  const Basis groundTruth { {{0.0, 0.0, 0.2, 0.2, 0.5, 0.6, 1.0, 1.0}}, 2};

  expectAllClose(estimate.knots(), groundTruth.knots(), 1e-6);
  EXPECT_EQ(estimate.order(), groundTruth.order());
}

/**
 * @brief Test conversion from breakpoints to knots for spline of order 2.
 *
 */
TEST_F(BasisTest, ToKnotsOrder2)
{
  const int order {2};
  const Eigen::ArrayXd bps {{0.0, 0.25, 0.5, 1.0}};
  const Eigen::ArrayXi conts {{0, 1, 0, 1}};

  const Eigen::ArrayXd valuesEst {Basis::toKnots(bps, conts, order)};
  const Eigen::ArrayXd valuesGtr {{0.0, 0.0, 0.25, 0.5, 0.5, 1.0}};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}
}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
