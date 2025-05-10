#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class BasisTest : public TestBase {
protected:
  void SetUp() {}

  const Eigen::ArrayXd m_knotsO3{{0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}};
  Basis m_basisO3{m_knotsO3, 3};
};

/**
 * @brief Test the determintation of greville sites for basis functions of
 * order 3.
 *
 */
TEST_F(BasisTest, GrevilleO3) {
  const Eigen::ArrayXd valuesEst{m_basisO3.greville()};
  const Eigen::ArrayXd valuesGtr{{0.0, 0.25, 0.5, 0.75, 1.0}};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}

/**
 * @brief Test the determination of breakpoints for basis functions of order 3.
 *
 */
TEST_F(BasisTest, BreakpointsO3) {
  const std::pair<Eigen::ArrayXd, Eigen::ArrayXi> valuesEst{
      m_basisO3.breakpoints()};
  const std::pair<Eigen::ArrayXd, Eigen::ArrayXi> valuesGtr{{{0.0, 0.5, 1.0}},
                                                            {{0, 1, 0}}};

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

  const Basis estimate{m_basisO3.combine(basisO2, m_basisO3.order())};

  const Basis groundTruth{
      {{0.0, 0.0, 0.0, 0.2, 0.2, 0.5, 0.5, 0.6, 0.6, 1.0, 1.0, 1.0}},
      m_basisO3.order()};

  expectAllClose(estimate.knots(), groundTruth.knots(), 1e-6);
  EXPECT_EQ(estimate.order(), groundTruth.order());
}

/**
 * @brief Test conversion from breakpoints to knots for spline of order 3.
 *
 */
TEST_F(BasisTest, ToKnotsO3) {
  const auto [bps, conts] = m_basisO3.breakpoints();

  const Eigen::ArrayXd valuesEst{Basis::toKnots(bps, conts, m_basisO3.order())};
  const Eigen::ArrayXd valuesGtr{m_knotsO3};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}
}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
