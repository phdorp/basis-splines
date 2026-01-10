#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Test conversion from breakpoints to knots for spline of order 3.
 *
 */
TEST_F(BasisTest, ToKnotsO3) {
  const auto [bps, conts] = m_basisO3->getBreakpoints();

  const Eigen::ArrayXd valuesEst{
      Basis::toKnots(bps, conts, m_basisO3->order())};
  const Eigen::ArrayXd valuesGtr{m_basisO3->knots()};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}

/**
 * @brief Test setting breakpoints successfully.
 *
 */
TEST_F(BasisTest, SetBreakpointsPositive) {
  // TEST: successful breakpoint setting
  // determine ground truth: set first breakpoint to 0.1, second to 0.6
  auto [breakpointsGtr, contsGtr] = m_basisO3->getBreakpoints();
  const Eigen::ArrayXi breakpointIdcs{{0, 1}};
  breakpointsGtr(breakpointIdcs) = Eigen::ArrayXd{{0.1, 0.6}};

  // expect: setting valit breakpoints does not throw invalid argument error
  EXPECT_NO_THROW(m_basisO3->setBreakpoints(breakpointsGtr(breakpointIdcs),
                                            breakpointIdcs));
  // determine estimate: get current breakpoints
  auto [breakpointsEst, contsEst] = m_basisO3->getBreakpoints();

  // expect: breakpoint estimates and ground truth almost equal
  expectAllClose(breakpointsGtr, breakpointsEst, 1e-8);
}

/**
 * @brief Test setting breakpoints resulting in a non-increasing order.
 *
 */
TEST_F(BasisTest, SetBreakpointsNegativeNonIncreasing) {
  // determine ground truth
  auto [breakpointsGtr, contsGtr] = m_basisO3->getBreakpoints();

  // expect: setting invalid breakpoints throws invalid arugment error
  EXPECT_THROW(m_basisO3->setBreakpoints({{0.1, 0.0}}, {{0, 1}}),
               std::invalid_argument);
  // determine estimate: get current breakpoints
  auto [breakpointsEst, contsEst] = m_basisO3->getBreakpoints();

  // expect: setting invalid breakpionts does not change basis
  expectAllClose(breakpointsGtr, breakpointsEst, 1e-8);
}
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}