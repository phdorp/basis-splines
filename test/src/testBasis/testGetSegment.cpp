#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Test retrieving first 2 segments from basis functions of order 3.
 * Determine clamped basis fro segment basis and test for correct knots and
 * order.
 *
 */
TEST_F(BasisTest, GetSegment01O3) {
  // retrieve segment basis
  const Basis basisSeg{m_basisO3Seg3->getSegment(0, 1)};

  // test knots and order of segment basis
  expectAllClose(basisSeg.knots(),
                 Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.4, 0.6, 0.6, 1.0}}, 1e-10);
  EXPECT_EQ(basisSeg.order(), m_basisO3Seg3->order());

  // test evaluation of segment basis
  const Eigen::ArrayXXd valuesEst{basisSeg(basisSeg.greville())};
  const Eigen::ArrayXXd valuesGtr{(*m_basisO3Seg3)(basisSeg.greville())(
      Eigen::all, Eigen::seqN(0, basisSeg.knots().size() - basisSeg.order()))};

  expectAllClose(valuesEst, valuesGtr, 1e-10);

  // determine clamped basis from segment basis
  const Basis basisClamped{basisSeg.getClamped()};

  // test breakpoints
  const auto [bps, conts] = basisClamped.getBreakpoints();
  expectAllClose(bps, Eigen::ArrayXd{{0.0, 0.4, 0.6}}, 1e-10);
  expectAllClose(conts, Eigen::ArrayXi{{0, 2, 0}}, 1e-10);

  // test order
  EXPECT_EQ(basisClamped.order(), basisSeg.order());
}

/**
 * @brief Test retrieving last 2 segments from basis functions of order 3.
 * Determine clamped basis fro segment basis and test for correct knots and
 * order.
 *
 */
TEST_F(BasisTest, GetSegment12O3) {
  // retrieve segment basis
  const Basis basisSeg{m_basisO3Seg3->getSegment(1, 2)};

  // test knots and order of segment basis
  expectAllClose(basisSeg.knots(),
                 Eigen::ArrayXd{{0.0, 0.0, 0.4, 0.6, 0.6, 1.0, 1.0, 1.0}},
                 1e-10);
  EXPECT_EQ(basisSeg.order(), m_basisO3Seg3->order());

  // test evaluation of segment basis
  expectAllClose(
      Eigen::ArrayXXd{(*m_basisO3Seg3)(
          m_points)(Eigen::all, Eigen::seqN(1, basisSeg.knots().size() -
                                                   basisSeg.order()))},
      Eigen::ArrayXXd{basisSeg(m_points)}, 1e-10);

  // determine clamped basis from segment basis
  const Basis basisClamped{basisSeg.getClamped()};

  // test breakpoints
  const auto [bps, conts] = basisClamped.getBreakpoints();
  expectAllClose(bps, Eigen::ArrayXd{{0.4, 0.6, 1.0}}, 1e-10);
  expectAllClose(conts, Eigen::ArrayXi{{0, 1, 0}}, 1e-10);

  // test order
  EXPECT_EQ(basisClamped.order(), basisSeg.order());
}
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}