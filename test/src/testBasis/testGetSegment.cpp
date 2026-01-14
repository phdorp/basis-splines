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

TEST_P(SegmentTest, GetSegment) {
  auto [begin, end] = m_segment;

  // retrieve segment basis
  const Basis basisSegment{m_basis.getSegment(begin, end)};

  const auto [breakpoints, contiunities] = basisSegment.getBreakpoints();

  const auto active = Eigen::seqN(
      std::accumulate(contiunities.begin(), contiunities.begin() + begin, 0), basisSegment.dim());
  EXPECT_TRUE(m_basis(m_points)(Eigen::all, active)
                  .isApprox(basisSegment(m_points), accAbsNumerical))
      << "Segment basis evaluation does not match expected values.";
}

INSTANTIATE_TEST_SUITE_P(
    SegmentTest, SegmentTest,
    testing::Combine(testing::Values(std::pair<int, int>{0, 0},
                                     std::pair<int, int>{1, 1},
                                     std::pair<int, int>{1, 2},
                                     std::pair<int, int>{0, 2},
                                     std::pair<int, int>{0, 1}),
                     testing::Values(2, 3),
                     testing::Values(Eigen::ArrayXd{{0.0, 0.5, 0.6, 1.0}}),
                     testing::Values(Eigen::ArrayXi{{0, 1, 1, 0}})));

} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}