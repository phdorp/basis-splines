#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {

TEST_P(UnitBasisTest, Greville) {
  expectAllClose(m_basis.greville(), greville(), accAbsNumerical);
}

TEST_P(UnitBasisTest, GetBreakpoints) {
  auto [estBreakpoints, estConts] = m_basis.getBreakpoints();
  auto [gtrBreakpoints, gtrConts] = getBreakpoints();

  EXPECT_TRUE(estBreakpoints.isApprox(gtrBreakpoints, accAbsNumerical));
  EXPECT_TRUE(estConts.cwiseEqual(gtrConts).all());
}

INSTANTIATE_TEST_SUITE_P(BasisOrder, UnitBasisTest, testing::Range(1, 6));

}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
