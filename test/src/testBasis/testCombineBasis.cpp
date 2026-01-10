#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Combine a basis of order 3 with a basis of order 2.
 *
 */
TEST_F(BasisTest, CombineO3O2) {
  const Eigen::ArrayXd knotsO2{{0.0, 0.0, 0.2, 0.5, 0.6, 1.0, 1.0}};
  const Basis basisO2{knotsO2, 2};

  const Basis estimate{m_basisO3->combine(basisO2, m_basisO3->order())};

  const Basis groundTruth{
      {{0.0, 0.0, 0.0, 0.2, 0.2, 0.5, 0.5, 0.6, 0.6, 1.0, 1.0, 1.0}},
      m_basisO3->order()};

  expectAllClose(estimate.knots(), groundTruth.knots(), 1e-6);
  EXPECT_EQ(estimate.order(), groundTruth.order());
}
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}