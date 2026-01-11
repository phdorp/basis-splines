#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Test summing two splines of order 3.
 *
 */
TEST_P(BinaryOperationBasisTest, SumTransformation) {
  // get gt from basis evaluations
  const Eigen::ArrayXd valuesGtr{m_spline(m_points) + m_splineOther(m_points)};

  // determine sum transformations
  Basis basisEst{};
  const auto [transformL, transformR] = m_basis.add(m_basisOther, basisEst);

  // get estimate by applying sum transformations
  const Eigen::ArrayXd valuesEst{
      basisEst(m_points) * (transformL * m_spline.getCoefficients() +
                            transformR * m_splineOther.getCoefficients())};

  // test if evaluations are almost equal
  Eigen::Index maxIndex;
  EXPECT_TRUE(valuesEst.isApprox(valuesGtr, accAbsNumerical))
      << "Function values do not match. Max error: "
      << (valuesEst - valuesGtr).abs().maxCoeff(&maxIndex) << " at index "
      << maxIndex << '.';
}

INSTANTIATE_TEST_SUITE_P(
    BinaryOperations, BinaryOperationBasisTest,
    testing::Combine(testing::Range(1, 3), testing::Range(2, 4),
                     testing::Range(2, 4),
                     testing::Values(Eigen::ArrayXd{{0.0, 0.5, 1.0}}),
                     testing::Values(Eigen::ArrayXd{{0.0, 0.5, 1.0}},
                                     Eigen::ArrayXd{{0.0, 0.3, 0.7, 1.0}})),
    BinaryOperationBasisTest::TestNameGenerator);
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}