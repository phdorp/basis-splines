#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Test product two splines of order 3.
 *
 */
TEST_P(BinaryOperationTest, ProductTransformation) {
  // get gt from basis evaluations
  const Eigen::ArrayXd valuesGtr{m_spline(m_points) * m_splineOther(m_points)};

  // determine product transformations
  Basis basisEst{};
  const Eigen::MatrixXd transform{m_basis.prod(m_basisOther, basisEst)};

  // get estimate by applying product transformations
  const Eigen::ArrayXd valuesEst{
      basisEst(m_points) * (transform * kron(m_spline.getCoefficients(),
                                             m_splineOther.getCoefficients()))};

  // test if evaluations are almost equal
  Eigen::Index maxIndex;
  EXPECT_TRUE(valuesEst.isApprox(valuesGtr, accAbsNumerical))
      << "Function values do not match. Max error: "
      << (valuesEst - valuesGtr).abs().maxCoeff(&maxIndex) << " at index "
      << maxIndex << '.';
}

/**
 * @brief Test summing two splines of order 3.
 *
 */
TEST_P(BinaryOperationTest, SumTransformation) {
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
    BinaryBasisOperations, BinaryOperationTest,
    testing::Combine(testing::Range(1, 3), testing::Range(1, 4),
                     testing::Range(1, 4),
                     testing::Values(Eigen::ArrayXd{{0.0, 0.5, 1.0}}),
                     testing::Values(Eigen::ArrayXd{{0.0, 0.5, 1.0}},
                                     Eigen::ArrayXd{{0.0, 0.3, 0.7, 1.0}})),
    BinaryOperationTest::TestNameGenerator);
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}