#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Test generation of derivative transformation matrix.
 *
 */
TEST_P(DerivativeBasisTest, MatrixTransformation) {
  const Eigen::ArrayXXd valuesGtr{m_splineDer.getCoefficients()};

  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      m_basis.derivative(basisEst, m_derivativeOrder) *
      m_spline.getCoefficients()};

  // test if coefficients are almost equal
  EXPECT_TRUE(valuesGtr.isApprox(valuesEst, accAbsNumerical))
      << "Coefficient values do not match.";

  // test if knots are almost equal
  EXPECT_TRUE(m_basisDer.knots().isApprox(basisEst.knots(), accAbsNumerical))
      << "Knot values do not match.";

  // test if order is equal
  EXPECT_EQ(m_basisDer.order(), basisEst.order())
      << "Basis orders do not match.";
}

/**
 * @brief Test generation of derivative value transformation.
 *
 */
TEST_P(DerivativeBasisTest, DirectTransformation) {
  const Eigen::ArrayXXd valuesGtr{m_splineDer.getCoefficients()};

  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basis.derivative(
      basisEst, m_spline.getCoefficients(), m_derivativeOrder)};

  // test if coefficients are almost equal
  EXPECT_TRUE(valuesGtr.isApprox(valuesEst, accAbsNumerical))
      << "Coefficient values do not match.";

  // test if knots are almost equal
  EXPECT_TRUE(m_basisDer.knots().isApprox(basisEst.knots(), accAbsNumerical))
      << "Knot values do not match.";

  // test if order is equal
  EXPECT_EQ(m_basisDer.order(), basisEst.order())
      << "Basis orders do not match.";
}

// Name generator for DerivativeBasisTest parameters
std::string DerivativeBasisTestNameGenerator(
    const testing::TestParamInfo<std::tuple<int, double>> &info) {
  return "DerivOrder" + std::to_string(std::get<0>(info.param)) +
         "_Scale" + std::to_string(static_cast<int>(std::get<1>(info.param)));
}

INSTANTIATE_TEST_SUITE_P(DerivativeOrder, DerivativeBasisTest,
                         testing::Combine(testing::Range(1, 3),
                                          testing::Range(1.0, 3.0)),
                         DerivativeBasisTestNameGenerator);

} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}