#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {

// Type aliases to avoid template syntax in TEST_P macros
using DerivativeTestType = DerivativeTest<std::tuple<int, double, int>>;
using IntegralTestType = IntegralTest<std::tuple<int, double, int>>;
using IntegralTestCustomType = IntegralTest<std::tuple<int, Basis>>;

/**
 * @brief Test generation of derivative transformation matrix.
 *
 */
TEST_P(DerivativeTestType, MatrixTransformation) {
  const Eigen::ArrayXXd valuesGtr{m_splineResult.getCoefficients()};

  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      m_basis.derivative(basisEst, m_operationOrder) *
      m_spline.getCoefficients()};

  // test if coefficients are almost equal
  EXPECT_TRUE(valuesGtr.isApprox(valuesEst, accAbsNumerical))
      << "Coefficient values do not match.";

  // test if knots are almost equal
  EXPECT_TRUE(m_basisResult.knots().isApprox(basisEst.knots(), accAbsNumerical))
      << "Knot values do not match.";

  // test if order is equal
  EXPECT_EQ(m_basisResult.order(), basisEst.order())
      << "Basis orders do not match.";
}

/**
 * @brief Test generation of derivative value transformation.
 *
 */
TEST_P(DerivativeTestType, DirectTransformation) {
  const Eigen::ArrayXXd valuesGtr{m_splineResult.getCoefficients()};

  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basis.derivative(
      basisEst, m_spline.getCoefficients(), m_operationOrder)};

  // test if coefficients are almost equal
  EXPECT_TRUE(valuesGtr.isApprox(valuesEst, accAbsNumerical))
      << "Coefficient values do not match.";

  // test if knots are almost equal
  EXPECT_TRUE(m_basisResult.knots().isApprox(basisEst.knots(), accAbsNumerical))
      << "Knot values do not match.";

  // test if order is equal
  EXPECT_EQ(m_basisResult.order(), basisEst.order())
      << "Basis orders do not match.";
}

INSTANTIATE_TEST_SUITE_P(DerivativeOrder, DerivativeTestType,
                         testing::Combine(testing::Range(0, 3),
                                          testing::Range(1.0, 3.0),
                                          testing::Range(1, 3)),
                         DerivativeTestType::TestNameGenerator);

/**
 * @brief Test generation of integral transformation matrix.
 *
 */
TEST_P(IntegralTestType, MatrixTransformation) {
  const Eigen::ArrayXXd valuesGtr{m_splineResult.getCoefficients()};

  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basis.integral(basisEst, m_operationOrder) *
                                  m_spline.getCoefficients()};

  // test if coefficients are almost equal
  EXPECT_TRUE(valuesGtr.isApprox(valuesEst, accAbsNumerical))
      << "Coefficient values do not match.";

  // test if knots are almost equal
  EXPECT_TRUE(m_basisResult.knots().isApprox(basisEst.knots(), accAbsNumerical))
      << "Knot values do not match.";

  // test if order is equal
  EXPECT_EQ(m_basisResult.order(), basisEst.order())
      << "Basis orders do not match.";
}

/**
 * @brief Test generation of integral value transformation.
 *
 */
TEST_P(IntegralTestType, DirectTransformation) {
  const Eigen::ArrayXXd valuesGtr{m_splineResult.getCoefficients()};

  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      m_basis.integral(basisEst, m_spline.getCoefficients(), m_operationOrder)};

  // test if coefficients are almost equal
  EXPECT_TRUE(valuesGtr.isApprox(valuesEst, accAbsNumerical))
      << "Coefficient values do not match.";

  // test if knots are almost equal
  EXPECT_TRUE(m_basisResult.knots().isApprox(basisEst.knots(), accAbsNumerical))
      << "Knot values do not match.";

  // test if order is equal
  EXPECT_EQ(m_basisResult.order(), basisEst.order())
      << "Basis orders do not match.";
}

INSTANTIATE_TEST_SUITE_P(IntegralOrder, IntegralTestType,
                         testing::Combine(testing::Range(0, 3),
                                          testing::Range(1.0, 3.0),
                                          testing::Range(1, 3)),
                         IntegralTestType::TestNameGenerator);
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}