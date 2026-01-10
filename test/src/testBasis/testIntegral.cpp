#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Test generation of integral transformation matrix.
 *
 */
TEST_F(BasisTest, IntMatO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXXd valuesGtr{m_splineO3Int.getCoefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basisO3->integral(basisEst, 1) *
                                  m_splineO3.getCoefficients()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Int.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of integral transformation matrix with scaled basis.
 *
 */
TEST_F(BasisTest, IntMatO3Scaled) {
  // scale basis with m_scalingFactor
  Basis basisO3Scale2{*m_basisO3};
  basisO3Scale2.setScale(m_scalingFactor);

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{basisO3Scale2.integral(basisEst, 1) *
                                  m_splineO3.getCoefficients()};

  // scale breakpoints with m_scalingFactor
  Basis basisO3Bps2{*m_basisO3};
  const Eigen::ArrayXd breakpoints{basisO3Bps2.getBreakpoints().first};
  basisO3Bps2.setBreakpoints(
      breakpoints * m_scalingFactor,
      Eigen::ArrayXi::LinSpaced(breakpoints.size(), 0, breakpoints.size()));

  // get derivative of breakpoint scaled basis
  Basis basisGtr{};
  const Eigen::ArrayXXd valuesGtr{basisO3Bps2.integral(basisGtr, 1) *
                                  m_splineO3.getCoefficients()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test generation of second integral transformation matrix.
 *
 */
TEST_F(BasisTest, IintMatO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXXd valuesGtr{m_splineO3Iint.getCoefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basisO3->integral(basisEst, 2) *
                                  m_splineO3.getCoefficients()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Iint.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of integral value transformation.
 *
 */
TEST_F(BasisTest, IntTransformO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXXd valuesGtr{m_splineO3Int.getCoefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      m_basisO3->integral(basisEst, m_splineO3.getCoefficients(), 1)};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Int.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of integral value transformation with scaled basis.
 *
 */
TEST_F(BasisTest, IntTransformO3Scaled) {
  // scale basis with m_scalingFactor
  Basis basisO3Scale2{*m_basisO3};
  basisO3Scale2.setScale(m_scalingFactor);

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      basisO3Scale2.integral(basisEst, m_splineO3.getCoefficients(), 1)};

  // scale breakpoints with m_scalingFactor
  Basis basisO3Bps2{*m_basisO3};
  const Eigen::ArrayXd breakpoints{basisO3Bps2.getBreakpoints().first};
  basisO3Bps2.setBreakpoints(
      breakpoints * m_scalingFactor,
      Eigen::ArrayXi::LinSpaced(breakpoints.size(), 0, breakpoints.size()));

  // get derivative of breakpoint scaled basis
  Basis basisGtr{};
  const Eigen::ArrayXXd valuesGtr{
      basisO3Bps2.integral(basisEst, m_splineO3.getCoefficients(), 1)};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test generation of second integral value transformation.
 *
 */
TEST_F(BasisTest, IintTransformO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXXd valuesGtr{m_splineO3Iint.getCoefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      m_basisO3->integral(basisEst, m_splineO3.getCoefficients(), 2)};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Iint.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}