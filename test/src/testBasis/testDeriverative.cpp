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
TEST_F(BasisTest, DerivMatO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXXd valuesGtr{m_splineO3Der.getCoefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basisO3->derivative(basisEst, 1) *
                                  m_splineO3.getCoefficients().matrix()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Der.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of derivative transformation matrix with scaled basis.
 *
 */
TEST_F(BasisTest, DerivMatO3Scaled) {
  // scale basis with m_scalingFactor
  Basis basisO3Scale2{*m_basisO3};
  basisO3Scale2.setScale(m_scalingFactor);

  // get derivative of scaled basis
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basisO3->derivative(basisEst, 1) *
                                  m_splineO3.getCoefficients().matrix()};

  // scale breakpoints with 2
  Basis basisO3Bps2{*m_basisO3};
  const Eigen::ArrayXd breakpoints{basisO3Bps2.getBreakpoints().first};
  basisO3Bps2.setBreakpoints(
      breakpoints * m_scalingFactor,
      Eigen::ArrayXi::LinSpaced(breakpoints.size(), 0, breakpoints.size()));

  // get derivative of breakpoint scaled basis
  Basis basisGtr{};
  const Eigen::ArrayXXd valuesGtr{m_basisO3->derivative(basisGtr, 1) *
                                  m_splineO3.getCoefficients().matrix()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test generation of second derivative transformation matrix.
 *
 */
TEST_F(BasisTest, DderivMatO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXXd valuesGtr{m_splineO3Dder.getCoefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{m_basisO3->derivative(basisEst, 2) *
                                  m_splineO3.getCoefficients().matrix()};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Dder.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of derivative value transformation.
 *
 */
TEST_F(BasisTest, DerivTransformO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXXd valuesGtr{m_splineO3Der.getCoefficients()};

  // get estimate from result spline
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      m_basisO3->derivative(basisEst, m_splineO3.getCoefficients(), 1)};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);

  // ground truth basis
  Basis basisGtr{*m_basisO3Der.get()};

  // test if knots are almost equal
  expectAllClose(basisGtr.knots(), basisEst.knots(), 1e-8);
  // test if order is equal
  EXPECT_EQ(basisGtr.order(), basisEst.order());
}

/**
 * @brief Test generation of second derivative value transformation with scaled
 * basis.
 *
 */
TEST_F(BasisTest, DerivTransformO3Scaled) {
  // scale basis with m_scalingFactor
  Basis basisO3Scale2{*m_basisO3};
  basisO3Scale2.setScale(m_scalingFactor);

  // get derivative of scaled basis
  Basis basisEst{};
  const Eigen::ArrayXXd valuesEst{
      basisO3Scale2.derivative(basisEst, m_splineO3.getCoefficients(), 1)};

  // scale breakpoints with m_scalingFactor
  Basis basisO3Bps2{*m_basisO3};
  const Eigen::ArrayXd breakpoints{basisO3Bps2.getBreakpoints().first};
  basisO3Bps2.setBreakpoints(
      breakpoints * m_scalingFactor,
      Eigen::ArrayXi::LinSpaced(breakpoints.size(), 0, breakpoints.size()));

  // get derivative of breakpoint scaled basis
  Basis basisGtr{};
  const Eigen::ArrayXXd valuesGtr{
      basisO3Bps2.derivative(basisEst, m_splineO3.getCoefficients(), 1)};

  // test if coefficients are almost equal
  expectAllClose(valuesGtr, valuesEst, 1e-8);
}
} // namespace Internal
} // namespace BasisSplines

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}