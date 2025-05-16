#include <Eigen/Core>
#include <gtest/gtest.h>
#include <iostream>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"
#include "basisSplines/spline.h"
#include "basisSplines/transform.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class TransformTest : public TestBase {

protected:
  static Eigen::ArrayXd polyO3(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{points.pow(2)};
  }

  static Eigen::ArrayXd polyO3Der(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{2 * points};
  }

  static Eigen::ArrayXd polyO3Dder(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd::Zero(points.size()) + 2;
  }

  static Eigen::ArrayXd polyO3Int(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{points.pow(3) / 3};
  }

  static Eigen::ArrayXd polyO3Iint(const Eigen::ArrayXd &points) {
    return Eigen::ArrayXd{points.pow(4) / 12};
  }

  void SetUp() {}

  // create basis of order 3
  const Eigen::ArrayXd m_knotsO3{
      {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}; /**<< knots of order 3 basis */
  std::shared_ptr<Basis> m_basisO3{
      std::make_shared<Basis>(m_knotsO3, 3)}; /**<< order 3 basis */

  // create order 3 interatpolation and transformation
  const Interpolate m_interpolateO3{
      m_basisO3}; /**<< interpolation for order 3 basis */
  const Transform m_transformO3{
      m_basisO3}; /**<< transfomration of order 3 spline coefficients */

  // create order 3 spline interpolating order 3 polynomial
  const Spline m_splineO3{
      m_basisO3, m_interpolateO3.fit(&polyO3)}; /**<< spline of order 3 */

  // create order 3 derivative spline
  std::shared_ptr<Basis> m_basisO3Der{std::make_shared<Basis>(
      m_basisO3->orderDecrease())}; /**<< order 3 derivative basis */
  const Interpolate m_interpolateO3Der{
      m_basisO3Der}; /**<< order 3 derivative interpolation */
  Spline m_splineO3Der{
      m_basisO3Der,
      m_interpolateO3Der.fit(&polyO3Der)}; /** order 3 derivative spline */

  // create order 3 second derivative spline
  std::shared_ptr<Basis> m_basisO3Dder{std::make_shared<Basis>(
      m_basisO3Der->orderDecrease())}; /**<< order 3 second derivative basis */
  const Interpolate m_interpolateO3Dder{
      m_basisO3Dder}; /**<< order 3 second derivative interpolation */
  Spline m_splineO3Dder{
      m_basisO3Dder, m_interpolateO3Dder.fit(
                         &polyO3Dder)}; /** order 3 second derivative spline */

  // create order 3 integral spline
  std::shared_ptr<Basis> m_basisO3Int{std::make_shared<Basis>(
      m_basisO3->orderIncrease())}; /**<< order 3 integral basis */
  const Interpolate m_interpolateO3Int{
      m_basisO3Int}; /**<< order 3 integral interpolation */
  Spline m_splineO3Int{
      m_basisO3Int,
      m_interpolateO3Int.fit(&polyO3Int)}; /** order 3 integral spline */

  // create order 3 second integral spline
  std::shared_ptr<Basis> m_basisO3Iint{std::make_shared<Basis>(
      m_basisO3Int->orderIncrease())}; /**<< order 3 integral basis */
  const Interpolate m_interpolateO3Iint{
      m_basisO3Iint}; /**<< order 3 integral interpolation */
  Spline m_splineO3Iint{
      m_basisO3Iint,
      m_interpolateO3Iint.fit(&polyO3Iint)}; /** order 3 integral spline */
};

/**
 * @brief Test generation of derivative transformation matrix.
 *
 */
TEST_F(TransformTest, DerivMatO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Der.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{m_transformO3.derivative() *
                                 m_splineO3.coefficients().matrix()};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test transformation of spline coefficient to derivative.
 *
 */
TEST_F(TransformTest, DerivCoeffO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Der.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{
      m_transformO3.derivative(m_splineO3.coefficients())};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test generation of second derivative transformation matrix.
 *
 */
TEST_F(TransformTest, DderivMatO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Dder.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{m_transformO3.derivative(2) *
                                 m_splineO3.coefficients().matrix()};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test transformation of spline coefficient to second derivative.
 *
 */
TEST_F(TransformTest, DderivCoeffO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Dder.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{
      m_transformO3.derivative(m_splineO3.coefficients(), 2)};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test generation of integral transformation matrix.
 *
 */
TEST_F(TransformTest, IntMatO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXd valuesGtr{m_splineO3Int.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{m_transformO3.integral() *
                                 m_splineO3.coefficients().matrix()};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test transformation of spline coefficient to integral.
 *
 */
TEST_F(TransformTest, IntCoeffO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Int.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{
      m_transformO3.integral(m_splineO3.coefficients())};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test generation of second integral transformation matrix.
 *
 */
TEST_F(TransformTest, IintMatO3) {
  // ground truth from spline fit to integral
  const Eigen::ArrayXd valuesGtr{m_splineO3Iint.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{m_transformO3.integral(2) *
                                 m_splineO3.coefficients().matrix()};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

/**
 * @brief Test transformation of spline coefficient to second integral.
 *
 */
TEST_F(TransformTest, IintCoeffO3) {
  // ground truth from spline fit to derivative
  const Eigen::ArrayXd valuesGtr{m_splineO3Iint.coefficients()};

  // get estimate from result spline
  const Eigen::ArrayXd valuesEst{
      m_transformO3.integral(m_splineO3.coefficients(), 2)};

  expectAllClose(valuesGtr, valuesEst, 1e-8);
}

}; // namespace Internal
}; // namespace BasisSplines

int main(int argc, char **argv) {
  std::srand(0);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
