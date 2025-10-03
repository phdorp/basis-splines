#ifndef INTERPOLATE_TEST_H
#define INTERPOLATE_TEST_H

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"
#include "basisSplines/spline.h"

#include "testBase.h"

namespace BasisSplines {
namespace Internal {

class InterpolateTest : public TestBase {
protected:
  void SetUp() {}

  const Eigen::ArrayXd m_knotsO2{{0.0, 0.0, 0.5, 1.0, 1.0}};
  std::shared_ptr<Basis> m_basisO2{std::make_shared<Basis>(m_knotsO2, 2)};
  const Spline m_splineO2{m_basisO2, Eigen::VectorXd{{0.0, 1.0, 0.25}}};
  const Interpolate m_interpO2{m_basisO2};

  const Eigen::ArrayXd m_knotsO3{{0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}};
  std::shared_ptr<Basis> m_basisO3{std::make_shared<Basis>(m_knotsO3, 3)};
  const Spline m_splineO3{
      m_basisO3,
      Eigen::MatrixXd::Random(m_knotsO3.size() - m_basisO3->order(), 2)};
  const Interpolate m_interpO3{m_basisO3};
};
} // namespace Internal
} // namespace BasisSplines

#endif