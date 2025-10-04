#ifndef SPLINE_TEST_H
#define SPLINE_TEST_H

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"
#include "basisSplines/spline.h"
#include "basisTest.h"

namespace BasisSplines {
namespace Internal {
class SplineTest : public BasisTest {
protected:
// MARK: protected properties
  const Eigen::ArrayXd m_knotsO2{{0.0, 0.0, 0.5, 1.0, 1.0}};

  std::shared_ptr<Basis> m_basisO2{std::make_shared<Basis>(m_knotsO2, 2)};

  const Spline m_splineO3Seg3{m_basisO3Seg3,
                              Eigen::MatrixXd::Random(m_basisO3Seg3->dim(), 2)};
};
} // namespace Internal
} // namespace BasisSplines

#endif