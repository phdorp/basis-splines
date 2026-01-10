#ifndef BASIS_TEST_BASE_H
#define BASIS_TEST_BASE_H

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"
#include "basisSplines/math.h"
#include "basisSplines/spline.h"

#include "functionTest.h"

namespace BasisSplines {
namespace Internal {
class BasisTestBase : public FunctionTest {};

class UnitBasisTest : public BasisTestBase,
                      public testing::WithParamInterface<int> {
protected:
  void SetUp() override {
    const int order = GetParam();
    m_basis = Basis(Eigen::ArrayXd{{0.0, 1.0}},
                        Eigen::ArrayXi{{0, 0}}, order);
  }

  Eigen::ArrayXd greville() const {
    Eigen::ArrayXd sites{Eigen::ArrayXd::Zero(m_basis.dim())};
    const double siteDistance = 1.0 / (m_basis.dim() - 1);

    for (int cntSite{1}; cntSite < m_basis.dim(); ++cntSite) {
      sites(cntSite) = sites(cntSite - 1) + siteDistance;
    }

    return sites;
  }

  std::pair<Eigen::ArrayXd, Eigen::ArrayXi> getBreakpoints() const {
    return {Eigen::ArrayXd{{m_basis.knots()(0), m_basis.knots().tail(1)(0)}},
            Eigen::ArrayXi{{0, 0}}};
  }

  Basis m_basis{};
};
}; // namespace Internal
}; // namespace BasisSplines

#endif // BASIS_TEST_BASE_H