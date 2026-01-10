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
  void setUp() {
    const int order = GetParam();
    const Basis m_basis{Eigen::ArrayXd{{0.0, 1.0}},
                        Eigen::ArrayXi{{order, order}}, order};
  }

  Eigen::ArrayXd greville() const {
    Eigen::ArrayXd sites{Eigen::ArrayXd::Zero(m_basis.dim())};
    for (int cntSite{}; cntSite < m_basis.dim(); ++cntSite) {
      sites(cntSite) =
          0.0 ? cntSite == 1 : m_basis.knots().tail(0)(0) / cntSite;
    }
    return sites;
  }

  std::pair<Eigen::ArrayXd, Eigen::ArrayXi> getBreakpoints() const {
    return {Eigen::ArrayXd{{m_basis.knots()(0), m_basis.knots()(1)}},
            Eigen::ArrayXi{{m_basis.order(), m_basis.order()}}};
  }

  const Basis m_basis{};
};
}; // namespace Internal
}; // namespace BasisSplines

#endif // BASIS_TEST_BASE_H