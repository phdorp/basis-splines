#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"
#include "cases/basisTestBase.h"

namespace BasisSplines {
namespace Internal {
/**
 * @brief Test summing two splines of order 3.
 *
 */
TEST_F(BasisTest, SumMatO3) {
  // instatiate left operand spline of order 3
  const Eigen::VectorXd coeffsL{Eigen::VectorXd::Random(m_basisO3->dim())};

  // instantiate right operand spline of order 3
  const Eigen::ArrayXd knotsR{{0.0, 0.0, 0.0, 0.25, 0.5, 0.8, 1.0, 1.0}};
  const Basis basisR{knotsR, 3};
  const Eigen::VectorXd coeffsR{Eigen::VectorXd::Random(basisR.dim())};

  // get gt from basis evaluations
  const Eigen::ArrayXd valuesGtr{(*m_basisO3)(m_points)*coeffsL +
                                 basisR(m_points) * coeffsR};

  // determine sum transformations
  Basis basisEst{};
  const auto [transformL, transformR] = m_basisO3->add(basisR, basisEst);

  // get estimate by applying sum transformations
  const Eigen::ArrayXd valuesEst{basisEst(m_points) *
                                 (transformL * coeffsL + transformR * coeffsR)};

  // test if evaluations are alomst equal
  expectAllClose(valuesGtr, valuesEst, 1e-10);

  // ground truth basis
  const Basis basisGtr{
      m_basisO3->combine(basisR, std::max(m_basisO3->order(), basisR.order()))};

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