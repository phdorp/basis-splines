#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/math.h"
#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class MathTest : public TestBase {
protected:
  const Eigen::ArrayXXd arr32{{1, 2}, {3, 4}, {5, 6}}; /**<< (3 x 2) matrix */
  const Eigen::ArrayXXd arr33{
      {1, 2, 3}, {4, 5, 6}, {7, 8, 9}}; /**<< (3 x 3) matrix */
};

/**
 * @brief Test Khatri-Rao product (3 x 2) and (3 x 3) matrix.
 *
 */
TEST_F(MathTest, khatriRao3x6) {
  const Eigen::ArrayXXd valuesGtr{
      {1, 2, 3, 2, 4, 6}, {12, 15, 18, 16, 20, 24}, {35, 40, 45, 42, 48, 54}};

  const Eigen::ArrayXXd valuesEst{khatriRao(arr32, arr33)};

  expectAllClose(valuesEst, valuesGtr, 1e-10);
}
}; // namespace Internal
}; // namespace BasisSplines