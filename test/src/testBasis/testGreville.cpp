#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/basis.h"
#include "basisSplines/math.h"

#include "cases/basisTest.h"

namespace BasisSplines {
namespace Internal {

/**
 * @brief Test the determination of Greville sites for basis functions of order 4.
 *
 * This test verifies that the Greville method correctly computes knot averages
 * for a B-spline basis of order 4. The Greville sites are computed as the average
 * of (order-1) consecutive knots.
 */
TEST_F(BasisTest, GrevilleBasisO4) {
  // Create knots for order 4 basis: [0, 0, 0, 0, 0.3, 0.7, 1, 1, 1, 1]
  // This gives us a basis with 6 basis functions (10 knots - 4 order = 6 dim)
  const Eigen::ArrayXd knotsO4{{0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0}};
  const Basis basisO4{knotsO4, 4};

  const Eigen::ArrayXd grevilleEst = basisO4.greville();

  const Eigen::ArrayXd grevilleGtr{{0.0, 0.1, 1.0/3.0, 2.0/3.0, 0.9, 1.0}};

  // Verify dimensions match
  EXPECT_EQ(grevilleEst.size(), basisO4.dim());
  EXPECT_EQ(grevilleEst.size(), grevilleGtr.size());

  // Verify Greville sites are computed correctly
  expectAllClose(grevilleEst, grevilleGtr, 1e-10);
}

/**
 * @brief Test the determination of individual Greville sites for basis functions of order 4.
 *
 * This test verifies that the single-index Greville method correctly computes
 * individual knot averages for specific basis function indices.
 */
TEST_F(BasisTest, GrevilleBasisO4Individual) {
  // Create the same order 4 basis as above
  const Eigen::ArrayXd knotsO4{{0.0, 0.0, 0.0, 0.0, 0.3, 0.7, 1.0, 1.0, 1.0, 1.0}};
  const Basis basisO4{knotsO4, 4};

  // Test individual Greville sites
  EXPECT_NEAR(basisO4.greville(0), 0.0, 1e-10);           // (0+0+0)/3
  EXPECT_NEAR(basisO4.greville(1), 0.1, 1e-10);           // (0+0+0.3)/3
  EXPECT_NEAR(basisO4.greville(2), 1.0/3.0, 1e-10);       // (0+0.3+0.7)/3
  EXPECT_NEAR(basisO4.greville(3), 2.0/3.0, 1e-10);       // (0.3+0.7+1)/3
  EXPECT_NEAR(basisO4.greville(4), 0.9, 1e-10);           // (0.7+1+1)/3
  EXPECT_NEAR(basisO4.greville(5), 1.0, 1e-10);           // (1+1+1)/3
}

} // namespace Internal
} // namespace BasisSplines