#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/math.h"

#include "cases/pairedMatrixTest.h"

namespace BasisSplines {
namespace Internal {

TEST_P(PairedMatrixTest, Kron) {
  const Eigen::MatrixXd result{kron(m_matL, m_matR)};

  const auto rows{m_matL.rows() * m_matR.rows()};
  EXPECT_EQ(result.rows(), rows);

  const auto cols{m_matL.cols() * m_matR.cols()};
  EXPECT_EQ(result.cols(), cols);

  expectAllClose(result, kronEye(m_matL, m_matR), 1e-10);
}

INSTANTIATE_TEST_SUITE_P(
    EyeMatrices, PairedMatrixTest,
    testing::Values(std::pair{Eigen::MatrixXd::Identity(1, 1),
                              Eigen::MatrixXd::Identity(1, 1)},
                    std::pair{Eigen::MatrixXd::Identity(2, 2),
                              Eigen::MatrixXd::Identity(2, 2)}));

TEST(MathTest, KronNonSquare) {
  Eigen::MatrixXd m_matL(2, 3);
  m_matL << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd m_matR(3, 2);
  m_matR << 7, 8, 9, 10, 11, 12;
  Eigen::MatrixXd result = kron(m_matL, m_matR);
  EXPECT_EQ(result.rows(), 6);
  EXPECT_EQ(result.cols(), 6);
}

TEST(MathTest, KronEmptyMatrix) {
  Eigen::MatrixXd m_matL(0, 2);
  Eigen::MatrixXd m_matR(2, 2);
  Eigen::MatrixXd result = kron(m_matL, m_matR);
  EXPECT_EQ(result.rows(), 0);
}

TEST(MathTest, KronKnownValues) {
  Eigen::MatrixXd m_matL(2, 2);
  m_matL << 1, 2, 3, 4;
  Eigen::MatrixXd m_matR(2, 2);
  m_matR << 0, 1, 2, 3;
  Eigen::MatrixXd expected(4, 4);
  expected << 0, 1, 0, 2, 2, 3, 4, 6, 0, 3, 0, 4, 6, 9, 8, 12;
  Eigen::MatrixXd result = kron(m_matL, m_matR);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}
} // namespace Internal
} // namespace BasisSplines