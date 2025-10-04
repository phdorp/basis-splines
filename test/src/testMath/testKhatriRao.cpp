#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/math.h"

#include "cases/pairedMatrixTest.h"

namespace BasisSplines {
namespace Internal {

TEST_P(PairedMatrixTest, KhatriRao) {
  const Eigen::MatrixXd result{khatriRao(m_matL, m_matR)};

  EXPECT_EQ(result.rows(), m_matL.rows());
  EXPECT_EQ(result.rows(), m_matR.rows());

  const auto matCols{m_matL.cols() * m_matR.cols()};
  EXPECT_EQ(result.cols(), matCols);

  expectAllClose(result, khatriRaoEye(m_matL, m_matR), 1e-10);
}

INSTANTIATE_TEST_SUITE_P(
    EyeMatrices, PairedMatrixTest,
    testing::Values(std::pair{Eigen::MatrixXd::Identity(1, 1),
                              Eigen::MatrixXd::Identity(1, 1)},
                    std::pair{Eigen::MatrixXd::Identity(2, 2),
                              Eigen::MatrixXd::Identity(2, 2)}));

// Edge case: non-square matrices
TEST(MathTest, KhatriRaoNonSquare) {
  Eigen::MatrixXd m_matL(2, 3);
  m_matL << 1, 2, 3, 4, 5, 6;
  Eigen::MatrixXd m_matR(2, 2);
  m_matR << 7, 8, 9, 10;
  Eigen::MatrixXd result = khatriRao(m_matL, m_matR);
  EXPECT_EQ(result.rows(), 2);
  EXPECT_EQ(result.cols(), 6);
}

// Edge case: empty matrices
TEST(MathTest, KhatriRaoEmptyMatrix) {
  Eigen::MatrixXd m_matL(0, 2);
  Eigen::MatrixXd m_matR(0, 2);
  Eigen::MatrixXd result = khatriRao(m_matL, m_matR);
  EXPECT_EQ(result.rows(), 0);
}

// Numerical accuracy: known values
TEST(MathTest, KhatriRaoKnownValues) {
  Eigen::MatrixXd m_matL(2, 2);
  m_matL << 1, 2, 3, 4;
  Eigen::MatrixXd m_matR(2, 2);
  m_matR << 5, 6, 7, 8;
  Eigen::MatrixXd expected(2, 4);
  expected << 5, 6, 10, 12, 21, 24, 28, 32;
  Eigen::MatrixXd result = khatriRao(m_matL, m_matR);
  EXPECT_TRUE(result.isApprox(expected, 1e-12));
}
}; // namespace Internal
}; // namespace BasisSplines