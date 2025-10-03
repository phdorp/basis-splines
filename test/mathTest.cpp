#include <Eigen/Core>
#include <gtest/gtest.h>

#include "basisSplines/math.h"

#include "pairedMatrixTest.h"

namespace BasisSplines {
namespace Internal {

TEST_P(IdenticalPairedMatrixTest, KhatriRao) {
  const Eigen::MatrixXd result{khatriRao(m_matL, m_matR)};

  EXPECT_EQ(result.rows(), m_matL.rows());
  EXPECT_EQ(result.rows(), m_matR.rows());

  const auto m_matCols{m_matL.cols() * m_matR.cols()};
  EXPECT_EQ(result.cols(), m_matCols);

  const double acc {1e-10};
  expectAllClose(result, khatriRaoEye(m_matL, m_matR), acc);
}

TEST_P(IdenticalPairedMatrixTest, Kron) {
  const Eigen::MatrixXd result{kron(m_matL, m_matR)};

  const auto rows{m_matL.rows() * m_matR.rows()};
  EXPECT_EQ(result.rows(), rows);

  const auto cols{m_matL.cols() * m_matR.cols()};
  EXPECT_EQ(result.cols(), cols);

  const double acc {1e-10};
  expectAllClose(result, kronEye(m_matL, m_matR), acc);
}

INSTANTIATE_TEST_SUITE_P(EyeMatrices, IdenticalPairedMatrixTest,
                         testing::Values(Eigen::MatrixXd::Identity(1, 1),
                                         Eigen::MatrixXd::Identity(2, 2),
                                         Eigen::MatrixXd::Identity(3, 3)));

}; // namespace Internal
}; // namespace BasisSplines