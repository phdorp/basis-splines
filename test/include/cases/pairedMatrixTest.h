#ifndef PAIRED_MATRIX_H
#define PAIRED_MATRIX_H

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "testBase.h"

namespace BasisSplines {
namespace Internal {

class MatrixTestBase : public TestBase {
protected:
  static Eigen::MatrixXd khatriRaoEye(const Eigen::MatrixXd &matL,
                                      const Eigen::MatrixXd &matR) {
    assert(matL.rows() == matR.rows());

    const auto rows{matL.rows()};
    const auto matCols{matL.cols() * matR.cols()};
    Eigen::MatrixXd result(rows, matCols);

    for (int cntRow{}; cntRow < rows; ++cntRow) {
      Eigen::MatrixXd sub{Eigen::MatrixXd::Zero(rows, rows)};
      sub(cntRow, cntRow) = 1.0;

      const auto cols{Eigen::seqN(cntRow * rows, rows)};
      result(Eigen::all, cols) = sub;
    }

    return result;
  }

  static Eigen::MatrixXd kronEye(const Eigen::MatrixXd &matL,
                                 const Eigen::MatrixXd &matR) {
    return Eigen::MatrixXd::Identity(matL.rows() * matR.rows(),
                                     matL.cols() * matR.cols());
  }
};

class PairedMatrixTest : public MatrixTestBase,
                         public testing::WithParamInterface<
                             std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> {
protected:
  void setUp() {
    const auto [matL, matR] = GetParam();
    const Eigen::MatrixXd m_matL = matL;
    const Eigen::MatrixXd m_matR = matR;
  }

  const Eigen::MatrixXd m_matL{};
  const Eigen::MatrixXd m_matR{};
};

class IdenticalPairedMatrixTest
    : public MatrixTestBase,
      public testing::WithParamInterface<Eigen::MatrixXd> {
protected:
  const Eigen::MatrixXd m_matL{GetParam()};
  const Eigen::MatrixXd m_matR{GetParam()};
};

} // namespace Internal
} // namespace BasisSplines

#endif