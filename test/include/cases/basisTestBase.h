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

using namespace std::placeholders;

class BasisTestBase : public FunctionTest {
protected:
  Basis m_basis{};
};

class UnitBasisTest : public BasisTestBase,
                      public testing::WithParamInterface<int> {
protected:
  void SetUp() override {
    const int order = GetParam();
    m_basis = Basis(Eigen::ArrayXd{{0.0, 1.0}}, Eigen::ArrayXi{{0, 0}}, order);
  }

  Eigen::ArrayXd greville() const {
    Eigen::ArrayXd sites{Eigen::ArrayXd::Ones(m_basis.dim())};
    const double siteDistance = 1.0 / (m_basis.dim() - 1);

    for (int cntSite{m_basis.dim() - 1}; cntSite > 0; --cntSite) {
      sites(cntSite - 1) = sites(cntSite) - siteDistance;
    }

    return sites;
  }

  std::pair<Eigen::ArrayXd, Eigen::ArrayXi> getBreakpoints() const {
    return {Eigen::ArrayXd{{m_basis.knots()(0), m_basis.knots().tail(1)(0)}},
            Eigen::ArrayXi{{0, 0}}};
  }
};

class OperationBasisTest
    : public BasisTestBase,
      public testing::WithParamInterface<std::tuple<int, double, int>> {
public:
  static std::string TestNameGenerator(
      const testing::TestParamInfo<std::tuple<int, double, int>> &info) {
    return "OperationOrder" + std::to_string(std::get<0>(info.param)) +
           "_Scale" +
           std::to_string(static_cast<int>(std::get<1>(info.param))) +
           "_Dimension" + std::to_string(std::get<2>(info.param));
  }

protected:
  void SetUp() override {
    m_scale = std::get<1>(GetParam());
    const Eigen::ArrayXd knots{
        Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}};
    const int order{3};
    m_basis = Basis(knots, order, m_scale);

    auto basis = std::make_shared<Basis>(m_basis);
    m_spline = Spline(
        basis, Interpolate(basis).fit(std::bind(&polynomial, _1, m_dimension)));

    m_operationOrder = std::get<0>(GetParam());
  }

  static Eigen::MatrixXd polynomial(const Eigen::ArrayXd &points,
                                    const int dimension = 1) {
    // polynomial of degree 2
    Eigen::MatrixXd values(Eigen::MatrixXd::Zero(points.size(), dimension));
    values << points.pow(2).matrix().replicate(1, dimension);
    return values;
  }

  Spline m_spline{};
  Basis m_basisResult{};
  Spline m_splineResult{};

  int m_operationOrder{};
  double m_scale{};
  int m_dimension{};
};

class DerivativeBasisTest : public OperationBasisTest {
protected:
  void SetUp() override {
    OperationBasisTest::SetUp();

    m_basisResult = m_basis.orderDecrease(m_operationOrder);
    auto basisResult = std::make_shared<Basis>(m_basisResult);
    m_splineResult = Spline(
        basisResult, Interpolate(basisResult)
                         .fit(std::bind(&polynomialDer, _1, m_operationOrder,
                                        m_scale, m_dimension)));
  }

private:
  static Eigen::MatrixXd polynomialDer(const Eigen::ArrayXd &points,
                                       int derivativeOrder = 1,
                                       double scale = 1.0,
                                       const int dimension = 1) {
    if (derivativeOrder == 2) {
      // second derivative of polynomial of degree 2
      return Eigen::MatrixXd::Constant(points.size(), dimension, 2.0) /
             std::pow(scale, 2);
    } else if (derivativeOrder == 1) {
      // derivative of polynomial of degree 2
      Eigen::MatrixXd values(points.size(), dimension);
      values << 2 * points.matrix().replicate(1, dimension) / scale;
      return values;
    } else if (derivativeOrder == 0) {
      return polynomial(points, dimension);
    } else {
      throw std::invalid_argument(
          "Only derivative orders 0, 1 and 2 are supported.");
    }
  }
};

class IntegralBasisTest : public OperationBasisTest {
protected:
  void SetUp() override {
    OperationBasisTest::SetUp();

    m_basisResult = m_basis.orderIncrease(m_operationOrder);
    auto basisResult = std::make_shared<Basis>(m_basisResult);
    m_splineResult = Spline(
        basisResult, Interpolate(basisResult)
                         .fit(std::bind(&polynomialInt, _1, m_operationOrder,
                                        m_scale, m_dimension)));
  }

private:
  static Eigen::MatrixXd polynomialInt(const Eigen::ArrayXd &points,
                                       int integralOrder = 1,
                                       double scale = 1.0,
                                       const int dimension = 1) {
    if (integralOrder == 2) {
      // second order integral of polynomial of degree 2
      Eigen::MatrixXd values(points.size(), dimension);
      values << points.pow(3).matrix().replicate(1, dimension) * scale / 12.0;
      return values;
    } else if (integralOrder == 1) {
      // first order integral of polynomial of degree 2
      Eigen::MatrixXd values(points.size(), dimension);
      values << points.pow(3).matrix().replicate(1, dimension) * scale / 3.0;
      return values;
    } else if (integralOrder == 0) {
      return polynomial(points, dimension);
    } else {
      throw std::invalid_argument(
          "Only derivative orders 0, 1 and 2 are supported.");
    }
  }
};

}; // namespace Internal
}; // namespace BasisSplines

#endif // BASIS_TEST_BASE_H