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

class DerivativeBasisTest
    : public BasisTestBase,
      public testing::WithParamInterface<std::tuple<int, double>> {
protected:
  void SetUp() override {
    m_scale = std::get<1>(GetParam());
    const Eigen::ArrayXd knots{
        Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}};
    const int order{3};
    m_basis = Basis(knots, order, m_scale);

    auto basis = std::make_shared<Basis>(m_basis);
    m_spline = Spline(basis, Interpolate(basis).fit(&polynomial));

    m_derivativeOrder = std::get<0>(GetParam());
    m_basisDer = m_basis.orderDecrease(m_derivativeOrder);

    auto basisDer = std::make_shared<Basis>(m_basisDer);
    m_splineDer = Spline(basisDer, Interpolate(basisDer).fit(std::bind(
                                       &polynomialDer, _1, m_derivativeOrder, m_scale)));
  }

  Basis m_basisDer{};
  Spline m_spline{};
  Spline m_splineDer{};

  int m_derivativeOrder{};
  double m_scale{};

private:
  static Eigen::MatrixXd polynomial(const Eigen::ArrayXd &points) {
    // polynomial of degree 2
    Eigen::MatrixXd values(points.size(), 1);
    values << points.pow(2);
    return values;
  }

  static Eigen::MatrixXd polynomialDer(const Eigen::ArrayXd &points,
                                       int derivativeOrder = 1, double scale = 1.0) {
    if (derivativeOrder == 2) {
      // second derivative of polynomial of degree 2
      return Eigen::ArrayXd::Constant(points.size(), 2.0) / std::pow(scale, 2);
    } else if (derivativeOrder == 1) {
      // derivative of polynomial of degree 2
      Eigen::MatrixXd values(points.size(), 1);
      values << 2 * points / scale;
      return values;
    } else if (derivativeOrder == 0) {
      return polynomial(points);
    } else {
      throw std::invalid_argument(
          "Only derivative orders 0, 1 and 2 are supported.");
    }
  }
};

}; // namespace Internal
}; // namespace BasisSplines

#endif // BASIS_TEST_BASE_H