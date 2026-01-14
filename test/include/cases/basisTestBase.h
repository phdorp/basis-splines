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

class BinaryOperationTest
    : public BasisTestBase,
      public testing::WithParamInterface<
          std::tuple<int, int, int, Eigen::ArrayXd, Eigen::ArrayXd>> {
public:
  static std::string TestNameGenerator(
      const testing::TestParamInfo<
          std::tuple<int, int, int, Eigen::ArrayXd, Eigen::ArrayXd>> &info) {
    return "Dimension" + std::to_string(std::get<0>(info.param)) + "_Order" +
           std::to_string(std::get<1>(info.param)) + "_OtherOrder" +
           std::to_string(std::get<2>(info.param)) + "_Breakpoints" +
           std::to_string(std::get<3>(info.param).size()) +
           "_OtherBreakpoints" + std::to_string(std::get<4>(info.param).size());
  }

protected:
  void SetUp() override {
    const Eigen::ArrayXd breakpoints{std::get<3>(GetParam())};
    Eigen::ArrayXi continuities{
        Eigen::ArrayXi::Constant(breakpoints.size(), m_order - 1)};
    continuities.head(1)(0) = 0;
    continuities.tail(1)(0) = 0;
    m_basis = Basis(breakpoints, continuities, m_order);

    const Eigen::VectorXd coeffs{Eigen::VectorXd::Random(m_basis.dim())};
    m_spline = Spline(std::make_shared<Basis>(m_basis), coeffs);

    const Eigen::ArrayXd breakpointsOther{std::get<4>(GetParam())};
    Eigen::ArrayXi continuitiesOther{
        Eigen::ArrayXi::Constant(breakpointsOther.size(), m_orderOther - 1)};
    continuitiesOther.head(1)(0) = 0;
    continuitiesOther.tail(1)(0) = 0;
    m_basisOther = Basis(breakpointsOther, continuitiesOther, m_orderOther);

    const Eigen::VectorXd coeffsOther{
        Eigen::VectorXd::Random(m_basisOther.dim())};
    m_splineOther = Spline(std::make_shared<Basis>(m_basisOther), coeffsOther);
  }

  Spline m_spline{};
  Basis m_basisOther{};
  Spline m_splineOther{};
  Basis m_basisResult{};
  Spline m_splineResult{};

  const int m_dimension{std::get<0>(GetParam())};
  const int m_order{std::get<1>(GetParam())};
  const int m_orderOther{std::get<2>(GetParam())};
};

// Base class containing common functionality for unary operations
class UnaryOperationTestBase : public BasisTestBase {
protected:
  void setupSplines() {
    auto basis = std::make_shared<Basis>(m_basis);
    m_spline = Spline(
        basis, Interpolate(basis).fit(std::bind(&polynomial, _1, m_dimension)));

    m_basisResult = getResultBasis();
    auto basisResult = std::make_shared<Basis>(m_basisResult);
    m_splineResult =
        Spline(basisResult,
               Interpolate(basisResult)
                   .fit(std::bind(&UnaryOperationTestBase::polynomialResult,
                                  this, _1)));
  }

  virtual Eigen::MatrixXd
  polynomialResult(const Eigen::ArrayXd &points) const = 0;

  virtual Basis getResultBasis() const = 0;

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

class SegmentTest
    : public BasisTestBase,
      public testing::WithParamInterface<std::tuple<
          std::pair<int, int>, int, Eigen::ArrayXd, Eigen::ArrayXi>> {
public:
protected:
  void SetUp() override {
    m_segment = std::get<0>(this->GetParam());
    m_order = std::get<1>(this->GetParam());
    m_breakpoints = std::get<2>(this->GetParam());
    m_continuities = std::get<3>(this->GetParam());

    m_basis = Basis(m_breakpoints, m_continuities, m_order);
  }

  std::pair<int, int> m_segment{};
  int m_order{};
  Eigen::ArrayXd m_breakpoints{};
  Eigen::ArrayXi m_continuities{};
};

// Primary template for UnaryOperationTest
template <typename ParamType = std::tuple<int, double, int>>
class UnaryOperationTest : public UnaryOperationTestBase,
                           public testing::WithParamInterface<ParamType> {
public:
  static std::string
  TestNameGenerator(const testing::TestParamInfo<ParamType> &info) {
    return "OperationOrder" + std::to_string(std::get<0>(info.param)) +
           "_Scale" +
           std::to_string(static_cast<int>(std::get<1>(info.param))) +
           "_Dimension" + std::to_string(std::get<2>(info.param));
  }

protected:
  void SetUp() override {
    m_operationOrder = std::get<0>(this->GetParam());
    m_scale = std::get<1>(this->GetParam());
    m_dimension = std::get<2>(this->GetParam());

    const Eigen::ArrayXd knots{
        Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}};
    const int order{3};
    m_basis = Basis(knots, order, m_scale);

    setupSplines();
  }
};

// Template specialization for custom Basis parameter
template <>
class UnaryOperationTest<std::tuple<int, Basis>>
    : public UnaryOperationTestBase,
      public testing::WithParamInterface<std::tuple<int, Basis>> {
public:
  static std::string TestNameGenerator(
      const testing::TestParamInfo<std::tuple<int, Basis>> &info) {
    return "OperationOrder" + std::to_string(std::get<0>(info.param)) +
           "_CustomBasis";
  }

protected:
  void SetUp() override {
    m_operationOrder = std::get<0>(GetParam());
    m_basis = std::get<1>(GetParam());
    m_scale = m_basis.scale();
    m_dimension = 1;

    setupSplines();
  }
};

template <typename ParamType = std::tuple<int, double, int>>
class DerivativeTest : public UnaryOperationTest<ParamType> {
private:
  Eigen::MatrixXd
  polynomialResult(const Eigen::ArrayXd &points) const override {
    if (this->m_operationOrder == 2) {
      // second derivative of polynomial of degree 2
      return Eigen::MatrixXd::Constant(points.size(), this->m_dimension, 2.0) /
             std::pow(this->m_scale, 2);
    } else if (this->m_operationOrder == 1) {
      // derivative of polynomial of degree 2
      Eigen::MatrixXd values(points.size(), this->m_dimension);
      values << 2 * points.matrix().replicate(1, this->m_dimension) /
                    this->m_scale;
      return values;
    } else if (this->m_operationOrder == 0) {
      return UnaryOperationTest<ParamType>::polynomial(points,
                                                       this->m_dimension);
    } else {
      throw std::invalid_argument(
          "Only derivative orders 0, 1 and 2 are supported.");
    }
  }

  Basis getResultBasis() const override {
    return this->m_basis.orderDecrease(this->m_operationOrder);
  }
};

template <typename ParamType = std::tuple<int, double, int>>
class IntegralTest : public UnaryOperationTest<ParamType> {
private:
  Eigen::MatrixXd
  polynomialResult(const Eigen::ArrayXd &points) const override {
    if (this->m_operationOrder == 2) {
      // second order integral of polynomial of degree 2
      Eigen::MatrixXd values(points.size(), this->m_dimension);
      values << points.pow(4).matrix().replicate(1, this->m_dimension) *
                    std::pow(this->m_scale, 2) / 12.0;
      return values;
    } else if (this->m_operationOrder == 1) {
      // first order integral of polynomial of degree 2
      Eigen::MatrixXd values(points.size(), this->m_dimension);
      values << points.pow(3).matrix().replicate(1, this->m_dimension) *
                    this->m_scale / 3.0;
      return values;
    } else if (this->m_operationOrder == 0) {
      return UnaryOperationTest<ParamType>::polynomial(points,
                                                       this->m_dimension);
    } else {
      throw std::invalid_argument(
          "Only integral orders 0, 1 and 2 are supported.");
    }
  }

  Basis getResultBasis() const override {
    return this->m_basis.orderIncrease(this->m_operationOrder);
  }
};

}; // namespace Internal
}
; // namespace BasisSplines

#endif // BASIS_TEST_BASE_H