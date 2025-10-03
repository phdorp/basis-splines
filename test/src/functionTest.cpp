#include "functionTest.h"

namespace BasisSplines {
namespace Internal {
Eigen::ArrayXd FunctionTest::getPointsSubset(double beginValue, double endValue,
                                             double acc) const {
  auto beginSubset{
      std::find_if(m_points.begin(), m_points.end(), [&](double point) {
        return std::abs(point - beginValue) <= acc;
      })};
  auto endSubset{
      std::find_if(m_points.begin(), m_points.end(), [&](double point) {
        return std::abs(point - endValue) <= acc;
      })};

  Eigen::ArrayXd subset(endSubset - beginSubset + 1);
  for (double &value : subset) {
    value = *beginSubset;
    ++beginSubset;
  }

  return subset;
}
} // namespace Internal
} // namespace BasisSplines