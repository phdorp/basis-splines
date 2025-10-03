#ifndef FUNCTION_TEST_H
#define FUNCTION_TEST_H

#include <Eigen/Core>
#include <gtest/gtest.h>

#include "testBase.h"

namespace BasisSplines {
namespace Internal {
class FunctionTest : public TestBase {
protected:
  // MARK: protected properties
  const Eigen::ArrayXd m_points{Eigen::ArrayXd::LinSpaced(101, 0.0, 1.0)};

  // MARK: protected methods

  /**
   * @brief Extracts a subset of points from m_points between specified boundary
   * values.
   *
   * This method searches for points in m_points that match the given beginValue
   * and endValue within the specified accuracy tolerance, then returns all
   * points between and including those boundary points as a new array.
   *
   * @param beginValue The starting boundary value to search for in m_points.
   * @param endValue The ending boundary value to search for in m_points.
   * @param acc The tolerance for matching boundary values (default: 1e-8)
   * @return Eigen::ArrayXd A subset array containing all points from beginValue
   * to endValue (inclusive).
   *
   * @note The method uses std::find_if to locate the first points that match
   * the boundary values within the given accuracy. If no matching points are
   * found, the behavior depends on the iterator positions returned by
   * std::find_if.
   */
  Eigen::ArrayXd getPointsSubset(double beginValue, double endValue,
                                 double acc = 1e-8) const;
};
} // namespace Internal
} // namespace BasisSplines

#endif