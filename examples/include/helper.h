#ifndef HELPER_H
#define HELPER_H

#include <Eigen/Core>
#include <matplot/matplot.h>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"

namespace Bs = BasisSplines;
namespace Mt = matplot;

/**
 * @brief Plot a spline function at the given points in an axis handle.
 * The plot includes the spline function, the coefficients at the greville
 * sites, and the breakpoints.
 *
 * @param spline spline object to plot.
 * @param points function evaluation points.
 * @param axesHandle axis handle for plotting.
 */
void plotSpline(const Bs::Spline &spline, const Eigen::ArrayXd &points,
                const Mt::axes_handle axesHandle) {
  // plot spline at evaluation points
  const Eigen::ArrayXd splineVals{spline(points)};
  axesHandle->plot(std::vector<double>{points.begin(), points.end()},
                   std::vector<double>{splineVals.begin(), splineVals.end()});

  // plot coefficients at greville sites
  const Eigen::ArrayXd greville{spline.basis()->greville()};
  matplot::plot(std::vector<double>{greville.begin(), greville.end()},
                std::vector<double>{spline.coefficients()(Eigen::all, 0).begin(),
                                    spline.coefficients()(Eigen::all, 0).end()},
                "-o");

  // plot breakpoints along spline
  const Eigen::ArrayXd bps = spline.basis()->getBreakpoints().first;
  const Eigen::ArrayXd splineValsBps{spline(bps)};
  matplot::scatter(
      std::vector<double>{bps.begin(), bps.end()},
      std::vector<double>{splineValsBps.begin(), splineValsBps.end()})
      ->marker_style(matplot::line_spec::marker_style::diamond)
      .marker_color({0.0, 0.0, 1.0})
      .marker_face_color({0.0, 0.0, 1.0});
}

#endif