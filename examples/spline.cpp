#include <Eigen/Core>
#include <matplot/matplot.h>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"

namespace Bs = BasisSplines;

int main(int argc, char *argv[]) {
  // basis of order 3 with 4 breakpoints
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.0, 0.4, 0.7, 0.7, 1.0, 1.0, 1.0}};
  std::shared_ptr<Bs::Basis> basis{std::make_shared<Bs::Basis>(knots, 3)};

  // spline definition
  const Eigen::ArrayXd coeffs{{0.0, 0.5, 0.25, -0.3, -1.0, 0.75}};
  const Bs::Spline spline{basis, coeffs};

  // evaluate spline at points between -0.1 and 1.1
  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1)};
  const Eigen::ArrayXd splineVals{spline(points)};

  // plot spline at evaluation points
  matplot::hold(true);
  matplot::plot(std::vector<double>{points.begin(), points.end()},
                std::vector<double>{splineVals.begin(), splineVals.end()});

  // plot coefficients at greville sites
  const Eigen::ArrayXd greville{basis->greville()};
  matplot::plot(std::vector<double>{greville.begin(), greville.end()},
                std::vector<double>{coeffs.begin(), coeffs.end()}, "-o");

  // plot breakpoints along spline
  const auto [bps, conts] = basis->breakpoints();
  const Eigen::ArrayXd splineValsBps{spline(bps)};
  matplot::scatter(
      std::vector<double>{bps.begin(), bps.end()},
      std::vector<double>{splineValsBps.begin(), splineValsBps.end()})
      ->marker_style(matplot::line_spec::marker_style::diamond)
      .marker_color({0.0, 0.0, 1.0})
      .marker_face_color({0.0, 0.0, 1.0});

  // enable grid, save and show figure
  matplot::grid(true);
  matplot::save(*(argv + 1));
  matplot::show();

  return 0;
}