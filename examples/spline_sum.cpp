#include <Eigen/Core>
#include <matplot/matplot.h>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"

namespace Bs = BasisSplines;
namespace Mt = matplot;

int main(int argc, char *argv[]) {

  std::vector<Bs::Spline> splines(3);

  // definition first spline of order 3 with 4 breakpoints
  splines[0] = Bs::Spline{
      std::make_shared<Bs::Basis>(
          Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.4, 0.7, 0.7, 1.0, 1.0, 1.0}}, 3),
      Eigen::ArrayXd{{0.0, 0.5, 0.25, -0.3, -1.0, 0.75}}};

  // definition second spline of order 4 with 3 breakpoints
  splines[1] = Bs::Spline{
      std::make_shared<Bs::Basis>(
          Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0}},
          4),
      Eigen::ArrayXd{{1.0, -1.0, 0.3, 0.4, -0.1, 0.0}}};

  // add two splines
  splines[2] = splines[0].add(splines[1]);

  // evaluate spline at points between -0.1 and 1.1
  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1)};

  int cSpline{};
  for (const Bs::Spline &spline : splines) {
    const Eigen::ArrayXd splineVals{spline(points)};

    // plot spline at evaluation points
    matplot::subplot(3, 1, cSpline);
    matplot::hold(true);
    matplot::plot(std::vector<double>{points.begin(), points.end()},
                  std::vector<double>{splineVals.begin(), splineVals.end()});

    // plot coefficients at greville sites
    const Eigen::ArrayXd greville{spline.basis()->greville()};
    matplot::plot(std::vector<double>{greville.begin(), greville.end()},
                  std::vector<double>{spline.coefficients().begin(),
                                      spline.coefficients().end()},
                  "-o");

    // plot breakpoints along spline
    const auto [bps, conts] = spline.basis()->getBreakpoints();
    const Eigen::ArrayXd splineValsBps{spline(bps)};
    matplot::scatter(
        std::vector<double>{bps.begin(), bps.end()},
        std::vector<double>{splineValsBps.begin(), splineValsBps.end()})
        ->marker_style(matplot::line_spec::marker_style::diamond)
        .marker_color({0.0, 0.0, 1.0})
        .marker_face_color({0.0, 0.0, 1.0});
    ++cSpline;

    matplot::grid(true);
  }

  // enable grid, save and show figure
  matplot::save(*(argv + 1));
  matplot::show();

  return 0;
}