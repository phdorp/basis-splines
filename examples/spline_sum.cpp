#include <Eigen/Core>
#include <matplot/matplot.h>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"
#include "helper.h"

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

  int cSpline{};
  for (const Bs::Spline &spline : splines) {

    // plot spline at evaluation points
    auto axesHandle{matplot::subplot(splines.size(), 1, cSpline++)};
    axesHandle->hold(true);
    axesHandle->grid(true);

    plotSpline(spline, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle);
  }

  // enable grid, save and show figure
  matplot::save(*(argv + 1));
  matplot::show();

  return 0;
}