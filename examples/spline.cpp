#include <Eigen/Core>
#include <matplot/matplot.h>
#include <string>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"
#include "helper.h"

namespace Bs = BasisSplines;
namespace Mt = matplot;

int main(int argc, char *argv[]) {

  std::vector<Bs::Spline> splines(2);

  // basis of order 3 with 4 breakpoints
  std::shared_ptr<Bs::Basis> basis{std::make_shared<Bs::Basis>(
      Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.4, 0.7, 0.7, 1.0, 1.0, 1.0}}, 3)};

  // first spline definition
  splines[0] =
      Bs::Spline{basis, Eigen::ArrayXd{{0.0, 0.5, 0.25, -0.3, -1.0, 0.75}}};

  // second spline definition
  splines[1] =
      Bs::Spline{basis, Eigen::ArrayXd{{1.0, 0.5, 2, -3, -1.0, 0.75}}};

  // plot splines
  int cSpline{};
  for (const Bs::Spline &spline : splines) {
    auto axesHandle{matplot::subplot(splines.size(), 2, cSpline)};
    axesHandle->hold(true);
    axesHandle->grid(true);
    axesHandle->title(std::format("Spline {}", splines.size() % ++cSpline));

    plotSpline(spline, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle);
  }

  // change breakpoint at 0 to 0.3 and breakpoint at 2 to 0.8
  basis->setBreakpoints({{0.3, 0.8}}, {{0, 2}});

  // plot splines with new basis
  for (const Bs::Spline &spline : splines) {
    auto axesHandle{matplot::subplot(splines.size(), 2, cSpline)};
    axesHandle->hold(true);
    axesHandle->grid(true);
    axesHandle->title(std::format("Spline new basis {}", splines.size() % ++cSpline));

    plotSpline(spline, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle);
  }

  // save and show figure
  matplot::save(*(argv + 1));
  matplot::show();

  return 0;
}