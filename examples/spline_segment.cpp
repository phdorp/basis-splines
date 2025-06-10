#include <Eigen/Core>
#include <matplot/matplot.h>
#include <string>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"
#include "helper.h"

namespace Bs = BasisSplines;
namespace Mt = matplot;

int main(int argc, char *argv[]) {

  // basis of order 3 with 4 breakpoints
  std::shared_ptr<Bs::Basis> basis{std::make_shared<Bs::Basis>(
      Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.4, 0.7, 0.7, 1.0, 1.0, 1.0}}, 3)};

  // spline of order 3
  const Bs::Spline spline{basis,
                          Eigen::ArrayXd{{0.0, 0.5, 0.25, -0.3, -1.0, 0.75}}};

  // plot spline
  int nAxes{3};
  auto axesHandle{Mt::subplot(nAxes, 1, 0)};
  axesHandle->hold(true);
  axesHandle->grid(true);
  axesHandle->ylim({-1.0, 1.0});
  plotSpline(spline, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle);

  // determine segment spline
  const Bs::Spline splineSeg{spline.getSegment(1, 1)};

  // plot spline segment
  axesHandle = Mt::subplot(nAxes, 1, 1);
  axesHandle->hold(true);
  axesHandle->grid(true);
  axesHandle->ylim({-1.0, 1.0});
  plotSpline(splineSeg, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle);

  // determine clamped segment spline
  const Bs::Spline splineClamped{splineSeg.getClamped()};

  // plot clamped segment spline
  axesHandle = Mt::subplot(nAxes, 1, 2);
  axesHandle->hold(true);
  axesHandle->grid(true);
  axesHandle->ylim({-1.0, 1.0});
  plotSpline(splineClamped, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1),
             axesHandle);

  // save and show figure
  Mt::save(*(argv + 1));
  Mt::show();

  return 0;
}