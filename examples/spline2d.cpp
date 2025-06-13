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

  // first spline definition
  const Bs::Spline spline{basis, Eigen::ArrayXXd{{-0.8, 0.0},
                                                 {-0.2, 1.0},
                                                 {0.3, -0.5},
                                                 {1.0, 0.3},
                                                 {1.0, 0.6},
                                                 {0.0, 0.8}}};

  // plot splines along each dimension
  for (int cDim{}; cDim < spline.dim(); ++cDim) {
    auto axesHandle{matplot::subplot(spline.dim(), 1, cDim)};
    axesHandle->hold(true);
    axesHandle->grid(true);
    axesHandle->title(std::format("Output {}", cDim));

    plotSpline(spline, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle,
               cDim);
  }

  // plot 2-dimensional spline
  auto axesHandle{Mt::figure()->current_axes()};
  axesHandle->hold(true);
  axesHandle->grid(true);
  plotSpline2d(spline, Eigen::ArrayXd::LinSpaced(121, 0.0, 1.0), axesHandle,
               {{0, 1}});

  // save and show figure
  matplot::save(*(argv + 1));
  matplot::show();

  return 0;
}