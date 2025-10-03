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
      Eigen::ArrayXd{{0.0, 0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0}}, 4)};

  // spline of order 3
  const Eigen::MatrixXd coeffs{{1.0, -1.0, 0.6, -0.7, 0.0, 0.0},
                               {0.0, 0.0, 0.0, -0.7, 1.0, -1.0}};
  Bs::Spline spline{basis, coeffs.transpose()};

  // setup figure
  auto figureHandle{Mt::figure()};
  figureHandle->size(800, 800);

  // plot roots
  const auto roots{spline.getRoots()};

  Eigen::Index nAxes{spline.dim()};

  for (int dim{}; dim < spline.dim(); ++dim) {
    auto axesHandle = Mt::subplot(figureHandle, nAxes, 1, dim);

    axesHandle->hold(true);
    axesHandle->grid(true);
    axesHandle->ylim({-1.0, 1.0});
    axesHandle->xlabel("x/1");
    axesHandle->ylabel(std::format("s_{}(x)/1", dim));

    plotSpline(spline, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle,
               dim);
    plotRoots(spline, roots[dim], axesHandle, dim);

    Mt::legend(Mt::on)->location(Mt::legend::general_alignment::bottomleft);
  }

  // save and show figure
  saveFigure(figureHandle, getFileName(argc, argv), getFileEnding(argc, argv));
  Mt::show();

  return 0;
}