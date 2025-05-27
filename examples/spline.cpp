#include <Eigen/Core>
#include <matplot/matplot.h>

#include "basisSplines/basis.h"
#include "basisSplines/spline.h"
#include "helper.h"

namespace Bs = BasisSplines;
namespace Mt = matplot;

int main(int argc, char *argv[]) {
  // basis of order 3 with 4 breakpoints
  const Eigen::ArrayXd knots{{0.0, 0.0, 0.0, 0.4, 0.7, 0.7, 1.0, 1.0, 1.0}};
  std::shared_ptr<Bs::Basis> basis{std::make_shared<Bs::Basis>(knots, 3)};

  // spline definition
  const Eigen::ArrayXd coeffs{{0.0, 0.5, 0.25, -0.3, -1.0, 0.75}};
  const Bs::Spline spline{basis, coeffs};

  // plot spline
  auto axesHandle{Mt::axes()};
  axesHandle->hold(true);
  plotSpline(spline, Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1), axesHandle);
  matplot::grid(true);

  // save and show figure
  matplot::save(*(argv + 1));
  matplot::show();

  return 0;
}