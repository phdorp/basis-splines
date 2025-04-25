#include <Eigen/Core>
#include <matplot/matplot.h>

#include "basisSplines/basis.h"

int main(int argc, char *argv[]) {
  // basis of order 3 with 3 breakpoints
  const BasisSplines::Basis basis{{{0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}},
                                  3};

  // evaluate basis between -0.1 and 1.1
  const Eigen::ArrayXd points{Eigen::ArrayXd::LinSpaced(121, -0.1, 1.1)};
  const Eigen::ArrayXXd basisVals{basis(points)};

  // plot all basis functions
  matplot::hold(true);
  for (auto col : basisVals.colwise())
    matplot::plot(std::vector<double>{points.begin(), points.end()},
                  std::vector<double>{col.begin(), col.end()});
  matplot::ylim({-0.1, 1.1});
  matplot::grid(true);
  matplot::save(*(argv + 1));
  matplot::show();

  return 0;
}