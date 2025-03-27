#include <Eigen/Core>

#include "testBase.h"
#include "basisSplines/basis.h"

namespace BasisSplines {
namespace Internal {
class BasisTest : public TestBase {
    protected:
    void SetUp()
    {

    }
};

TEST_F(BasisTest, BasisEvalOrder1)
{
    const Eigen::ArrayXd knots {{0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}};
    const int order {1};
    const Eigen::ArrayXd points {{0.0, 0.75}};

    const Basis basis {knots, order};
    const Eigen::ArrayXXd valuesEst {basis(points)};

    const Eigen::ArrayXXd valuesGtr {
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0}
    };

    expectAllClose(valuesEst, valuesGtr, 1e-10);
}

TEST_F(BasisTest, BasisEvalOrder2)
{
    const Eigen::ArrayXd knots {{0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}};
    const int order {2};
    const Eigen::ArrayXd points {{0.1, 0.75}};

    const Basis basis {knots, order};
    const Eigen::ArrayXXd valuesEst {basis(points)};

    const Eigen::ArrayXXd valuesGtr {
        {0.0, 0.8, 0.2, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.5, 0.5, 0.0}
    };

    expectAllClose(valuesEst, valuesGtr, 1e-10);
}

TEST_F(BasisTest, GrevilleOrder2)
{
    const Eigen::ArrayXd knots {{0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}};
    const int order {2};

    const Basis basis {knots, order};
    const Eigen::ArrayXd valuesEst {basis.greville()};
    const Eigen::ArrayXd valuesGtr {{0.0, 0.0, 0.5, 0.5, 1.0, 1.0}};

    expectAllClose(valuesEst, valuesGtr, 1e-10);
}
}; // namespace Internal
}; // namespace BasisSplines


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
