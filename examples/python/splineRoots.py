import basisSplines as bs
import matplotlib.pyplot as plt
import numpy as np
from helper import helper


def main():
    # basis of order 3 with 4 breakpoints
    basis = bs.Basis(np.array([0.0, 0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0, 1.0]), 4)

    # spline of order 3 with 2 dimensions
    coeffs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.6, 0.0], [-0.7, -0.7], [0.0, 1.0], [0.0, -1.0]])
    spline = bs.Spline(basis, coeffs)

    # plot roots
    roots = spline.getRoots()

    nAxes = spline.dim()

    # setup figure
    _, axs = plt.subplots(nAxes, 1, figsize=(8, 8))
    points = np.linspace(-0.1, 1.1, 121)

    for dim in range(spline.dim()):
        axs[dim].grid(True)
        axs[dim].set_ylim([-1.0, 1.0])
        axs[dim].set_xlabel("x/1")
        axs[dim].set_ylabel(f"s_{dim}(x)/1")

        helper.plotSpline(spline, points, axs[dim], dim)
        helper.plotRoots(spline, roots[dim], axs[dim], dim)

        axs[dim].legend(loc="lower left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
