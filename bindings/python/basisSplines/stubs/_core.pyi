from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = ["Basis", "Spline"]

class Basis:
    """

    Basis of piecewise polynomial functions represented by truncated powers.

    The basis is defined by its order and an increasing sequence of knots.
    It provides properties that are derived from the knots and degree.
    Allows the combination of two splines bases.
    """

    @staticmethod
    def toBreakpoints(
        knots: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
        order: typing.SupportsInt,
        accuracy: typing.SupportsFloat = 1e-06,
    ) -> tuple[
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
    ]:
        """
        Map knots to breakpoints and their order of continuity.

        Args:
             knots (np.ndarray): Query points mapped to breakpoints.
             order (int): Basis order.
             accuracy (float, optional): Tolerance for assigning knots to breakpoint. Default is 1e-6.
        Returns:
             Tuple[np.ndarray, np.ndarray]: Breakpoints and their continuities.
        """

    @staticmethod
    @typing.overload
    def toKnots(
        breakpoints: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
        continuities: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"],
        order: typing.SupportsInt,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Map breakpoints and continuities to knots.

        Args:
             breakpoints (np.ndarray): Breakpoints mapped to knots.
             continuities (np.ndarray): Breakpoint continuities.
             order (int): Basis order.
        Returns:
             np.ndarray: Knot representation of given breakpoints.
        """

    @staticmethod
    @typing.overload
    def toKnots(
        breakpoints: tuple[
            typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
            typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"],
        ],
        order: typing.SupportsInt,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Map a pair of breakpoints and continuities to knots.

        Args:
             breakpoints (Tuple[np.ndarray, np.ndarray]): Pair of breakpoints and continuities.
             order (int): Basis order.
        Returns:
             np.ndarray: Knot representation.
        """

    def __call__(
        self,
        points: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
        accBps: typing.SupportsFloat = 1e-06,
        accSegment: typing.SupportsFloat = 1e-06,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Evaluate the truncated power basis at the given points.

        Args:
             points (np.ndarray): Evaluation points.
             accBps (float, optional): Minimum distance between breakpoints. Default is 1e-6.
             accSegment (float, optional): Accuracy point assignment to knot segment for order 1 basis values. Default is 1e-6.
        Returns:
             np.ndarray: Values of truncated powers with points.size rows and self.dim columns.
        """

    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor for an empty basis.
        """

    @typing.overload
    def __init__(
        self,
        knots: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"],
        order: typing.SupportsInt,
        scale: typing.SupportsFloat = 1.0,
    ) -> None:
        """
        Construct a new basis for the given knots, order, and knot scale.

        Args:
             knots (np.ndarray): Knot locations.
             order (int): Basis order.
             scale (float, optional): Knot scaling factor. Default is 1.0.
        """

    def add(self, basis: Basis, basisOut: Basis) -> tuple[
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"],
    ]:
        """
        Determine transformation matrices for left and right operand coefficients to get sum coefficients.

        Args:
             basis (Basis): Right operand basis.
             basisOut (Basis): Sum basis.
        Returns:
             Tuple[np.ndarray, np.ndarray]: Transformation matrices Tl and Tr.
        """

    def combine(
        self,
        basis: Basis,
        order: typing.SupportsInt,
        accScale: typing.SupportsFloat = 1e-06,
        accBps: typing.SupportsFloat = 1e-06,
    ) -> Basis:
        """
        Combine the knots of this and another basis to new basis of given order.

        Args:
             basis (Basis): Other basis to combine with.
             order (int): Result basis order.
             accScale (float, optional): Accepted difference between this and other basis scaling.
             accBps (float, optional): Tolerance for assigning knots to breakpoint.
        Returns:
             Basis: Combined basis.
        """

    @typing.overload
    def derivative(
        self, basis: Basis, orderDer: typing.SupportsInt = 1
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Determines a matrix to transform the spline coefficients to derivative coefficients.

        Args:
             basis (Basis): Basis of reduced order.
             orderDer (int, optional): Derivative order. Default is 1.
        Returns:
             np.ndarray: Transformation matrix.
        """

    @typing.overload
    def derivative(
        self,
        basis: Basis,
        values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"],
        orderDer: typing.SupportsInt = 1,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Transforms the given values to the derivative of this basis.

        Args:
             basis (Basis): Basis of reduced order.
             values (np.ndarray): Basis values or spline coefficients.
             orderDer (int, optional): Derivative order. Default is 1.
        Returns:
             np.ndarray: Derivative values.
        """

    def dim(self) -> int:
        """
        Determine basis dimensionality.

        Returns:
             int: Basis dimensionality.
        """

    def getBreakpoints(self, accuracy: typing.SupportsFloat = 1e-06) -> tuple[
        typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"],
        typing.Annotated[numpy.typing.NDArray[numpy.int32], "[m, 1]"],
    ]:
        """
        Determine basis breakpoints and continuities at breakpoints.

        Args:
             accuracy (float, optional): Tolerance for assigning knots to breakpoint. Default is 1e-6.
        Returns:
             Tuple[np.ndarray, np.ndarray]: Breakpoints and continuities.
        """

    def getClamped(self) -> Basis:
        """
        Determine basis with knots clamped to basis segment.

        Returns:
             Basis: Clamped basis.
        """

    def getScale(self) -> float:
        """
        Get the basis scaling factor.

        Returns:
             float: Scaling factor.
        """

    def getSegment(self, arg0: typing.SupportsInt, arg1: typing.SupportsInt) -> Basis:
        """
        Determine a basis representing the first to the last segment of this basis.

        Args:
             first (int): Index of the first segment.
             last (int): Index of the last segment.
        Returns:
             Basis: Segment basis.
        """

    def greville(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Determine the Greville sites representing the knot averages.

        Returns:
             np.ndarray: Greville sites.
        """

    def insertKnots(
        self, knotsIn: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> Basis:
        """
        Create a new basis with knots including knotsIn and this basis knots.

        Args:
             knotsIn (np.ndarray): Knots to insert to this basis' knots.
        Returns:
             Basis: New basis including the given knots.
        """

    @typing.overload
    def integral(
        self, basis: Basis, orderInt: typing.SupportsInt = 1
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Determines a matrix to transform the spline coefficients to integral coefficients.

        Args:
             basis (Basis): Basis spline.
             orderInt (int, optional): Integral order. Default is 1.
        Returns:
             np.ndarray: Transformation matrix.
        """

    @typing.overload
    def integral(
        self,
        basis: Basis,
        values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"],
        orderInt: typing.SupportsInt = 1,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Transforms the given values to the integral of this basis.

        Args:
             basis (Basis): Basis of increased order.
             values (np.ndarray): Basis values or spline coefficients.
             orderInt (int, optional): Integral order. Default is 1.
        Returns:
             np.ndarray: Integral values.
        """

    def knots(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Determine basis knots.

        Returns:
             np.ndarray: Basis knots.
        """

    def order(self) -> int:
        """
        Determine basis order.

        Returns:
             int: Basis order.
        """

    def orderDecrease(self, change: typing.SupportsInt = 1) -> Basis:
        """
        Determine new basis with order decreased by change.

        Args:
             change (int, optional): Order to decrease. Default is 1.
        Returns:
             Basis: Basis with reduced order.
        """

    def orderElevation(self, change: typing.SupportsInt = 1) -> Basis:
        """
        Determine new basis with order increased by change, preserving breakpoint continuity.

        Args:
             change (int, optional): Order to increase. Default is 1.
        Returns:
             Basis: Basis with increased order.
        """

    def orderIncrease(self, change: typing.SupportsInt = 1) -> Basis:
        """
        Determine new basis with order increased by change.

        Args:
             change (int, optional): Order to increase. Default is 1.
        Returns:
             Basis: Basis with increased order.
        """

    def prod(
        self, basis: Basis, basisOut: Basis
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Determine transformation matrix for coefficients to get product coefficients.

        Args:
             basis (Basis): Right operand basis.
             basisOut (Basis): Product basis.
        Returns:
             np.ndarray: Transformation matrix T.
        """

    def setBreakpoints(
        self,
        breakpointsNew: typing.Annotated[
            numpy.typing.ArrayLike, numpy.float64, "[m, 1]"
        ],
        idcs: typing.Annotated[numpy.typing.ArrayLike, numpy.int32, "[m, 1]"],
    ) -> None:
        """
        Set breakpoints at given breakpoint indices.

        Args:
             breakpointsNew (np.ndarray): New breakpoints.
             idcs (np.ndarray): Breakpoint indices to set.
        """

    def setScale(self, scale: typing.SupportsFloat) -> None:
        """
        Set the basis scaling factor.

        Args:
             scale (float): Scaling factor.
        """

class Spline:
    """

    Polynomial spline in basis form.

    Represents a multidimensional spline S(t) determined by its coefficients C for a given basis B(t).

    S(t) = C^T B(t)
    """

    def __call__(
        self, points: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Evaluate spline at given points.

        Args:
             points (np.ndarray): Evaluation points.
        Returns:
             np.ndarray: Spline function values at points. Rows = number of points, columns = output dimensionality.
        """

    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor for an empty spline.
        """

    @typing.overload
    def __init__(
        self,
        basis: Basis,
        coefficients: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"],
    ) -> None:
        """
        Construct a new spline in basis form from a basis and coefficients.

        Args:
             basis (Basis): Spline basis.
             coefficients (np.ndarray): Spline coefficients. Number of rows must correspond with basis dimensionality.
        """

    def __neg__(self) -> Spline:
        """
        Create new spline with negated spline coefficients.

        Returns:
             Spline: Spline with negated coefficients.
        """

    def add(
        self,
        other: Spline,
        accScale: typing.SupportsFloat = 1e-06,
        accBps: typing.SupportsFloat = 1e-06,
    ) -> Spline:
        """
        Create new spline as sum of this and other spline.

        Args:
             other (Spline): Right spline summand.
             accScale (float, optional): Accepted difference between this and other splines' basis scaling. Default is 1e-6.
             accBps (float, optional): Tolerance for assigning knots to breakpoint. Default is 1e-6.
        Returns:
             Spline: Spline sum.
        """

    def basis(self) -> Basis:
        """
        Get the spline basis.

        Returns:
             Basis: Spline basis.
        """

    def derivative(self, orderDer: typing.SupportsInt = 1) -> Spline:
        """
        Create new spline as derivative of this spline.

        Args:
             orderDer (int, optional): Derivative order. Default is 1.
        Returns:
             Spline: Derivative spline.
        """

    def dim(self) -> int:
        """
        Get the spline output dimensionality.

        Returns:
             int: Spline output dimensionality.
        """

    def getClamped(self) -> Spline:
        """
        Determine spline with knots clamped to spline segment.

        Returns:
             Spline: Clamped spline.
        """

    def getCoefficients(
        self,
    ) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Get the spline coefficients.

        Returns:
             np.ndarray: Spline coefficients. Rows correspond with basis dimensionality, columns with output dimensionality.
        """

    def getSegment(self, first: typing.SupportsInt, last: typing.SupportsInt) -> Spline:
        """
        Determine a spline representing the first and the last segment of this spline.

        Args:
             first (int): Index of the first segment.
             last (int): Index of the last segment.
        Returns:
             Spline: Segment spline.
        """

    def insertKnots(
        self, knots: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"]
    ) -> Spline:
        """
        Create new spline including the given and this splines' knots.

        Args:
             knots (np.ndarray): Knots to insert to this basis' knots.
        Returns:
             Spline: New spline including the given knots.
        """

    def integral(self, orderInt: typing.SupportsInt = 1) -> Spline:
        """
        Create new spline as integral of this spline.

        Args:
             orderInt (int, optional): Integral order. Default is 1.
        Returns:
             Spline: Integral spline.
        """

    def orderElevation(self, change: typing.SupportsInt) -> Spline:
        """
        Create new spline with order increased by change.

        Args:
             change (int): Positive order change.
        Returns:
             Spline: New spline with increased order.
        """

    def prod(
        self,
        other: Spline,
        accScale: typing.SupportsFloat = 1e-06,
        accBps: typing.SupportsFloat = 1e-06,
    ) -> Spline:
        """
        Create new spline as product of this and other spline.

        Args:
             other (Spline): Right product spline.
             accScale (float, optional): Accepted difference between this and other splines' basis scaling. Default is 1e-6.
             accBps (float, optional): Tolerance for assigning knots to breakpoint. Default is 1e-6.
        Returns:
             Spline: Spline product.
        """

    def setCoefficients(
        self,
        coefficients: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"],
    ) -> None:
        """
        Set the spline coefficients.

        Args:
             coefficients (np.ndarray): New spline coefficients. Must match shape of current coefficients.
        """
