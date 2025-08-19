#include "basisSplines/basis.h"
#include "basisSplines/interpolate.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace BasisSplines {
PYBIND11_MODULE(basis_splines, handle) {
  py::class_<Basis>(handle, "Basis")
      .def(py::init<Eigen::ArrayXd, int, double>())
      .def("insertKnots", &Basis::insertKnots)
      .def("combine", &Basis::combine)
      .def("orderDecrease", &Basis::orderDecrease)
      .def("orderIncrease", &Basis::orderIncrease)
      .def("orderElevation", &Basis::orderElevation)
      .def("derivative", py::overload_cast<Basis&, int>(&Basis::derivative, py::const_))
      .def("derivative", py::overload_cast<Basis&, const Eigen::MatrixXd&, int>(&Basis::derivative, py::const_))
      .def("integral", py::overload_cast<Basis&, int>(&Basis::integral, py::const_))
      .def("integral", py::overload_cast<Basis&, const Eigen::MatrixXd&, int>(&Basis::integral, py::const_))
      .def("add", &Basis::add<Interpolate>)
      .def("prod", &Basis::prod<Interpolate>)
      .def("dim", &Basis::dim)
      .def("order", &Basis::order)
      .def("knots", &Basis::knots, py::return_value_policy::reference_internal)
      .def("setBreakpoints", &Basis::setBreakpoints)
      .def("getBreakpoints", &Basis::getBreakpoints)
      .def("getScale", &Basis::getScale)
      .def("setScale", &Basis::setScale)
      .def("__call__", &Basis::operator())
      .def("greville", &Basis::greville)
      .def("getSegment", py::overload_cast<int, int>(&Basis::getSegment, py::const_))
      .def("getClamped", &Basis::getClamped)
      .def_static("toKnots", py::overload_cast<const Eigen::ArrayXd&, const Eigen::ArrayXi&, int>(&Basis::toKnots))
      .def_static("toKnots_pair", py::overload_cast<const std::pair<Eigen::ArrayXd, Eigen::ArrayXi>&, int>(&Basis::toKnots))
      .def_static("toBreakpoints", &Basis::toBreakpoints);

//   using PointsD = Points<Eigen::Dynamic>;
//   py::class_<PointsD>(handle, "Points")
//       .def(py::init<Eigen::ArrayXd, Eigen::ArrayXd>())
//       .def("x", py::overload_cast<const int>(&PointsD::x, py::const_))
//       .def("x", py::overload_cast<>(&PointsD::x, py::const_))
//       .def("y", py::overload_cast<const int>(&PointsD::y, py::const_))
//       .def("y", py::overload_cast<>(&PointsD::y, py::const_))
//       .def("setX", &PointsD::setX)
//       .def("setY", &PointsD::setY);

//   using PathD = Path<Eigen::Dynamic>;
//   py::class_<PathD> path(handle, "Path");

//   using PolychainD = Polychain<Eigen::Dynamic>;
//   py::class_<PolychainD>(handle, "Polychain", path)
//       .def(py::init<Eigen::ArrayXd, Eigen::ArrayXd>())
//       .def("setPoints", &PolychainD::setPoints)
//       .def("__call__", &PolychainD::operator())
//       .def("lengths", &PolychainD::lengths);

//   using TransformD = Transform<Eigen::Dynamic>;
//   py::class_<TransformD>(handle, "Transform")
//       .def(py::init<std::shared_ptr<PathD>>())
//       .def("posFrenet", &TransformD::posFrenet)
//       .def("posCartes", &TransformD::posCartes)
//       .def("velFrenet", &TransformD::velFrenet)
//       .def("velCartes", &TransformD::velCartes)
//       .def("accFrenet", &TransformD::accFrenet)
//       .def("accCartes", &TransformD::accCartes);
}
} // namespace BasisSplines