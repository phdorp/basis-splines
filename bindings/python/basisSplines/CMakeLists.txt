FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG v3.0.0
  DOWNLOAD_EXTRACT_TIMESTAMP NEW
)
FetchContent_MakeAvailable(pybind11)

FILE(GLOB
    MAIN_FILES
    *.cpp
)

pybind11_add_module(basis_splines basisSplines.cpp)
target_link_libraries(
  basis_splines
  PUBLIC
  basisSplines
)
install(TARGETS basis_splines DESTINATION .)