cmake_minimum_required(VERSION 3.16)
project(openfhe_numpy
    VERSION 0.0.1
    DESCRIPTION "Python package 'openfhe_numpy' with C++ backend for OpenFHE matrix operations"
    LANGUAGES CXX
)

# -----------------------------------------------------------------------------
# Options
# -----------------------------------------------------------------------------
option(ENABLE_DEBUG_LOGGING "Enable debug logging (defines DEBUG)" OFF)
option(BUILD_STATIC_LIB    "Build static core library" OFF)
option(BUILD_EXAMPLES      "Build C++ example executables" ON)
option(BUILD_TESTING       "Enable building of tests" OFF)

# -----------------------------------------------------------------------------
# Standards
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenFHE 1.2.3 CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(OpenMP QUIET)

# -----------------------------------------------------------------------------
# Core C++ library (private)
# -----------------------------------------------------------------------------
file(GLOB_RECURSE CORE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src/*.cpp
)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/bindings?\\.cpp$")

if(BUILD_STATIC_LIB)
    add_library(openfhe_core STATIC ${CORE_SOURCES})
else()
    add_library(openfhe_core SHARED ${CORE_SOURCES})
endif()

target_include_directories(openfhe_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
        ${OpenFHE_INCLUDE}
        ${OpenFHE_INCLUDE}/third-party/include
        ${OpenFHE_INCLUDE}/core
        ${OpenFHE_INCLUDE}/pke
        ${OpenFHE_INCLUDE}/binfhe
)

if(BUILD_STATIC_LIB)
    target_link_libraries(openfhe_core
        PUBLIC
            ${OpenFHE_STATIC_LIBRARIES}
            ${OpenFHE_PKE_STATIC_LIBRARIES}
            ${OpenFHE_BINFHE_STATIC_LIBRARIES}
    )
else()
    target_link_libraries(openfhe_core
        PUBLIC
            ${OpenFHE_SHARED_LIBRARIES}
            ${OpenFHE_PKE_SHARED_LIBRARIES}
            ${OpenFHE_BINFHE_SHARED_LIBRARIES}
    )
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(openfhe_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# -----------------------------------------------------------------------------
# Build Python extension module 'openfhe_numpy'
# -----------------------------------------------------------------------------
pybind11_add_module(openfhe_numpy
    cpp/src/bindings.cpp
)

target_link_libraries(openfhe_numpy
    PRIVATE
        openfhe_core
        Python3::Python
)

target_include_directories(openfhe_numpy
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
        ${OpenFHE_INCLUDE}
        ${OpenFHE_INCLUDE}/third-party/include
        ${OpenFHE_INCLUDE}/core
        ${OpenFHE_INCLUDE}/pke
        ${OpenFHE_INCLUDE}/binfhe
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(openfhe_numpy PRIVATE OpenMP::OpenMP_CXX)
endif()

# Place the compiled extension into python/src/openfhe_numpy/
set_target_properties(openfhe_numpy PROPERTIES
    OUTPUT_NAME "openfhe_numpy"
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/src/openfhe_numpy
)

# -----------------------------------------------------------------------------
# Install Python package into user site-packages
# -----------------------------------------------------------------------------
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.USER_SITE)"
  OUTPUT_VARIABLE PY_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Installing Python package to user site: ${PY_SITE}")

install(DIRECTORY python/src/openfhe_numpy
  DESTINATION ${PY_SITE}
  FILES_MATCHING PATTERN "*.py" PATTERN "*.so"
)

# -----------------------------------------------------------------------------
# Examples (build-only)
# -----------------------------------------------------------------------------
if(BUILD_EXAMPLES)
    set(EXAMPLE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/examples)
    file(MAKE_DIRECTORY ${EXAMPLE_OUTPUT_DIR})

    file(GLOB EXAMPLES ${CMAKE_CURRENT_SOURCE_DIR}/cpp/examples/*.cpp)
    list(FILTER EXAMPLES EXCLUDE REGEX ".*/simple-matrix-operations\\.cpp$")
    foreach(ex IN LISTS EXAMPLES)
        get_filename_component(name ${ex} NAME_WE)
        add_executable(${name} ${ex})
        target_include_directories(${name}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
        )
        target_link_libraries(${name} PUBLIC openfhe_core)
        set_target_properties(${name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_OUTPUT_DIR}
        )
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Testing (optional)
# -----------------------------------------------------------------------------
if(BUILD_TESTING)
    enable_testing()
endif()

# -----------------------------------------------------------------------------
# Packaging (CPack)
# -----------------------------------------------------------------------------
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_NAME "openfhe_numpy")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})
set(CPACK_GENERATOR "TGZ;ZIP")
include(CPack)
