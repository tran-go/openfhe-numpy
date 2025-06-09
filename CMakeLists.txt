# Minimum required CMake version - needs at least 3.16 for modern features
cmake_minimum_required(VERSION 3.16)

# Project declaration with name, version, and languages
project(OpenFHE-Numpy VERSION 0.0.1 LANGUAGES CXX)

# Version variables
set(OPENFHE_NUMPY_VERSION_MAJOR 0)
set(OPENFHE_NUMPY_VERSION_MINOR 0)
set(OPENFHE_NUMPY_VERSION_PATCH 1)
set(OPENFHE_NUMPY_VERSION ${OPENFHE_NUMPY_VERSION_MAJOR}.${OPENFHE_NUMPY_VERSION_MINOR}.${OPENFHE_NUMPY_VERSION_PATCH})

# Build Options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

option(ENABLE_DEBUG_LOGGING "Enable debug logging (defines DEBUG)" OFF)
option(BUILD_EXAMPLES "Build C++ example executables" ON)
option(BUILD_TESTING "Enable building of tests" OFF)

# Custom OpenFHE options
option(WITH_CUSTOM_OPENFHE "Use OpenFHE from a custom location" OFF)
set(CUSTOM_OPENFHE_ROOT "" CACHE PATH "Path to custom OpenFHE installation")
set(CUSTOM_OPENFHE_PYTHON "" CACHE PATH "Path to custom OpenFHE Python wrapper")

# Set up OpenFHE search paths if custom installation is specified
if(WITH_CUSTOM_OPENFHE)
    if(NOT CUSTOM_OPENFHE_ROOT)
        message(FATAL_ERROR "WITH_CUSTOM_OPENFHE is ON but CUSTOM_OPENFHE_ROOT is not specified")
    endif()

    set(OpenFHE_DIR "${CUSTOM_OPENFHE_ROOT}/lib/cmake/OpenFHE" CACHE PATH "Path to OpenFHE cmake config" FORCE)
    list(APPEND CMAKE_PREFIX_PATH "${CUSTOM_OPENFHE_ROOT}")

    message(STATUS "Using custom OpenFHE installation: ${CUSTOM_OPENFHE_ROOT}")

    if(CUSTOM_OPENFHE_PYTHON)
        message(STATUS "Using custom OpenFHE Python wrapper: ${CUSTOM_OPENFHE_PYTHON}")
        # Add Python wrapper to PYTHONPATH at build time
        set(ENV{PYTHONPATH} "${CUSTOM_OPENFHE_PYTHON}:$ENV{PYTHONPATH}")
    endif()
endif()

if(APPLE)
    set(CMAKE_CXX_VISIBILITY_PRESET default)
endif()

if(POLICY CMP0148)
  cmake_policy(SET CMP0148 NEW)
endif()

# Dependencies
find_package(OpenFHE 1.3.0 REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP QUIET)
find_package(Python REQUIRED COMPONENTS Interpreter Development)

link_directories(${OpenFHE_LIBDIR})

# Debug OpenFHE
message(STATUS "OpenFHE version: ${OpenFHE_VERSION}")
message(STATUS "OpenFHE libraries: ${OpenFHE_LIBRARIES}")
message(STATUS "OpenFHE include path: ${OpenFHE_INCLUDE}")
message(STATUS "OpenFHE library path: ${OpenFHE_LIBDIR}")

# Project Structure - Updated for new directory layout
set(OpenFHE_Numpy_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(OpenFHE_Numpy_CPP_DIR ${OpenFHE_Numpy_SRC_DIR}/cpp)
set(OpenFHE_Numpy_SOURCES ${OpenFHE_Numpy_CPP_DIR}/lib)
set(OpenFHE_Numpy_INCLUDES ${OpenFHE_Numpy_CPP_DIR}/include)
set(OpenFHE_Numpy_BINDINGS ${OpenFHE_Numpy_CPP_DIR}/lib/numpy_bindings.cpp)
set(OpenFHE_Numpy_PYTHON ${OpenFHE_Numpy_SRC_DIR})
set(OpenFHE_Numpy_BUILD_PYTHON ${CMAKE_BINARY_DIR}/openfhe_numpy)

file(MAKE_DIRECTORY ${OpenFHE_Numpy_BUILD_PYTHON})

# Configuration Header
configure_file(
    "${OpenFHE_Numpy_SRC_DIR}/version.py.in"
    "${OpenFHE_Numpy_BUILD_PYTHON}/version.py" @ONLY)

# Include Paths
include_directories(
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_Numpy_INCLUDES})

# Source Files
file(GLOB_RECURSE CORE_SOURCES ${OpenFHE_Numpy_SOURCES}/*.cpp)
list(FILTER CORE_SOURCES EXCLUDE REGEX ".*/numpy_bindings\\.cpp$")

add_library(openfhe_numpy_cpp_core STATIC ${CORE_SOURCES})

target_include_directories(openfhe_numpy_cpp_core PUBLIC
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_Numpy_INCLUDES})

target_link_libraries(openfhe_numpy_cpp_core PUBLIC
    ${OpenFHE_SHARED_LIBRARIES}
    ${OpenFHE_PKE_SHARED_LIBRARIES}
    ${OpenFHE_BINFHE_SHARED_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(openfhe_numpy_cpp_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python Extension
pybind11_add_module(_onp_cpp MODULE ${OpenFHE_Numpy_BINDINGS})
target_link_libraries(_onp_cpp PRIVATE openfhe_numpy_cpp_core)

# Set output location for the compiled Python extension
set_target_properties(_onp_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${OpenFHE_Numpy_BUILD_PYTHON}
    INSTALL_RPATH "$ORIGIN;${OpenFHE_LIBDIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Copy Python files to build directory
file(GLOB_RECURSE PYTHON_SOURCE_FILES
    "${OpenFHE_Numpy_PYTHON}/*.py"
    "${OpenFHE_Numpy_PYTHON}/operations/*.py"
    "${OpenFHE_Numpy_PYTHON}/tensor/*.py"
    "${OpenFHE_Numpy_PYTHON}/utils/*.py")

foreach(PY_FILE ${PYTHON_SOURCE_FILES})
    file(RELATIVE_PATH REL_PATH "${OpenFHE_Numpy_PYTHON}" "${PY_FILE}")
    configure_file(${PY_FILE} "${OpenFHE_Numpy_BUILD_PYTHON}/${REL_PATH}" COPYONLY)
endforeach()

# Installation Configuration
include(GNUInstallDirs)

# For pip/scikit-build installations
if(DEFINED SKBUILD)
    set(PYTHON_INSTALL_PATH "${SKBUILD_PROJECT_NAME}")
    message(STATUS "Building for pip/scikit-build: ${PYTHON_INSTALL_PATH}")
else()
    # Determine installation path based on CMAKE_INSTALL_PREFIX
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
            OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(PYTHON_INSTALL_PATH "${PYTHON_SITE_PACKAGES}")
    else()
        set(PYTHON_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

message(STATUS "Python package will be installed to: ${PYTHON_INSTALL_PATH}")

# Install the Python extension and files
if(DEFINED SKBUILD)
    # For pip installations
    install(TARGETS _onp_cpp
            LIBRARY DESTINATION "${PYTHON_INSTALL_PATH}")

    # Install Python modules
    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/operations"
            DESTINATION "${PYTHON_INSTALL_PATH}"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/tensor"
            DESTINATION "${PYTHON_INSTALL_PATH}"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/utils"
            DESTINATION "${PYTHON_INSTALL_PATH}"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    # Install root Python files
    install(FILES
            "${OpenFHE_Numpy_PYTHON}/__init__.py"
            "${OpenFHE_Numpy_PYTHON}/_version.py"
            "${OpenFHE_Numpy_BUILD_PYTHON}/version.py"
            DESTINATION "${PYTHON_INSTALL_PATH}")
else()
    # For custom installations, install directly to specified path
    install(TARGETS _onp_cpp
            LIBRARY DESTINATION "${PYTHON_INSTALL_PATH}")

    # Install Python modules with directory structure
    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/operations"
            DESTINATION "${PYTHON_INSTALL_PATH}"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/tensor"
            DESTINATION "${PYTHON_INSTALL_PATH}"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    install(DIRECTORY "${OpenFHE_Numpy_PYTHON}/utils"
            DESTINATION "${PYTHON_INSTALL_PATH}"
            FILES_MATCHING PATTERN "*.py"
            PATTERN "__pycache__" EXCLUDE)

    # Install root Python files
    install(FILES
            "${OpenFHE_Numpy_PYTHON}/__init__.py"
            "${OpenFHE_Numpy_PYTHON}/_version.py"
            "${OpenFHE_Numpy_BUILD_PYTHON}/version.py"
            DESTINATION "${PYTHON_INSTALL_PATH}")
endif()

# Build Examples
if(BUILD_EXAMPLES)
    set(EXAMPLE_OUTPUT_DIR ${CMAKE_BINARY_DIR}/examples)
    file(MAKE_DIRECTORY ${EXAMPLE_OUTPUT_DIR})

    find_library(OPENFHE_CORE_LIB OPENFHEcore PATHS ${OpenFHE_LIBDIR} REQUIRED)
    find_library(OPENFHE_PKE_LIB OPENFHEpke PATHS ${OpenFHE_LIBDIR} REQUIRED)
    find_library(OPENFHE_BINFHE_LIB OPENFHEbinfhe PATHS ${OpenFHE_LIBDIR} REQUIRED)

    file(GLOB EXAMPLES ${OpenFHE_Numpy_CPP_DIR}/examples/*.cpp)
    foreach(ex IN LISTS EXAMPLES)
        get_filename_component(name ${ex} NAME_WE)
        file(READ ${ex} FILE_CONTENT)
        if(NOT FILE_CONTENT MATCHES "^\\s*//")
            add_executable(${name} ${ex})
            target_link_libraries(${name} PRIVATE
                openfhe_numpy_cpp_core
                ${OPENFHE_PKE_LIB}
                ${OPENFHE_CORE_LIB}
                ${OPENFHE_BINFHE_LIB})
            set_target_properties(${name} PROPERTIES
                RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_OUTPUT_DIR}
                INSTALL_RPATH "${OpenFHE_LIBDIR}")
        endif()
    endforeach()
endif()

# Testing
if(BUILD_TESTING)
    enable_testing()
endif()

# CPack configuration for creating binary packages
include(CPack)
set(CPACK_PACKAGE_NAME "openfhe-numpy")
set(CPACK_PACKAGE_VENDOR "OpenFHE Team")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NumPy-like API for homomorphic encryption with OpenFHE")
set(CPACK_PACKAGE_VERSION "${OPENFHE_NUMPY_VERSION}")
set(CPACK_PACKAGE_VERSION_MAJOR "${OPENFHE_NUMPY_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${OPENFHE_NUMPY_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${OPENFHE_NUMPY_VERSION_PATCH}")
