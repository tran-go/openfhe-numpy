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

# Default locations for OpenFHE components
if(UNIX AND NOT APPLE)
    # Linux default paths
    set(DEFAULT_OPENFHE_ROOT "/usr/local" CACHE PATH "Default path to OpenFHE C++ installation")
    set(DEFAULT_OPENFHE_PYTHON "/usr/local/lib/python3/dist-packages/openfhe" CACHE PATH "Default path to OpenFHE Python wrapper")
    set(DEFAULT_OPENFHE_NUMPY "/usr/local/lib/python3/dist-packages/openfhe_numpy" CACHE PATH "Default path to OpenFHE-Numpy installation")
elseif(APPLE)
    # macOS default paths
    set(DEFAULT_OPENFHE_ROOT "/usr/local" CACHE PATH "Default path to OpenFHE C++ installation")
    set(DEFAULT_OPENFHE_PYTHON "/usr/local/lib/python3/site-packages/openfhe" CACHE PATH "Default path to OpenFHE Python wrapper")
    set(DEFAULT_OPENFHE_NUMPY "/usr/local/lib/python3/site-packages/openfhe_numpy" CACHE PATH "Default path to OpenFHE-Numpy installation")
elseif(WIN32)
    # Windows default paths
    set(DEFAULT_OPENFHE_ROOT "C:/Program Files/OpenFHE" CACHE PATH "Default path to OpenFHE C++ installation")
    set(DEFAULT_OPENFHE_PYTHON "C:/Python3/Lib/site-packages/openfhe" CACHE PATH "Default path to OpenFHE Python wrapper")
    set(DEFAULT_OPENFHE_NUMPY "C:/Python3/Lib/site-packages/openfhe_numpy" CACHE PATH "Default path to OpenFHE-Numpy installation")
endif()

# Custom OpenFHE options
option(WITH_CUSTOM_OPENFHE "Use OpenFHE from a custom location" OFF)
set(CUSTOM_OPENFHE_ROOT "${DEFAULT_OPENFHE_ROOT}" CACHE PATH "Path to custom OpenFHE installation")
set(CUSTOM_OPENFHE_PYTHON "${DEFAULT_OPENFHE_PYTHON}" CACHE PATH "Path to custom OpenFHE Python wrapper")
set(CUSTOM_OPENFHE_NUMPY "${DEFAULT_OPENFHE_NUMPY}" CACHE PATH "Path to custom OpenFHE-Numpy installation")

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
else()
    # Use default paths when custom installation is not specified
    set(OpenFHE_DIR "${DEFAULT_OPENFHE_ROOT}/lib/cmake/OpenFHE" CACHE PATH "Path to OpenFHE cmake config" FORCE)
    list(APPEND CMAKE_PREFIX_PATH "${DEFAULT_OPENFHE_ROOT}")

    message(STATUS "Using default OpenFHE installation: ${DEFAULT_OPENFHE_ROOT}")
    message(STATUS "Using default OpenFHE Python wrapper: ${DEFAULT_OPENFHE_PYTHON}")

    # Add Python wrapper to PYTHONPATH at build time
    set(ENV{PYTHONPATH} "${DEFAULT_OPENFHE_PYTHON}:$ENV{PYTHONPATH}")
endif()
