cmake_minimum_required (VERSION 3.5.1)

### To use gcc/g++ on a Macintosh, you must set the Compilers
### here, not inside the project
##if(APPLE)
##       set(CMAKE_C_COMPILER "/usr/local/bin/gcc-7")
##       set(CMAKE_CXX_COMPILER "/usr/local/bin/g++-7")
##endif()
### TODO: for now, we use CLang for Mac
###
### In order to create OpenFHE's static libraries you should enable
### the BUILD_STATIC option. For that, you run "cmake .. -DBUILD_STATIC=ON".
### After having your link completed you will find static libs
### with the suffix "_static" in ./build/libs/.
### Examples: OPENFHEpke_static.a, OPENFHEcore_static.a, etc.
### After you run "make install" in your build directory, you can build your custom application.
### If you need your application to be linked statically, then run "cmake .. -DBUILD_STATIC=ON"


project(OpenFHE-Matrix CXX)
set(OPENFHE_MATRIX_VERSION_MAJOR 0)
set(OPENFHE_MATRIX_VERSION_MINOR 0)
set(OPENFHE_MATRIX_VERSION_PATCH 1)
set(OPENFHE_MATRIX_VERSION ${OPENFHE_MATRIX_VERSION_MAJOR}.${OPENFHE_MATRIX_VERSION_MINOR}.${OPENFHE_MATRIX_VERSION_PATCH})

set(CMAKE_CXX_STANDARD 17)
# option( BUILD_STATIC "Set to ON to include static versions of the library" OFF)

find_package(OpenFHE CONFIG REQUIRED)
if (OpenFHE_FOUND)
    message(STATUS "FOUND PACKAGE OpenFHE")
    message(STATUS "OpenFHE Version: ${BASE_OPENFHE_VERSION}")
    message(STATUS "OpenFHE installed as shared libraries: ${OpenFHE_SHARED}")
    message(STATUS "OpenFHE include files location: ${OpenFHE_INCLUDE}")
    message(STATUS "OpenFHE lib files location: ${OpenFHE_LIBDIR}")
    message(STATUS "OpenFHE Native Backend size: ${OpenFHE_NATIVE_SIZE}")
else()
    message(FATAL_ERROR "PACKAGE OpenFHE NOT FOUND")
endif ()

set( CMAKE_CXX_FLAGS ${OpenFHE_CXX_FLAGS} )

include_directories( ${OPENMP_INCLUDES} )
include_directories( ${OpenFHE_INCLUDE} )
include_directories( ${OpenFHE_INCLUDE}/third-party/include )
include_directories( ${OpenFHE_INCLUDE}/core )
include_directories( ${OpenFHE_INCLUDE}/pke )
include_directories( ${OpenFHE_INCLUDE}/binfhe )
### add directories for other OpenFHE modules as needed for your project

link_directories( ${OpenFHE_LIBDIR} )
link_directories( ${OPENMP_LIBRARIES} )
if(BUILD_STATIC)
    set( CMAKE_EXE_LINKER_FLAGS "${OpenFHE_EXE_LINKER_FLAGS} -static")
    link_libraries( ${OpenFHE_STATIC_LIBRARIES} )
else()
    set( CMAKE_EXE_LINKER_FLAGS ${OpenFHE_EXE_LINKER_FLAGS} )
    link_libraries( ${OpenFHE_SHARED_LIBRARIES} )
endif()

# Set the installation prefix to a custom directory
# set(CMAKE_INSTALL_PREFIX "/home/tango/Software/xlib")


# Print the value of PROJECT_SOURCE_DIR
message("***** INSTALL IS AT ${CMAKE_INSTALL_PREFIX}; to change, run cmake with -DCMAKE_INSTALL_PREFIX=/your/path")
message("***** CMAKE_SOURCE_DIR = ${CMAKE_SOURCE_DIR}")
message("***** PROJECT_BINARY_DIR = ${PROJECT_BINARY_DIR}")
message("***** PROJECT_SOURCE_DIR = ${PROJECT_SOURCE_DIR}")


# Add the library
add_library(OpenFHE-Matrix SHARED)
target_sources(OpenFHE-Matrix
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/src/enc_matrix.cpp
    ${PROJECT_SOURCE_DIR}/src/utils.cpp
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include/enc_matrix.h
    ${PROJECT_SOURCE_DIR}/include/utils.h
)

# Specify the output directory for the library
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

target_include_directories(OpenFHE-Matrix
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Add the executable to test the library
add_executable(example test/example.cpp)


# Link the library to the executable
target_link_libraries(example PRIVATE OpenFHE-Matrix)


### set up for install
install(TARGETS OpenFHE-Matrix DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/openfhe_matrix)

# set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
# set(INSTALL_INCLUDE_DIR include/openfhe_matrix CACHE PATH "Installation directory for headers")
message("***** INSTALL IS AT ${CMAKE_INSTALL_PREFIX}; to change, run cmake with -DCMAKE_INSTALL_PREFIX=/your/path")
# install(TARGETS  OpenFHE-Matrix
#         LIBRARY DESTINATION lib
#         ARCHIVE DESTINATION lib
#         RUNTIME DESTINATION bin
#         INCLUDES DESTINATION include
# )


