cmake_minimum_required(VERSION 3.17)
project(random_matrix)

option(USE_MPI "Compile with MPI options" OFF)

MESSAGE(${USE_MPI})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (${USE_MPI})
    find_package(MPI REQUIRED COMPONENTS CXX)
    set(HDF5_PREFER_PARALLEL TRUE)
endif ()

find_package(HDF5 REQUIRED COMPONENTS C)
find_package(Eigen3 REQUIRED NO_MODULE)

MESSAGE(${HDF5_IS_PARALLEL} ${HDF5_INCLUDE_DIRS})

if (${HDF5_IS_PARALLEL} AND ${USE_MPI})
    add_compile_options(-DUSE_MPI)
    set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} ")
    set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS}")
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    link_libraries(${MPI_CXX_LIBRARIES})
    include_directories(${MPI_CXX_INCLUDE_PATH})
elseif(${USE_MPI})
     MESSAGE("HDF5 is not parallel. Continuing without MPI support.")
endif()

add_compile_options(-Wall -Wextra -pedantic -Werror)
include_directories(${HDF5_INCLUDE_DIRS})
link_libraries(Eigen3::Eigen ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

add_executable(random_matrix src/main.cpp)