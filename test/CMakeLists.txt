cmake_minimum_required(VERSION 2.8)
project( gigraph )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( gigraph  generic.cpp boev.cpp report.cpp)
target_link_libraries( gigraph ${OpenCV_LIBS} )
# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -march=native -msse4.2 -mavx -mfma -mfpmath=sse -mabm -ffast-math -funroll-loops")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()
