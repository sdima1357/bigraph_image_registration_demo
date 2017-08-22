cmake_minimum_required(VERSION 2.8)
project( Bigraph )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Bigraph  Bigraph.cpp )
target_link_libraries( Bigraph ${OpenCV_LIBS} )
# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3 -std=c++11 -march=native -ffast-math -funroll-loops")        ## Optimize
    set(CMAKE_EXE_LINKER_FLAGS "-s")  ## Strip binary
endif()
