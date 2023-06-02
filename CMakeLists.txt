cmake_minimum_required(VERSION 3.23)
project(mlp)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_executable(mlp main.cpp mlp.cpp mlp.h)

target_link_libraries (mlp Eigen3::Eigen)