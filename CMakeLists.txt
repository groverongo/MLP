cmake_minimum_required(VERSION 3.23)
project(mlp)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(./include ./lib)

file(GLOB archivos
    src/main.cpp 
    lib/*.cpp 
    include/*.h 
)

add_executable(mlp 
    src/main.cpp 
    ${archivos}
)

target_link_libraries (mlp Eigen3::Eigen)