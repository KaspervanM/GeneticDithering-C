cmake_minimum_required(VERSION 3.30)
project(GeneticDithering C)

set(CMAKE_C_STANDARD 23)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Include the netpbm project added as a submodule.
add_subdirectory(netpbm-c)

# Define your executable target.
add_executable(GeneticDithering main.c)

# If you need to set include directories manually,
# you can add the netpbm directory to the include path.
target_include_directories(GeneticDithering PRIVATE netpbm-c)

# Link your executable with the netpbm library and OpenMP.
target_link_libraries(GeneticDithering PRIVATE netpbm OpenMP::OpenMP_C)
