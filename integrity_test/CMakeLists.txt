cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(LearnVulkan VERSION 0.0.0 LANGUAGES CXX)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

# main executable
set(MAIN_EXE main)
add_executable(${MAIN_EXE} src/main.cpp)

target_link_libraries(${MAIN_EXE} PRIVATE Vulkan::Vulkan glfw)
