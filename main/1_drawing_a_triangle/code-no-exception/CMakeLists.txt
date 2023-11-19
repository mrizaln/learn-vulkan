cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(LearnVulkan VERSION 0.0.0 LANGUAGES CXX)

# suppress subobject warnings on gcc
message("compiler id: ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    message(">>> suppressing subobject warning")
    add_compile_options(-Wno-subobject-linkage)
endif()

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)

include_directories(./include)

# main executable
set(MAIN_EXE main)
add_executable(${MAIN_EXE} src/main.cpp)

target_link_libraries(${MAIN_EXE} PRIVATE Vulkan::Vulkan glfw)

# copy assets to build directory
add_custom_command(TARGET ${MAIN_EXE} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/assets $<TARGET_FILE_DIR:${MAIN_EXE}>/assets)
