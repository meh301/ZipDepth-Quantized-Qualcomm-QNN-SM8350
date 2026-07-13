cmake_minimum_required(VERSION 3.22.1)
project(zipdepth_demo C)

add_library(zipdepth_demo SHARED zipdepth_jni.c)

target_compile_features(zipdepth_demo PRIVATE c_std_11)
target_compile_options(zipdepth_demo PRIVATE -O3 -Wall -Wextra)
target_include_directories(zipdepth_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(zipdepth_demo log android dl)
