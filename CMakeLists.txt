cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

project(kdd99 CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

include(${PROJECT_SOURCE_DIR}/cmake/conan.cmake)
conan_cmake_run(
  CONANFILE conanfile.txt
  BASIC_SETUP
  CMAKE_TARGETS # individual targets to link to
  BUILD
    missing 
)

set(CMAKE_CXX_STANDARD 17)

file(GLOB HRDS
    include/kdd99/*.h
)
file(GLOB SRCS
    src/*.cpp
)

add_library(${PROJECT_NAME}
    ${SRCS}
    ${HRDS}
)

target_link_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/contrib/catboost
)

target_link_libraries(${PROJECT_NAME}
    catboostmodel
    dl
    pthread
)

target_include_directories(${PROJECT_NAME} PUBLIC
    include
)

set(DEMO_COMMON_HDRS
    demo/helpers.h
)

set(DEMO_COMMON_SRCS
    demo/helpers.cpp
)

file(GLOB TEST_SRCS
    tests/*.cpp
)

add_executable( fashio_mnist
    ${TEST_SRCS}
    ${DEMO_COMMON_SRCS}
    ${DEMO_COMMON_HDRS}
)

target_include_directories(fashio_mnist PUBLIC
    include
    demo
)

target_link_libraries(fashio_mnist
    CONAN_PKG::gtest
    ${PROJECT_NAME}
)

add_custom_target(${PROJECT_NAME}_test
    $<TARGET_FILE:fashio_mnist>
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)