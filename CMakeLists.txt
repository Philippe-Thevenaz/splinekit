cmake_minimum_required(VERSION 3.23)
project(padding LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (MSVC)
  add_compile_options(/W4 /permissive- /Zc:preprocessor /Zc:__cplusplus)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Absolute source paths (robust on Windows) ----
set(PADDING_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ansi_c/spline_padding/src")

add_library(padding SHARED)

target_sources(padding
  PRIVATE
    "${PADDING_SRC_DIR}/spline_padding.c"
  PUBLIC
    FILE_SET spline_padding
    TYPE HEADERS
    FILES
      "${PADDING_SRC_DIR}/spline_padding.h"
)
target_compile_features(padding PUBLIC c_std_99)

install(
  TARGETS padding
  LIBRARY DESTINATION splinelkit
  RUNTIME DESTINATION splinelkit
  ARCHIVE DESTINATION splinelkit
)
