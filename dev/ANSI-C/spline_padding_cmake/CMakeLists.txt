cmake_minimum_required(VERSION 3.23)

project(Padding)

add_library(Padding SHARED)

target_sources(Padding
  PRIVATE
    spline_padding.c
  PUBLIC
    FILE_SET spline_padding
    TYPE HEADERS
    FILES
      spline_padding.h
)