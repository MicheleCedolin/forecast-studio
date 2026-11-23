cmake_minimum_required(VERSION 3.16)
project(tsfcore C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# catch22 (GPL-3), C sources taken from Rcatch22 v0.1.13.
file(GLOB CATCH22_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/catch22/*.c)
add_library(tsfcore SHARED src/util.cpp src/basic.cpp src/ets.cpp src/arima.cpp src/autoarima.cpp src/intermittent.cpp src/theta_sf.cpp src/ces_sf.cpp src/sf_common.cpp src/mstl_sf.cpp src/mfles_sf.cpp src/tbats.cpp src/global.cpp src/catboost_bridge.cpp src/lightgbm_bridge.cpp src/stlf.cpp src/bagged.cpp src/loess.cpp src/lbfgsb.cpp src/temporal.cpp src/cpoly.cpp src/simple_sf.cpp src/capi.cpp ${CATCH22_SOURCES})
target_include_directories(tsfcore PRIVATE src)
set_target_properties(tsfcore PROPERTIES PREFIX "")        # tsfcore.dll / tsfcore.so (no "lib" prefix)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # -O3 without -ffast-math: IEEE semantics (no reassociation; baseline x86-64 has no FMA), so results match R/numpy.
  target_compile_options(tsfcore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
  # catch22 is compiled like R builds Rcatch22 (-O2, warnings off: the code is kept verbatim).
  target_compile_options(tsfcore PRIVATE $<$<COMPILE_LANGUAGE:C>:-O2 -std=gnu99 -w>)
  if(MINGW)
    # No dependency on MinGW runtime DLLs: the DLL runs on any Windows machine.
    target_link_options(tsfcore PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
elseif(MSVC)
  target_compile_options(tsfcore PRIVATE /O2 /W3 /fp:precise)
endif()
