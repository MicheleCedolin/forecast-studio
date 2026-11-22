cmake_minimum_required(VERSION 3.16)
project(tsfcore CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(tsfcore SHARED src/util.cpp src/basic.cpp src/ets.cpp src/arima.cpp src/autoarima.cpp src/intermittent.cpp src/capi.cpp)
target_include_directories(tsfcore PRIVATE src)
set_target_properties(tsfcore PROPERTIES PREFIX "")        # tsfcore.dll / tsfcore.so (no "lib" prefix)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # -O3 without -ffast-math: IEEE semantics (no reassociation; baseline x86-64 has no FMA), so results match R/numpy.
  target_compile_options(tsfcore PRIVATE -O3 -Wall -Wextra)
  if(MINGW)
    # No dependency on MinGW runtime DLLs: the DLL runs on any Windows machine.
    target_link_options(tsfcore PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
elseif(MSVC)
  target_compile_options(tsfcore PRIVATE /O2 /W3 /fp:precise)
endif()
