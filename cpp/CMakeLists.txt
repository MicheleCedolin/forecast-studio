cmake_minimum_required(VERSION 3.16)
project(tsfcore C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# catch22 (GPL-3), C sources taken from Rcatch22 v0.1.13.
file(GLOB CATCH22_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/catch22/*.c)
add_library(tsfcore SHARED src/util.cpp src/basic.cpp src/ets.cpp src/arima.cpp src/autoarima.cpp src/intermittent.cpp src/theta_sf.cpp src/ces_sf.cpp src/sf_common.cpp src/mstl_sf.cpp src/mfles_sf.cpp src/tbats.cpp src/global.cpp src/catboost_bridge.cpp src/lightgbm_bridge.cpp src/stlf.cpp src/bagged.cpp src/loess.cpp src/lbfgsb.cpp src/temporal.cpp src/cpoly.cpp src/simple_sf.cpp src/fracdiff.cpp src/supsmu.cpp src/lapack_svd.cpp src/tslm.cpp src/struct_ts.cpp src/dshw.cpp src/nnetar.cpp src/capi.cpp src/tsfio.cpp src/metrics.cpp src/run.cpp src/tsfeat.cpp ${CATCH22_SOURCES})
target_include_directories(tsfcore PRIVATE src)
set_target_properties(tsfcore PROPERTIES PREFIX "")        # tsfcore.dll / tsfcore.so (no "lib" prefix)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # -O3 without -ffast-math: IEEE semantics (no reassociation; baseline x86-64 has no FMA), so results match R/numpy.
  target_compile_options(tsfcore PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
  # catch22 is compiled like R builds Rcatch22 (-O2, warnings off: the code is kept verbatim).
  target_compile_options(tsfcore PRIVATE $<$<COMPILE_LANGUAGE:C>:-O2 -std=gnu99 -w>)
  # Linux: -DTSF_STATIC_RUNTIME=ON puts libstdc++ / libgcc inside tsfcore.so. Needed when the compiler is newer than the
  # system's (an old cluster with a conda or devtoolset compiler): the library then loads without that compiler's
  # C++ runtime. libquadmath stays a shared library (see below); a conda compiler records its folder in the run path.
  option(TSF_STATIC_RUNTIME "Link libstdc++ and libgcc statically (non-Windows)" OFF)
  if(TSF_STATIC_RUNTIME AND NOT MINGW)
    target_link_options(tsfcore PRIVATE -static-libgcc -static-libstdc++)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # correctly rounded sin / cos for R's Fourier terms (autoarima.cpp)
    # A link test, not find_library: on Linux libquadmath lives in GCC's own directory, which find_library does not
    # search, while autoarima.cpp uses sinq / cosq whenever <quadmath.h> exists (unresolved symbol at load time).
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES quadmath)
    check_cxx_source_compiles("#include <quadmath.h>
int main(int argc, char**) { return static_cast<int>(sinq(static_cast<__float128>(argc))); }" HAVE_QUADMATH)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(HAVE_QUADMATH)
      # always the shared libquadmath: its static archive is not position independent and cannot go into tsfcore.so
      target_link_libraries(tsfcore PRIVATE quadmath)
    endif()
  endif()
  if(MINGW)
    # No dependency on MinGW runtime DLLs: the DLL runs on any Windows machine.
    target_link_options(tsfcore PRIVATE -static -static-libgcc -static-libstdc++)
  endif()
elseif(MSVC)
  target_compile_options(tsfcore PRIVATE /O2 /W3 /fp:precise)
endif()
