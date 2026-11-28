#!/usr/bin/env bash
# Reproduces the "Build tsfcore.wasm" step of .github/workflows/pages.yml (without the __BUILD__ sed)
set -euo pipefail
S=wasm_work
source $S/emsdk/emsdk_env.sh >/dev/null 2>&1
cd /home/user/forecast-studio
B=$S/wasm_check/build_wasm
mkdir -p $B
objects=""
pids=()
for f in cpp/src/*.cpp; do
  o="$B/$(basename "$f" .cpp).o"
  em++ -std=c++17 -O3 -fexceptions -Icpp/src -c "$f" -o "$o" &
  pids+=($!)
  objects="$objects $o"
  if (( ${#pids[@]} >= 4 )); then wait ${pids[0]}; pids=("${pids[@]:1}"); fi
done
for p in "${pids[@]}"; do wait $p; done
for f in cpp/src/catch22/*.c; do
  o="$B/catch22_$(basename "$f" .c).o"
  emcc -O2 -std=gnu99 -w -c "$f" -o "$o"
  objects="$objects $o"
done
em++ -O3 -fexceptions $objects -o web/tsfcore.js \
  -sMODULARIZE=1 -sEXPORT_NAME=createTsfCore -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB -sSTACK_SIZE=8MB \
  -sEXPORTED_FUNCTIONS=_malloc,_free,_tsf_version,_tsf_methods,_tsf_run,_tsf_forecast_batch,_tsf_io_open,_tsf_io_open_buffer,_tsf_io_close,_tsf_io_n_series,_tsf_io_n_attributes,_tsf_io_series,_tsf_io_attr,_tsf_io_header_json,_tsf_io_stats_json,_tsf_metrics_names,_tsf_metrics_table,_tsf_metrics_summary,_tsf_primary_period,_tsf_catch22,_tsf_catch22_count,_tsf_catch22_name,_tsf_boxcox_lambda,_tsf_tsfeat_new,_tsf_tsfeat_add,_tsf_tsfeat_columns,_tsf_tsfeat_table,_tsf_tsfeat_free,_tsf_tsfeat_startx \
  -sEXPORTED_RUNTIME_METHODS=FS,UTF8ToString,stringToUTF8,lengthBytesUTF8,HEAPU8,HEAPF64,HEAP64,HEAP32
ls -l web/tsfcore.js web/tsfcore.wasm
