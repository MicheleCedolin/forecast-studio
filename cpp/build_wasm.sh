#!/usr/bin/env bash
# Builds tsfcore for the browser (WebAssembly) and writes web/tsfcore.mjs + web/tsfcore.wasm, which the web edition
# (web/) loads in a Web Worker. Usage (from the repository root, with the Emscripten SDK activated):
#     source /path/to/emsdk/emsdk_env.sh && bash cpp/build_wasm.sh
# Same sources and the same IEEE semantics as the native build (-O3, no -ffast-math). Differences from the native
# library: one thread per instance (the page starts one worker per CPU core instead), no x87 precision switch
# (long double is software binary128 in WebAssembly), and the CatBoost / LightGBM bridges are compiled but unusable
# (they load external libraries). tests/test_web_edition.py checks the result against the study's R forecasts.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/cpp/src"
out="$root/web"
build="$root/cpp/build-wasm"
mkdir -p "$build" "$out"

command -v em++ >/dev/null || { echo "em++ not found: activate the Emscripten SDK first" >&2; exit 1; }

# Every TSF_API function of capi.cpp is exported.
exports="$(grep -o '^TSF_API [A-Za-z0-9_ *]*tsf_[a-z0-9_]*' "$src/capi.cpp" | grep -o 'tsf_[a-z0-9_]*' | sed 's/^/_/' | paste -sd, -)"

pids=()
for f in "$src"/*.cpp; do
  em++ -std=c++17 -O3 -fwasm-exceptions -I"$src" -c "$f" -o "$build/$(basename "$f" .cpp).o" & pids+=($!)
done
for f in "$src"/catch22/*.c; do
  emcc -O2 -std=gnu99 -w -c "$f" -o "$build/c22_$(basename "$f" .c).o" & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

em++ -O3 -fwasm-exceptions "$build"/*.o -o "$out/tsfcore.mjs" \
  -sMODULARIZE -sEXPORT_ES6 -sENVIRONMENT=web,worker,node \
  -sALLOW_MEMORY_GROWTH -sSTACK_SIZE=5MB -sWASM_LEGACY_EXCEPTIONS=1 \
  -sEXPORTED_FUNCTIONS="$exports,_malloc,_free" \
  -sEXPORTED_RUNTIME_METHODS=HEAPU8,UTF8ToString,stringToUTF8,lengthBytesUTF8

ls -l "$out/tsfcore.mjs" "$out/tsfcore.wasm"
