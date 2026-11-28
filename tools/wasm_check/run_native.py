"""Runs jobs.json through the native src/tsfstudio/native/tsfcore.so with tsf_run (same config lines as the wasm side).
usage: python run_native.py jobs.json outdir [job-name-filter]"""
import ctypes, json, os, re, shutil, sys, tempfile, time
REPO = "/home/user/forecast-studio"
lib = ctypes.CDLL(REPO + "/src/tsfstudio/native/tsfcore.so")
lib.tsf_version.restype = ctypes.c_char_p
lib.tsf_methods.restype = ctypes.c_char_p
lib.tsf_run.restype = ctypes.c_int32
lib.tsf_run.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
dp, i64p, i32p = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32)
lib.tsf_forecast_batch.restype = ctypes.c_int32
lib.tsf_forecast_batch.argtypes = [ctypes.c_char_p, dp, i64p, ctypes.c_int32, ctypes.c_int32, i32p, dp, ctypes.c_int32, ctypes.c_int32, dp, i32p, ctypes.c_char_p, ctypes.c_int32]
jobs_file, out_root = sys.argv[1], sys.argv[2]
filt = sys.argv[3] if len(sys.argv) > 3 else None
jobs = [j for j in json.load(open(jobs_file)) if not filt or re.search(filt, j["name"])]
os.makedirs(out_root, exist_ok=True)
info = {"version": lib.tsf_version().decode(), "methods": lib.tsf_methods().decode(), "batch_rc": {}}
names = [s for s in open(os.path.join(os.path.dirname(jobs_file), "registry_names.txt")).read().split(",") if s] + ["catboost", "lightgbm", "no_such_method"]
n = 40
y = (ctypes.c_double * n)(*[10 + (i % 4) + 0.1 * i for i in range(n)])
off = (ctypes.c_int64 * 2)(0, n); per = (ctypes.c_double * 2)(4, 8); out = (ctypes.c_double * 4)(); fl = (ctypes.c_int32 * 1)(); msg = ctypes.create_string_buffer(256)
for nm in names:
    info["batch_rc"][nm] = lib.tsf_forecast_batch(nm.encode(), y, off, 1, 4, None, per, 2 if nm == "dshw" else 1, 1, out, fl, msg, 256)
json.dump(info, open(os.path.join(out_root, "info.json"), "w"), indent=1)
tmp = tempfile.mkdtemp(dir=out_root)
for job in jobs:
    run_dir, m = ctypes.create_string_buffer(1024), ctypes.create_string_buffer(2048)
    t0 = time.perf_counter()
    rc = lib.tsf_run(job["file"].encode(), tmp.encode(), "\n".join(job["config"]).encode(), run_dir, 1024, m, 2048)
    sec = time.perf_counter() - t0
    od = os.path.join(out_root, job["name"]); os.makedirs(od, exist_ok=True)
    if rc == 0:
        d = run_dir.value.decode()
        for f in ("forecasts.csv", "fallbacks.csv", "manifest.json"):
            if os.path.exists(os.path.join(d, f)): shutil.copy(os.path.join(d, f), od)
        shutil.rmtree(d)
    json.dump({"rc": rc, "seconds": sec, "message": m.value.decode()}, open(os.path.join(od, "result.json"), "w"))
    print(job["name"], rc, f"{sec:.2f}s", m.value.decode(), flush=True)
os.rmdir(tmp)
