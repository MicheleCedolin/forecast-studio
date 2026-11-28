import json
D = "/home/user/forecast-studio/web/data/"
files = {"m1_yearly": "m1_yearly_dataset.tsf", "m1_quarterly": "m1_quarterly_dataset.tsf", "m1_monthly": "m1_monthly_dataset.tsf",
         "tourism_quarterly": "tourism_quarterly_dataset.tsf", "nn5_weekly": "nn5_weekly_dataset.tsf", "hospital": "hospital_dataset.tsf",
         "car_parts": "car_parts_dataset_without_missing_values.tsf", "covid_deaths": "covid_deaths_dataset.tsf",
         "nn5_daily": "nn5_daily_dataset_without_missing_values.tsf"}
FAST, SLOW = 100, 30
models = {"tslm": FAST, "stm": FAST, "otm": FAST, "dstm": FAST, "theta": FAST, "ets": SLOW, "arima": SLOW,
          "bats": SLOW, "tbats": SLOW, "struct_ts": SLOW, "nnetar": SLOW}
jobs = []
for ds, f in files.items():
    for mdl, ns in models.items():
        jobs.append({"name": f"{mdl}__{ds}", "file": D + f, "config": [f"method={mdl}", f"dataset_id={ds}", f"max_series={ns}", "threads=1", "eol=lf"]})
for seas in ("7,28", "7,364"):
    jobs.append({"name": f"dshw__nn5_daily_{seas.replace(',', '_')}", "file": D + files["nn5_daily"],
                 "config": ["method=dshw", "dataset_id=nn5_daily", f"seasonality={seas}", f"max_series={FAST}", "threads=1", "eol=lf"]})
json.dump(jobs, open("wasm_work/wasm_check/jobs.json", "w"), indent=1)
print(len(jobs))
