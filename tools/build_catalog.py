"""catalog.yaml üretici (tek seferlik, izlenebilirlik için depoda tutulur).

Girdiler
  * reference/experiments/fixed_horizon.R            -> lag, ufuk, tamsayı dönüşümü, yerel/global yöntem listesi
  * reference/experiments/deep_learning_experiments.py -> derin model listesi (ve lag/ufuk çapraz kontrolü)
  * DRIVE_FILES (aşağıda)                            -> Drive `tsf_data` klasöründeki gerçek dosya adları/boyutları
Çıktı: src/tsfstudio/data/catalog.yaml

Kullanım: python3 -I tools/build_catalog.py
"""
from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# (dosya adı, bayt, paket) — Drive `tsf_data` (main) ve `tsf_data/not used` (extra) dökümünden
DRIVE_FILES = [
    ("traffic_hourly_dataset.tsf", 103476945, "main"), ("nn5_daily_dataset_without_missing_values.tsf", 1479606, "main"),
    ("m4_weekly_dataset.tsf", 2736081, "main"), ("m4_hourly_dataset.tsf", 1688095, "main"),
    ("m4_daily_dataset.tsf", 76262290, "main"), ("kdd_cup_2018_dataset_without_missing_values.tsf", 9925024, "main"),
    ("m3_monthly_dataset.tsf", 960208, "main"), ("traffic_weekly_dataset.tsf", 676465, "main"),
    ("tourism_monthly_dataset.tsf", 516457, "main"), ("fred_md_dataset.tsf", 487237, "main"),
    ("sample.tsf", 351888, "main"), ("car_parts_dataset_without_missing_values.tsf", 344870, "main"),
    ("m1_monthly_dataset.tsf", 313600, "main"), ("m3_quarterly_dataset.tsf", 253744, "main"),
    ("vehicle_trips_dataset_without_missing_values.tsf", 147792, "main"), ("hospital_dataset.tsf", 233355, "main"),
    ("tourism_quarterly_dataset.tsf", 226526, "main"), ("nn5_weekly_dataset.tsf", 215630, "main"),
    ("m3_yearly_dataset.tsf", 125743, "main"), ("tourism_yearly_dataset.tsf", 88771, "main"),
    ("m3_other_dataset.tsf", 80003, "main"), ("covid_deaths_dataset.tsf", 158234, "main"),
    ("m1_quarterly_dataset.tsf", 53754, "main"), ("m1_yearly_dataset.tsf", 30557, "main"),
    ("weather_dataset.tsf", 212401586, "extra"), ("temperature_rain_dataset_without_missing_values.tsf", 114753808, "extra"),
    ("solar_10_minutes_dataset.tsf", 24110029, "extra"), ("rideshare_dataset_without_missing_values.tsf", 7072724, "extra"),
    ("pedestrian_counts_dataset.tsf", 11901419, "extra"), ("m4_yearly_dataset.tsf", 6876712, "extra"),
    ("m4_quarterly_dataset.tsf", 17623972, "extra"), ("m4_monthly_dataset.tsf", 77554719, "extra"),
    ("london_smart_meters_dataset_without_missing_values.tsf", 977782457, "extra"),
    ("kaggle_web_traffic_weekly_dataset.tsf", 74880015, "extra"), ("electricity_hourly_dataset.tsf", 35932338, "extra"),
    ("dominick_dataset.tsf", 75954427, "extra"), ("australian_electricity_demand_dataset.tsf", 13422464, "extra"),
    ("bitcoin_dataset_without_missing_values.tsf", 583250, "extra"), ("electricity_weekly_dataset.tsf", 334919, "extra"),
    ("solar_weekly_dataset.tsf", 57779, "extra"), ("cif_2016_dataset.tsf", 108787, "extra"),
]
# M5 (bu çalışmada üretildi; boyutlar m5_tsf/out ile aynı)
M5_FILES = [("m5_daily_dataset.tsf", 99129403), ("m5_item_state_daily_dataset.tsf", 31992314)]

DOMAIN = {
    "m1": "M1 competition (mixed)", "m3": "M3 competition (mixed)", "m4": "M4 competition (mixed)",
    "tourism": "Tourism", "nn5_daily": "ATM cash withdrawals (NN5)", "nn5_weekly": "ATM cash withdrawals (NN5)",
    "traffic": "Road occupancy (traffic)", "fred_md": "US macroeconomics (FRED-MD)", "hospital": "Hospital patient counts",
    "covid_deaths": "COVID-19 deaths", "car_parts": "Spare-parts demand (intermittent)", "vehicle_trips": "Vehicle trips",
    "kdd_cup": "Air quality (KDD Cup 2018)", "sample": "Sample dataset", "electricity": "Electricity consumption",
    "solar": "Solar power", "aus_elecdemand": "Australian electricity demand", "melbourne_pedestrian_counts": "Pedestrian counts",
    "rideshare": "Ride-hailing", "bitcoin": "Cryptocurrency", "cif_2016": "CIF 2016 competition (banking)",
    "dominick": "Retail sales (Dominick's)", "kaggle_web_traffic_weekly": "Web traffic", "london_smart_meters": "Smart-meter energy",
    "weather": "Weather", "temperature_rain": "Temperature and rainfall", "m5": "Retail unit sales (M5)",
}


def split_args(s: str) -> list[str]:
    return [x.strip() for x in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', s)]


def b(v: str) -> bool:
    return v.strip().upper() == "TRUE"


def parse_r_driver():
    src = (ROOT / "reference/experiments/fixed_horizon.R").read_text(encoding="utf-8", errors="replace")
    loc, glo = collections.defaultdict(dict), collections.defaultdict(dict)
    for m in re.finditer(r'^(#?)\s*do_fixed_horizon_(local|global)_forecasting\((.*)\)\s*$', src, re.M):
        comment, kind, args = m.groups()
        if comment:
            continue
        a = split_args(args)
        if kind == "local":
            ds, meth, f, extra = a[0].strip('"'), a[1].strip('"'), a[2].strip('"'), a[3:]
            loc[ds][meth] = (f, extra)
        else:
            ds, lag, f, meth, extra = a[0].strip('"'), int(a[1]), a[2].strip('"'), a[3].strip('"'), a[4:]
            glo[ds][meth] = (lag, f, extra)
    return loc, glo


def parse_dl():
    src = (ROOT / "reference/experiments/deep_learning_experiments.py").read_text(encoding="utf-8", errors="replace")
    dl = collections.defaultdict(dict)
    for m in re.finditer(r'^get_deep_nn_forecasts\((.*)\)\s*$', src, re.M):
        a = split_args(m.group(1))
        dl[a[0].strip('"')][a[3].strip('"')] = (int(a[1]), a[2].strip('"'), a[4:])
    return dl


def extras(extra: list[str]):
    key = index = None
    horizon = None
    integer = False
    # sıra: key, index, horizon, integer_conversion
    if len(extra) >= 1 and extra[0] != "NULL":
        key = extra[0].strip('"')
    if len(extra) >= 2 and extra[1] != "NULL":
        index = extra[1].strip('"')
    if len(extra) >= 3 and extra[2] != "NULL":
        horizon = int(extra[2])
    if len(extra) >= 4:
        integer = b(extra[3])
    return key, index, horizon, integer


def domain_of(ds_id: str) -> str:
    if ds_id in DOMAIN:
        return DOMAIN[ds_id]
    for k, v in DOMAIN.items():
        if ds_id.startswith(k):
            return v
    return "Other"


def main():
    loc, glo = parse_r_driver()
    dl = parse_dl()
    file_info = {n: (sz, grp) for n, sz, grp in DRIVE_FILES}
    entries, problems, missing = [], [], []
    seen_files = set()
    for ds_id in sorted(set(loc) | set(glo)):
        ref_file = (next(iter(glo[ds_id].values()))[1] if ds_id in glo else next(iter(loc[ds_id].values()))[0])
        if ref_file not in file_info:
            missing.append({"id": ds_id, "file": ref_file})
            continue
        size, grp = file_info[ref_file]
        seen_files.add(ref_file)
        # yerel/global/derin ayarları çapraz kontrol et
        ex_sets = {tuple(map(str, extras(e)[:4])) for (_, e) in loc.get(ds_id, {}).values()}
        ex_sets |= {tuple(map(str, extras(e)[:4])) for (_, _, e) in glo.get(ds_id, {}).values()}
        if len(ex_sets) > 1:
            problems.append(f"{ds_id}: yerel/global çağrılarda farklı key/index/ufuk/tamsayı: {ex_sets}")
        ex = (next(iter(loc[ds_id].values()))[1] if ds_id in loc else next(iter(glo[ds_id].values()))[2])
        key, index, horizon, integer = extras(ex)
        lags = {v[0] for v in glo.get(ds_id, {}).values()}
        if len(lags) != 1:
            problems.append(f"{ds_id}: global lag tutarsız/yok: {lags}")
        lag = next(iter(lags)) if lags else None
        dl_lags = {v[0] for v in dl.get(ds_id, {}).values()}
        if dl_lags and dl_lags != lags:
            problems.append(f"{ds_id}: derin model lag {dl_lags} != global lag {lags}")
        entries.append({
            "id": ds_id,
            "file": ref_file,
            "bytes": size,
            "bundle": grp,
            "domain": domain_of(ds_id),
            "lag": lag,
            "lag_origin": "study",
            "horizon_override": horizon,
            "integer_conversion": integer,
            "use_timestamps": bool(key and index),
            "local_methods": sorted(loc.get(ds_id, {}), key=["ses", "ets", "theta", "tbats", "arima", "dhr_arima"].index),
            "global_methods": sorted(glo.get(ds_id, {})),
            "deep_methods": sorted(dl.get(ds_id, {}), key=["feed_forward", "deepar", "nbeats", "wavenet", "transformer"].index),
        })
    # çalışmada kullanılmamış ama elimizdeki dosyalar
    for name, size, grp in DRIVE_FILES:
        if name in seen_files:
            continue
        ds_id = name.replace("_dataset_without_missing_values.tsf", "").replace("_dataset.tsf", "").replace(".tsf", "")
        sug = {"london_smart_meters": 336, "sample": 25}.get(ds_id)
        entries.append({
            "id": ds_id, "file": name, "bytes": size, "bundle": grp, "domain": domain_of(ds_id),
            "lag": sug, "lag_origin": "suggested", "horizon_override": 8 if ds_id == "sample" else None,
            "integer_conversion": False, "use_timestamps": True,
            "local_methods": [], "global_methods": [], "deep_methods": [],
        })
    for name, size in M5_FILES:
        ds_id = "m5_item_store" if name == "m5_daily_dataset.tsf" else "m5_item_state"
        entries.append({
            "id": ds_id, "file": name, "bytes": size, "bundle": "m5", "domain": domain_of("m5"),
            "lag": 28, "lag_origin": "suggested", "horizon_override": None, "integer_conversion": True,
            "use_timestamps": True, "local_methods": [], "global_methods": [], "deep_methods": [],
            "wrmsse": name == "m5_daily_dataset.tsf",
        })
    out = {
        "schema": 1,
        "source_note": ("lag/ufuk/tamsayı dönüşümü/yöntem listeleri kullanıcının fixed_horizon.R ve "
                        "deep_learning_experiments.py dosyalarından çıkarıldı; lag_origin=suggested olanlar çalışmada yoktur."),
        "datasets": sorted(entries, key=lambda e: (e["bundle"], e["id"])),
        "referenced_but_not_bundled": missing,
    }
    dst = ROOT / "src/tsfstudio/data/catalog.yaml"
    dst.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    print(f"yazıldı: {dst}  | veri kümesi: {len(entries)} | pakette olmayan: {[m['id'] for m in missing]}")
    for p in problems:
        print("UYARI:", p)
    total = sum(e["bytes"] for e in entries)
    print(f"toplam: {total/1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
