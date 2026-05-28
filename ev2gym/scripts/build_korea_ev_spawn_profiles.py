import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Korean strings are written with unicode escapes so this script stays readable
# even on Windows consoles with a non-UTF-8 code page.
COL_STATION = "\ucda9\uc804\uc18c\uba85"
COL_ADDRESS = "\uc8fc\uc18c"
COL_ENERGY = "\ucda9\uc804\ub7c9"
COL_START = "\ucda9\uc804\uc2dc\uc791\uc2dc\uac01"
COL_END = "\ucda9\uc804\uc885\ub8cc\uc2dc\uac01"

PRIVATE_KEYWORDS = [
    "\uc544\ud30c\ud2b8", "APT", "apt", "\uc8fc\uac70", "\ub808\uc9c0\ub358\uc2a4",
    "\uc624\ud53c\uc2a4\ud154", "\ube4c\ub77c", "\uc8fc\ud0dd", "\ud0c0\uc6b4",
    "\ud558\uc774\uce20", "\ud478\ub974\uc9c0\uc624", "\uc790\uc774",
    "\ub798\ubbf8\uc548", "\ud790\uc2a4\ud14c\uc774\ud2b8",
    "\uc544\uc774\ud30c\ud06c", "\ub86f\ub370\uce90\uc2ac", "\ub354\uc0f5",
    "e\ud3b8\ud55c", "\uc774\ud3b8\ud55c", "\ub450\uc0b0\uc704\ube0c",
]

WORKPLACE_KEYWORDS = [
    "\uc9c0\uc0ac", "\ubcf8\ubd80", "\uccad\uc0ac", "\uc13c\ud130", "\uad6c\uccad",
    "\uc2dc\uccad", "\uad70\uccad", "\ud68c\uc0ac", "\ub300\ud559\uad50",
    "\ub300\ud559", "\ucea0\ud37c\uc2a4", "\ubcd1\uc6d0", "\uacf5\uc0ac",
    "\uacf5\ub2e8", "\uc5f0\uad6c\uc6d0", "\uc5f0\uad6c\uc18c",
    "\uc0ac\uc5c5\uc18c", "\uc804\ub825\uc9c0\uc0ac", "\ud55c\uc804",
    "\ud55c\uad6d\uc804\ub825", "\ud559\uad50",
]

SCENARIOS = ["private", "public", "workplace"]
PROFILE_STEP_MINUTES = 15


def classify_scenario(row):
    text = f"{row.get(COL_STATION, '')} {row.get(COL_ADDRESS, '')}"

    if any(keyword in text for keyword in PRIVATE_KEYWORDS):
        return "private"
    if any(keyword in text for keyword in WORKPLACE_KEYWORDS):
        return "workplace"
    return "public"


def time_label(ts, step_minutes):
    minute = (ts.minute // step_minutes) * step_minutes
    return f"{ts.hour:02d}:{minute:02d}"


def all_times(step_minutes):
    periods = int(24 * 60 / step_minutes)
    return [
        f"{(i * step_minutes) // 60:02d}:{(i * step_minutes) % 60:02d}"
        for i in range(periods)
    ]


def build_arrival_distribution(df, weekend):
    source = df[df["is_weekend"] == weekend]
    times = all_times(PROFILE_STEP_MINUTES)
    result = pd.DataFrame({"Arrival time": times})

    for scenario in SCENARIOS:
        scenario_rows = source[source["scenario"] == scenario]
        counts = scenario_rows.groupby("arrival_profile_step").size()
        total = counts.sum()
        if total == 0:
            result[scenario] = 0.0
        else:
            result[scenario] = result["Arrival time"].map(
                lambda label: counts.get(label, 0) / total * 100
            )

    return result


def fill_profile_table(df, value_col, output_time_col):
    times = all_times(PROFILE_STEP_MINUTES)
    result = pd.DataFrame({output_time_col: times})
    global_mean = df[value_col].mean()

    for scenario in SCENARIOS:
        scenario_rows = df[df["scenario"] == scenario]
        scenario_mean = scenario_rows[value_col].mean()
        if np.isnan(scenario_mean):
            scenario_mean = global_mean

        by_time = scenario_rows.groupby("arrival_profile_step")[value_col].mean()
        result[scenario] = result[output_time_col].map(
            lambda label: by_time.get(label, scenario_mean)
        )

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Build Korean EV spawn profile CSVs from KEPCO Seoul charging-session data."
    )
    parser.add_argument(
        "--input",
        default=r"c:\Users\MYCOM\Downloads\한국전력공사_서울시 전기차 충전소 충전량_20220331.xlsx",
        help="Path to the KEPCO Seoul charging-session Excel file.",
    )
    parser.add_argument(
        "--output-dir",
        default="ev2gym/data",
        help="Directory where generated CSV files will be written.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(input_path)
    df[COL_START] = pd.to_datetime(df[COL_START], errors="coerce")
    df[COL_END] = pd.to_datetime(df[COL_END], errors="coerce")
    df[COL_ENERGY] = pd.to_numeric(df[COL_ENERGY], errors="coerce")

    df = df.dropna(subset=[COL_START, COL_END, COL_ENERGY]).copy()
    df["duration_hours"] = (
        df[COL_END] - df[COL_START]
    ).dt.total_seconds() / 3600

    df = df[
        (df[COL_ENERGY] > 0.1)
        & (df["duration_hours"] >= 5 / 60)
        & (df["duration_hours"] <= 48)
    ].copy()

    df["scenario"] = df.apply(classify_scenario, axis=1)
    df["is_weekend"] = df[COL_START].dt.weekday >= 5
    df["arrival_profile_step"] = df[COL_START].apply(
        lambda ts: time_label(ts, PROFILE_STEP_MINUTES)
    )

    weekday_arrival = build_arrival_distribution(df, weekend=False)
    weekend_arrival = build_arrival_distribution(df, weekend=True)
    demand = fill_profile_table(df, COL_ENERGY, "Arrival Time")
    session_length = fill_profile_table(df, "duration_hours", "Arrival Time")

    weekday_arrival.to_csv(
        output_dir / "korea_distribution_of_arrival.csv",
        index=False,
        encoding="utf-8-sig",
    )
    weekend_arrival.to_csv(
        output_dir / "korea_distribution_of_arrival_weekend.csv",
        index=False,
        encoding="utf-8-sig",
    )
    demand.to_csv(
        output_dir / "korea_mean_demand_per_arrival.csv",
        index=False,
        encoding="utf-8-sig",
    )
    session_length.to_csv(
        output_dir / "korea_mean_session_length_per.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        df.groupby(["scenario", "is_weekend"])
        .agg(
            sessions=(COL_ENERGY, "size"),
            mean_energy_kwh=(COL_ENERGY, "mean"),
            mean_duration_hours=("duration_hours", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        output_dir / "korea_ev_spawn_profile_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("Generated Korean EV spawn profile files:")
    for filename in [
        "korea_distribution_of_arrival.csv",
        "korea_distribution_of_arrival_weekend.csv",
        "korea_mean_demand_per_arrival.csv",
        "korea_mean_session_length_per.csv",
        "korea_ev_spawn_profile_summary.csv",
    ]:
        print(f"  - {output_dir / filename}")
    print("\nScenario/session summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
