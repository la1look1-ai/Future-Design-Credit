from pathlib import Path
import tempfile

import yaml

from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxMPCGurobi


BASE_CONFIG = Path("ev2gym/example_config_files/V2GProfitMax.yaml")
SEEDS = range(101, 111)
HORIZONS = [120, 180, 240, 300]


def make_config(horizon_minutes, tmp_dir):
    with BASE_CONFIG.open("r", encoding="utf-8") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["forecast_horizon_minutes"] = horizon_minutes
    config_path = Path(tmp_dir) / f"V2GProfitMax_h{horizon_minutes}.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return str(config_path)


def run(config_file, horizon_minutes, seed):
    env = EV2Gym(
        config_file=config_file,
        verbose=False,
        save_replay=False,
        save_plots=False,
        seed=seed,
    )
    env.reset(seed=seed)
    agent = V2GProfitMaxMPCGurobi(env, verbose=False)

    for _ in range(env.simulation_length):
        actions = agent.get_action(env)
        _, _, done, _, stats = env.step(actions)
        if done:
            return {
                "seed": seed,
                "horizon_hours": horizon_minutes / 60,
                "profit": stats["total_profits"],
                "served": stats["total_ev_served"],
                "charged": stats["total_energy_charged"],
                "discharged": stats["total_energy_discharged"],
                "avg_sat": stats["average_user_satisfaction"],
                "mean_energy_sat": stats["energy_user_satisfaction"],
                "min_energy_sat": stats["min_energy_user_satisfaction"],
            }

    raise RuntimeError(f"Simulation did not finish for horizon {horizon_minutes}")


def main():
    results = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_files = {
            horizon: make_config(horizon, tmp_dir)
            for horizon in HORIZONS
        }
        for seed in SEEDS:
            seed_results = []
            for horizon in HORIZONS:
                result = run(config_files[horizon], horizon, seed)
                results.append(result)
                seed_results.append(result)

            best = max(seed_results, key=lambda row: row["profit"])
            print(f"seed={seed}")
            for row in seed_results:
                marker = " <==" if row is best else ""
                print(
                    f"  {row['horizon_hours']:.0f}h "
                    f"profit={row['profit']:.2f} "
                    f"charged={row['charged']:.2f} "
                    f"discharged={row['discharged']:.2f}"
                    f"{marker}"
                )

    print("SUMMARY")
    for horizon in HORIZONS:
        horizon_hours = horizon / 60
        horizon_rows = [
            row for row in results
            if row["horizon_hours"] == horizon_hours
        ]
        avg_profit = sum(row["profit"] for row in horizon_rows) / len(horizon_rows)
        avg_charged = sum(row["charged"] for row in horizon_rows) / len(horizon_rows)
        avg_discharged = (
            sum(row["discharged"] for row in horizon_rows) / len(horizon_rows)
        )
        wins = 0
        for seed in SEEDS:
            seed_rows = [row for row in results if row["seed"] == seed]
            best_profit = max(row["profit"] for row in seed_rows)
            if any(
                row["horizon_hours"] == horizon_hours
                and row["profit"] == best_profit
                for row in seed_rows
            ):
                wins += 1

        print(
            f"{horizon_hours:.0f}h avg_profit={avg_profit:.2f} "
            f"avg_charged={avg_charged:.2f} "
            f"avg_discharged={avg_discharged:.2f} "
            f"wins={wins}/{len(list(SEEDS))}"
        )


if __name__ == "__main__":
    main()
