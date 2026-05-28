from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import SafeRandomAgent
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxMPCGurobi


CONFIG = "ev2gym/example_config_files/V2GProfitMax.yaml"


def run(agent_name, agent_factory, seed):
    env = EV2Gym(
        config_file=CONFIG,
        verbose=False,
        save_replay=False,
        save_plots=False,
        seed=seed,
    )
    env.reset(seed=seed)
    agent = agent_factory(env)

    for _ in range(env.simulation_length):
        actions = agent.get_action(env)
        _, _, done, _, stats = env.step(actions)
        if done:
            return {
                "agent": agent_name,
                "seed": seed,
                "served": stats["total_ev_served"],
                "profit": stats["total_profits"],
                "charged": stats["total_energy_charged"],
                "discharged": stats["total_energy_discharged"],
                "avg_sat": stats["average_user_satisfaction"],
                "mean_energy_sat": stats["energy_user_satisfaction"],
                "min_energy_sat": stats["min_energy_user_satisfaction"],
            }

    raise RuntimeError("Simulation did not finish")


def main():
    rows = []
    for seed in range(101, 111):
        random_stats = run(
            "SafeRandom", lambda env: SafeRandomAgent(verbose=False), seed)
        mpc_stats = run(
            "MPC-Gurobi", lambda env: V2GProfitMaxMPCGurobi(env, verbose=False), seed)
        diff = mpc_stats["profit"] - random_stats["profit"]
        rows.append((random_stats, mpc_stats, diff))
        print(random_stats)
        print(mpc_stats)
        print(
            f"seed={seed} profit_diff_mpc_minus_random="
            f"{diff:.2f}"
        )

    random_profit = sum(row[0]["profit"] for row in rows) / len(rows)
    mpc_profit = sum(row[1]["profit"] for row in rows) / len(rows)
    avg_diff = sum(row[2] for row in rows) / len(rows)
    mpc_wins = sum(1 for row in rows if row[2] > 0)
    print("SUMMARY")
    print(f"runs={len(rows)}")
    print(f"safe_random_avg_profit={random_profit:.2f}")
    print(f"mpc_avg_profit={mpc_profit:.2f}")
    print(f"avg_profit_diff_mpc_minus_random={avg_diff:.2f}")
    print(f"mpc_wins={mpc_wins}/{len(rows)}")


if __name__ == "__main__":
    main()
