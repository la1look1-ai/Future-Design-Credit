from ev2gym.models.ev2gym_env import EV2Gym
from ev2gym.baselines.heuristics import SafeRandomAgent
from ev2gym.baselines.mpc.V2GProfitMax import V2GProfitMaxMPCGurobi


CONFIG = "ev2gym/example_config_files/V2GProfitMax.yaml"
SUMMARY_KEYS = [
    "total_ev_served",
    "total_profits",
    "total_energy_charged",
    "total_energy_discharged",
    "average_user_satisfaction",
    "energy_user_satisfaction",
    "min_energy_user_satisfaction",
]


def run(label, agent_factory, seed):
    env = EV2Gym(
        config_file=CONFIG,
        verbose=False,
        save_replay=True,
        save_plots=False,
        seed=seed,
    )
    env.reset(seed=seed)
    agent = agent_factory(env)

    try:
        for _ in range(env.simulation_length):
            actions = agent.get_action(env)
            _, _, done, _, stats = env.step(actions)
            if done:
                replay_path = f"./replay/replay_{env.sim_name}.pkl"
                summary = {key: stats[key] for key in SUMMARY_KEYS}
                print(f"{label} seed={seed} replay={replay_path}")
                print(summary)
                return summary
    except Exception as exc:
        print(f"{label} seed={seed} FAILED: {type(exc).__name__}: {exc}")
        return None

    print(f"{label} seed={seed} did not finish")
    return None


def main():
    for i, seed in enumerate([101, 102, 103, 104], start=1):
        run(f"SafeRandom-{i}", lambda env: SafeRandomAgent(verbose=False), seed)

    run("MPC-Gurobi", lambda env: V2GProfitMaxMPCGurobi(env, verbose=False), 201)


if __name__ == "__main__":
    main()
