'''
This file contains the eMPC class, which is used to control the ev2gym environment using the eMPC algorithm.

Authors: Cesar Diaz-Londono, Stavros Orfanoudakis
'''


import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from ev2gym.baselines.mpc.mpc import MPC


class V2GProfitMaxMPCGurobi(MPC):

    algo_name = "V2G Profit Max MPC (Gurobi)"

    def __init__(self, env, control_horizon=None, verbose=False, **kwargs):
        if control_horizon is None:
            horizon_minutes = env.config.get('forecast_horizon_minutes', 240)
            control_horizon = int(horizon_minutes / env.timescale)
        super().__init__(env, control_horizon, verbose)

    def get_action(self, env):
        t = env.current_step
        active_evs = []
        port_index = 0
        for cs in env.charging_stations:
            for ev in cs.evs_connected:
                if ev is not None:
                    active_evs.append((port_index, cs, ev))
                port_index += 1

        if not active_evs:
            return np.zeros(self.n_ports)

        h = min(self.control_horizon, env.simulation_length - t)
        n_active = len(active_evs)
        charge_prices = self.get_charge_decision_prices(t, h)
        discharge_prices = self.get_discharge_decision_prices(t, h)

        model = gp.Model("v2g_profit_mpc_gurobi")
        model.setParam('OutputFlag', 1 if self.verbose else 0)
        model.params.MIPGap = 0.01

        ch = model.addVars(n_active, h, lb=0, vtype=GRB.CONTINUOUS, name="ch")
        dis = model.addVars(n_active, h, lb=0, vtype=GRB.CONTINUOUS, name="dis")
        energy = model.addVars(n_active, h + 1, lb=0,
                               vtype=GRB.CONTINUOUS, name="energy")
        zbin = model.addVars(n_active, h, vtype=GRB.BINARY, name="mode")

        objective = gp.LinExpr()
        for k, (_, cs, ev) in enumerate(active_evs):
            max_ch_power = min(cs.get_max_power(), ev.max_ac_charge_power)
            max_dis_power = min(abs(cs.get_min_power()), abs(ev.max_discharge_power))
            if isinstance(ev.charge_efficiency, dict):
                ch_eff = max(ev.charge_efficiency.values()) / 100
            else:
                ch_eff = ev.charge_efficiency
            if isinstance(ev.discharge_efficiency, dict):
                dis_eff = max(ev.discharge_efficiency.values()) / 100
            else:
                dis_eff = ev.discharge_efficiency

            model.addConstr(energy[k, 0] == ev.current_capacity,
                            name=f"initial_energy_{k}")
            for step in range(h):
                absolute_step = t + step
                model.addConstr(ch[k, step] <= max_ch_power * zbin[k, step],
                                name=f"charge_mode_{k}_{step}")
                model.addConstr(dis[k, step] <= max_dis_power * (1 - zbin[k, step]),
                                name=f"discharge_mode_{k}_{step}")
                model.addConstr(
                    energy[k, step + 1] ==
                    energy[k, step] +
                    self.T * ch_eff * ch[k, step] -
                    self.T * dis_eff * dis[k, step],
                    name=f"energy_balance_{k}_{step}")
                model.addConstr(energy[k, step + 1] <= ev.battery_capacity,
                                name=f"battery_max_{k}_{step}")
                model.addConstr(energy[k, step + 1] >= 0,
                                name=f"battery_min_{k}_{step}")

                steps_until_departure = max(ev.time_of_departure - t + 1, 0)
                reachable_capacity = min(
                    ev.battery_capacity,
                    ev.current_capacity +
                    steps_until_departure * self.T * ch_eff * max_ch_power)
                if (absolute_step >= ev.time_of_departure and
                        reachable_capacity >= ev.desired_capacity - 1.5):
                    model.addConstr(energy[k, step + 1] >= ev.desired_capacity,
                                    name=f"departure_energy_{k}_{step}")

                objective += self.T * charge_prices[step] * ch[k, step]
                objective += -self.T * discharge_prices[step] * dis[k, step]

        for step in range(h):
            net_power = gp.quicksum(ch[k, step] - dis[k, step]
                                    for k in range(n_active))
            tr_max_power = env.transformers[0].max_power
            if not np.isscalar(tr_max_power):
                tr_max_power = tr_max_power[t + step]
            model.addConstr(net_power <= tr_max_power,
                            name=f"transformer_max_{step}")

        model.setObjective(objective, GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.Status.OPTIMAL:
            if self.verbose:
                print(f'Gurobi MPC failed with status {model.status}; using safety action')
            return self.enforce_departure_capacity(env, np.zeros(self.n_ports))

        action = np.zeros(self.n_ports)
        e = 0.001
        for k, (port_index, cs, _) in enumerate(active_evs):
            net_power = ch[k, 0].x - dis[k, 0].x
            if net_power > e:
                action[port_index] = net_power / cs.get_max_power()
            elif net_power < -e:
                action[port_index] = net_power / abs(cs.get_min_power())

        return self.enforce_departure_capacity(env, action)


class V2GProfitMaxMPC(MPC):

    algo_name = "V2G Profit Max MPC"

    def __init__(self, env, control_horizon=None, verbose=False, **kwargs):
        if control_horizon is None:
            horizon_minutes = env.config.get('forecast_horizon_minutes', 240)
            control_horizon = int(horizon_minutes / env.timescale)
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports
        self.nb = 2 * self.na

    def get_action(self, env):
        '''
        Re-optimizes every timestep. Actual SMP is used for the current
        published hour, and forecast SMP is used for future hours in horizon.
        '''
        t = env.current_step
        active_evs = []
        port_index = 0
        for cs in env.charging_stations:
            for ev in cs.evs_connected:
                if ev is not None:
                    active_evs.append((port_index, cs, ev))
                port_index += 1

        if not active_evs:
            return np.zeros(self.n_ports)

        h = min(self.control_horizon, env.simulation_length - t)
        n_active = len(active_evs)
        charge_prices = self.get_charge_decision_prices(t, h)
        discharge_prices = self.get_discharge_decision_prices(t, h)

        ch_offset = 0
        dis_offset = n_active * h
        energy_offset = 2 * n_active * h
        n_vars = energy_offset + n_active * (h + 1)

        def ch_idx(k, step):
            return ch_offset + k * h + step

        def dis_idx(k, step):
            return dis_offset + k * h + step

        def energy_idx(k, step):
            return energy_offset + k * (h + 1) + step

        c = np.zeros(n_vars)
        bounds = [(0, None)] * n_vars
        ev_params = []

        for k, (_, cs, ev) in enumerate(active_evs):
            max_ch_power = min(cs.get_max_power(), ev.max_ac_charge_power)
            max_dis_power = min(abs(cs.get_min_power()), abs(ev.max_discharge_power))
            if isinstance(ev.charge_efficiency, dict):
                ch_eff = max(ev.charge_efficiency.values()) / 100
            else:
                ch_eff = ev.charge_efficiency
            if isinstance(ev.discharge_efficiency, dict):
                dis_eff = max(ev.discharge_efficiency.values()) / 100
            else:
                dis_eff = ev.discharge_efficiency
            ev_params.append((max_ch_power, max_dis_power, ch_eff, dis_eff))

            for step in range(h):
                bounds[ch_idx(k, step)] = (0, max_ch_power)
                bounds[dis_idx(k, step)] = (0, max_dis_power)
                c[ch_idx(k, step)] = self.T * charge_prices[step]
                c[dis_idx(k, step)] = -self.T * discharge_prices[step]

            for step in range(h + 1):
                if step == 0:
                    bounds[energy_idx(k, step)] = (
                        ev.current_capacity, ev.current_capacity)
                else:
                    bounds[energy_idx(k, step)] = (
                        0, ev.battery_capacity)

        eq_rows = n_active * (h + 1)
        A_eq = lil_matrix((eq_rows, n_vars))
        b_eq = np.zeros(eq_rows)
        row = 0
        for k, (_, _, ev) in enumerate(active_evs):
            max_ch_power, max_dis_power, ch_eff, dis_eff = ev_params[k]
            A_eq[row, energy_idx(k, 0)] = 1
            b_eq[row] = ev.current_capacity
            row += 1

            for step in range(h):
                A_eq[row, energy_idx(k, step + 1)] = 1
                A_eq[row, energy_idx(k, step)] = -1
                A_eq[row, ch_idx(k, step)] = -self.T * ch_eff
                A_eq[row, dis_idx(k, step)] = self.T * dis_eff
                row += 1

        ub_rows = n_active * h + h + n_active * h
        A_ub = lil_matrix((ub_rows, n_vars))
        b_ub = np.zeros(ub_rows)
        row = 0
        for k, (_, _, ev) in enumerate(active_evs):
            max_ch_power, max_dis_power, _, _ = ev_params[k]
            for step in range(h):
                A_ub[row, ch_idx(k, step)] = 1
                A_ub[row, dis_idx(k, step)] = 1
                b_ub[row] = max(max_ch_power, max_dis_power)
                row += 1

        for step in range(h):
            for k in range(n_active):
                A_ub[row, ch_idx(k, step)] = 1
                A_ub[row, dis_idx(k, step)] = -1
            tr_max_power = env.transformers[0].max_power
            if not np.isscalar(tr_max_power):
                tr_max_power = tr_max_power[t + step]
            b_ub[row] = tr_max_power
            row += 1

        for k, (_, _, ev) in enumerate(active_evs):
            max_ch_power, _, ch_eff, _ = ev_params[k]
            for step in range(h):
                absolute_step = t + step
                A_ub[row, energy_idx(k, step + 1)] = -1
                if absolute_step >= ev.time_of_departure:
                    steps_until_departure = max(ev.time_of_departure - t + 1, 0)
                    reachable_capacity = min(
                        ev.battery_capacity,
                        ev.current_capacity +
                        steps_until_departure * self.T * ch_eff * max_ch_power)
                    if reachable_capacity >= ev.desired_capacity - 0.1:
                        b_ub[row] = -ev.desired_capacity
                    else:
                        b_ub[row] = 0
                else:
                    b_ub[row] = 0
                row += 1

        result = linprog(c,
                         A_ub=A_ub.tocsr(),
                         b_ub=b_ub,
                         A_eq=A_eq.tocsr(),
                         b_eq=b_eq,
                         bounds=bounds,
                         method='highs')

        if not result.success:
            if self.verbose:
                print(f'MPC optimization failed, using safety action: {result.message}')
            return self.enforce_departure_capacity(env, np.zeros(self.n_ports))

        action = np.zeros(self.n_ports)
        e = 0.001
        for k, (port_index, cs, ev) in enumerate(active_evs):
            net_power = result.x[ch_idx(k, 0)] - result.x[dis_idx(k, 0)]
            if net_power > e:
                action[port_index] = net_power / cs.get_max_power()
            elif net_power < -e:
                action[port_index] = net_power / abs(cs.get_min_power())

        return self.enforce_departure_capacity(env, action)


class V2GProfitMaxOracle(MPC):

    algo_name = "Optimal (Offline)"

    def __init__(self, env, verbose=False, **kwargs):
        """
        Initialize the MPC baseline.
        Args:
            env: The environment to be used for the MPC baseline.
            horizon: The horizon of the MPC baseline.
            verbose: Whether to print debug information.
        """
        control_horizon = env.simulation_length
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports
        self.nb = 2 * self.na

        self.actions = None

    def get_action(self, env):
        """
        This function computes the MPC actions for the economic problem including V2G.
        """

        t = env.current_step

        if self.actions is not None:
            return self.enforce_departure_capacity(env, self.actions[t, :])

        # update transformer limits using the exact values (not forecasts)
        self.update_tr_power_oracle(t)

        # reconstruct self.x_next using the environment
        self.reconstruct_state(t)
        self.calculate_XF_V2G(t)

        # Station models: Amono and Bmono
        self.v2g_station_models(t)

        # Complete model calculation Gu, and inequality constraints Au and bu
        self.calculate_InequalityConstraints(t)

        # Set power limits
        self.set_power_limits_V2G(t)

        # Print information if verbose
        if self.verbose:
            self.print_info(t)

        # Generate the min cost function
        f = []

        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.ch_prices[t + i])
                f.append(-self.T * self.disch_prices[t + i])

        f = np.array(f).reshape(-1, 1)

        nb = self.nb
        n = self.n_ports
        h = self.control_horizon

        model = gp.Model("optimization_model")
        u = model.addVars(range(nb*h),
                          vtype=GRB.CONTINUOUS,
                          name="u")  # Power

        # Binary for charging or discharging
        Zbin = model.addVars(range(n*h),
                             vtype=GRB.BINARY,
                             name="Zbin")

        # Constraints
        model.addConstrs((gp.quicksum(self.AU[i, j] * u[j]
                                      for j in range(nb*h))
                          <= self.bU[i]
                          for i in range(nb*h)), name="constr1")  # Constraint with prediction model

        # Constraints for charging P
        model.addConstrs((0 <= u[j]
                          for j in range(0, nb*h, 2)), name="constr3a")

        model.addConstrs((u[j] <= self.UB[j] * Zbin[j//2]
                          for j in range(0, nb*h, 2)), name="constr3b")

        # Constraints for discharging P
        model.addConstrs((0 <= u[j]
                          for j in range(1, nb*h, 2)),
                         name="constr4a")

        model.addConstrs((u[j] <= self.UB[j]*(1-Zbin[j//2])
                          for j in range(1, nb*h, 2)),
                         name="constr4b")

        # Add the transformer constraints
        # for tr_index in range(self.number_of_transformers):
        #     for i in range(self.control_horizon):
        #         model.addConstr((gp.quicksum((u[j] - u[j+1])
        #                                      for index, j in enumerate(
        #                                          range(i*self.nb, (i+1)*self.nb, 2))
        #                                      if self.cs_transformers[index] == tr_index) +
        #                          self.tr_loads[tr_index, i] +
        #                          self.tr_pv[tr_index, i] <=
        #                          self.tr_power_limit[tr_index, i]),
        #                         name=f'constr5_{tr_index}_t{i}')

        obj_expr = gp.LinExpr()
        for i in range(nb*h):
            obj_expr.addTerms(f[i], u[i])

        model.setObjective(obj_expr, GRB.MINIMIZE)
        model.params.NonConvex = 2
        model.params.MIPGap = 0.01

        # model.write('model.lp')
        model.optimize()

        if model.status != GRB.Status.OPTIMAL:
            print(f'Objective value: {model.status}')
            print("Optimal solution not found !!!!!")
            exit()

        a = np.zeros((self.simulation_length, nb*h))
        # z_bin = np.zeros((n*h, 1))

        for step in range(self.simulation_length):
            for i in range(2*self.n_ports):
                a[step, i] = u[step*2*self.n_ports + i].x

        # build normalized actions
        actions = np.zeros((self.simulation_length, self.n_ports))
        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports, 2)}')

        e = 0.001
        for step in range(self.simulation_length):
            for i in range(0, 2*self.n_ports, 2):
                if a[step, i] > e and a[step, i + 1] > e:
                    raise ValueError(f'Charging and discharging at the same time\
                                        {i} {a[step,i]} {a[step,i+1]}')
                elif a[step, i] > e:
                    actions[step, i//2] = a[step, i]/self.max_ch_power[i//2]
                elif a[step, i + 1] > e:
                    actions[step, i//2] = -a[step, i+1] / \
                        abs(self.max_disch_power[i//2])

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        self.actions = actions
        
        return self.enforce_departure_capacity(env, actions[t, :])


class V2GProfitMaxLoadsOracle(MPC):

    algo_name = "Optimal (Offline)"

    def __init__(self, env, verbose=False, **kwargs):
        """
        Initialize the MPC baseline.
        Args:
            env: The environment to be used for the MPC baseline.
            horizon: The horizon of the MPC baseline.
            verbose: Whether to print debug information.
        """
        control_horizon = env.simulation_length
        super().__init__(env, control_horizon, verbose)

        self.na = self.n_ports
        self.nb = 2 * self.na

        self.actions = None

    def get_action(self, env):
        """
        This function computes the MPC actions for the economic problem including V2G.
        """

        t = env.current_step

        if self.actions is not None:
            return self.enforce_departure_capacity(env, self.actions[t, :])

        # update transformer limits using the exact values (not forecasts)
        self.update_tr_power_oracle(t)

        # reconstruct self.x_next using the environment
        self.reconstruct_state(t)
        self.calculate_XF_V2G(t)

        # Station models: Amono and Bmono
        self.v2g_station_models(t)

        # Complete model calculation Gu, and inequality constraints Au and bu
        self.calculate_InequalityConstraints(t)

        # Set power limits
        self.set_power_limits_V2G(t)

        # Print information if verbose
        if self.verbose:
            self.print_info(t)

        # Generate the min cost function
        f = []

        for i in range(self.control_horizon):
            for j in range(self.n_ports):
                f.append(self.T * self.ch_prices[t + i])
                f.append(-self.T * self.disch_prices[t + i])

        f = np.array(f).reshape(-1, 1)

        nb = self.nb
        n = self.n_ports
        h = self.control_horizon

        model = gp.Model("optimization_model")
        u = model.addVars(range(nb*h),
                          vtype=GRB.CONTINUOUS,
                          name="u")  # Power

        # Binary for charging or discharging
        Zbin = model.addVars(range(n*h),
                             vtype=GRB.BINARY,
                             name="Zbin")

        # Constraints
        model.addConstrs((gp.quicksum(self.AU[i, j] * u[j]
                                      for j in range(nb*h))
                          <= self.bU[i]
                          for i in range(nb*h)), name="constr1")  # Constraint with prediction model

        # Constraints for charging P
        model.addConstrs((0 <= u[j]
                          for j in range(0, nb*h, 2)), name="constr3a")

        model.addConstrs((u[j] <= self.UB[j] * Zbin[j//2]
                          for j in range(0, nb*h, 2)), name="constr3b")

        # Constraints for discharging P
        model.addConstrs((0 <= u[j]
                          for j in range(1, nb*h, 2)),
                         name="constr4a")

        model.addConstrs((u[j] <= self.UB[j]*(1-Zbin[j//2])
                          for j in range(1, nb*h, 2)),
                         name="constr4b")

        # Add the transformer constraints
        for tr_index in range(self.number_of_transformers):
            for i in range(self.control_horizon):
                model.addConstr((gp.quicksum((u[j] - u[j+1])
                                             for index, j in enumerate(
                                                 range(i*self.nb, (i+1)*self.nb, 2))
                                             if self.cs_transformers[index] == tr_index) +
                                 self.tr_loads[tr_index, i] +
                                 self.tr_pv[tr_index, i] <=
                                 self.tr_power_limit[tr_index, i]),
                                name=f'constr5_{tr_index}_t{i}')

        obj_expr = gp.LinExpr()
        for i in range(nb*h):
            obj_expr.addTerms(f[i], u[i])

        model.setObjective(obj_expr, GRB.MINIMIZE)
        model.params.NonConvex = 2
        model.params.MIPGap = 0.01

        # model.write('model.lp')
        model.optimize()

        if model.status != GRB.Status.OPTIMAL:
            print(f'Objective value: {model.status}')
            print("Optimal solution not found !!!!!")
            exit()

        a = np.zeros((self.simulation_length, nb*h))
        # z_bin = np.zeros((n*h, 1))

        for step in range(self.simulation_length):
            for i in range(2*self.n_ports):
                a[step, i] = u[step*2*self.n_ports + i].x

        # build normalized actions
        actions = np.zeros((self.simulation_length, self.n_ports))
        if self.verbose:
            print(f'Actions:\n {a.reshape(-1,self.n_ports, 2)}')

        e = 0.001
        for step in range(self.simulation_length):
            for i in range(0, 2*self.n_ports, 2):
                if a[step, i] > e and a[step, i + 1] > e:
                    raise ValueError(f'Charging and discharging at the same time\
                                        {i} {a[step,i]} {a[step,i+1]}')
                elif a[step, i] > e:
                    actions[step, i//2] = a[step, i]/self.max_ch_power[i//2]
                elif a[step, i + 1] > e:
                    actions[step, i//2] = -a[step, i+1] / \
                        abs(self.max_disch_power[i//2])

        if self.verbose:
            print(f'actions: {actions.shape} \n {actions}')

        self.actions = actions
        
        return self.enforce_departure_capacity(env, actions[t, :])
