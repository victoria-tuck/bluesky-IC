import sys
from pathlib import Path
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import time
import json
import math
from pathlib import Path
from multiprocessing import Pool
import logging

from fisher.FisherGraphBuilder import FisherGraphBuilder

logging.basicConfig(filename='solver_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

fisher_objective = True
TOL_ERROR = 1e-2
MAX_NUM_ITERATIONS = 5000
UPDATED_APPROACH = True


def construct_market(flights, timing_info, routes, vertiport_usage, default_good_valuation=1, dropout_good_valuation=-1, BETA=1):
    """

    """
    # # building the graph
    # market_auction_time=timing_info["start_time"]
    # start_time_graph_build = time.time()
    # builder = FisherGraphBuilder(vertiport_usage, timing_info)
    # market_graph = builder.build_graph(flights)
    # print(f"Time to build graph: {time.time() - start_time_graph_build}")

    print("Constructing market...")
    start_time_market_construct = time.time()
    # goods_list = list(market_graph.edges) + ['default_good'] + ['dropout_good']
    goods_list = []
    w = []
    u = []
    agent_constraints = []
    agent_goods_lists = []
    
    for flight_id, flight in flights.items():


        builder = FisherGraphBuilder(vertiport_usage, timing_info)
        agent_graph = builder.build_graph(flight)
        origin_vertiport = flight["origin_vertiport_id"]
        start_node_time = flight["appearance_time"]
    

        # Add constraints
        nodes = list(agent_graph.nodes)
        edges = list(agent_graph.edges)
        starting_node = origin_vertiport + "_" + str(start_node_time)
        nodes.remove(starting_node)
        nodes = [starting_node] + nodes
        inc_matrix = nx.incidence_matrix(agent_graph, nodelist=nodes, edgelist=edges, oriented=True).toarray()
        rows_to_delete = []
        for i, row in enumerate(inc_matrix):
            if -1 not in row:
                rows_to_delete.append(i)
        A = np.delete(inc_matrix, rows_to_delete, axis=0)
        A[0] = -1 * A[0]
        valuations = []
        for edge in edges:
            valuations.append(agent_graph.edges[edge]["valuation"])

        b = np.zeros(len(A))
        b[0] = 1
        
        A_with_default_good = np.hstack((A, np.zeros((A.shape[0], 1)), np.zeros((A.shape[0], 1)))) # outside/default and dropout good
        A_with_default_good[0, -1] = 1 # droupout good
        # A_with_default_good = np.hstack((A, np.zeros((A.shape[0], 1)))) # outside/default good
        goods = edges + ['default_good'] + ['dropout_good']
        # Appending values for default and dropout goods
        valuations.append(default_good_valuation) # Small positive valuation for default good
        valuations.append(dropout_good_valuation) # Small positive valuation for dropout good

        w.append(flight["budget_constraint"])
        u.append(valuations)
        agent_constraints.append((A_with_default_good, b))
        agent_goods_lists.append(goods)
        goods_list += edges

    goods_list = goods_list + ['default_good'] + ['dropout_good']
    supply = find_capacity(goods_list, routes, vertiport_usage)
    
  

    print(f"Time to construct market: {time.time() - start_time_market_construct}")
    return (u, agent_constraints, agent_goods_lists), (w, supply, BETA), (goods_list)


def run_market(initial_values, agent_settings, market_settings, bookkeeping, rational=False, price_default_good=10, rebate_frequency=1):
    """
    
    """
    def social_welfare(x, p, u, supply):

        welfare = 0
        #assuming the utilties are stacked vertically with the same edge order per agent (as x)
        # and removing dropuout and default from eveyr agent utility list
        utility_vector = [item for sublist in u for item in sublist[:-2]]
        current_capacity = supply[:-2] - np.sum(x[:,:-2], axis=0) # should this be ceil or leave it as fraciton for now?
        agent_welfare = np.dot(utility_vector, x[:,:-2].T) / len(x) #- np.dot(p[:-2], x[:,:-2].T)
        welfare = np.sum(agent_welfare)

        return welfare
    print(f"running with rebate frequency: {rebate_frequency}")
    u, agent_constraints, agent_goods_lists = agent_settings
    y, p, r = initial_values
    w, supply, beta = market_settings
    goods_list = bookkeeping

    x_iter = 0
    prices = []
    rebates = []
    overdemand = []
    agent_allocations = []
    market_clearing = []
    yplot= []
    error = [] * len(agent_constraints)
    abs_error = [] * len(agent_constraints)
    social_welfare_vector = []
    
    # Algorithm 1
    num_agents = len(agent_goods_lists)
    tolerance = num_agents * np.sqrt(len(supply)-2) * TOL_ERROR  # -1 to ignore default goods
    market_clearing_error = float('inf')
    x_iter = 0
    start_time_algorithm = time.time()  
    
    problem = None
    while x_iter <= MAX_NUM_ITERATIONS:  # max(abs(np.sum(opt_xi, axis=0) - C)) > epsilon:
        x, adjusted_budgets = update_agents(w, u, p, r, agent_constraints, goods_list, agent_goods_lists, y, beta, x_iter, rebate_frequency, rational=rational)
        agent_allocations.append(x) # 
        overdemand.append(np.sum(x[:,:-2], axis=0) - supply[:-2].flatten())
        x_ij = np.sum(x[:,:-2], axis=0) # removing default and dropout good
        excess_demand = x_ij - supply[:-2]
        clipped_excess_demand = np.where(p[:-2] > 0,excess_demand, np.maximum(0, excess_demand)) # price removing default and dropout good
        market_clearing_error = np.linalg.norm(clipped_excess_demand, ord=2)
        market_clearing.append(market_clearing_error)

        iter_constraint_error = 0
        iter_constraint_x_y = 0
        for agent_index in range(len(agent_constraints)):
            agent_x = np.array([x[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index]])
            agent_y = np.array([y[agent_index, goods_list.index(good)] for good in agent_goods_lists[agent_index][:-2]])
            constraint_error = agent_constraints[agent_index][0] @ agent_x - agent_constraints[agent_index][1]
            abs_constraint_error = np.sqrt(np.sum(np.square(constraint_error)))
            iter_constraint_error += abs_constraint_error 
            agent_error = np.sqrt(np.sum(np.square(agent_x[:-2] - agent_y)))
            iter_constraint_x_y += agent_error
            if x_iter == 0:
                error.append([abs_constraint_error])
                abs_error.append([agent_error])
            else:
                error[agent_index].append(abs_constraint_error)
                abs_error[agent_index].append(agent_error)
    
        
        # if x_iter % rebate_frequency == 0:
        if True:
            update_rebates = True
        else:
            update_rebates = False
        # Update market
        k, y, p, r, problem = update_market(x, (1, p, r), (supply, beta), agent_constraints, agent_goods_lists, goods_list, price_default_good, problem, update_rebates=update_rebates)
        yplot.append(y)
        rebates.append([[rebate] for rebate in r])
        prices.append(p)
        current_social_welfare = social_welfare(x, p, u, supply)
        social_welfare_vector.append(current_social_welfare)

        x_iter += 1
        if (market_clearing_error <= tolerance) and (iter_constraint_error <= 0.0001) and (x_iter>=10) and (iter_constraint_x_y <= 0.1):
            break
        if x_iter ==  1000:
            break




        print("Iteration: ", x_iter, "- MCE: ", round(market_clearing_error, 5), "-Ax-b. Err: ", iter_constraint_error, " - Tol: ", round(tolerance,3), "x-y error:", iter_constraint_x_y)
        logging.info(f"Iteration: {x_iter}, Market Clearing Error: {market_clearing_error}, Tolerance: {tolerance}")
    
        # if market_clearing_error <= tolerance:
        #     break

    print(f"Time to run algorithm: {time.time() - start_time_algorithm}")

    data_to_plot ={
        "x_iter": x_iter,
        "prices": prices,
        "p": p,
        "overdemand": overdemand,
        "error": error,
        "abs_error": abs_error,
        "rebates": rebates,
        "agent_allocations": agent_allocations,
        "market_clearing": market_clearing,
        "agent_constraints": agent_constraints,
        "yplot": yplot,
        "social_welfare_vector": social_welfare_vector
    }


    # data_to_plot = [x_iter, prices, p, overdemand, error, abs_error, rebates, agent_allocations, market_clearing, agent_constraints, yplot, social_welfare_vector]

    last_prices = np.array(prices[-1])
    # final_prices = last_prices[last_prices > 0]

    # print(f"Error: {[error[i][-1] for i in range(len(error))]}")
    # print(f"Overdemand: {overdemand[-1][:]}")
    return x, last_prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot


def update_market(x_val, values_k, market_settings, constraints, agent_goods_lists, goods_list, price_default_good, problem, update_rebates=False):
    '''
    Update market consumption, prices, and rebates
    '''
    start_time = time.time()
    shape = np.shape(x_val)
    num_agents = shape[0]
    num_goods = shape[1]
    k, p_k_val, r_k = values_k
    supply, beta = market_settings
    
    # Update consumption
    # y = cp.Variable((num_agents, num_goods - 2)) # dropout and default removed
    # y = cp.Variable((num_agents, num_goods - 1)) # dropout removed (4)
    warm_start = False
    if problem is None:
        y = cp.Variable((num_agents, num_goods - 2)) 
        y_bar = cp.Variable(num_goods - 2)
        p_k = cp.Parameter(num_goods, name='p_k')
        x = cp.Parameter((num_agents, num_goods), name='x')
        # Do we remove drop out here or not? - remove the default and dropout good
        # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-1] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-1], 2))) # (4) (5)
        # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-2] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply[:-2], 2)))
        # objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x - y, 'fro')) - (beta / 2) * cp.square(cp.norm(cp.sum(y, axis=0) - supply, 2)))
        y_sum = cp.sum(y, axis=0)
        objective = cp.Maximize(-(beta / 2) * cp.square(cp.norm(x[:,:-2] - y, 'fro')) - (beta / 2) * cp.square(cp.norm(y_sum + y_bar - supply[:-2], 2))  - p_k[:-2].T @ y_bar)
        cp_constraints = [y_bar >= 0, y<=1, y_bar<=supply[:-2]] # remove default and dropout good
        problem = cp.Problem(objective, cp_constraints)
        warm_start = False
        # problem = cp.Problem(objective)
    else:
        y = problem.variables()[0]
        y_bar = problem.variables()[1]
        p_k = problem.param_dict['p_k']
        x = problem.param_dict['x']
    p_k.value = p_k_val
    x.value = x_val
    build_time = time.time() - start_time

    start_time = time.time()
    solvers = [cp.CLARABEL, cp.SCS, cp.OSQP, cp.ECOS, cp.CVXOPT]
    for solver in solvers:
        try:
            result = problem.solve(solver=solver, warm_start=warm_start, ignore_dpp=True)
            logging.info(f"Problem solved with solver {solver}")
            break
        except cp.error.SolverError as e:
            logging.error(f"Solver {solver} failed: {e}")
            continue
        except Exception as e:
            logging.error(f"An unexpected error occurred with solver {solver}: {e}")
            continue
    solve_time = time.time() - start_time
    print(f"Market: Build time: {build_time} - Solve time: {solve_time} with solver {solver}")

    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        logging.error("Failed to solve the problem with all solvers.")
    else:
        logging.info("Optimization result: %s", result)

    y_k_plus_1 = y.value

    # Update prices
    # p_k_plus_1 = np.zeros(p_k.shape)
    # try the options (default good): 
    # (3) do not update price, dont add it in optimization, 
    # (4) update price and add it in optimization, 
    # (5) dont update price but add it in optimization
    p_k_plus_1 = p_k_val[:-2] + beta * (np.sum(y_k_plus_1, axis=0) - supply[:-2]) #(3) default 
    # p_k_plus_1 = p_k[:-1] + beta * (np.sum(y_k_plus_1, axis=0) - supply[:-1]) #(4)
    # p_k_plus_1 = p_k[:-2] + beta * (np.sum(y_k_plus_1[:,:-2], axis=0) - supply[:-2]) #(5)
    # p_k_plus_1 = p_k + beta * (np.sum(y_k_plus_1, axis=0) - supply)
    for i in range(len(p_k_plus_1)):
        if p_k_plus_1[i] < 0:
            p_k_plus_1[i] = 0   
    # p_k_plus_1[-1] = 0  # dropout good
    # p_k_plus_1[-2] = price_default_good  # default good
    p_k_plus_1 = np.append(p_k_plus_1, [price_default_good,0]) # default and dropout good
    # p_k_plus_1 = np.append(p_k_plus_1, 0)  #  (4) update default good


    # Update each agent's rebates
    if update_rebates:
        r_k_plus_1 = []
        for i in range(num_agents):
            agent_constraints = constraints[i]
            agent_x = np.array([x.value[i, goods_list.index(good)] for good in agent_goods_lists[i]])
            if UPDATED_APPROACH:
                # agent_x = np.array([x[i, goods_list[:-1].index(good)] for good in agent_goods_lists[i][:-1]])
                constraint_violations = np.array([agent_constraints[0][j] @ agent_x - agent_constraints[1][j] for j in range(len(agent_constraints[1]))])
            else:
                constraint_violations = np.array([max(agent_constraints[0][j] @ agent_x - agent_constraints[1][j], 0) for j in range(len(agent_constraints[1]))])

            r_k_plus_1.append(r_k[i] + beta * constraint_violations[0])
    else:
        r_k_plus_1 = r_k
    return k + 1, y_k_plus_1, p_k_plus_1, r_k_plus_1, problem


def update_agents(w, u, p, r, constraints, goods_list, agent_goods_lists, y, beta, x_iter, update_frequency, rational=False, parallel=False):
    num_agents, num_goods = len(w), len(p)

    agent_indices = range(num_agents)
    agent_prices = [np.array([p[goods_list.index(good)] for good in agent_goods_lists[i]]) for i in agent_indices]
    agent_utilities = [np.array(u[i]) for i in agent_indices]
    agent_ys = [np.array([y[i, goods_list.index(good)] for good in agent_goods_lists[i][:-2]]) for i in agent_indices]
    # agent_ys = [np.array([y[i, goods_list[:-2].index(good)] for good in agent_goods_lists[i][:-2]]) for i in agent_indices] # removing dropout and detault good (3)
    # agent_ys = [np.array([y[i, goods_list[:-1].index(good)] for good in agent_goods_lists[i][:-1]]) for i in agent_indices] # removing dropout (4)
    args = [(w[i], agent_utilities[i], agent_prices[i], r[i], constraints[i], agent_ys[i], beta, x_iter, update_frequency, rational) for i in agent_indices]

    # Update agents in parallel or not depending on parallel flag
    # parallel = True
    if not parallel:
        results = []
        adjusted_budgets = []
        build_times = []
        solve_times = []
        for arg in args:
            updates =  update_agent(*arg)
            results.append(updates[0])
            adjusted_budgets.append(updates[1])
            build_times.append(updates[2][0])
            solve_times.append(updates[2][1])
        # results = [update_agent(*arg) for arg in args]
        print(f"Average build time: {np.mean(build_times)} - Average solve time: {np.mean(solve_times)}")
    else:
        num_processes = 4 # increase based on available resources
        with Pool(num_processes) as pool:
            pooled_results = pool.starmap(update_agent, args)
            results = [result[0] for result in pooled_results]
            adjusted_budgets = [result[1] for result in pooled_results]
            build_times = [result[2][0] for result in pooled_results]
            solve_times = [result[2][1] for result in pooled_results]
        print(f"Average build time: {np.mean(build_times)} - Average solve time: {np.mean(solve_times)}")

    x = np.zeros((num_agents, num_goods))
    for i, agent_x in zip(agent_indices, results):
        for good in goods_list:
            if good in agent_goods_lists[i]:
                x[i, goods_list.index(good)] = agent_x[agent_goods_lists[i].index(good)]
    return x, adjusted_budgets


def update_agent(w_i, u_i, p, r_i, constraints, y_i, beta, x_iter, update_frequency, rational=False, solver=cp.SCS):
    """
    Update individual agent's consumption given market settings and constraints
    """
    start_time = time.time()
    # Individual agent optimization
    A_i, b_i = constraints
    # A_bar = A_i[0]

    num_constraints = len(b_i)
    num_goods = len(p)

    if x_iter % update_frequency == 0:
        # lambda_i = r_i.T @ b_i # update lambda
        lambda_i = r_i * b_i[0]
        w_adj = w_i + lambda_i
        # print(w_adj)
        w_adj = max(w_adj, 0)
    else:
        w_adj = w_i
    # w_adj = abs(w_adj) 

    # print(f"Adjusted budget: {w_adj}")
    # optimizer check
    x_i = cp.Variable(num_goods)
    if rational:
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        objective = cp.Maximize(u_i.T @ x_i + regularizers + lagrangians)
        cp_constraints = [x_i >= 0, x_i<= 1]
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj]
        # objective = cp.Maximize(u_i.T @ x_i)
        # cp_constraints = [x_i >= 0, p.T @ x_i <= w_adj, A_i @ x_i <= b_i]
    elif UPDATED_APPROACH:
        # objective_terms = v_i.T @ x_i[:-2] + v_i_o * x_i[-2] + v_i_d * x_i[-1]
        objective_terms = u_i.T @ x_i
        # regularizers = - (beta / 2) * cp.square(cp.norm(x_i[:-1] - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2))  #(4)
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i[:-2] - y_i, 2)) - (beta / 2) *cp.square(cp.norm(A_i[0][:] @ x_i - b_i[0], 2)) # - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2))        
        # regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.square(cp.norm(A_i @ x_i - b_i, 2)) 
        # lagrangians = - p.T @ x_i - r_i.T @ (A_i @ x_i - b_i) # the price of dropout good is 0
        lagrangians = - p.T @ x_i - r_i * (A_i[0][:] @ x_i - b_i[0])
        nominal_objective = w_adj * cp.log(objective_terms)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0, x_i<= 1, A_i[1:] @ x_i == b_i[1:]]
        # cp_constraints = [x_i >= 0, A_bar @ x_i[:-2] + x_i[-2] >= 0]
    else:
        regularizers = - (beta / 2) * cp.square(cp.norm(x_i - y_i, 2)) - (beta / 2) * cp.sum([cp.square(cp.maximum(A_i[t] @ x_i - b_i[t], 0)) for t in range(num_constraints)])
        lagrangians = - p.T @ x_i - cp.sum([r_i[t] * cp.maximum(A_i[t] @ x_i - b_i[t], 0) for t in range(num_constraints)])
        nominal_objective = w_adj * cp.log(u_i.T @ x_i)
        objective = cp.Maximize(nominal_objective + lagrangians + regularizers)
        cp_constraints = [x_i >= 0, x_i<= 1]
    # check_time = time.time()
    problem = cp.Problem(objective, cp_constraints)
    # problem.solve(solver=solver, verbose=False)

    build_time = time.time() - start_time
    start_time = time.time()
    solvers = [cp.SCS, cp.CLARABEL, cp.MOSEK, cp.OSQP, cp.ECOS, cp.CVXOPT]
    for solver in solvers:
        try:
            result = problem.solve(solver=solver)
            logging.info(f"Agent Opt - Problem solved with solver {solver}")
            break
        except cp.error.SolverError as e:
            logging.error(f"Agent Opt - Solver {solver} failed: {e}")
            continue
        except Exception as e:
            logging.error(f"Agent Opt - An unexpected error occurred with solver {solver}: {e}")
            continue
    solve_time = time.time() - start_time
    
    # Check if the problem was solved successfully
    if problem.status != cp.OPTIMAL:
        logging.error("Agent Opt - Failed to solve the problem with all solvers.")
    else:
        logging.info("Agent opt - Optimization result: %s", result)


    return x_i.value, w_adj, (build_time, solve_time)


def find_capacity(goods_list, route_data, vertiport_data):
    # Create a dictionary for route capacities, for now just connectin between diff vertiports
    route_dict = {(route["origin_vertiport_id"], route["destination_vertiport_id"]): route["capacity"] for route in route_data}

    capacities = np.zeros(len(goods_list)) 
    for i, (origin, destination) in enumerate(goods_list[:-2]): # excluding default/outside good - consider changing this to remove "dropout_good" and "default_good"
        origin_base = origin.split("_")[0]
        destination_base = destination.split("_")[0]
        if origin_base != destination_base:
            # Traveling between vertiport
            capacity = route_dict.get((origin_base, destination_base), None)
        else:
            # Staying within a vertiport
            if origin.endswith('_arr'):
                origin_time = origin.replace('_arr', '')
                node = vertiport_data._node.get(origin_time)
                capacity = node.get('landing_capacity') - node.get('landing_usage') 
            elif destination.endswith('_dep'):
                destination_time = destination.replace('_dep', '')
                node = vertiport_data._node.get(destination_time)
                capacity = node.get('takeoff_capacity') - node.get('takeoff_usage') 
            else:
                node = vertiport_data._node.get(origin)
                capacity = node.get('hold_capacity') - node.get('hold_usage') 

        capacities[i] = capacity
    
    capacities[-2] = 100 # default/outside good
    capacities[-1] = 100 # dropout good

    return capacities
