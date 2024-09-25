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

logging.basicConfig(filename='solver_log.txt', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Add the bluesky package to the path
top_level_path = Path(__file__).resolve().parent.parent
print(str(top_level_path))
sys.path.append(str(top_level_path))

from VertiportStatus import VertiportStatus
from fisher.sampling_graph import build_edge_information, agent_probability_graph_extended, sample_path, plot_sample_path_extended, process_allocations, mapping_agent_to_full_data, mapping_goods_from_allocation
from fisher.fisher_int_optimization import int_optimization
from fisher.fisher_market import build_graph, construct_market, run_market
from write_csv import write_output, save_data

# TOL_ERROR = 1e-2
MAX_NUM_ITERATIONS = 5000
# BETA = 1
# dropout_good_valuation = -1
# default_good_valuation = 1
# price_default_good = 10


def fisher_allocation_and_payment(vertiport_usage, flights, timing_info, routes_data, vertiports, 
                                  output_folder=None, save_file=None, initial_allocation=True, design_parameters=None):

    market_auction_time=timing_info["start_time"]
    # Build Fisher Graph
    market_graph = build_graph(vertiport_usage, timing_info)

    #Extracting design parameters
    if design_parameters:
        price_default_good = design_parameters["price_default_good"]
        default_good_valuation = design_parameters["default_good_valuation"]
        dropout_good_valuation = design_parameters["dropout_good_valuation"]
        BETA = design_parameters["beta"]
        rebate_frequency = design_parameters["rebate_frequency"]
    else:
        BETA = 1 # chante to 1/T
        dropout_good_valuation = -1
        default_good_valuation = 1
        price_default_good = 10
        rebate_frequency = 1

    # Construct market
    agent_information, market_information, bookkeeping = construct_market(market_graph, flights, timing_info, routes_data, vertiport_usage, 
                                                                          default_good_valuation=default_good_valuation, 
                                                                          dropout_good_valuation=dropout_good_valuation, BETA=BETA)

    # Run market
    goods_list, times_list = bookkeeping
    num_goods, num_agents = len(goods_list), len(flights)
    u, agent_constraints, agent_goods_lists = agent_information
    # y = np.random.rand(num_agents, num_goods-2)*10
    y = np.zeros((num_agents, num_goods))
    desired_goods = track_desired_goods(flights, goods_list)
    for i, agent_ids in enumerate(desired_goods):
        dept_id = desired_goods[agent_ids]["desired_good_arr"]
        arr_id = desired_goods[agent_ids]["desired_good_dep"]
        y[i][dept_id]= 1
        y[i][arr_id] = 1
    # y = np.random.rand(num_agents, num_goods)
    p = np.zeros(num_goods)
    p[-2] = price_default_good 
    p[-1] = 0 # dropout good
    r = [np.zeros(len(agent_constraints[i][1])) for i in range(num_agents)]
    # x, p, r, overdemand = run_market((y,p,r), agent_information, market_information, bookkeeping, plotting=True, rational=False)
    x, prices, r, overdemand, agent_constraints, adjusted_budgets, data_to_plot = run_market((y,p,r), agent_information, market_information, 
                                                             bookkeeping, rational=False, price_default_good=price_default_good, 
                                                             rebate_frequency=rebate_frequency)
    
    extra_data = {
        'x': x,
        'prices': prices,
        'r': r,
        'agent_constraints': agent_constraints,
        'adjusted_budgets': adjusted_budgets,
        'data_to_plot': data_to_plot
    }
    save_data(output_folder, "fisher_data", market_auction_time, **extra_data)
    data_to_plot.append(desired_goods)
    plotting_market(data_to_plot, output_folder, market_auction_time)
    
    # Building edge information for mapping - move this to separate function
    # move this part to a different function
    edge_information = build_edge_information(goods_list)
    agent_allocations, agent_dropout_x, agent_indices, agent_edge_information = process_allocations(x, edge_information, agent_goods_lists)

    _ , capacity, _ = market_information
    int_allocations = []
    utilities = []
    dropouts = [] # remove
    agents_data = {}
    allocated_flights = [] #remove
    budgets = [] #remove
    constraints = [] #remove
    agents_indices = [] #remove
    int_allocations_full = []
    start_time_sample = time.time()
    flights_list = list(flights.keys())
    print("Sampling edges ...")
    for i, flight_id in enumerate(flights_list):
        frac_allocations = agent_allocations[i]
        start_node= list(agent_edge_information[i].values())[0][0]
        extended_graph, agent_allocation = agent_probability_graph_extended(agent_edge_information[i], frac_allocations, flight_id, output_folder)
        sampled_path_extended, sampled_edges, int_allocation, dropout_flag = sample_path(extended_graph, start_node, agent_allocation, agent_dropout_x[i])
        if dropout_flag:
            dropouts.append(flight_id) #remove
            agents_data[flight_id] = {'status': 'dropped', "payment":  0}
            agents_data[flight_id]['good_allocated'] = None
            agents_data[flight_id]['request'] = None
        else:
            # print("Sampled Path:", sampled_path_extended)
            # print("Sampled Edges:", sampled_edges)
            # plot_sample_path_extended(extended_graph, sampled_path_extended, agent_number, output_folder)
            allocated_flights.append(flight_id) # remove 
            agents_data[flight_id] = {'status': 'int_allocated'}
            agents_data[flight_id]['int_allocation'] = int_allocation
            int_allocations.append(int_allocation)
            int_allocation_full = mapping_agent_to_full_data(edge_information, sampled_edges)
            int_allocations_full.append(int_allocation_full)           
            utilities.append(u[i]) # removing default and dropout good
            budgets.append(adjusted_budgets[i]) #remove
            constraints.append(agent_constraints[i]) #remove
            agents_indices.append(agent_indices) #remove

        agents_data[flight_id]['utility'] = u[i]
        agents_data[flight_id]['constraints'] = agent_constraints[i]
        agents_data[flight_id]['adjusted_budget'] = adjusted_budgets[i]
        agents_data[flight_id]['agent_edge_indices'] = agent_indices[i]
        agents_data[flight_id]['fisher_allocation'] = agent_allocations[i]
        agents_data[flight_id]['agent_goods_list'] = agent_goods_lists[i]
        agents_data[flight_id]['deconficted_goods'] = None

    if int_allocations_full:
        int_allocations_full = np.array(int_allocations_full)
        print(f"Time to sample: {time.time() - start_time_sample:.5f}")
        # IOP for contested goods
        
        capacity = capacity[:-2] # removing default and dropout good
        prices = prices[:-2] # removing default and dropout good
        new_allocations, new_prices = int_optimization(int_allocations_full, capacity, budgets, prices, utilities, constraints, agents_indices, int_allocations, output_folder)

        # if new_allocations.any():
        payment = np.sum(new_prices * new_allocations, axis=1)
        end_capacity = capacity - np.sum(new_allocations, axis=0)
        new_allocations_goods = mapping_goods_from_allocation(new_allocations, goods_list)
        # Allocation and Rebased
        allocation = []
        rebased = []
        index = 0
        for key, value in agents_data.items():
            if agents_data[key]['status'] == 'int_allocated':
                agents_data[key]["deconflicted_goods"] = new_allocations_goods[index]
                dep_node, arrival_node = find_dep_and_arrival_nodes(new_allocations_goods[index])
                if dep_node:
                    good = (dep_node, arrival_node)
                    allocation.append((key, good)) #remove
                    agents_data[key]["status"] = "allocated"
                    agents_data[key]['payment'] = payment[index]
                    agents_data[key]["request"] = "001"
                    agents_data[key]["good_allocated"] =  good
                    index += 1
                else:
                    rebased.append((key, '000')) #remove
                    agents_data[key]['payment'] = 0
                    agents_data[key]["request"] = "000"
                    agents_data[key]["good_allocated"] =  None

    else: # if there are only dropouts
        payment = np.zeros(len(flights))
        end_capacity = capacity
        new_prices = prices
        allocation = None
        rebased = None
        new_allocations_goods = None

    # for i, flight_id in enumerate(allocated_flights):
    #     dep_node, arrival_node = find_dep_and_arrival_nodes(new_allocations_goods[i])
    #     if dep_node:
    #         good = (dep_node, arrival_node)
    #         allocation.append((flight_id, good))
    #         agents_data["status"] = "allocated"
    #     else:
    #         rebased.append((flight_id, '000'))
    #         agents_data["status"] = "rebased"
    # else:
    #     payment = 0
    #     end_capacity = capacity
    #     allocation = None
    #     rebased = None
    #     new_allocations_goods = None
    # end_agent_status_data = (allocation, rebased, dropouts) #change

    write_output(flights, edge_information, prices, new_prices, capacity, end_capacity, 
                agents_data, market_auction_time, output_folder)
    
    return allocation, rebased, None


def plotting_market(data_to_plot, output_folder, market_auction_time=None):
    x_iter, prices, p, overdemand, error, abs_error, rebates, agent_allocations, market_clearing, agent_constraints, yplot, social_welfare, desired_goods = data_to_plot
    def plt_helper(figsize, data, labels, filename):
        plt.figure(figsize=figsize)
        if callable(data[0]):
            data[0](*data[1])
        else:
            plt.plot(data[0], data[1])
        plt.legend()
        plt.xlabel(labels["x"])
        plt.ylabel(labels["y"])
        plt.title(labels["title"])
        plt.savefig(filename)
        plt.close()
    
    def get_filename(base_name):
        if market_auction_time:
            return f"{output_folder}/{base_name}_a{market_auction_time}.png"
        else:
            return f"{output_folder}/{base_name}.png"
    
    # Plot Price evolution
    def price_evol_plt(p, x_iter, prices):
        for good_index in range(len(p) - 2):
            plt.plot(range(1, x_iter + 1), [prices[i][good_index] for i in range(len(prices))])
            plt.plot(range(1, x_iter + 1), [prices[i][-2] for i in range(len(prices))], 'b--', label="Default Good")
            plt.plot(range(1, x_iter + 1), [prices[i][-1] for i in range(len(prices))], 'r-.', label="Dropout Good")
    price_evol_labels = {"x": 'x_iter', "y": 'Prices', "title": 'Price evolution'}
    plt_helper((10, 5), (price_evol_plt, (p, x_iter, prices)), price_evol_labels, get_filename("price_evolution"))

    # Plot Overdemand evolution
    overdemand_labels = {"x": 'x_iter', "y": 'Demand - Supply', "title": 'Overdemand evolution'}
    plt_helper((10, 5), (range(1, x_iter + 1), overdemand), overdemand_labels, get_filename("overdemand_evolution"))

    # Plot Constraint error evolution
    def constraint_error_plt(error, x_iter, agent_constraints):
        for agent_index in range(len(agent_constraints)):
            plt.plot(range(1, x_iter + 1), error[agent_index])
    constraint_error_labels = {"x": 'x_iter', "y": 'Constraint error', "title": 'Constraint error evolution $\sum (linear constraints)^2$'}
    plt_helper((10, 5), (constraint_error_plt, (error, x_iter, agent_constraints)), constraint_error_labels, get_filename("constraint_error_evolution"))

    # Plot Absolute error evolution
    def abs_error_plt(abs_error, x_iter, agent_constraints):
        for agent_index in range(len(agent_constraints)):
            plt.plot(range(1, x_iter + 1), abs_error[agent_index])
    abs_error_labels = {"x": 'x_iter', "y": 'Constraint error', "title": 'Absolute error evolution $\sum (x_i - y_i)^2$'}
    plt_helper((10, 5), (abs_error_plt, (abs_error, x_iter, agent_constraints)), abs_error_labels, get_filename("absolute_error_evolution"))

    # Plot Rebate evolution
    def rebate_plt(rebates, x_iter):
        for constraint_index in range(len(rebates[0])):
            plt.plot(range(1, x_iter + 1), [rebates[i][constraint_index] for i in range(len(rebates))])
    rebate_labels = {"x": 'x_iter', "y": 'Rebate', "title": 'Rebate evolution'}
    plt_helper((10, 5), (rebate_plt, (rebates, x_iter)), rebate_labels, get_filename("rebate_evolution"))

    # Agent allocation evolution
    # plt.figure(figsize=(10, 5))
    # for agent_index in range(len(agent_allocations[0])):
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][:-2] for i in range(len(agent_allocations))])
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][-2] for i in range(len(agent_allocations))], 'b--', label=f"{agent_index} - Default Good")
    #     plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_index][-1] for i in range(len(agent_allocations))], 'r-.', label=f"{agent_index} - Dropout Good")
    # plt.legend()
    # plt.xlabel('x_iter')
    # plt.title("Agent allocation evolution")
    # plt.savefig(get_filename("agent_allocation_evolution"))
    # plt.close()

    # Plot Desired goods evolution
    def desired_goods_plt(agent_allocations, x_iter, desired_goods):
        for agent in enumerate(desired_goods):
            agent_id = agent[0]
            agent_name = agent[1]       
            dep_index = desired_goods[agent_name]["desired_good_dep"]
            arr_index = desired_goods[agent_name]["desired_good_arr"]
            plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_id][dep_index] for i in range(len(agent_allocations))], '--', label=f"{agent_name}_dep good")
            plt.plot(range(1, x_iter + 1), [agent_allocations[i][agent_id][arr_index] for i in range(len(agent_allocations))], '-', label=f"{agent_name}_arr good")
    desired_goods_labels = {"x": 'x_iter', "y": 'Allocation', "title": 'Desired Goods Agent allocation evolution'}
    plt_helper((10, 5), (desired_goods_plt, (agent_allocations, x_iter, desired_goods)), desired_goods_labels, get_filename("desired_goods_allocation_evolution"))

    # Plot Market Clearing Error
    market_clearing_labels = {"x": 'x_iter', "y": 'Market Clearing Error', "title": 'Market Clearing Error'}
    plt_helper((10, 5), (range(1, x_iter + 1), market_clearing), market_clearing_labels, get_filename("market_clearing_error"))

    # Plot y
    def y_plt(yplot, x_iter):
        for agent_index in range(len(yplot[0])):
            plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][:-2] for i in range(len(yplot))])
            plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][-2] for i in range(len(yplot))], 'b--', label="Default Good")
            plt.plot(range(1, x_iter + 1), [yplot[i][agent_index][-1] for i in range(len(yplot))], 'r-.', label="Dropout Good")
    y_labels = {"x": 'x_iter', "y": 'y', "title": 'y evolution'}
    plt_helper((10, 5), (y_plt, (yplot, x_iter)), y_labels, get_filename("y_evolution"))

    # Plot Social Welfare
    social_welfare_labels = {"x": 'x_iter', "y": 'Social Welfare', "title": 'Social Welfare'}
    plt_helper((10, 5), (range(1, x_iter + 1), social_welfare), social_welfare_labels, get_filename("social_welfare"))


def load_json(file=None):
    """
    Load a case file for a fisher market test case from a JSON file.
    """
    if file is None:
        return None
    assert Path(file).is_file(), f"File {file} does not exist."

    # Load the JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"Opened file {file}")
    return data


def find_dep_and_arrival_nodes(edges):
    dep_node_found = False
    arrival_node_found = False
    
    for edge in edges:
        if "dep" in edge[0]:
            dep_node_found = edge[0]
            arrival_node_found = edge[1]
            assert "arr" in arrival_node_found, f"Arrival node not found: {arrival_node_found}"
            return dep_node_found, arrival_node_found
    
    return dep_node_found, arrival_node_found


def track_desired_goods(flights, goods_list):
    "return the index of the desired goods for each flight"
    desired_goods = {}
    for i, flight_id in enumerate(flights.keys()):
        origin_vertiport = flights[flight_id]["origin_vertiport_id"]
        desired_dep_time = flights[flight_id]["requests"]["001"]["request_departure_time"]
        desired_vertiport = flights[flight_id]["requests"]["001"]["destination_vertiport_id"]
        desired_arrival_time = flights[flight_id]["requests"]["001"]["request_arrival_time"]
        desired_good_arr = (f"{origin_vertiport}_{desired_dep_time}", f"{origin_vertiport}_{desired_dep_time}_dep")
        desired_good_dep = (f"{desired_vertiport}_{desired_arrival_time}_arr", f"{desired_vertiport}_{desired_arrival_time}")
        good_id_arr = goods_list.index(desired_good_arr)
        good_id_dep = goods_list.index(desired_good_dep)
        desired_goods[flight_id] = {"desired_good_arr": good_id_arr, "desired_good_dep": good_id_dep}

    return desired_goods


if __name__ == "__main__":
    pass
