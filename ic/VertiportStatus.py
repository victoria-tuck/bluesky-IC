import networkx as nx
import matplotlib.pyplot as plt



class VertiportStatus(nx.DiGraph):
    """
    Class for storing the status of vertiports.
    """
    
    def __init__(self, vertiports, edges, timing, flights, data=None, **attr):
        """
        Create time-extended vertiport graph.

        Args:
            vertiports (dictionary): Dictionary of vertiports with ids as keys containing 
                latitude, longitude, landing_capacity, takeoff_capacity, and hold_capacity.
            edges (list): (or routes) List of edges where an edge has an origin_vertiport_id and destination_vertport_id.
            timing (dictionary): Dictionary of timing information for the simulation including
                start_time, end_time, and time_step.
        """
        super().__init__(data, **attr)
        # self.time_steps = list(range(timing["start_time"], timing["end_time"] + timing["time_step"], timing["time_step"]))
        self.vertiports = vertiports
        self.ts = timing["time_step"]
        self.dissapearance_ts = timing["dissapear_ts"]
        self.route_info = edges

        # Create time extended graph of vertiports
        for key, value in flights.items():
            agent_appareance_time = value["appearance_time"]
            agent_desired_departure_time = value["requests"]["001"]["request_departure_time"]
            agent_desired_arrival_time = value["requests"]["001"]["request_arrival_time"]
            agent_disappearance_time = agent_desired_arrival_time + self.dissapearance_ts
            #change the hardcoded 3 to a random value
            time_steps = list(range(agent_appareance_time, agent_disappearance_time, self.ts))
            origin_vertiport = value["origin_vertiport_id"]
            for step in time_steps:
                # for all the parked vertiports in orginal vertiport
                time_extended_vertiport_id = origin_vertiport + "_" + str(step) 
                self.add_node(time_extended_vertiport_id, **vertiports[origin_vertiport])                
                self.add_node_attributes(time_extended_vertiport_id, origin_vertiport, step)
                self.create_route_edges(origin_vertiport, origin_vertiport, step, step + 1)


            # for all the destination vertiports
            destination_vertiport = value["requests"]["001"]["destination_vertiport_id"]
            for dest_step in range(agent_desired_arrival_time, agent_disappearance_time, self.ts):
                destination_vertiport = value["requests"]["001"]["destination_vertiport_id"]
                time_extended_dest_vertiport_id = destination_vertiport + "_" + str(dest_step) 
                self.add_node(time_extended_dest_vertiport_id, **vertiports[destination_vertiport])  
                self.add_node_attributes(time_extended_dest_vertiport_id, destination_vertiport, dest_step)
                self.create_route_edges(destination_vertiport, destination_vertiport, 
                                        dest_step, dest_step + 1)
        

    def create_route_edges(self, origin_vertiport, destination_vertiport, dept_time, arr_time):
        # Define the time-extended edge
        edge_start = f"{origin_vertiport}_{dept_time}"
        edge_end = f"{destination_vertiport}_{arr_time}"
    
        # Check if the edge already exists in the graph
        if not self.has_edge(edge_start, edge_end):
            # Add the edge if it doesn't exist
            self.add_edge(edge_start, edge_end)
        # time_steps = list(range(timing["start_time"], timing["end_time"] + timing["time_step"], timing["time_step"]))
        # for step in time_steps:
        #     for edge in edges:
        #         arrival_time = step + edge["travel_time"]
        #         if arrival_time > timing["end_time"]:
        #             continue
        #         assert arrival_time in time_steps, f"Timing setup incorrect. Arrival time {arrival_time} not in time steps."
        #         time_extended_start = edge["origin_vertiport_id"] + "_" + str(step)
        #         time_extended_end = edge["destination_vertiport_id"] + "_" + str(arrival_time)
        #         self.add_edge(time_extended_start, time_extended_end)



    def add_node_attributes(self, time_extended_vertiport_id, vertiport_id, step):
        self.nodes[time_extended_vertiport_id]["landing_usage"] = 0
        self.nodes[time_extended_vertiport_id]["takeoff_usage"] = 0
        self.nodes[time_extended_vertiport_id]["hold_usage"] = 0
        self.nodes[time_extended_vertiport_id]["time"] = step
        self.nodes[time_extended_vertiport_id]["landing_capacity"] = self.vertiports[vertiport_id]["landing_capacity"]
        self.nodes[time_extended_vertiport_id]["takeoff_capacity"] = self.vertiports[vertiport_id]["takeoff_capacity"]
        self.nodes[time_extended_vertiport_id]["hold_capacity"] = self.vertiports[vertiport_id]["hold_capacity"]
        self.nodes[time_extended_vertiport_id]["vertiport_id"] = vertiport_id   

    def add_aircraft(self, flights):
        """
        Add starting vertiport usage information.

        Args:
            flights (list): List of flights with at least origin_vertiport_id information.
        """
        for flight_id, flight in flights.items():
            start_vertiport = flight["origin_vertiport_id"]
            start_time = flight["appearance_time"]
            end_time = flight["requests"]["001"]["request_arrival_time"] + self.dissapearance_ts

            time_steps = list(range(start_time, end_time, self.ts))
            # This might need to be changed to only hold the capacity in the current simulation time
            for time in time_steps:
                time_extended_start = start_vertiport + "_" + str(time)
                self.nodes[time_extended_start]["hold_usage"] += 1
                assert self.nodes[time_extended_start]["hold_usage"] <= self.nodes[time_extended_start]["hold_capacity"], \
                    f"Vertiport {start_vertiport} at time {time} over capacity."  


    def move_aircraft(self, origin_vertiport, request):
        """
        Move aircraft from one vertiport to another.

        Args:
            origin_vertiport (str): Vertiport id where the aircraft is coming from.
            request (dict): Dictionary containing information about the aircraft movement request
                including destination_vertiport_id, request_departure_time, and request_arrival_time.
        """
        destination_vertiport = request["destination_vertiport_id"]
        departure_time = request["request_departure_time"]
        arrival_time = request["request_arrival_time"]

        # Update the hold usage of the origin and destination vertiports
        for time in self.time_steps:
            # Move the aircraft from the origin vertiport
            if time < departure_time:
                continue
            time_extended_origin = origin_vertiport + "_" + str(time)
            self.nodes[time_extended_origin]["hold_usage"] -= 1
            assert self.nodes[time_extended_origin]["hold_usage"] >= 0, \
                f"Vertiport {origin_vertiport} at time {time} has negative hold usage."
            
            # Add the aircraft to the destination vertiport
            if time >= arrival_time:
                time_extended_destination = destination_vertiport + "_" + str(time)
                self.nodes[time_extended_destination]["hold_usage"] += 1
                assert self.nodes[time_extended_destination]["hold_usage"] >= 0, \
                    f"Vertiport {destination_vertiport} at time {time} has negative hold usage."
        
        # Add the aircrafts takeoff and landing usage
        time_extended_origin = origin_vertiport + "_" + str(departure_time)
        time_extended_destination = destination_vertiport + "_" + str(arrival_time)
        self.nodes[time_extended_origin]["takeoff_usage"] += 1
        self.nodes[time_extended_destination]["landing_usage"] += 1


def draw_graph(graph):
    """
        Draw networkx graph.

        Args:
            graph (nx.Graph): Graph to draw.
        """
    # Todo: This could be significantly improved
    # Draw the graph
    pos = nx.shell_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')

    # Show the plot
    plt.show()