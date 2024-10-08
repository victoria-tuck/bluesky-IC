import json
import random

DISCRETIZATION_STEP = 15

def is_vertiport(id):
    return (id == "V001") or (id == "V002") or (id == "V003") or (id == "V004")

def this_round(input_time):
    return round(input_time / DISCRETIZATION_STEP)

# Load the original JSON data from a file (for example)
with open("toulouse_case2.json", "r") as f:
    data = json.load(f)
flights = data["flights"]

empty_flights = []
# Iterate over the flights and modify the "requests" section
max_time = 0
for flight_id, flight_data in flights.items():
    requests = flight_data["requests"]
    if len(requests) == 0:
        empty_flights.append(flight_id)
        continue
    request_start = int(flight_data["appearance_time"])
    sampled_appearance_time = random.randint(max(request_start - 600, 0), max(request_start - 120, 0))
    rounded_appearance_time = this_round(sampled_appearance_time)
    
    # Create the sector_path and sector_times
    old_to_new_sectors = {"V001": "V001", "V002": "V002", "V003": "V003", "V004": "V004",
                          "V005": "S001", "V006": "S002", "V007": "S003", "V008": "S004", "V009": "S005"}
    sector_path = [old_to_new_sectors[sector] for sector in requests["001"]["destination_vertiport_id"]]
    # sector_path = [req["destination_vertiport_id"] for req in requests]
    # fixed_sector_path = [old_to_new_sectors[sector] for sector in sector_path]
    sector_times = [this_round(req_time) for req_time in [requests["001"]["request_departure_time"][0]] + requests["001"]["request_arrival_time"]]
    # sector_times = [this_round(int(req["request_departure_time"])) for req in requests]

    if len(requests) > 0:
        # sector_times.append(this_round(int(requests[-1]["request_arrival_time"])))  # Add the last arrival time
        destination = requests["001"]["destination_vertiport_id"][-1]
        if is_vertiport(destination):
            destination_vertiport_id = destination
            extended_sector_path = sector_path
            extended_sector_times = sector_times
        else:
            destination_vertiport_id = None
            extended_sector_path = sector_path + sector_path[:-1][::-1]
            sector_times_diff = [time1 - time2 for time1, time2 in zip(sector_times[1:], sector_times[:-1])][::-1]
            print(sector_times_diff)
            # sector_times_diff = (sector_times[:-1] - sector_times[1:]).reverse()
            # Adding 30 seconds in between outgoing and return trip
            extended_sector_times = sector_times[:-1] + [sector_times[-1] + sum(sector_times_diff[:ind+1]) + this_round(30) for ind in range(len(sector_times_diff))]
    else:
        destination_vertiport_id = None
        extended_sector_times = sector_times
        extended_sector_path = sector_path
    
    if len(sector_times) > 0:
        departure_time = extended_sector_times[0]
        arrival_time = extended_sector_times[-1]
        max_time = max(max_time, max(extended_sector_times))
    else:    
        departure_time = 0
        arrival_time = 0
    # Replace the requests section with the new format
    flight_data["appearance_time"] = rounded_appearance_time
    flight_data["budget_constraint"] = random.randint(1, 300)
    flight_data["requests"] = {"000": {
            "bid": 1,
            "valuation": 1,
            "request_departure_time": 0,
            "request_arrival_time": 0,
            "destination_vertiport_id": flight_data["origin_vertiport_id"]
        },
        "001": {
            "bid": random.randint(1, 300),
            "valuation": random.randint(1, 300),
            "sector_path": extended_sector_path,
            "sector_times": extended_sector_times,
            "destination_vertiport_id": destination_vertiport_id,
            "request_departure_time": departure_time,
            "request_arrival_time": arrival_time        
        }
    }
for flight_id in empty_flights:
    del flights[flight_id]
data["flights"] = flights
data["timing_info"] = {
    "start_time": 1,
    "end_time": max_time + 1,
    "time_step": 1,
    "auction_frequency": 20
}
data["congestion_params"] = {
    "lambda": 0.1,
    "C": {
        "V001": [
            0.0,
            0.2,
            0.6,
            1.2,
            2.0,
            3.0,
            4.2,
            5.6,
            7.2,
            9.0,
            11.0
        ]
    }
}

# Save the modified data back to a file (or use the data as needed)
with open("modified_toulouse_case.json", "w") as f:
    json.dump(data, f, indent=4)
