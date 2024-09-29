import json
import random

def is_vertiport(id):
    return (id == "V001") or (id == "V002") or (id == "V003") or (id == "V004")

# Load the original JSON data from a file (for example)
with open("toulouse_case.json", "r") as f:
    data = json.load(f)
flights = data["flights"]

# Iterate over the flights and modify the "requests" section
for flight_id, flight_data in flights.items():
    requests = flight_data["requests"]
    request_start = int(flight_data["appearance_time"])
    sampled_appearance_time = random.randint(max(request_start - 600, 0), max(request_start - 120, 0))
    
    # Create the sector_path and sector_times
    sector_path = [req["destination_vertiport_id"] for req in requests]
    sector_times = [int(req["request_departure_time"]) for req in requests]

    if len(requests) > 0:
        sector_times.append(int(requests[-1]["request_arrival_time"]))  # Add the last arrival time
        destination = requests[-1]["destination_vertiport_id"]
        if is_vertiport(requests[-1]["destination_vertiport_id"]):
            destination_vertiport_id = requests[-1]["destination_vertiport_id"]
        else:
            destination_vertiport_id = None
    else:
        destination_vertiport_id = None
    
    # Replace the requests section with the new format
    flight_data["appearance_time"] = sampled_appearance_time
    flight_data["requests"] = {
        "bid": 1,
        "valuation": 1,
        "sector_path": sector_path,
        "sector_times": sector_times,
        "destination_vertiport_id": destination_vertiport_id
    }
data["flights"] = flights

# Save the modified data back to a file (or use the data as needed)
with open("modified_toulouse_case.json", "w") as f:
    json.dump(data, f, indent=4)
