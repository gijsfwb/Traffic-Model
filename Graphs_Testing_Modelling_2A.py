import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats


###### RUN THE SIMULATION WITHOUT THE SPEED INCREASE
node_names = [
             "Groningen",
             "Begin Afsluitdijk",
             "Eind Afsluitdijk",
             "Emmeloord",
             "Lelystad",
             "Almere",
             "Utrecht",
             "Amsterdam",
             "Den haag",
             ]
#The network is built of a list of nodes (cities) and a list of roads
#Each node contains a list with the indices of the roads leaving that node
nodes = [[0,1,2],[3],[5],[4],[6],[7,8],[10],[9],[]]
#Each road has a startnode, endnode, speed, lane number and length
#In addition to a (currently unused) capacity multiplier
startnodes = [0,0,0,1,3,2,4,5,5,6,7]
endnodes = [1,3,6,2,4,7,5,7,6,8,8]
speeds = [100,100,100,100,100,100,100,100,100,100,100]
#2 Lanes per road seems like the most accurate choice, all roads are mostly 2
#lanes but have segments with more. Area of improvement would be to allow variable
#lane amounts (by splitting roads perhaps)
lanes = [2,2,2,2,2,2,2,2,2,2,2]
lengths = [100e3,110e3,191e3,29e3,15e3,85e3,23e3,46e3,39e3,36e3,60e3]
capacity_multipliers = [1,1,1,1,1,1,1,1,1,1,1]
#Initializing lists
roads = []
cars = []
roads = []
choice_vals = []
#Variables that determine capacity of a road
#and the travel time on a road based on the occupancy
car_length = 4.5
d_spacing = 55
t = 0
alpha = 0.15
beta = 4
#Time for the roads to fill up before we start considering the data
cutoff_time = 480
def generate_paths(node, path):
    #Generalized function to generate all paths from a given node to the end
    #of the network
    #Not necessary for this small network but might be useful when considering
    #a large one
    paths = []
    
    # If the node is the last node (i.e., it has no outgoing edges), return the current path
    if node >= len(nodes) or not nodes[node]:
        return [path]
    
    # Explore each outgoing edge
    for index in nodes[node]:
        edge = roads[index]
        new_path = path + [index]  # Create a new path with the current edge
        paths.extend(generate_paths(edge.endnode, new_path))  # Collect paths recursively

    return paths
class road:
    #The road objects hold all relevant information for each road
    def __init__(self,startnode, endnode,length,max_speed,n_lanes,capacity_multiplier):
        self.startnode = startnode
        self.endnode = endnode
        self.length = length # m
        self.max_speed = max_speed # km/h
        self.n_lanes = n_lanes
        self.freeflow_time = length/max_speed * 60/1000 #conversion factor so result is in minutes
        self.travel_time = self.freeflow_time
        self.cars_on_road = 0
        self.capacity = int(n_lanes*length/(car_length+d_spacing)*capacity_multiplier)
        self.total_cars = 0
        self.occupancy = []
        self.paths = []
class car:
    #The car objects hold the relevant information for each car
    def __init__(self,position):
        self.position = position
        self.finished = False
        self.time_on_road = 0
        self.time_to_reach_node = 0
        self.total_time = 0
        self.roads_taken = [0]
for i in range(len(startnodes)):
    #Creating the roads
    roads.append(road(startnodes[i],endnodes[i],lengths[i],speeds[i],lanes[i],capacity_multipliers[i]))
for rd in roads:
    #For each road, we can generate every possible path that can be taken
    #when exited
    rd.paths = generate_paths(rd.endnode,[])
#Startpaths and nodepaths hold the paths from Groningen to Den Haag in terms of
#roads taken and nodes taken respectively
startpaths = generate_paths(0,[])
nodepaths = []
for i in range(len(startpaths)):
    nodepaths.append([0])
    for road in startpaths[i]:
        nodepaths[i].append(roads[road].endnode)

#Timestep loop:
while t<960:
    t+=1
    print(t)
    #Adding cars to the network
    #Area of improvement would be a more realistic traffic inflow, perhaps
    #with specific inflow and outflow for individual roads
    numcars = int(np.random.normal(95,1))
    for i in range(numcars):
        cars.append(car(0))
        
    for rd in roads:
    #In each timestep, we calculate the travel time of a road instead of doing
    #so for every car. This is much faster and gives almost indistinguishable
    #results
        rd.travel_time = rd.freeflow_time*(1+alpha*pow(rd.cars_on_road/rd.capacity,beta))
        rd.occupancy.append(rd.cars_on_road/rd.capacity)
    #We keep track of the total number of cars for relevant plotting variables
        if t>cutoff_time:
            rd.total_cars += rd.cars_on_road
    if t<=cutoff_time:
        cars_before_cutoff = len(cars)
        
    #Main body of the timestep loop, where we go over each car and move it
    #Through the network if needed
    for vehicle in cars:
        if not vehicle.finished:
            vehicle.total_time += 1
            vehicle.time_on_road += 1

            #if vehicle has reached the end of the road, move it to a new road (if possible)
            if vehicle.time_on_road+1 > vehicle.time_to_reach_node:
                if vehicle.position == 0:
                    paths = startpaths
                else:
                    #Determining the possible paths through the network from
                    #Here and their corresponding travel times
                    paths = vehicle.position.paths
                    if vehicle.position.endnode == 8:
                        vehicle.roads_taken.append(vehicle.position.endnode)
                        vehicle.position.cars_on_road -= 1
                        vehicle.position = 8
                        vehicle.finished = True
                        continue
                attractiveness = []      
                #Calculating the attractiveness of each path
                for path in paths:
                    total_time = 0
                    if roads[path[0]].cars_on_road >= roads[path[0]].capacity:
                        total_time = -1
                        continue
                    for index in path:
                        total_time += roads[index].travel_time
                    if total_time != -1:
                        attractiveness.append(1/total_time)
                    else:
                        attractiveness.append(0)
                p_choice = []
                normalize_attractiveness = sum(attractiveness)
                #Picking a road probabilistically based on the attractiveness
                #If no roads are available, it tries again next timestep
                if normalize_attractiveness != 0:
                    for i in range(len(attractiveness)):
                        p_choice.append(attractiveness[i]/normalize_attractiveness)
                    choice_val = np.random.random()
                    for i in range(len(p_choice)):
                        if choice_val < sum(p_choice[0:i+1]):
                            if vehicle.position != 0:
                                vehicle.position.cars_on_road -= 1
                                vehicle.roads_taken.append(vehicle.position.endnode)
                            vehicle.time_on_road = 0
                            vehicle.position = roads[paths[i][0]]
                            vehicle.position.cars_on_road += 1
                            vehicle.time_to_reach_node = vehicle.position.travel_time + np.random.normal(0,2)
                            break
                        
#Create a list with the respective finish times of every possible path
finished = []
travel_time = []
paths = []
for cr in cars:
    finished.append(cr.finished)
    travel_time.append([cr.total_time,cr.finished])
    paths.append(cr.roads_taken)
    
finish_times = []
for i in range(len(nodepaths)):
    finish_times.append([])
for pathindex in range(len(nodepaths)):   
    for i in range(cars_before_cutoff-1,len(cars)):
        if finished[i]:
            if cars[i].roads_taken == nodepaths[pathindex]:
                finish_times[pathindex].append(cars[i].total_time)


###### RUN THE SIMULATION WITH THE SPEED INCREASE

#Same code as before until line 360 (without the comments), most convenient way
#to do the comparison
node_names_new = [
    "Groningen",
    "Begin Afsluitdijk",
    "Eind Afsluitdijk",
    "Emmeloord",
    "Lelystad",
    "Almere",
    "Utrecht",
    "Amsterdam",
    "Den haag",
]
nodes_new = [[0, 1, 2], [3], [5], [4], [6], [7, 8], [10], [9], []]
startnodes_new = [0, 0, 0, 1, 3, 2, 4, 5, 5, 6, 7]
endnodes_new = [1, 3, 6, 2, 4, 7, 5, 7, 6, 8, 8]
speeds_new = [100, 100, 100, 130, 130, 100, 100, 100, 100, 100, 100]
lanes_new = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
lengths_new = [100e3, 110e3, 191e3, 29e3, 15e3, 85e3, 23e3, 46e3, 39e3, 36e3, 60e3]
capacity_multipliers_new = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

roads_new = []
cars_new = []
car_length_new = 4.5
d_spacing_new = 55
time_new = 0
alpha_new = 0.15
beta_new = 4
cutoff_time_new = 480

# Function to generate paths
def generate_paths_new(node, path):
    paths_new = []
    if node >= len(nodes_new) or not nodes_new[node]:
        return [path]
    for index in nodes_new[node]:
        edge_new = roads_new[index]
        new_path_new = path + [index]
        paths_new.extend(generate_paths_new(edge_new.endnode, new_path_new))
    return paths_new

# Road class definition
class RoadNew:
    def __init__(self, startnode, endnode, length, max_speed, n_lanes, capacity_multiplier):
        self.startnode = startnode
        self.endnode = endnode
        self.length = length
        self.max_speed = max_speed
        self.n_lanes = n_lanes
        self.freeflow_time = length / max_speed * 60 / 1000
        self.travel_time = self.freeflow_time
        self.cars_on_road = 0
        self.capacity = int(n_lanes * length / (car_length_new + d_spacing_new) * capacity_multiplier)
        self.total_cars = 0
        self.occupancy = []
        self.paths = []

# Car class definition
class CarNew:
    def __init__(self, position):
        self.position = position
        self.finished = False
        self.time_on_road = 0
        self.time_to_reach_node = 0
        self.total_time = 0
        self.roads_taken = [0]

# Initialize roads
for i in range(len(startnodes_new)):
    roads_new.append(RoadNew(startnodes_new[i], endnodes_new[i], lengths_new[i], speeds_new[i], lanes_new[i], capacity_multipliers_new[i]))

for rd in roads_new:
    rd.paths = generate_paths_new(rd.endnode, [])

startpaths_new = generate_paths_new(0, [])
nodepaths_new = []
for i in range(len(startpaths_new)):
    nodepaths_new.append([0])
    for road in startpaths_new[i]:
        nodepaths_new[i].append(roads_new[road].endnode)

# Timestep loop
while time_new < 960:
    time_new += 1
    numcars_new = int(np.random.normal(85, 10))
    for i in range(numcars_new):
        cars_new.append(CarNew(0))
    for rd in roads_new:
        rd.travel_time = rd.freeflow_time * (1 + alpha_new * pow(rd.cars_on_road / rd.capacity, beta_new))
        rd.occupancy.append(rd.cars_on_road / rd.capacity)
        if time_new > cutoff_time_new:
            rd.total_cars += rd.cars_on_road
    if time_new <= cutoff_time_new:
        cars_before_cutoff_new = len(cars_new)
    for vehicle in cars_new:
        if not vehicle.finished:
            vehicle.total_time += 1
            vehicle.time_on_road += 1
            if vehicle.time_on_road + 1 > vehicle.time_to_reach_node:
                if vehicle.position == 0:
                    paths_new = startpaths_new
                else:
                    paths_new = vehicle.position.paths
                    if vehicle.position.endnode == 8:
                        vehicle.roads_taken.append(vehicle.position.endnode)
                        vehicle.position.cars_on_road -= 1
                        vehicle.position = 8
                        vehicle.finished = True
                        continue
                attractiveness_new = []
                for path in paths_new:
                    total_time_new = 0
                    if roads_new[path[0]].cars_on_road >= roads_new[path[0]].capacity:
                        total_time_new = -1
                        continue
                    for index in path:
                        total_time_new += roads_new[index].travel_time
                    if total_time_new != -1:
                        attractiveness_new.append(1 / total_time_new)
                    else:
                        attractiveness_new.append(0)
                p_choice_new = []
                normalize_attractiveness_new = sum(attractiveness_new)
                if normalize_attractiveness_new != 0:
                    for i in range(len(attractiveness_new)):
                        p_choice_new.append(attractiveness_new[i] / normalize_attractiveness_new)
                    choice_val_new = np.random.random()
                    for i in range(len(p_choice_new)):
                        if choice_val_new < sum(p_choice_new[0:i + 1]):
                            if vehicle.position != 0:
                                vehicle.position.cars_on_road -= 1
                                vehicle.roads_taken.append(vehicle.position.endnode)
                            vehicle.time_on_road = 0
                            vehicle.position = roads_new[paths_new[i][0]]
                            vehicle.position.cars_on_road += 1
                            vehicle.time_to_reach_node = vehicle.position.travel_time + np.random.normal(0, 2)
                            break

finished_new = []
travel_time_new = []
paths_new = []
for cr in cars_new:
    finished_new.append(cr.finished)
    travel_time_new.append([cr.total_time, cr.finished])
    paths_new.append(cr.roads_taken)

finish_times_new = []
for i in range(len(nodepaths_new)):
    finish_times_new.append([])
for pathindex in range(len(nodepaths_new)):
    for i in range(cars_before_cutoff_new - 1, len(cars_new)):
        if finished_new[i]:
            if cars_new[i].roads_taken == nodepaths_new[pathindex]:
                finish_times_new[pathindex].append(cars_new[i].total_time)

# Code for Fig. 4.2
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14, 6))

# Plot occupancy before the speed increase
for rd in roads:
    route_label = f"{node_names[rd.startnode]} to {node_names[rd.endnode]}"
    ax1.plot(rd.occupancy, label=route_label)
ax1.set_title('Occupancy of Roads Before Speed Increase')
ax1.set_ylabel('Occupancy')

# Plot occupancy after the speed increase
for rd_new in roads_new:
    route_label_new = f"{node_names_new[rd_new.startnode]} to {node_names_new[rd_new.endnode]}"
    ax2.plot(rd_new.occupancy, label=route_label_new)
ax2.set_title('Occupancy of Roads After Speed Increase')
ax2.set_xlabel('Time')
ax2.set_ylabel('Occupancy')

# Create a single legend for both plots below the figure
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4)

# General layout adjustments
plt.tight_layout()
plt.show()

#Code voor fig. 4.3
# Flatten all travel time data into single arrays for GMM fitting
all_times_new = np.concatenate(finish_times_new)
all_times = np.concatenate(finish_times)

# Fit GMM for finish_times_new
initial_means_new = np.array([np.mean(times) for times in finish_times_new]).reshape(-1, 1)
total_length_new = sum(len(times) for times in finish_times_new)
initial_weights_new = np.array([len(times) / total_length_new for times in finish_times_new])
gmm_new = GaussianMixture(n_components=4, means_init=initial_means_new, weights_init=initial_weights_new)
gmm_new.fit(all_times_new.reshape(-1, 1))
means_new = gmm_new.means_.flatten()
stds_new = np.sqrt(gmm_new.covariances_).flatten()
weights_new = gmm_new.weights_.flatten()

# Fit GMM for finish_times
initial_means = np.array([np.mean(times) for times in finish_times]).reshape(-1, 1)
total_length = sum(len(times) for times in finish_times)
initial_weights = np.array([len(times) / total_length for times in finish_times])
gmm = GaussianMixture(n_components=4, means_init=initial_means, weights_init=initial_weights)
gmm.fit(all_times.reshape(-1, 1))
means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_).flatten()
weights = gmm.weights_.flatten()

# Generate x-axis values for plotting the normal distributions
x_vals_new = np.linspace(np.min(all_times_new), np.max(all_times_new), 1000)
x_vals = np.linspace(np.min(all_times), np.max(all_times), 1000)

# Define route names
path_names = ['Afsluitdijk', 'Almere-Amsterdam', 'Almere-Utrecht', 'Utrecht']

# Create figure and GridSpec layout
fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.7])

# 2x2 distribution plots for each route
for i, (route_times_new, route_times, name) in enumerate(zip(finish_times_new, finish_times, path_names)):
    ax = fig.add_subplot(gs[i // 2, i % 2])

    # Plot the histogram and normal fit for the new travel times
    route_mean_new = np.mean(route_times_new)
    route_std_new = np.std(route_times_new)
    ax.hist(route_times_new, bins=10, density=True, alpha=0.5, color='gray')
    normal_pdf_new = norm.pdf(x_vals_new, route_mean_new, route_std_new)
    ax.plot(x_vals_new, normal_pdf_new, color='red', label=f'Normal Fit with Speed Increase: μ={route_mean_new:.2f}, σ={route_std_new:.2f}')

    # Plot the histogram and normal fit for the original travel times
    route_mean = np.mean(route_times)
    route_std = np.std(route_times)
    ax.hist(route_times, bins=10, density=True, alpha=0.3, color='orange')
    normal_pdf = norm.pdf(x_vals, route_mean, route_std)
    ax.plot(x_vals, normal_pdf, color='green', label=f'Normal Fit without Speed Increase: μ={route_mean:.2f}, σ={route_std:.2f}')

    # Formatting
    ax.set_title(f'Traveltime Distribution - {name}')
    ax.set_xlabel('Traveltime (minutes)')
    ax.set_ylabel('Probability Density')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15))

# Combined histogram and GMM plot on bottom row
ax_combined = fig.add_subplot(gs[2, :])
ax_combined.hist(all_times, bins=int(np.ptp(all_times)), range=(np.min(all_times), np.max(all_times)), density=True, alpha=0.5, color='gray', linewidth=1.2, label='Travel Times before Speed Increase')
ax_combined.hist(all_times_new, bins=int(np.ptp(all_times_new)), range=(np.min(all_times_new), np.max(all_times_new)), density=True, alpha=0.3, color='orange', linewidth=1.2, label='Travel Times after Speed Increase')

# Plot GMM fits
gmm_pdf_new = sum(weights_new[i] * norm.pdf(x_vals_new, means_new[i], stds_new[i]) for i in range(len(weights_new)))
ax_combined.plot(x_vals_new, gmm_pdf_new, color='red', linestyle='-', linewidth=2, label='GMM Fit after Speed Increase')
gmm_pdf_old = sum(weights[i] * norm.pdf(x_vals, means[i], stds[i]) for i in range(len(weights)))
ax_combined.plot(x_vals, gmm_pdf_old, color='green', linestyle='-', linewidth=2, label='GMM Fit before Speed Increase')

# Formatting for the combined plot
ax_combined.set_title('Combined Travel Time Distributions and GMM Fits')
ax_combined.set_xlabel('Travel Time (minutes)')
ax_combined.set_ylabel('Probability Density')
ax_combined.legend(loc='upper right')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()


# Testing for complex model
all_times_new = np.concatenate(finish_times_new)
all_times = np.concatenate(finish_times)

travel_times_before = all_times
travel_times_after = all_times_new

u_statistic, p_value = stats.mannwhitneyu(travel_times_before, travel_times_after, alternative='greater')

print("Mann-Whitney U-test for the complex model:")
print(f"U-statistic: {u_statistic}, P-value: {p_value}")

alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: there is a significant reduction in travel time.")
else:
    print("Fail to reject the null hypothesis: no significant reduction in travel time.")


print("\n t-test for each individual route:")
for i, (route_before, route_after) in enumerate(zip(finish_times, finish_times_new)):
    t_statistic, p_value = stats.ttest_ind(route_before, route_after, equal_var=False, alternative='greater')

    print(f"Route: {path_names[i]}")
    print(f"T-statistic: {t_statistic}, P-value: {p_value}")

    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: there is a significant reduction in travel time.")
    else:
        print("Fail to reject the null hypothesis: no significant reduction in travel time.")
    print("------")
    
