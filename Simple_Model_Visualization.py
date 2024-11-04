import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
#The network is built of a list of nodes (cities) and a list of roads
#Each node contains a list with the indices of the roads leaving that node
nodes = [[0,1],[2],[3],[4,5,6],[7],[8],[]]
#Each road has a startnode, endnode, speed, lane number and length
#In addition to a capacity multiplier
startnodes = [0,0,1,2,3,3,3,4,5]
endnodes = [1,2,3,3,4,6,5,6,6]
#Lengths, lanes and speeds are given in the assignment
lengths = [3e4,5e4,4e4,6e4,2e4,2e4,3e4,1e4,5e3]
speeds =[100,100,100,100,100,100,100,100,100]
lanes = [2,2,2,2,2,1,2,2,2]
capacity_multipliers = [1,1,1,1,1,1,1,1,1]
#Initializing lists
cars = []
roads = []
#Variables that determine capacity of a road
#and the travel time on a road based on the occupancy
car_length = 4.5
d_spacing = 55
alpha = 0.15
beta = 4
#Time to pass before considering the data
cutoff_time = 120

t = 0

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
class car:
    #The car objects hold the relevant information for each car
    def __init__(self,position):
        self.position = position
        self.finished = False
        self.time_on_road = 0
        self.time_to_reach_node = 0
        self.total_time = 0
        self.roads_taken = []
for i in range(9):
    #Creating the roads
    roads.append(road(startnodes[i],endnodes[i],lengths[i],speeds[i],lanes[i],capacity_multipliers[i]))
    
#Timestep loop
while t< 240:
    t+=1
    print(t)
    #Adding cars
    numcars = int(np.random.normal(85,1.5))
    for i in range(numcars):
        cars.append(car(0))
    for rd in roads:
        rd.travel_time = rd.freeflow_time*(1+alpha*pow(rd.cars_on_road/rd.capacity,beta))
        rd.occupancy.append(f"{t/480};{rd.cars_on_road/rd.capacity}")
        if t>cutoff_time:
            rd.total_cars += rd.cars_on_road
    for vehicle in cars:
        if vehicle.finished:
            continue
        else:
            vehicle.total_time += 1
            vehicle.time_on_road += 1
            #move the vehicle from city A to a road (if possible)
            #Looks at all available roads and picks one based on travel time of
            #that road
            if vehicle.position == 0:
                attractiveness = []
                for index in nodes[vehicle.position]:
                    if roads[index].cars_on_road < roads[index].capacity:
                        attractiveness.append(1/roads[index].travel_time)
                    else:
                        attractiveness.append(0)
                normalize_attractiveness = sum(attractiveness)
                p_choice = []
                if normalize_attractiveness != 0:
                    for i in range(len(attractiveness)):
                         p_choice.append(attractiveness[i]/normalize_attractiveness)
                    choice_val = np.random.random()
                    for i in range(len(p_choice)):
                        if choice_val < sum(p_choice[0:i+1]):
                            vehicle.time_on_road = 0
                            vehicle.position = roads[nodes[vehicle.position][i]]
                            vehicle.position.cars_on_road += 1
                            vehicle.time_to_reach_node = vehicle.position.travel_time + np.random.normal(0,2)
                            break

            #if vehicle has reached the end of the road, move it to a new node (if possible)
            elif vehicle.time_on_road >= vehicle.time_to_reach_node:
                if vehicle.position.endnode == 6:
                    vehicle.position.cars_on_road -= 1
                    vehicle.position = 6
                    vehicle.finished = True
                    continue
                attractiveness = []
                for index in nodes[vehicle.position.endnode]:
                    if roads[index].cars_on_road < roads[index].capacity:
                        attractiveness.append(1/roads[index].travel_time)
                    else:
                        attractiveness.append(0)
                normalize_attractiveness = sum(attractiveness)
                p_choice = []
                if normalize_attractiveness != 0:
                    for i in range(len(attractiveness)):
                         p_choice.append(attractiveness[i]/normalize_attractiveness)
                    choice_val = np.random.random()
                    for i in range(len(p_choice)):
                        if choice_val < sum(p_choice[0:i+1]):
                            vehicle.position.cars_on_road -= 1
                            vehicle.time_on_road = 0
                            vehicle.roads_taken.append(vehicle.position)
                            vehicle.position = roads[nodes[vehicle.position.endnode][i]]
                            vehicle.position.cars_on_road += 1
                            vehicle.time_to_reach_node = vehicle.position.travel_time + np.random.normal(0,2)
                            break

# Data for analysis
finished = []
travel_time = []
paths = []
for cr in cars:
    finished.append(cr.finished)
    travel_time.append([cr.total_time,cr.finished])
    paths.append([cr.roads_taken,cr.total_time])
avg_times = []
binsize = 1
for i in range(int(len(cars)/binsize)):
    total_bin = 0
    if sum(finished[i*binsize:(i+1)*binsize]) == binsize:
        for j in range(i*binsize,(i+1)*binsize):
            total_bin += cars[j].total_time
            avg_bin = total_bin/binsize
        avg_times.append(avg_bin)
        

# Plotting the histogram
plt.hist(avg_times, bins=int(max(avg_times)-min(avg_times)), range=(min(avg_times), max(avg_times)), color='skyblue', edgecolor='black', density=True)
plt.xlabel('Travel Time (minutes)')
plt.ylabel('Density')
plt.title('Histogram of Off-Peak Travel Times with Double Gaussian Fit')
plt.grid(True)

# Fitting a double Gaussian distribution (using Gaussian Mixture Model)
gmm = GaussianMixture(n_components=2)
gmm.fit(np.array(avg_times).reshape(-1, 1))
means = gmm.means_.flatten()
stds = np.sqrt(gmm.covariances_).flatten()
weights = gmm.weights_.flatten()

# Generating the x-axis for the Gaussian curves
x_vals = np.linspace(min(avg_times), max(avg_times), 1000)

# Plotting the individual Gaussian curves
gaussian1 = weights[0] * norm.pdf(x_vals, means[0], stds[0])
gaussian2 = weights[1] * norm.pdf(x_vals, means[1], stds[1])
plt.plot(x_vals, gaussian1, color='red', linestyle='--', label=f'Gaussian 1:$μ$={means[0]:.2f}, $σ$={stds[0]:.2f}, $w_0$ = {weights[0]:.2f}')
plt.plot(x_vals, gaussian2, color='green', linestyle='--', label=f'Gaussian 2: $μ$={means[1]:.2f}, $σ$={stds[1]:.2f}, $w_1$ = {weights[1]:.2f}')

# Plotting the combined Gaussian curve
combined_gaussian = gaussian1 + gaussian2
plt.plot(x_vals, combined_gaussian, color='blue', label='Combined Gaussian')

# Adding legend
plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)  # Adjusts legend position below plot
plt.show()
