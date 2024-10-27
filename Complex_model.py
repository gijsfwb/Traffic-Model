#TODO
#Slimmere routekiezer - GEDAAN
#Lanes correct maken
import numpy as np
import matplotlib.pyplot as plt




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
nodes = [[0,1,2],[3],[5],[4],[6],[7,8],[10],[9],[]]
startnodes = [0,0,0,1,3,2,4,5,5,6,7]
endnodes = [1,3,6,2,4,7,5,7,6,8,8]
speeds = [100,100,100,100,100,100,100,100,100,100,100]
lanes = [2,2,2,2,2,2,2,2,2,2,2] #zeer onduidelijk maar 2 lijkt prima te kloppen
lengths = [100e3,110e3,191e3,29e3,15e3,85e3,23e3,46e3,39e3,36e3,60e3]
capacity_multipliers = [1,1,1,1,1,1,1,1,1,1,1]
# for i in range(len(startnodes)):
#     print(f"van {node_names[startnodes[i]]} naar {node_names[endnodes[i]]} is {lengths[i]}")
roads = []
cars = []
roads = []
choice_vals = []
car_length = 4.5
d_spacing = 55
t = 0
alpha = 0.15
beta = 4
cutoff_time = 480
def generate_paths(node, path):
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
    def __init__(self,position):
        self.position = position
        self.finished = False
        self.time_on_road = 0
        self.time_to_reach_node = 0
        self.total_time = 0
        self.roads_taken = [0]
for i in range(len(startnodes)):
    roads.append(road(startnodes[i],endnodes[i],lengths[i],speeds[i],lanes[i],capacity_multipliers[i]))
for rd in roads:
    rd.paths = generate_paths(rd.endnode,[])
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
    numcars = int(np.random.normal(80,1))
    for i in range(numcars):
        cars.append(car(0))
    for rd in roads:
        rd.travel_time = rd.freeflow_time*(1+alpha*pow(rd.cars_on_road/rd.capacity,beta))
        rd.occupancy.append(rd.cars_on_road/rd.capacity)
        if t>cutoff_time:
            rd.total_cars += rd.cars_on_road
    if t<=cutoff_time:
        cars_before_cutoff = len(cars)
    for vehicle in cars:
        if not vehicle.finished:
            vehicle.total_time += 1
            vehicle.time_on_road += 1

            #if vehicle has reached the end of the road, move it to a new road (if possible)
            if vehicle.time_on_road+1 > vehicle.time_to_reach_node:
                if vehicle.position == 0:
                    paths = startpaths
                else:
                    paths = vehicle.position.paths
                    if vehicle.position.endnode == 8:
                        vehicle.roads_taken.append(vehicle.position.endnode)
                        vehicle.position.cars_on_road -= 1
                        vehicle.position = 8
                        vehicle.finished = True
                        continue
                attractiveness = []      
                
                for path in paths:
                    total_time = 0
                    for index in path:
                        if roads[index].cars_on_road < roads[index].capacity:
                            total_time += roads[index].travel_time
                        else:
                            total_time = -1
                            break
                    if total_time != -1:
                        attractiveness.append(1/total_time)
                    else:
                        attractiveness.append(0)
                p_choice = []
                normalize_attractiveness = sum(attractiveness)
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

finished = []
travel_time = []
paths = []
for cr in cars:
    finished.append(cr.finished)
    travel_time.append([cr.total_time,cr.finished])
    paths.append(cr.roads_taken)
# avg_times = []
# binsize = 1
# for i in range(int((cars_before_cutoff-1)/binsize),int(len(cars)/binsize)):
#     total_bin = 0
#     if sum(finished[i*binsize:(i+1)*binsize]) == binsize:
#         for j in range(i*binsize,(i+1)*binsize):            
#             total_bin += cars[j].total_time
#             avg_bin = total_bin/binsize
#         avg_times.append(avg_bin)
    

    
finish_times = []
for i in range(len(nodepaths)):
    finish_times.append([])
for pathindex in range(len(nodepaths)):   
    for i in range(cars_before_cutoff-1,len(cars)):
        if finished[i]:
            if cars[i].roads_taken == nodepaths[pathindex]:
                finish_times[pathindex].append(cars[i].total_time)
for rd in roads:
    plt.plot(rd.occupancy)
    print(f"Road from {node_names[rd.startnode]} to {node_names[rd.endnode]} had average occupancy: {rd.total_cars/(rd.capacity*(t-cutoff_time))}")
plt.legend(startnodes)
plt.show()
#plt.plot(avg_times)
#plt.show()
timemax = max([max(times) for times in finish_times])
timemin = min([min(times) for times in finish_times])
plt.hist(finish_times,bins = int(timemax-timemin),range=(timemin,timemax),density=True,histtype='barstacked',stacked = False)
path_names = ['Afsluitdijk','Almere-Amsterdam','Almere-Utrecht','Utrecht']
plt.legend(path_names)
#for i in range(len(finish_times)):
#     plt.hist(finish_times[i],bins=int(max(finish_times[i])-min(finish_times[i])),range=(min(finish_times[i]),max(finish_times[i])),density=True,histtype='barstacked',stacked = True)
plt.show()
