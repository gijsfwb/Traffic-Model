import numpy as np
import matplotlib.pyplot as plt
nodes = [[0,1],[2],[3],[4,5,6],[7],[8],[]]
startnodes = [0,0,1,2,3,3,3,4,5]
endnodes = [1,2,3,3,4,6,5,6,6]
lengths = [3e4,5e4,4e4,6e4,2e4,2e4,3e4,1e4,5e3]
speeds =[100,100,100,100,100,100,100,100,100]
lanes = [2,2,2,2,2,1,2,2,2]
capacity_multipliers = [1,1,1,1,1,1,1,1,1]
cars = []
roads = []
car_length = 4.5
d_spacing = 55
t = 0
alpha = 0.15
beta = 4
cutoff_time = 120
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
class car:
    def __init__(self,position):
        self.position = position
        self.finished = False
        self.time_on_road = 0
        self.time_to_reach_node = 0
        self.total_time = 0
        self.roads_taken = []
for i in range(9):
    roads.append(road(startnodes[i],endnodes[i],lengths[i],speeds[i],lanes[i],capacity_multipliers[i]))


#Timestep loop:
while t<480:
    t+=1
    print(t)
    numcars = int(np.random.normal(95,1))
    for i in range(numcars):
        cars.append(car(0))
    for rd in roads:
        rd.travel_time = rd.freeflow_time*(1+alpha*pow(rd.cars_on_road/rd.capacity,beta))
        rd.occupancy.append(rd.cars_on_road/rd.capacity)
        if t>cutoff_time:
            rd.total_cars += rd.cars_on_road
    for vehicle in cars:
        if vehicle.finished:
            continue
        else:
            vehicle.total_time += 1
            vehicle.time_on_road += 1
            #move the vehicle from city A to a road (if possible)
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


#plt.plot(avg_times)
#plt.hist(avg_times,bins=int(max(avg_times)-min(avg_times)),range=(min(avg_times),max(avg_times)))


for rd in roads:
    plt.plot(rd.occupancy)
    print(f"road from {rd.startnode} to {rd.endnode} had average occupancy: {rd.total_cars/(rd.capacity*(t-cutoff_time))}")
plt.show()




import csv
cities = [
    {'Id':1, 'label':'City 1'},
    {'Id':2, 'label':'City 2'},
    {'Id':3, 'label':'A'},
    {'Id':4, 'label':'B'},
    {'Id':5, 'label':'C'},
    {'Id':6, 'label':'D'},
    {'Id':7, 'label':'E'},
]

def write_cities_csv(filename):
    with open(filename,'w',newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["Id", "label"])
        writer.writeheader()
        writer.writerows(cities)

write_cities_csv('cities.csv')





