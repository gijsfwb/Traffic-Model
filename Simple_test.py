import numpy as np
nodes = [[0,1],[2],[3],[4,5,6],[7],[8],[]]
startnodes = [0,0,1,2,3,3,3,4,5]
endnodes = [1,2,3,3,4,6,5,6,6]
lengths = [3e4,5e4,4e4,6e4,2e4,3e4,2e4,1e4,5e3]
speeds =[100,100,100,100,100,100,100,100,100]
lanes = [2,2,2,2,2,2,1,2,2]
cars = []
roads = []
car_length = 4.5
d_spacing = 55
t = 0
alpha = 0.15
beta = 4
class road:
    def __init__(self,startnode, endnode,length,max_speed,n_lanes):
        self.startnode = startnode
        self.endnode = endnode
        self.length = length # m
        self.max_speed = max_speed #km/h
        self.n_lanes = n_lanes 
        self.freeflow_time = length/max_speed * 60/1000    #conversion factor so result is in minutes
        self.travel_time = self.freeflow_time
        self.cars_on_road = 0
        self.capacity = int(n_lanes*length/(car_length+d_spacing))
class car:
    def __init__(self,position):
        self.position = position
        self.finished = False
        self.time_on_road = 0
        self.time_to_reach_node = 0
        self.total_time = 0

for i in range(9):
    roads.append(road(startnodes[i],endnodes[i],lengths[i],speeds[i],lanes[i]))

        
#Timestep loop:

while t<240:
    t+=1
    print(t)
    numcars = int(np.random.normal(95,1))
    for i in range(numcars):
        cars.append(car(0))
    for rd in roads:
        rd.travel_time = rd.freeflow_time*(1+alpha*pow(rd.cars_on_road/rd.capacity,beta))
    for vehicle in cars:
        if vehicle.finished:
            continue
        vehicle.total_time +=1
        #if current position is a node, change it to road if capacity allows
        if isinstance(vehicle.position,int):
            if vehicle.position == 6:
                vehicle.finished = True
                continue
#               OLD CODE WITHOUT PATH CHOOSING           
#               if roads[nodes[vehicle.position][0]].cars_on_road < roads[nodes[vehicle.position][0]].capacity:
#               vehicle.position = roads[nodes[vehicle.position][0]]
#               vehicle.position.cars_on_road +=1
#               vehicle.time_to_reach_node = vehicle.position.travel_time + np.random.normal(0,2)
            attractiveness = []            
            for index in nodes[vehicle.position]:
                if roads[index].cars_on_road < roads[index].capacity:
                    attractiveness.append(roads[index].travel_time)
                else:
                    attractiveness.append(0)
            normalize_attractiveness = sum(attractiveness)
            p_choice = []
            if normalize_attractiveness != 0:
                for i in range(len(attractiveness)):
                    if attractiveness != 0:
                        p_choice.append(attractiveness[i]/normalize_attractiveness)
                choice_val = np.random.random()
                for i in range(len(p_choice)):
                    if choice_val < sum(p_choice[0:i+1]):
                        vehicle.position = roads[nodes[vehicle.position][i]]
                        vehicle.position.cars_on_road +=1
                        vehicle.time_to_reach_node = vehicle.position.travel_time + np.random.normal(0,2)
                        break

        #if current position is a road, see whether the car reached the next node
        else:
            vehicle.time_on_road += 1
            if vehicle.time_on_road >= vehicle.time_to_reach_node:
                vehicle.position.cars_on_road -=1
                vehicle.position = vehicle.position.endnode
                vehicle.time_on_road = 0

finished = []
for cr in cars:
    finished.append(cr.finished)
