import logging
import copy
class EchoDynamic:
    def __init__(self, uid, attackers, active_attackers, communication, my_neighbors, victims=[], dist={}):
        self.uid = uid
        self.attackers = attackers
        self.active_attackers = active_attackers
        self.communication = communication
        self.my_neighbors = my_neighbors
        self.victims = victims
        self.updates = {}
        self.iteration = -1
        self.dist = dist

    def add_update(self,data,uid):
        if 'NotWorking' not in data:
            self.updates[uid] = copy.deepcopy(data)

    def is_attacker(self):
        return self.uid in self.attackers
    
    def is_active_attacker(self):
        return self.uid in self.active_attackers

    def send_not_working(self, iteration):
        self.updates = {}
        self.iteration = iteration
        if (self.uid in self.attackers and self.uid not in self.active_attackers):
            for x in self.my_neighbors:
                self.communication.send(
                    x,
                    {
                        "CHANNEL": "DPSGD",
                        "iteration": iteration,
                        "NotWorking": True,
                    },
                )
        else:
            for x in self.active_attackers:
                if x != self.uid:
                    self.communication.send(
                        x,
                        {
                            "CHANNEL": "DPSGD",
                            "iteration": iteration,
                            "NotWorking": True,
                        },
                    )

    def send_echo(self):
        if self.is_active_attacker():
            neighbors_this_round = set(self.updates.keys())
            for neighbor in neighbors_this_round:
                data = self.updates[neighbor]
                logging.debug("Sending to neighbor: {} data: {}".format(neighbor,data))
                self.communication.send(neighbor, data)

            for x in self.my_neighbors:
                if x not in neighbors_this_round:
                    self.communication.send(
                        x,
                        {
                            "CHANNEL": "DPSGD",
                            "iteration": self.iteration,
                            "NotWorking": True,
                        },
                    )

class EchoStatic:
    def __init__(self, uid, attackers, active_attackers, communication, my_neighbors, victims=[], dist={}):
        self.uid = uid
        self.attackers = attackers
        self.active_attackers = active_attackers
        self.communication = communication
        self.my_neighbors = my_neighbors
        self.victims = victims
        self.updates = {}
        self.iteration = -1
        self.dist = dist

    def send_not_working(self, iteration):
        self.updates = {}
        self.iteration = iteration
        if (self.uid in self.attackers and self.uid not in self.active_attackers):
            for x in self.my_neighbors:
                self.communication.send(
                    x,
                    {
                        "CHANNEL": "DPSGD",
                        "iteration": iteration,
                        "NotWorking": True,
                    },
                )
        else:
            for x in self.active_attackers:
                if x != self.uid:
                    self.communication.send(
                        x,
                        {
                            "CHANNEL": "DPSGD",
                            "iteration": iteration,
                            "NotWorking": True,
                        },
                    )

    def add_update(self,data,uid):
        self.updates[uid] = copy.deepcopy(data)
        self.updates[uid]['iteration'] = self.updates[uid]['iteration']+1

    def is_attacker(self):
        return self.uid in self.attackers
    
    def is_active_attacker(self):
        return self.uid in self.active_attackers
    
    def send_echo(self, neighbor):
        victim=self.victims[0]
        data = self.updates[victim]
        self.communication.send(neighbor, data)


class EchoTargeted:
    def __init__(self, uid, attackers, active_attackers, communication, my_neighbors, victims=[], dist={}):
        self.uid = uid
        self.attackers = attackers
        self.active_attackers = active_attackers
        self.communication = communication
        self.my_neighbors = my_neighbors
        self.victims = victims
        self.updates = {}
        self.last_update = {}
        for neighbor in my_neighbors:
            self.updates[neighbor] = {}
        self.updates[self.uid] = {}
        self.iteration = -1
        self.dist = dist

    def add_update(self,data,uid):
        if 'NotWorking' not in data and (uid==self.victims[0] or uid == self.uid):
            self.updates[uid][data['iteration']] = copy.deepcopy(data)
            # self.last_update[uid] = max(self.last_update[uid],data['iteration'])

    def is_attacker(self):
        return self.uid in self.attackers
    
    def is_active_attacker(self):
        return self.uid in self.active_attackers

    def send_not_working(self, iteration):
        # self.updates = {}
        self.iteration = iteration
        if (self.uid in self.attackers and self.uid not in self.active_attackers):
            for x in self.my_neighbors:
                self.communication.send(
                    x,
                    {
                        "CHANNEL": "DPSGD",
                        "iteration": iteration,
                        "NotWorking": True,
                    },
                )
        else:
            for x in self.active_attackers:
                if x != self.uid:
                    self.communication.send(
                        x,
                        {
                            "CHANNEL": "DPSGD",
                            "iteration": iteration,
                            "NotWorking": True,
                        },
                    )

    def send_echo(self):
        if self.is_active_attacker():
            victim=self.victims[0]
            # send to everybody the difference between the last two updates of the victim
            if len(self.updates[victim]) >= 1:
                my_keys = sorted(self.updates[self.uid].keys(), reverse=True)
                victim_keys = sorted(self.updates[victim].keys(), reverse=True)
                diff = self.updates[victim][victim_keys[0]]['params'] - self.updates[victim][victim_keys[1]]['params']
                diff = diff*2

                data = {
                    "CHANNEL": "DPSGD",
                    "iteration": self.iteration,
                    "params": diff
                }

                logging.debug("Sending to neighbor: {} data: {}".format(victim,data))
                self.communication.send(victim, data)
                
                for neighbor in self.my_neighbors:
                    if neighbor != victim:
                        self.communication.send(
                            neighbor,
                            {
                                "CHANNEL": "DPSGD",
                                "iteration": self.iteration,
                                "NotWorking": True,
                            },
                        )
            else:
                for neighbor in self.my_neighbors:
                        self.communication.send(
                            neighbor,
                            {
                                "CHANNEL": "DPSGD",
                                "iteration": self.iteration,
                                "NotWorking": True,
                            },
                        )
            # select from neighbors_this_round the neighbor that has the lowest value in dist
            # if len(self.updates) > 0:
            #     closest = min(list(self.updates.keys()), key=lambda x: self.dist[x])
            #     data = self.updates[closest]

            #     for neighbor in self.my_neighbors:
            #         logging.debug("Sending to neighbor: {} data: {}".format(neighbor,data))
            #         self.communication.send(neighbor, data)
            # else:
            #     for neighbor in self.my_neighbors:
            #         self.communication.send(
            #             neighbor,
            #             {
            #                 "CHANNEL": "DPSGD",
            #                 "iteration": self.iteration,
            #                 "NotWorking": True,
            #             },
            #         )
        
