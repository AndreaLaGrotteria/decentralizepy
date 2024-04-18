import logging
import copy

class Echo:
    def __init__(self, uid, attackers, active_attackers, communication, my_neighbors):
        self.uid = uid
        self.attackers = attackers
        self.active_attackers = active_attackers
        self.communication = communication
        self.my_neighbors = my_neighbors
        self.updates = {}
        self.iteration = -1

    def add_update(self,data,uid):
        if 'NotWorking' not in data:
            self.updates[uid] = copy.deepcopy(data)

    def is_attacker(self):
        return self.uid in self.attackers

    def send_not_working(self, iteration):
        self.updates = {}
        self.iteration = iteration
        if (self.uid in self.attackers and self.uid not in self.active_attackers) or (iteration == 0):
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
        neighbors_this_round = list(self.updates.keys())
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
        
