import importlib
import json
import logging
import math
import os
from collections import deque
from random import Random

import torch
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.attacks.DatasetSharing import DatasetSharing
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node
from decentralizepy.datasets.CIFAR10 import LeNet

import copy
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference import membership_inference
import numpy as np
import random

class EL_Local(Node):
    """
    This class defines the node on overlay graph

    """

    def save_plot(self, l, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)
    
    def plot_multiple(self, res_list, labels, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        for i,l in enumerate(res_list):
            y_axis = [l[key] for key in l.keys()]
            x_axis = list(map(int, l.keys()))
            plt.plot(x_axis, y_axis, label=labels[i])
        plt.xlabel(xlabel)
        plt.title(title)
        plt.legend()
        plt.savefig(filename)

    def get_neighbors(self, node=None):
        tmp = set(self.rng.sample(self.my_neighbors, self.degree))
        self.datasets.add_neighbourset(self.uid, tmp)
        return tmp

    def receive_DPSGD(self):
        return self.receive_channel("DPSGD", block=True)

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_neighbors:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def run(self):
        """
        Start the decentralized learning

        """
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1
        do_echo = True
        self.rng = Random()
        self.rng.seed(self.dataset.random_seed + self.uid)

        self.connect_neighbors()
        logging.info("Connected to all neighbors")

        logging.info("Total number of neighbor: {}".format(len(self.my_neighbors)))

        for iteration in range(self.iterations):
            # Local Phase
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1
            

            self.iteration = iteration
            self.trainer.train(self.dataset)
            echo = do_echo and self.uid in self.active_attackers

            neighbors_this_round = self.get_neighbors()

            to_send = self.sharing.get_data_to_send()

            if not echo:
                # Communication Phase
                if self.uid in self.attackers:
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
                    to_send["CHANNEL"] = "DPSGD"
                    for neighbor in neighbors_this_round:
                        logging.debug("Sending to neighbor: {} data: {}".format(neighbor,to_send))
                        self.communication.send(neighbor, to_send)

                    for x in self.my_neighbors:
                        if x not in neighbors_this_round:
                            self.communication.send(
                                x,
                                {
                                    "CHANNEL": "DPSGD",
                                    "iteration": iteration,
                                    "NotWorking": True,
                                },
                            )
            else:
                for x in self.attackers:
                    if x != self.uid:
                        self.communication.send(
                            x,
                            {
                                "CHANNEL": "DPSGD",
                                "iteration": iteration,
                                "NotWorking": True,
                            },
                        )

            while not self.received_from_all():
                response = self.receive_DPSGD()
                if response:
                    sender, data = response
                    logging.debug(
                        "Received Model from {} of iteration {}: {}".format(
                            sender,
                            data["iteration"],
                            "NotWorking" if "NotWorking" in data else data,
                        )
                    )
                    if sender not in self.peer_deques:
                        self.peer_deques[sender] = deque()

                    if data["iteration"] == self.iteration:
                        self.peer_deques[sender].appendleft(data)
                    else:
                        self.peer_deques[sender].append(data)     

            averaging_deque = dict()
            atleast_one = False
            for x in self.my_neighbors:
                if x in self.peer_deques and len(self.peer_deques[x]) > 0:
                    this_message = self.peer_deques[x][0]
                    if (
                        this_message["iteration"] == self.iteration
                        and "NotWorking" not in this_message
                    ):
                        averaging_deque[x] = self.peer_deques[x]
                        atleast_one = True
                    elif this_message["iteration"] == self.iteration:
                        self.peer_deques[x].popleft()
                        logging.debug(
                            "Discarding message from {} of iteration {}".format(
                                x, this_message["iteration"]
                            )
                        )

            update_buffer = {}
            if atleast_one:
                self.sharing._averaging(averaging_deque,update_buffer,echo)
            else:
                if not echo:
                    self.sharing.communication_round += 1

            self.model_update_buffer = update_buffer

            if echo:
                # Communication Phase
                neighbors_this_round = list(update_buffer.keys())
                for neighbor in neighbors_this_round:
                    total = {}
                    for key, value in self.model_update_buffer[neighbor].items():
                        total[key] = value
                    self.model.load_state_dict(total)
                    to_send = self.sharing.get_data_to_send_attack(degree=len(self.my_neighbors))
                    to_send["CHANNEL"] = "DPSGD"
                    # del data["degree"]
                    logging.debug("Sending to neighbor: {} data: {}".format(neighbor,to_send))
                    self.communication.send(neighbor, to_send)

                
                for x in self.my_neighbors:
                    if x not in neighbors_this_round:
                        self.communication.send(
                            x,
                            {
                                "CHANNEL": "DPSGD",
                                "iteration": iteration,
                                "NotWorking": True,
                            },
                        )

                self.sharing.communication_round += 1

            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "received_this_round": {},
                    "mlp_train_acc_baseline": {},
                    "mlp_test_acc_baseline": {},
                    "mlp_acc_baseline": {},
                    "mlp_train_acc_update": {},
                    "mlp_test_acc_update": {},
                    "mlp_acc_update": {},
                    "mlp_train_acc_marginalized": {},
                    "mlp_test_acc_marginalized": {},
                    "mlp_acc_marginalized": {},
                    "loss_mia_update": {},
                    "ent_mia_update": {},
                    "loss_mia_marginalized": {},
                    "ent_mia_marginalized": {},
                    "mia_all": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data
            if hasattr(self.communication, "received_this_round"):
                results_dict["received_this_round"][
                    iteration + 1
                ] = self.communication.received_this_round

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )


                if self.uid in self.attackers:
                    pass
                else:
                    mias = self.mia_local()
                    results_dict["mia_all"][iteration + 1] = mias
                    results_dict["loss_mia_update"][iteration + 1] = mias[0]
                    results_dict["ent_mia_update"][iteration + 1] = mias[1]
                    self.plot_multiple([results_dict["loss_mia_update"]],["update"],"Membership Inference Attack Loss","Communication Rounds",os.path.join(self.log_dir, "{}_mia_loss.png".format(self.rank)))
                    self.plot_multiple([results_dict["ent_mia_update"]],["update"],"Membership Inference Attack Entropy","Communication Rounds",os.path.join(self.log_dir, "{}_mia_entropy.png".format(self.rank)))


            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)
        self.my_neighbors = self.graph.neighbors(self.uid)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")


    def find_victim(self, set, victim):
        if victim in set:
            return victim
        for v in set:
            if victim in self.datasets.get_neighbourset(v):
                return v
        return random.choice(set)


    def compute_modified_entropy(self, p, y, epsilon=0.00001):
        """ Computes label informed entropy from 'Systematic evaluation of privacy risks of machine learning models' USENIX21 """
        assert len(y) == len(p)
        n = len(p)

        entropy = np.zeros(n)

        for i in range(n):
            pi = p[i]
            yi = y[i]
            for j, pij in enumerate(pi):
                if j == yi:
                    # right class
                    entropy[i] -= (1-pij)*np.log(pij+epsilon)
                else:
                    entropy[i] -= (pij)*np.log(1-pij+epsilon)

        return entropy


    def ths_searching_space(self, nt, train, test):
        """ it defines the threshold searching space as nt points between the max and min value for the given metrics """
        thrs = np.linspace(
            min(train.min(), test.min()),
            max(train.max(), test.max()), 
            nt
        )
        return thrs

    def mia_best_th(self, train_set, nt=150):
        """ Perfom naive, metric-based MIA with 'optimal' threshold """
        
        def search_th(Etrain, Etest):
            R = np.empty(len(thrs))
            for i, th in enumerate(thrs):
                tp = (Etrain < th).sum()
                tn = (Etest >= th).sum()
                acc = (tp + tn) / (Etrain.shape[0] + Etest.shape[0])
                R[i] = acc
            return R.max()
        
        # evaluating model on train and test set
        # I need loss, accuracy and Output
        _, Ltrain, Ptrain, Ytrain = self.dataset.testMIA(self.model, self.loss, train_set)
        _, Ltest, Ptest, Ytest = self.dataset.testMIA(self.model, self.loss, self.dataset.get_testset())
        
        # it takes a subset of results on test set with size equal to the one of the training test 
        n = Ptrain.shape[0]
        Ptest = Ptest[:n]
        Ytest = Ytest[:n]
        Ltest = Ltest[:n]

        # if self.uid == 1:
        #     logging.info(Ltrain)
        #     # print("METRIC",metric,"\n")
        #     logging.info(Ptrain)
        #     logging.info(Ytrain)
            
        # performs optimal threshold for loss-based MIA 
        thrs = self.ths_searching_space(nt, Ltrain, Ltest)
        loss_mia = search_th(Ltrain, Ltest)
        
        # computes entropy
        Etrain = self.compute_modified_entropy(Ptrain, Ytrain)
        Etest = self.compute_modified_entropy(Ptest, Ytest)
        
        # performs optimal threshold for entropy-based MIA 
        thrs = self.ths_searching_space(nt, Etrain, Etest)
        ent_mia = search_th(Etrain, Etest)
        
        return loss_mia, ent_mia


    def mia_for_each_nn(self, modify_model, update_buffer):
        """ Run MIA for each attacker's neighbors """
        
        nn = sorted(list(update_buffer.keys()))
        model_copy = copy.deepcopy(self.model)

        # mias = np.zeros((len(nn), 2))
        mias = {}
        for i, v in enumerate(nn):
            modify_model(update_buffer, v, model_copy)
                        
            train_set = self.datasets.get_dataset(v).get_trainset()
            
            mias[i] = self.mia_best_th(train_set)
            
        return mias
    
    def mia_local(self):
        train_set = self.dataset.get_trainset()
        mias = self.mia_best_th(train_set)
        return mias

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False

        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )

    def isolate_victim(self, model_update_buffer, victim_id, model_copy):
        """ Computes marginalized model  """
        with torch.no_grad():
            total = dict()
            weight = 1 / (len(model_update_buffer))
            for id,update in model_update_buffer.items():
                if id!=victim_id:
                    for key, value in update.items():
                        if key in total:
                            total[key] += value * weight
                        else:
                            total[key] = value * weight
            
            weight = len(model_update_buffer)
            for key, value in model_update_buffer[victim_id].items():
                total[key] -= value * weight

            for key, value in self.model.state_dict().items():
                total[key] += value * weight
            
            model_copy.load_state_dict(total)

        return total
    
    def isolate_updates(self, model_update_buffer, _, model_copy):
        """ Computes marginalized model  """
        with torch.no_grad():
            total = dict()
            weight = 1 / (len(model_update_buffer)+1)
            for id,update in model_update_buffer.items():
                for key, value in update.items():
                    if key in total:
                        total[key] += value * weight
                    else:
                        total[key] = value * weight

            for key, value in self.model.state_dict().items():
                total[key] += value * weight
            
            model_copy.load_state_dict(total)

        return total
    
    def isolate_update(self, model_update_buffer, victim, model_copy):
        """ Computes marginalized model  """
        with torch.no_grad():
            total = dict()
            update = model_update_buffer[victim]
            for key, value in update.items():
                if key in total:
                    total[key] += value 
                else:
                    total[key] = value 
            
            model_copy.load_state_dict(total)

        return total

    # def MIA_received_model(dl,attacker):
    #     get_model = lambda attacker, u: attacker.model_update_buffer[u.name]
        
    #     return mia_for_each_nn(dl, attacker, get_model)

    # def mia_for_each_nn(dl, attacker, get_model):
    #     """ Run MIA for each attacker's neighbors """
    
    #     nn = sorted(attacker.neighbors, key=lambda x:int(x.name))
    #     model = deepCopyModel(attacker.model)

    #     mias = np.zeros((len(nn), 2))
    #     for i, v in enumerate(nn):
    #         var = get_model(attacker, v)
            
    #         assign_list_variables(model.trainable_variables, var)
            
    #         train_set = v.train_set
            
    #         mias[i] = mia_best_th(model, train_set, dl)
            
    #     return mias

    # def deepCopyModel(model):
    #     _model = copy.deepcopy(model)     
    #     return _model


    # def assign_list_variables(A, B):
    #     """ A <- B """
    #     assert len(A) == len(B)
    #     n = len(A)
    #     for i in range(n):
    #         A[i].assign(B[i])


    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        datasets=None,
        victim=-1,
        attackers=[-1],
        active_attackers=[-1],
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )
        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()


        self.victim = victim
        self.attackers = attackers
        self.active_attackers = active_attackers
        self.model_update_buffer = {}

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)
        self.true_neighbors = set(self.my_neighbors)-set(self.attackers)
        self.datasets = datasets
        self.datasets.add_dataset(self.uid, self.dataset)


        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        datasets=None,
        victim=-1,
        attackers=[-1],
        active_attackers=[-1],
        *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.procs_per_machine), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            datasets,
            victim,
            attackers,
            active_attackers,
            *args
        )

        nodeConfigs = config["NODE"]
        self.degree = (
            nodeConfigs["graph_degree"] if "graph_degree" in nodeConfigs else 2
        )

        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
