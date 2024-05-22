import logging
from multiprocessing import Manager
from pathlib import Path
from shutil import copy

from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Linear import Linear
from decentralizepy.node.EpidemicLearning.EL_Local import EL_Local


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        print("Section: ", section)
        for key, value in config.items(section):
            print((key, value))
    print(dict(config.items("DATASET")))
    return config


if __name__ == "__main__":
    args = utils.get_args()

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    config = read_ini(args.config_file)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))

    copy(args.config_file, args.log_dir)
    copy(args.graph_file, args.log_dir)
    utils.write_args(args, args.log_dir)

    g = Graph()
    g.read_graph_from_file(args.graph_file, args.graph_type)
    n_machines = args.machines
    procs_per_machine = args.procs_per_machine[0]

    l = Linear(n_machines, procs_per_machine)
    m_id = args.machine_id

    attackers = []
    for i in range(my_config["ATTACK"]["num_attackers"]):
        attackers.append(i)
    active_attackers = []
    for i in range(my_config["ATTACK"]["num_active_attackers"]):
        active_attackers.append(i)
    victims = []
    if my_config["ATTACK"]["victims"] != '[]':
        victims = my_config["ATTACK"]["victims"].replace("[", "").replace("]", "").split(",")
        victims = [int(i) for i in victims]
    elif my_config["ATTACK"]["num_victims"] == 'all':
        victims = [i for i in range(procs_per_machine) if i not in attackers]
    else:
        # pick randomly num_victims victim from the non-attackers
        import random
        victims = random.sample([i for i in range(procs_per_machine) if i not in attackers], int(my_config["VICTIMS"]["num_victims"]))


    # print("victim: ", victims)

    manager = Manager()
    dist = manager.dict()
    for i in range(procs_per_machine):
        dist[i] = float("inf")

    processes = []
    for r in range(procs_per_machine):
        processes.append(
            mp.Process(
                target=EL_Local,
                args=[
                    r,
                    m_id,
                    l,
                    g,
                    my_config,
                    args.iterations,
                    args.log_dir,
                    args.weights_store_dir,
                    log_level[args.log_level],
                    args.test_after,
                    args.train_evaluate_after,
                    args.reset_optimizer,
                    attackers,
                    active_attackers,
                    victims,
                    dist
                ],
            )
        )

    for p in processes:
        p.start()

    for p in processes:
        p.join()
