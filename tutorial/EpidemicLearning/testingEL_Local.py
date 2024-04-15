import logging
from pathlib import Path
from shutil import copy

from localconfig import LocalConfig
from multiprocessing import Manager
from torch import multiprocessing as mp

from decentralizepy.attacks.DatasetSharing import DatasetSharing
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

    processes = []
    manager = Manager()
    d = manager.dict()
    for i in range(procs_per_machine):
        d[i] = None
    n = manager.dict()
    for i in range(procs_per_machine):
        n[i] = None

    datasets = DatasetSharing(d,n)

    victim = 3
    attackers = [0,1,2]
    
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
                    datasets,
                    victim,
                    attackers
                ],
            )
        )

    for p in processes:
        p.start()

    for p in processes:
        p.join()
