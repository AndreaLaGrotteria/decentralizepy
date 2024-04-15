import importlib
import json
import logging
import math
import os
from matplotlib import pyplot as plt
import sys

def plot_multiple(res_list, labels, title, xlabel, filename):
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

def plot(dir,uid,victim):          
    with open(
        os.path.join(dir, "{}_results.json".format(uid)),
        "r",
    ) as inf:
        results_dict = json.load(inf)

    
    # plot_multiple([results_dict["loss_mia_update"]],["update"],"Membership Inference Attack Loss","Communication Rounds",os.path.join(dir, "{}_mia_loss.png".format(uid)))

    loss, ent, ids = parse_dict(results_dict["mia_all"])

    # find index of victim in ids
    victim_id = ids.index(int(victim))


    plot_multiple([loss[victim_id], results_dict["loss_mia_update"]],[victim, "avg"],"Membership Inference Attack Loss","Communication Rounds",os.path.join(dir, "{}_mia_all.png".format(uid)))

def parse_dict(data):
    # To achieve the desired structure, we need to iterate over each node in each iteration and collect the loss values.
    loss_result = []
    ent_result = []
    idx = []

    # Find the maximum number of nodes in any iteration to determine the number of dictionaries to create in the list
    max_nodes = max(len(iteration_values) for iteration_values in data.values())

    # Iterate over each node index to create a dictionary for each
    for node_index in range(max_nodes):
        idx.append(node_index)
        loss_dict = {}
        ent_dict = {}
        for iteration, nodes in data.items():
            # Check if the current node exists in this iteration
            if str(node_index) in nodes:
                # Add the iteration with its loss value for this node
                loss_dict[iteration] = nodes[str(node_index)][0]
                ent_dict[iteration] = nodes[str(node_index)][1]
        loss_result.append(loss_dict)
        ent_result.append(ent_dict)
        

    return loss_result, ent_result, idx

    
if __name__ == "__main__":
    assert len(sys.argv) == 4
    plot(sys.argv[1], sys.argv[2], sys.argv[3])