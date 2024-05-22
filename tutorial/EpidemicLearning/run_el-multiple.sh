#!/bin/bash

log2() {
    result=$(echo "l($1)/l(2)" | bc -l)
    printf "%.0f\n" "$result"
}

n=32
d=$(log2 $n)
decpy_path=../../eval # Path to eval folder
graph=fullyConnected_$n.edges # Absolute path of the graph file generated using the generate_graph.py script
run_path=../../eval/data/echo-$n # Path to the folder where the graph and config file will be copied and the results will be stored
config_file=config_EL-multi.ini

mkdir $run_path

env_python=~/miniconda3/envs/decpy/bin/python3 # Path to python executable of the environment | conda recommended
machines=1 # number of machines in the runtime
eval_file=testingEL_Local.py # decentralized driver code (run on each machine)
log_level=INFO # DEBUG | INFO | WARN | CRITICAL

m=0 # machine id corresponding consistent with ip.json
echo M is $m

procs_per_machine=$n 
echo procs per machine is $procs_per_machine

# List of possible values for each setting
partition_niid_values=(False dirichlet)
full_epochs_values=(True False)
num_active_attackers_values=(1 0)

seed=90
alpha=
lr=0.015
partition=false
full_epochs=False
num_active=1
shard=4

# Template with placeholders for parameters
config_template="[DATASET]
dataset_package = decentralizepy.datasets.CIFAR10
dataset_class = CIFAR10
model_class = LeNet
train_dir = ../../eval/data/
test_dir = ../../eval/data/
sizes = 
random_seed = %d
partition_niid = %s
shards = %s
alpha = %s

[OPTIMIZER_PARAMS]
optimizer_package = torch.optim
optimizer_class = SGD
lr = %s

[TRAIN_PARAMS]
training_package = decentralizepy.training.Training
training_class = Training
rounds = %d
full_epochs = %s
batch_size = 5
shuffle = True
loss_package = torch.nn
loss_class = CrossEntropyLoss

[COMMUNICATION]
comm_package = decentralizepy.communication.TCP
comm_class = TCP
addresses_filepath = ip.json

[SHARING]
sharing_package = decentralizepy.sharing.PlainAverageSharing
sharing_class = PlainAverageSharing
compress = False

[NODE]
graph_degree = %d

[ATTACK]
attack_package = decentralizepy.attacks.Echo
attack_class = EchoDynamic
num_attackers = 1
num_active_attackers = %d
victims = [%d]
num_victims = all
"

# Iterate over all combinations of settings
for num_active in 0 1; do
    # Number of rounds and iterations for each setting
    iterations=200
    test_after=25
    if [ "$full_epochs" == "True" ]; then
        rounds=1
        iterations=$(echo "($iterations + 5) / 6" | bc)
        test_after=$(echo "($test_after + 5) / 6" | bc)
    else
        rounds=10
    fi

    # Fill in the template and save to the new config file
    printf "$config_template" "$seed" "$partition" "$shard" "$alpha" "$lr" "$rounds" "$full_epochs" "$d" "$num_active" "$victims" > "$config_file"

    cp $graph $config_file $run_path
    
    # log_dir=$run_path/alpha_${alpha}_seed_${seed}_niid_${partition}_epochs_${full_epochs}_attackers_${num_active}_victims_[${victims}]_$(date +%s)/machine$m
    log_dir=$run_path/exp_$(date +%s)/machine$m

    mkdir -p $log_dir

    printf "Running with partition_niid=$partition, alpha=$alpha full_epochs=$full_epochs, num_active_attackers=$num_active, victims=$victims, rounds=$rounds, iterations=$iterations, test_after=$test_after \n"

    $env_python $eval_file -ro 0 -tea $test_after -ld $log_dir -mid $m -ps $procs_per_machine -ms $machines -is $iterations -gf $run_path/$graph -ta $test_after -cf $run_path/$config_file -ll $log_level -wsd $log_dir

done
