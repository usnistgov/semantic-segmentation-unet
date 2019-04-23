#!/usr/bin/bash

# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --job-name=unet
#SBATCH -o unet_%N.%j.out
#SBATCH --time=24:0:0


# job configuration
test_every_n_steps=1000
batch_size=8 # 4x across the gpus

train_lmdb_file="train-hes.lmdb"
test_lmdb_file="test-hes.lmdb"

input_data_directory="/wrk/mmajursk/small-data-cnns/UNet"
output_directory="/wrk/mmajursk/small-data-cnns/UNet"

experiment_name="unet-$(date +%Y%m%dT%H%M%S)"

# MODIFY THESE OPTIONS
# **************************

echo "Experiment: $experiment_name"
scratch_dir="/scratch/${SLURM_JOB_ID}"

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
term_handler()
{
        echo "function term_handler called.  Cleaning up and Exiting"
        # Do nothing
        exit -1
}

# associate the function "term_handler" with the TERM signal
trap 'term_handler' TERM


# make working directory
mkdir -p ${scratch_dir}
echo "Created Directory: $scratch_dir"

# copy data to node
echo "Copying data to Node"
cp -r ${input_data_directory}/${train_lmdb_file} ${scratch_dir}/${train_lmdb_file}
cp -r ${input_data_directory}/${test_lmdb_file} ${scratch_dir}/${test_lmdb_file}
echo "data copy to node complete"
echo "Working directory contains: "
ls ${scratch_dir}


module load powerAI/tensorflow-1.5.4
echo "Modules loaded"


results_dir="$output_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

mkdir -p "$results_dir/src"
cp -r . "$results_dir/src"

# launch training script with required options
echo "Launching Training Script"
python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$scratch_dir/$train_lmdb_file" --test_database="$scratch_dir/$test_lmdb_file" --output_dir="$results_dir" | tee "$results_dir/log.txt"

echo "Job completed"
