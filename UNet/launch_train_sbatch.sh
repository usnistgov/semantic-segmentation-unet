#!/usr/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=unet
#SBATCH -o unet_%N.%j.out
#SBATCH --time=24:0:0


timestamp="$(date +%Y%m%dT%H%M%S)"
experiment_name="unet-${timestamp}"
echo "Experiment: $experiment_name"
working_dir="/scratch/${SLURM_JOB_ID}"

# define the handler function
# note that this is not executed here, but rather
# when the associated signal is sent
term_handler()
{
        echo "function term_handler called.  Cleaning up and Exiting"
        # Do nothing since working directory cleanup is handled for you
        exit -1
}

# associate the function "term_handler" with the TERM signal
trap 'term_handler' TERM

wrk_directory="/wrk/mmajursk/tf_tutorial/"

# job configuration
test_every_n_steps=250
batch_size=16 # Nx across the gpus

# make working directory
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

# copy data to node
echo "Copying data to Node"
train_lmdb_file="train-hes-large.lmdb"
test_lmdb_file="test-hes-large.lmdb"
cp -r /wrk/mmajursk/tf_tutorial/data/${train_lmdb_file} ${working_dir}/${train_lmdb_file}
cp -r /wrk/mmajursk/tf_tutorial/data/${test_lmdb_file} ${working_dir}/${test_lmdb_file}
# train_lmdb_file="train-hes.lmdb"
# test_lmdb_file="test-hes.lmdb"
#cp -r ${wrk_directory}/data/${train_lmdb_file} ${working_dir}/${train_lmdb_file}
#cp -r ${wrk_directory}/data/${test_lmdb_file} ${working_dir}/${test_lmdb_file}
echo "data copy to node complete"
echo "Working directory contains: "
ls ${working_dir}


module load powerAI/tensorflow-1.5.4
echo "Modules loaded"


results_dir="$wrk_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

mkdir -p "$results_dir/src"
cp -r . "$results_dir/src"

# launch training script with required options
echo "Launching Training Script"
python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" | tee "$results_dir/log.txt"

echo "Job completed"
