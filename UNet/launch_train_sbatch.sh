#!/usr/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=160
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --job-name=unet
#SBATCH -o unet_%N.%j.out
#SBATCH --mail-user=mmajursk@nist.gov
#SBATCH --mail-type=FAIL
#SBATCH --time=48:0:0


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
        rm -rf ${working_dir}
        exit -1
}

# associate the function "term_handler" with the TERM signal
trap 'term_handler' TERM

wrk_directory="/wrk/mmajursk/small-data-cnns/UNet"

# job configuration
test_every_n_steps=1000
batch_size=8 # 4x across the gpus

# make working directory
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

# copy data to node
echo "Copying data to Node"
train_lmdb_file="train-HES.lmdb"
test_lmdb_file="test-HES.lmdb"
cp -r ${wrk_directory}/${train_lmdb_file} ${working_dir}/${train_lmdb_file}
cp -r ${wrk_directory}/${test_lmdb_file} ${working_dir}/${test_lmdb_file}
echo "data copy to node complete"
echo "Working directory contains: "
ls ${working_dir}

# # **********************
# # Conda setup
# module load anaconda3
# conda create --name tf python=3.6
# conda activate tf

# conda install tensorflow-gpu
# conda install scikit-image
# conda install scikit-learn
# pip install lmdb
# load any modules
module load anaconda3
conda activate tf
echo "Modules loaded"

results_dir="$wrk_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

mkdir -p "$results_dir/src"
cp -r . "$results_dir/src"

# launch training script with required options
echo "Launching Training Script"
python train_unet.py --test_every_n_steps=${test_every_n_steps} --batch_size=${batch_size} --train_database="$working_dir/$train_lmdb_file" --test_database="$working_dir/$test_lmdb_file" --output_dir="$results_dir" | tee "$results_dir/log.txt"

# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"
