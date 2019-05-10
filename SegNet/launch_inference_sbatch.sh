#!/usr/bin/bash

# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=segnet
#SBATCH -o segnet_%N.%j.out
#SBATCH --time=24:0:0


# job configuration

input_data_directory="/wrk/mmajursk/Concrete_Feldman/images"
output_directory="/wrk/mmajursk/Concrete_Feldman"

checkpoint_filepath="/wrk/mmajursk/Concrete_Feldman/segnet-20190510T123041-0/checkpoint/model.ckpt"
number_classes=4
image_height=712
image_width=950

experiment_name="segnet-inference-$(date +%Y%m%dT%H%M%S)"

# MODIFY THESE OPTIONS
# **************************

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


# load any modules
module load powerAI/tensorflow-1.5.4
echo "Modules loaded"

for N in 0 1 2 3
do
    results_dir="$output_directory/$experiment_name-$N"

    mkdir -p ${results_dir}
    echo "Results Directory: $results_dir"

    mkdir -p "$results_dir/src"
    cp -r . "$results_dir/src"

    # launch training script with required options
    echo "Launching Training Script"
    python inference.py --checkpoint_filepath=${checkpoint_filepath} --image_folder=${input_data_directory} --output_folder=${results_dir} --number_classes=${number_classes} --image_height=${image_height} --image_width=${image_width} | tee "$results_dir/log.txt"

done


echo "Job completed"
