#!/usr/bin/bash

# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --job-name=segnet
#SBATCH -o segnet_%N.%j.out
#SBATCH --time=48:0:0


# job configuration
batch_size=2 # 4x across the gpus
number_classes=4

image_folder="/wrk/mmajursk/Concrete_Feldman/data/rawFOV"
output_directory="/wrk/mmajursk/Concrete_Feldman/output"
img_height=712
img_width=950

checkpoint_filepath="/wrk/mmajursk/Concrete_Feldman/segnet-20190507T141113/checkpoint/model.ckpt"

experiment_name="segnet-$(date +%Y%m%dT%H%M%S)"

# MODIFY THESE OPTIONS
# **************************


echo "Experiment: $experiment_name"

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


module load powerAI/tensorflow-1.5.4
echo "Modules loaded"

mkdir -p ${output_directory}
echo "Output Directory: $output_directory"


# launch training script with required options
echo "Launching Script"
python inference.py --gpu=0 --checkpoint_filepath="$checkpoint_filepath" --image_folder="$image_folder" --output_folder="$output_directory" --number_classes=${number_classes} --image_height=${img_height} --image_width=${img_width}

echo "Job completed"
