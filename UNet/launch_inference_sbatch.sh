#!/usr/bin/bash

# **************************
# MODIFY THESE OPTIONS

#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=unet
#SBATCH -o unet_%N.%j.out
#SBATCH --time=24:0:0


# job configuration

input_data_directory="/wrk/mmajursk/Concrete_Feldman/images"
output_directory="/wrk/mmajursk/Concrete_Feldman"

checkpoint_filepath="/wrk/mmajursk/Concrete_Feldman/unet-20190510T113248-0/checkpoint/model.ckpt"
number_classes=4

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


mkdir -p ${output_directory}
echo "Results Directory: $output_directory"

mkdir -p "$output_directory/src"
cp -r . "$output_directory/src"

# launch training script with required options
echo "Launching Training Script"
python inference.py --checkpoint_filepath=${checkpoint_filepath} --image_folder=${input_data_directory} --output_folder=${output_directory} --number_classes=${number_classes} | tee "$output_directory/log.txt"


echo "Job completed"

