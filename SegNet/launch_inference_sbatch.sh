#!/usr/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=segnet
#SBATCH -o segnet_%N.%j.out
#SBATCH --mail-user=mmajursk@nist.gov
#SBATCH --mail-type=FAIL

timestamp="$(date +%Y-%m-%dT%H:%M:%S)"
experiment_name="segnet-infer-${timestamp}"
echo "Experiment: $experiment_name"

# this is the root directory for results
wrk_directory="/wrk/pnb"

# make working directory
working_dir="/scratch/pnb/unet/$experiment_name"
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

#make results directory
results_dir="$wrk_directory/$experiment_name"
mkdir -p ${results_dir}
echo "Results Directory: $results_dir"

# job configuration
checkpoint_filepath=${wrk_directory}/segnet-2019-03-01T15\:05\:08/checkpoint/model.ckpt
infer_folder="rawTiles"
image_folder=${wrk_directory}/data/${infer_folder}/
number_classes=4
echo "job config: checkpoint_filepath=" $checkpoint_filepath " image_folder=" $image_folder " output_folder=" $results_dir " number_classes=" ${number_classes}

# copy data to node
echo "Copying data to Node"
cp -r ${wrk_directory}/data/${infer_folder}/ ${working_dir}/
echo "data copy to node complete"
echo "Working directory contains: "
ls ${working_dir}

# load any modules
module load powerAI/tensorflow
echo "Modules loaded"


# launch inference script with required options
echo "Launching Inference Script"
python inference.py --gpu=0 --checkpoint_filepath="$checkpoint_filepath" --image_folder="$image_folder" --output_folder="$results_dir" --number_classes=${number_classes} | tee "$results_dir/log.txt"

# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"
