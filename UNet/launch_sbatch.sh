#!/usr/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --job-name=mm
#SBATCH -o %N.%j.out

# job configuration
#epoch_count=50 # 10x within the script to reduce test frequency
batch_size=8
dataset=${1}
model=${2}
nb_classes=${3}
epoch_count=${4}

wrk_directory="/wrk/mmajursk/small-data-cnns/data/"
source_code_directory="/wrk/mmajursk/small-data-cnns/src/aug/"


# make working directory
working_dir="/scratch/mmajursk/$dataset/aug-$model/"
rm -rf ${working_dir}
mkdir -p ${working_dir}
echo "Created Directory: $working_dir"

# copy data to node
echo "Copying data to Node"
cd ${working_dir}

fn="$dataset.zip"
src="$wrk_directory$fn"
cp ${src} ${working_dir}
unzip ${working_dir}/${fn} > /dev/null
echo "data copy to node complete"
echo "Working directory contains: "
ls ${working_dir}

# load any modules
module load powerAI/tensorflow
echo "Modules loaded"

declare -a dataset_sizes=("100" "200" "300" "400" "500")
#declare -a dataset_sizes=("100")
declare -a reps=("0" "1" "2" "3")
#declare -a reps=("0")
for rep in "${reps[@]}"
do
    for dataset_size in "${dataset_sizes[@]}"
    do
        results_dir="$wrk_directory/$dataset/aug-$model-$dataset_size-r$rep"
        mkdir -p ${results_dir}
        echo "Results Directory: $results_dir"

        # launch training script with required options
        echo "Launching Training Script"
        cd ${source_code_directory}
        python ${source_code_directory}/train_${model}.py  --batch_size=${batch_size} --epoch_count=${epoch_count} --output_dir="$results_dir" --train_data_folder="$working_dir/$dataset/train" --test_data_folder="$working_dir/$dataset/test" --number_training_examples=${dataset_size} --number_classes=${nb_classes}
        echo "Python script terminated"
    done
done

# cleanup (delete src, data)
echo "Performing Node Cleanup"
rm -rf ${working_dir}

echo "Job completed"