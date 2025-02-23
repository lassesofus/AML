#!/bin/bash
#BSUB -J flow_mnist
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 5:00
#BSUB -R "select[gpu32gb]"
#BSUB -u lassesofus@gmail.com
#BSUB -o %J.out
#BSUB -e %J.err

source /zhome/e3/3/139772/Desktop/AML/aml/bin/activate

# Define the ranges for the parameters
num_transformations_list=(8)
num_hidden_list=(3136)
epochs_list=(15)

# Loop over each combination of parameters
for num_transformations in "${num_transformations_list[@]}"; do
    for num_hidden in "${num_hidden_list[@]}"; do
        for epochs in "${epochs_list[@]}"; do
            # Create a folder named after the combination of parameters
            output_dir="/zhome/e3/3/139772/Desktop/AML/AML/Module_1_Generative_Models/Week_2_Normalizing_Flows/output_t${num_transformations}_h${num_hidden}_e${epochs}"
            mkdir -p $output_dir

            # Run the training script
            python3 /zhome/e3/3/139772/Desktop/AML/AML/Module_1_Generative_Models/Week_2_Normalizing_Flows/flow_2_5.py train --masking cb --num_transformations $num_transformations --num_hidden $num_hidden --epochs $epochs --device cuda --model $output_dir/model.pt --output_dir $output_dir --samples $output_dir/samples_mnist.png

            # Run the sampling script
            python3 /zhome/e3/3/139772/Desktop/AML/AML/Module_1_Generative_Models/Week_2_Normalizing_Flows/flow_2_5.py sample --masking cb --num_transformations $num_transformations --num_hidden $num_hidden --model $output_dir/model.pt --device cuda --samples $output_dir/samples_mnist.png

            # Move the output files to the created folder
            mv *.out *.err $output_dir/

        done
    done
done