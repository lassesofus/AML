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

python3 /zhome/e3/3/139772/Desktop/AML/AML/Module_1_Generative_Models/Week_2_Normalizing_Flows/flow_2_5.py train --masking cb --epochs 1000 --device cuda
