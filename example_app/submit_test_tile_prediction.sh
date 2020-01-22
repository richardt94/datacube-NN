#!/bin/bash
#PBS -P ge3
#PBS -q normal
#PBS -l walltime=10:00:00
#PBS -l ncpus=24
#PBS -l mem=95GB
#PBS -m abe
#PBS -M richard.taylor@ga.gov.au
#PBS -l wd
#PBS -l storage=gdata/r78+gdata/v10+gdata/fk4

module use /g/data/v10/public/modules/modulefiles

module load dea

python test_tile_prediction.py
