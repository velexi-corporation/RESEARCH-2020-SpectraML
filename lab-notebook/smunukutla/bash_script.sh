cd /Users/Srikar/Desktop/Velexi/spectra-ml/lab-notebook/smunukutla
# $name of variable will put the value of the variable in
# Bash script
# reading from variable dollar sign
# TOP_DIR=`builtin cd $(dirname "${BASH_SOURCE[0]}") && pwd`
# export DATA_DIR=$TOP_DIR/data

python 2019-07-24-SAM\ -\ 1D\ CNN.py > accuracies.txt
# python 2019-08-03-SAM\ -\ Random\ Forest.py >> accuracies.txt
# python 2019-07-14-SAM\ -\ 2D\ CNN.py

# (any shell command)
