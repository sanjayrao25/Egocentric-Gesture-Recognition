#!/bin/bash

UCF_DIR="/home/sanjay/Music/improved_trajectory_release/CS221_Project-master/videos/" #videos path
TRAIN_LIST="/home/sanjay/Music/improved_trajectory_release/CS221_Project-master/video_list.txt" #train file
GMM_OUT="/home/sanjay/Music/improved_trajectory_release/CS221_Project-master/gmm_list" #output path

python gmm.py 120 $UCF_DIR $TRAIN_LIST $GMM_OUT --pca

trainlist01="/home/sanjay/Music/improved_trajectory_release/CS221_Project-master/video_list.txt" #train file
testlist01="/home/sanjay/Music/improved_trajectory_release/CS221_Project-master/testlist01.txt" #test file

training_output="/home/sanjay/Music/improved_trajectory_release/fisher_trainoutput" #train output
testing_output="/home/sanjay/Music/improved_trajectory_release/fisher_testoutput"  #test output

python computeFVs.py $UCF_DIR $trainlist01 $training_output $GMM_OUT
python computeFVs.py $UCF_DIR $testlist01 $testing_output $GMM_OUT

CLASS_INDEX="/home/sanjay/Music/improved_trajectory_release/CS221_Project-master/class_index.txt" #
CLASS_INDEX_OUT="./class_index"
python compute_UCF101_class_index.py $CLASS_INDEX $CLASS_INDEX_OUT

python classify_experiment.py
