# -wes237b-assignment_5

Simple key-word search inference script to run on Jetson TX2

A small sample test set is provided. The full test set can be located here:
http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz

Just unzip the test set and place it into the "test_sets" folder. Then provide the name of the unzipped folder in the script's "--test_set" argument.

The models were trained on the cloud using Google Research's "kws_streaming" repo:
https://github.com/google-research/google-research/blob/master/kws_streaming/experiments/kws_experiments_35_labels.md

The full training set can be found here:
http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz