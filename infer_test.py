import os
import tempfile

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1

import time

from argparse import ArgumentParser

from tensorflow.python.ops import gen_audio_ops as audio_ops
from tensorflow.python.ops import io_ops

labels = ["_silence_", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

def load_wav_file(filename):
  """Loads an audio file and returns a float PCM-encoded array of samples.

  Args:
    filename: Path to the .wav file to load.

  Returns:
    Numpy array holding the sample data as floats between -1.0 and 1.0.
  """
  with tf1.Session(graph=tf1.Graph()) as sess:
    wav_filename_placeholder = tf1.placeholder(tf.string, [])
    wav_loader = io_ops.read_file(wav_filename_placeholder)
    wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1)
    return sess.run(
        wav_decoder,
        feed_dict={wav_filename_placeholder: filename}).audio.flatten()

def win_label(tensor):
    arr = tensor['dense_4'].numpy()
    winner = arr.argmax()
    return labels[winner]


def getFile(path):
    return tf.convert_to_tensor(load_wav_file(path));

def test_label(infer, label_name, verbose):
    dataset_path = TEST_SET_DIR + "/" + label_name
    
    try:
        files = os.listdir(dataset_path)
    except:
        print("No test set for label " + label_name + ", skipping...")
        return [0, 0, 0]
    
    start_time = time.monotonic()
    total_count = 0
    success = 0
    for index, file in enumerate(files):
        if index % 2000 == 1999:
            print(str(index+1) +" datas has loaded")
        audio_data = getFile(os.path.join(dataset_path, file))
        tensor = infer(audio_data)
        label = win_label(tensor)
        if verbose:
            print(label)
        total_count += 1
        if label == label_name:
            success += 1
            
    elapsed_time = time.monotonic() - start_time 
    print("\"" + label_name + "\" test complete, ran " + str(total_count) + " times, " + str(success) + " hits");
    print("Elapsed time: " + str(elapsed_time) + " s (" + str(elapsed_time/total_count) + " s per run)");
    print(str(success / total_count) + "% success")
    
    return [total_count, success, elapsed_time]
    

parser = ArgumentParser()
parser.add_argument('--model', type=str, default="quick_trained_model")
parser.add_argument('--test_set', type=str, default="small_test_set")
parser.add_argument('--label', type=str, default="all")
parser.add_argument('--verbose', type=bool, default=False)
args = parser.parse_args()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

SAVED_MODEL_DIR = "models/" + args.model
TEST_SET_DIR = "test_sets/" + args.test_set

print("loading " + SAVED_MODEL_DIR + "...")
loaded = tf.saved_model.load(SAVED_MODEL_DIR)
infer = loaded.signatures["serving_default"]
print("model " + args.model + " loaded")

if args.label == "all":
    print("running test set " + args.test_set + " on all labels");
    total_count = 0
    success = 0
    total_time = 0
    for label in labels:
        res = test_label(infer, label, args.verbose)
        total_count += res[0]
        success += res[1]
        total_time += res[2]
    
    print("all tests complete, ran " + str(total_count) + " times, " + str(success) + " hits");
    print("Total time: " + str(total_time) + " s, (" + str(total_time/total_count) + " per run)");
    print(str(success / total_count) + "% success")
        
else:
    print("running test set " + args.test_set + " on label " + args.label);
    test_label(infer, args.label, args.verbose)