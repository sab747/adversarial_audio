"""
Original Author: Moustafa Alzantot (malzantot@ucla.edu)

Modifications made by Tommy White and Sabrina Jain at Dartmouth for CS89.

"""

import numpy as np
import tensorflow as tf
from speech_commands import label_wav
import os, sys
import csv
import argparse

def load_graph(filename):
    with tf.compat.v1.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

def load_labels(filename):
    return [line.rstrip() for line in tf.compat.v1.gfile.FastGFile(filename)]


def load_audiofile(filename):
    with open(filename, 'rb') as fh:
        return fh.read()


# setup calling parameters for main function of this file
argp = argparse.ArgumentParser()
argp.add_argument('--output-dir', '-o', type=str, default='output')
argp.add_argument('--labels-path', '-lp', type=str, required=True, help='Path to labels file')
argp.add_argument('--graph-path', '-gp', type=str, required=True, help='Path to frozen graph file')
argp.add_argument('--results-path', '-rp', type=str, default='eval_output.csv', help='path for CSV file of evaluation results')
argp.add_argument('--verbose', '-v', action='store_true')


if __name__ == '__main__':
    args = argp.parse_args()
    output_dir = args.output_dir
    labels_file = args.labels_path
    graph_file = args.graph_path
    output_file = args.results_path
    verbose = args.verbose

    labels = load_labels(labels_file)
    n_labels = len(labels)
    result_mat = np.zeros((n_labels, n_labels))
    input_node_name = 'wav_data:0'
    output_node_name = 'labels_softmax:0'
    load_graph(graph_file)
    
    ## set header of output file
    output_fh = open(output_file, 'w')
    fieldnames = ['filename', 'original', 'target', 'predicted']
    for label in labels:
        fieldnames.append(label)
    csv_writer = csv.DictWriter(output_fh, fieldnames=fieldnames)
    csv_writer.writeheader()
    if verbose: print(fieldnames)
        
    count = 0
    with tf.compat.v1.Session() as sess:
        output_node = sess.graph.get_tensor_by_name(output_node_name) 
        for src_idx, src_label in enumerate(labels):
            for target_idx, target_label in enumerate(labels):
                case_dir = format("%s/%s/%s" %(output_dir, target_label, src_label))
                if os.path.exists(case_dir):
                    wav_files =[format('%s/%s' %(case_dir, f)) for f in os.listdir(case_dir) if f.endswith('.wav')]
                    for wav_filename in wav_files:
                        wav_data = load_audiofile(wav_filename)
                        
                        preds = sess.run(output_node, feed_dict = {
                                input_node_name: wav_data
                        })  # feed forward our audio file and identify the class with highest probability from the network's output vector
                        wav_pred = np.argmax(preds[0])
                        
                        if wav_pred == target_idx:  # update our result matrix if we managed to trick the classifier
                            result_mat[src_idx][wav_pred] += 1
                        
                        # write out the data
                        row_dict = dict()
                        row_dict['filename'] = wav_filename
                        row_dict['original'] = src_label
                        row_dict['target'] = target_label
                        row_dict['predicted'] = labels[wav_pred]
                        
                        for i in range(preds[0].shape[0]):
                            row_dict[labels[i]] = preds[0][i]
                        
                        csv_writer.writerow(row_dict)
                        count += 1
        
        if verbose: 
            print(result_mat)
            print(f'{int(np.sum(result_mat))} / {count} attacks successful')
                        

