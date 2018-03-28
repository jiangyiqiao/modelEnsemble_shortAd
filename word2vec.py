# encoding: UTF-8
import tensorflow as tf
import numpy as np
import os
import csv
from gensim.models import Word2Vec
import re
import itertools
from collections import Counter
import os
import cPickle
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# Eval Parameters
tf.flags.DEFINE_string("checkpoint_dir", "models/word2vec/checkpoints", "Checkpoint directory from training run")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# validate
# ==================================================

# validate checkout point file
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# validate word2vec model file
trained_word2vec_model_file = os.path.join(FLAGS.checkpoint_dir, "..", "trained_word2vec.model")
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))

# validate training params file
training_params_file = os.path.join(FLAGS.checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = cPickle.load(f)
    return output_dict


def padding_sentences(texts, padding_token, padding_sentence_length = None):
    sentences=[]
    for text in texts:
        sentences+=text

    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    if len(sentences) > max_sentence_length:
        sentences = sentences[:max_sentence_length]
    else:
        sentences.extend([padding_token] * (max_sentence_length - len(sentences)))
    return (sentences, max_sentence_length)


def embedding_sentences(sentences, embedding_size = 128, window = 5, min_count = 5, file_to_load = None, file_to_save = None):
    if file_to_load is not None:
        w2vModel = Word2Vec.load(file_to_load)
    else:
        if file_to_save is not None:
            w2vModel = Word2Vec(sentences, size = embedding_size, window = window, min_count = min_count, workers = multiprocessing.cpu_count())
        if file_to_save is not None:
            w2vModel.save(file_to_save)
    embeddingDim = w2vModel.vector_size
    embeddingUnknown = [0 for i in range(embeddingDim)]
    this_vector = []
    for word in sentences:
        if word in w2vModel.wv.vocab:
            this_vector.append(w2vModel[word])
        else:
            this_vector.append(embeddingUnknown)
    return [this_vector]

# Predict
def predict(text):
    params = loadDict(training_params_file)
    num_labels = int(params['num_labels'])
    max_document_length = 190         #由已训练好模型决定，不可该，除非另外训练模型

    x_raw=[]
    for i in text:
        x_raw.append(i)
    
    # Get Embedding vector x_test
    sentences, max_document_length = padding_sentences(x_raw, '<PADDING>', padding_sentence_length = max_document_length) 

    x_test = np.array(embedding_sentences(sentences, file_to_load = trained_word2vec_model_file))

    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
        # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            result  = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
            return np.ndarray.tolist(result) #numpy.ndarray  返回list
