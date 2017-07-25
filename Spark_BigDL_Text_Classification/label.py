#coding=utf-8

import re
import itertools
import datetime as dt
import matplotlib
import jieba
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from data import *
import pickle
import time

def text_to_words(review_text,stopwords):
    words = list(jieba.cut(review_text.replace('\n','')))
    # Filter stop words
    words = filterCmt(words,stopwords)
    print(words)
    return words

def analyze_texts(data_rdd,stopwords):
    def index(w_c_i):
        ((w, c), i) = w_c_i
        return (w, (i + 1, c))
    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0],stopwords)) \
        .map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b) \
        .sortBy(lambda w_c: - w_c[1]).zipWithIndex() \
        .map(lambda w_c_i: index(w_c_i)).collect()

# pad([1, 2, 3, 4, 5], 0, 6)
def pad(l, fill_value, width):
    if len(l) >= width:
        return l[0: width]
    else:
        l.extend([fill_value] * (width - len(l)))
        return l


def to_vec(token, b_w2v, embedding_dim):
    if token in b_w2v:
        return b_w2v[token]
    else:
        return pad([], 0, embedding_dim)

def to_sample(vectors, label, embedding_dim):
    # flatten nested list
    flatten_features = list(itertools.chain(*vectors))
    features = np.array(flatten_features, dtype='float').reshape(
        [sequence_len, embedding_dim])

    if model_type.lower() == "cnn":
        features = features.transpose(1, 0)
    return Sample.from_ndarray(features, np.array(label))

def build_model(class_num):
    model = Sequential()
    if model_type.lower() == "lstm":
        model.add(Recurrent()
                  .add(LSTM(embedding_dim, 128, p)))
        model.add(Select(2, -1))
    elif model_type.lower() == "gru":
        model.add(Recurrent()
                  .add(GRU(embedding_dim, 128, p)))
        model.add(Select(2, -1))
    else:
        raise ValueError('model can only be lstm, or gru')

    model.add(Linear(128, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model


def map_predict_label(l):
    return np.array(l).argmax()
def map_groundtruth_label(l):
    return l[0] - 1

def train(sc,
          batch_size,
          sequence_len, max_words, embedding_dim, training_split,params):
    
    print('Processing text dataset')
    raw_data = pd.read_csv(params["data"],low_memory=False,encoding='utf-8')
    texts = getTrain(raw_data,params["act"],params["target"],params["target_value"],params["subsample_num"])
    
    stopwords = getStopWords(params["path_to_stopwords"])
    data_rdd = sc.parallelize(texts, 2)
    word_to_ic = analyze_texts(data_rdd,stopwords)

    # Only take the top wc between [10, sequence_len]
    word_to_ic = dict(word_to_ic[10: max_words])
    bword_to_ic = sc.broadcast(word_to_ic)

    w2v,all_cmts=get_w2v(texts)
    filtered_w2v = dict((w, v) for w, v in w2v.items() if w in word_to_ic)
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    tokens_rdd = data_rdd.map(lambda text_label:
                              ([w for w in text_to_words(text_label[0],stopwords) if
                                w in bword_to_ic.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(
        lambda tokens_label: (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                       ([to_vec(w, bfiltered_w2v.value,
                                                embedding_dim) for w in
                                         tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], embedding_dim))

    train_rdd, test_rdd = sample_rdd.randomSplit(
        [training_split, 1-training_split])

    optimizer = Optimizer(
        model=build_model(3),
        training_rdd=train_rdd,
        criterion=ClassNLLCriterion(),
        end_trigger=MaxEpoch(max_epoch),
        batch_size=batch_size,
        optim_method="adam")
    optimizer.set_validation(
        batch_size = batch_size,
        val_rdd = test_rdd,
        trigger = EveryEpoch()
    )
    
    # Start to train
    train_model = optimizer.optimize()
    print("Train over!")
    predictions = train_model.predict(test_rdd)
    y_pred = np.array([ map_predict_label(s) for s in predictions.collect()])
    y_true = np.array([map_groundtruth_label(s.label) for s in test_rdd.collect()])
    correct = 0
    for i in range(0, y_pred.size):
        if (y_pred[i] == y_true[i]):
            correct += 1

    accuracy = float(correct) / y_pred.size
    print ('-'*20+'\n'
            +'Prediction accuracy on test  set is: ',accuracy)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="120")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="10")
    parser.add_option("--model", dest="model_type", default="gru")
    parser.add_option("-p", "--p", dest="p", default="0.0")

    (options, args) = parser.parse_args(sys.argv)
    batch_size = int(options.batchSize)
    embedding_dim = int(options.embedding_dim)
    max_epoch = int(options.max_epoch)
    p = float(options.p)
    model_type = options.model_type
    
    sequence_len = 50
    max_words = 1000
    training_split = 0.8
    params = {}
    _dir = "cellar/"
    params["path_to_model"]=_dir+"model"
    params["path_to_word_to_ic"]=_dir+"word_to_ic.pkl"
    params["path_to_filtered_w2v"]=_dir+"filtered_w2v.pkl"
    
    params["path_to_stopwords"]="stopwords"        
    params["data"] = "data.csv"
    params["logDir"] = "logs/"
    params["act"] = "ApplePay"
    params["subsample_num"] = 400
    params["target"] = "fav"
    params["target_value"] = [u"差",u"中",u"好"]
    params["label_name"] = "noise"
    

    # Initialize env
    sc = SparkContext(appName="sa",conf=create_spark_conf())
    init_engine()

    # Train model
    train(sc,batch_size,
            sequence_len, max_words, embedding_dim, training_split,params)
    sc.stop()
    
