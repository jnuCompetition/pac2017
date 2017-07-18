#coding=utf-8

import re
import itertools
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser
from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from data import *
import pickle

def text_to_words(review_text):
    words = list(jieba.cut(review_text.replace('\n','')))
    print(words)
    return words

def analyze_texts(data_rdd):
    def index(w_c_i):
        ((w, c), i) = w_c_i
        return (w, (i + 1, c))
    return data_rdd.flatMap(lambda text_label: text_to_words(text_label[0])) \
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
        raise ValueError('model can only be cnn, lstm, or gru')

    model.add(Linear(128, 100))
    model.add(Linear(100, class_num))
    model.add(LogSoftMax())
    return model

def map_predict_label(l):
    return np.array(l).argmax()

def predict(sentences,embedding_dim,params):

    train_model,bword_to_ic,bfiltered_w2v=unpickler(params["path_to_model"],params["path_to_word_to_ic"],params["path_to_filtered_w2v"])
    # Predict
    data_rdd = sc.parallelize(sentences, 2)
    tokens_rdd = data_rdd.map(lambda text_label:
                          ([w for w in text_to_words(text_label[0]) if
                            w in bword_to_ic.value], text_label[1]))
    padded_tokens_rdd = tokens_rdd.map(
        lambda tokens_label: (pad(tokens_label[0], "##", sequence_len), tokens_label[1]))
    vector_rdd = padded_tokens_rdd.map(lambda tokens_label:
                                   ([to_vec(w, bfiltered_w2v.value,
                                            embedding_dim) for w in
                                     tokens_label[0]], tokens_label[1]))
    sample_rdd = vector_rdd.map(
        lambda vectors_label: to_sample(vectors_label[0], vectors_label[1], embedding_dim))

    predictions = train_model.predict(sample_rdd)
    print('Predicted labels:')
    print(','.join(str(map_predict_label(s)) for s in predictions.take(1)))

def saveFig(train_summary):
    # Train results
    loss = np.array(train_summary.read_scalar("Loss"))
    plt.figure(figsize = (12,12))
    plt.plot(loss[:,0],loss[:,1],label='loss')
    plt.xlim(0,loss.shape[0]+10)
    plt.title("loss")
    plt.savefig('NLP.jpg')

def pickler(train_model,word_to_ic,filtered_w2v):
    _dir = "cellar/"
    path_to_train_model = _dir+"model"
    path_to_word_to_ic = _dir+"word_to_ic.pkl"
    path_to_filtered_w2v = _dir+"filtered_w2v.pkl"
    pickle.dump(word_to_ic,open(path_to_word_to_ic,"wb"))
    pickle.dump(filtered_w2v,open(path_to_filtered_w2v,"wb"))
    train_model.save(path_to_train_model,True)

def unpickler(path_to_train_model,path_to_word_to_ic,
                                    path_to_filtered_w2v):
    word_to_ic = pickle.load(open(path_to_word_to_ic,"rb"))
    filtered_w2v = pickle.load(open(path_to_filtered_w2v,"rb"))
    train_model = Model.load(path_to_train_model)
    bword_to_ic = sc.broadcast(word_to_ic)
    bfiltered_w2v = sc.broadcast(filtered_w2v)
    return train_model,bword_to_ic,bfiltered_w2v


def train(sc,
          batch_size,
          sequence_len, max_words, embedding_dim, training_split,params):
    
    print('Processing text dataset')
    raw_data = pd.read_csv(params["data"],low_memory=False,encoding='utf-8')
    texts = getTrain(raw_data,params["act"],params["target"],params["target_value"])
    
    data_rdd = sc.parallelize(texts, 2)
    word_to_ic = analyze_texts(data_rdd)

    # Only take the top wc between [10, sequence_len]
    word_to_ic = dict(word_to_ic[10: max_words])
    bword_to_ic = sc.broadcast(word_to_ic)

    w2v=get_w2v(texts)
    filtered_w2v = dict((w, v) for w, v in w2v.items() if w in word_to_ic)
    bfiltered_w2v = sc.broadcast(filtered_w2v)

    tokens_rdd = data_rdd.map(lambda text_label:
                              ([w for w in text_to_words(text_label[0]) if
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
        optim_method=Adam())

    # Save to log
    app_name='NLP-'+dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary = TrainSummary(log_dir=params["logDir"],app_name=app_name)
    train_summary.set_summary_trigger("Parameters", SeveralIteration(2))
    optimizer.set_train_summary(train_summary)
    
    # Start to train
    train_model = optimizer.optimize()
    
    saveFig(train_summary)
    pickler(train_model,word_to_ic,filtered_w2v)


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-a", "--action", dest="action", default="train")
    parser.add_option("-b", "--batchSize", dest="batchSize", default="120")
    parser.add_option("-e", "--embedding_dim", dest="embedding_dim", default="50")
    parser.add_option("-m", "--max_epoch", dest="max_epoch", default="15")
    parser.add_option("--model", dest="model_type", default="lstm")
    parser.add_option("-p", "--p", dest="p", default="0.0")
    (options, args) = parser.parse_args(sys.argv)
    
    if options.action == "train":
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
        
        params["data"] = "data.csv"
        params["logDir"] = "logs/"
        params["act"] = "ApplePay"
        params["target"] = "ptotal"
        params["target_value"] = [u"差",u"中",u"好"]

        # Initialize env
        sc = SparkContext(appName="sa",conf=create_spark_conf())
        init_engine()

        # Train model
        train(sc,
              batch_size,
              sequence_len, max_words, embedding_dim, training_split,params)
        
        # Predict model
        sentences = [('ApplePay的服务比其他产品要好呀!',0)]
        predict(sentences,embedding_dim,params)
        
        sc.stop()
    elif options.action == "test":
        pass
