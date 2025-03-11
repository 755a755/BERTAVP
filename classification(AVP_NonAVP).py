import os, sys, re

from keras_bert import load_trained_model_from_checkpoint
import pandas as pd
import tensorflow as tf
from keras.utils import np_utils
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Input, Dropout, Embedding, Flatten, MaxPooling1D, Conv1D, SimpleRNN, LSTM, GRU, \
    Multiply, GlobalMaxPooling1D, Lambda
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config

from protein_encoding import PC_6
from transformers import TFAutoModelForSequenceClassification, BertTokenizer, TFBertModel, BertModel, \
    BertForPreTraining, PretrainedConfig, \
    BertConfig

if tf.config.list_physical_devices('GPU') != []:
    visible_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([visible_devices[1]], 'GPU')


# with tf.device("/gpu:0"):
def focal_loss(gamma=1., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(K.epsilon() + pt_1)) \
            - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(
                1. - pt_0 + K.epsilon()))  # K.epsilon()函数是其中之一，它返回一个非常小的正数，通常是机器精度的一部分，用于避免数值计算中的不稳定性

    return focal_loss_fixed


def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


class myCallback(tf.keras.callbacks.Callback):
    data = []

    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        loss = logs.get('loss')
        val_acc = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        self.data.append([epoch, acc, loss, val_acc, val_loss])

    def on_train_end(self, logs=None):
        pass

    def get_data(self):
        return self.data

    def reset_data(self, logs=None):
        self.data = []


def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    F1 = fbeta_score(labels, pred_y, beta=1)
    return acc, precision, sensitivity, specificity, MCC, F1


def seq_to_num(line, seq_length):
    seq = np.zeros(seq_length)
    for j in range(len(line)):
        seq[j] = protein_dict[line[j]]
    return seq


def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>',
                 records) == None:  # Scan through string looking for a match to the pattern, returning a match object, or None if no match was found
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])
    return myFasta


# re.sub将array[1:].upper()的字符串里“非ARNDCQEGHILKMFPSTWYV-”的统统换成“-”符号

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=10, verbose=1)


def build_model():
    inputs_1 = Input(shape=(123,), dtype=tf.int32)  # inputs_1 = Input(shape=(123,),dtype=tf.int32)
    inputs_mask = Input(shape=(123,), dtype=tf.int32)
    inputs_3 = Input(shape=(2453, 1))  # inputs3是专门给CNN做处理的

    bert_tf = TFBertModel.from_pretrained('../bert-base-uncased')

    for l in bert_tf.layers:
        l.trainable = True
    x_1 = bert_tf(inputs_1, attention_mask=inputs_mask)[0]  # x_1 shape = (None,123,768),用TFBertModel的用法

    conv1d_1 = layers.Conv1D(filters=64, strides=1, padding='SAME', kernel_size=2)(x_1)
    conv1d_2 = layers.Conv1D(filters=64, strides=1, padding='SAME', kernel_size=5)(x_1)
    conv1d_3 = layers.Conv1D(filters=64, strides=1, padding='SAME', kernel_size=8)(x_1)
    max_pooling1d_1 = layers.AvgPool1D(pool_size=2, name='max_pooling1d_1')(conv1d_1)
    max_pooling1d_2 = layers.AvgPool1D(pool_size=2, name='max_pooling1d_2')(conv1d_2)  # max加不了因为会对第二个维度缩减
    max_pooling1d_3 = layers.AvgPool1D(pool_size=2, name='max_pooling1d_3')(conv1d_3)
    concatenate = layers.Concatenate()([max_pooling1d_1, max_pooling1d_2, max_pooling1d_3])
    print(concatenate.shape)
    x = layers.Bidirectional(LSTM(units=128, return_sequences=True))(concatenate)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    channel1 = Dense(256, activation='relu')(x)
    print(channel1.shape)
    # ------------------Bert通道------------------------------

    Conv1 = layers.Conv1D(filters=512, strides=1, padding='SAME', kernel_size=2, activation='relu')(inputs_3)
    Pool1 = layers.AvgPool1D(2, name='Pool1')(Conv1)
    Conv2 = layers.Conv1D(filters=256, strides=1, padding='SAME', kernel_size=2, activation='relu')(Pool1)
    Pool2 = layers.AvgPool1D(2, name='Pool2')(Conv2)
    Conv3 = layers.Conv1D(filters=256, strides=1, padding='SAME', kernel_size=2, activation='relu')(Pool2)
    Pool3 = layers.AvgPool1D(2, name='Pool3')(Conv3)
    x_3 = Flatten()(Pool3)
    x_3 = Dense(512, activation='relu')(x_3)
    x_3 = Dropout(0.1)(x_3)
    channel2 = Dense(256, activation='relu')(x_3)
    print(channel2.shape)
    # ------------------CNN通道（专门是为了提取物理化学特征和氨基酸组成特征）--------------
    concat = layers.Concatenate()([channel1, channel2])
    print(concat.shape)
    x = Dropout(0.1)(concat)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(2, activation='softmax')(x)
    model = keras.Model([inputs_1, inputs_mask, inputs_3], outputs)
    model.compile(loss=focal_loss(), optimizer=tf.optimizers.Adam(0.00001), metrics=["accuracy"])

    return model


tokenizer = BertTokenizer.from_pretrained("./data/vocab.txt")
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
protein_dict = dict((c, i) for i, c in enumerate(amino_acids))

# label，主要的训练集是2762个阳性，10089个阴性，测试集应该是有2522个阴性（0），691个阳性（1）
label_pos = np.ones((2762, 1), dtype=int)
label_neg = np.zeros((10089, 1), dtype=int)
label = np.append(label_pos, label_neg)

seq_length = 121

# 对input做了seq2num的操作
file = open('./data/dataset/main dataset/first/firstStage.faa', encoding="utf-8")

all_line = file.readlines()

fasta = []
input_data = []
for i in range(len(all_line)):
    if i % 2 == 1:  # 奇数行的时候，把奇数行的全部增加到fasta里去，结果是将文件里的肽序列全部拉了进fasta
        fasta.append(all_line[i][0:-1])

list2 = [i.ljust(121, 'O') for i in fasta]  # ljust是补充序列的
temp = ' '
list3 = []
for y in fasta:
    for i in range(len(y)):
        temp = temp + y[i] + ' '
    list3.append(temp)
    temp = ' '

token = tokenizer(list3, return_tensors='tf', padding=True, truncation=True)  # 我加了Padding之后好像就不用补零了
input_data = np.array(token["input_ids"])
input_mask = np.array(token["attention_mask"])
input_token_ids = np.array(token["token_type_ids"])

# --------------------------------load feature ---------------------------------------------------------------------
input_data3 = pd.read_csv("./data/feature/first_stage_AAC+CKSAAP+PAAC+PHYC.csv")  # Amino acid features
input_data3 = pd.DataFrame(input_data3)
input_data3 = np.array(input_data3.values)  # AAC，CKSAAP,PAAC的编码为12851*2444，加上phy10一共是12851*2453,应该是物理化学特性只有九个
input_data3 = input_data3[:, 1:]
transfer = MinMaxScaler(feature_range=[-1, 1])  # 这个是我自己用的数据预处理方式，最小最大值缩放
input_data3 = transfer.fit_transform(input_data3)
# --------------------------------load feature ---------------------------------------------------------------------

callback = myCallback()

original_time = time.time()

evaluation = []
evaluation.append(['Original time: {0}'.format(original_time)])
data = []
end_time = 0
file_dir = './model_firststage{0},{1}'.format(time.strftime("%Y-%m-%d", time.localtime()), original_time)
os.makedirs(file_dir)
mean_acc = []
mean_precision = []
mean_sensitivity = []
mean_specificity = []
mean_MCC = []
mean_AUC = []
mean_AUPR = []
mean_F1 = []

for i in range(5):
    fault_list = []
    model = build_model()
    model.summary()
    evaluation.append((['model_trainable_weights is'], ['{0}'.format(len(model.trainable_weights))]))

    seed = i + 520
    # 形成训练集和测试集
    train_set, test_set, train_set2, test_set2, train_set3, test_set3, label_train, label_test = train_test_split(
        input_data, input_mask, input_data3,
        label, test_size=0.2,
        train_size=0.8,
        random_state=seed,
        shuffle=True)

    fold_time = 0
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0o425)
    start_time = time.time()
    for train, test in kfold.split(train_set, label_train):
        if (fold_time == 0):
            label_train = np_utils.to_categorical(
                label_train)  # 因为现在的dense是二，并且kfold.split只能输入一维数组，所以先分五折交叉再one-hot

        evaluation.append(['model{0}_{1} and start time:{2}'.format(i, fold_time, start_time)])

        model.fit([train_set[train], train_set2[train], train_set3[train]], label_train[train],
                  validation_data=([train_set[test], train_set2[test], train_set3[test]], label_train[test]),
                  epochs=20,
                  batch_size=64,
                  callbacks=[callback])
        # 每折结束都保存模型

        evaluation.append(['epoch', 'acc', 'loss', 'val_acc', 'val_loss'])
        data = callback.get_data()
        for y in range(len(data)):
            evaluation.append(data[y])
        callback.reset_data()
        fold_time = fold_time + 1

    model.save_weights('{2}/model{0}_{1}.h5'.format(i, fold_time, file_dir))
    end_time = time.time()
    evaluation.append(['model{0} end time:{1},and model cost :{2}'.format(i, end_time, end_time - start_time)])
    preds = model.predict([test_set, test_set2, test_set3])
    preds = preds[:, 1]
    pred_y = np.rint(preds)
    # ----------------------------我把预测错误的打印了出来-------------------
    # for index,p in enumerate(pred_y):
    #     if p != label_test[index-1]:
    #         fault_list.append(([index-1],test_set[index-1]))
    # fault_list = pd.DataFrame(fault_list)
    # fault_list.to_csv(r"model{0}_{1}.csv".format(i, fold_time), header=None, index=False)

    acc, precision, sensitivity, specificity, MCC, F1 = calculate_performace(len(label_test), pred_y, label_test)
    fpr, tpr, _ = roc_curve(label_test, preds)
    AUC = auc(fpr, tpr)
    pre, rec, _ = precision_recall_curve(label_test, preds)
    AUPR = auc(rec, pre)

    print('model%d,acc=%f,precision=%f,sensitivity=%f,specificity=%f,MCC=%f,AUC=%f,AUPR=%f, F1=%f'
          % (i, acc, precision, sensitivity, specificity, MCC, AUC, AUPR, F1))
    evaluation.append(['acc', 'precision', 'sensitivity', 'specificity', 'MCC', 'AUC', 'AUPR', 'F1'])
    evaluation.append([str(acc), str(precision), str(sensitivity), str(specificity), str(MCC), str(AUC), str(AUPR),
                       str(F1)])
    mean_acc.append(acc)
    mean_precision.append(precision)
    mean_sensitivity.append(sensitivity)
    mean_specificity.append(specificity)
    mean_MCC.append(MCC)
    mean_AUC.append(AUC)
    mean_AUPR.append(AUPR)
    mean_F1.append(F1)

evaluation.append(
    ['mean_acc', 'mean_precision', 'mean_sensitivity', 'mean_specificity', 'mean_MCC', 'mean_AUC', 'mean_AUPR',
     'mean_F1'])
evaluation.append([np.mean(mean_acc), np.mean(mean_precision), np.mean(mean_sensitivity), np.mean(mean_specificity),
                   np.mean(mean_MCC), np.mean(mean_AUC), np.mean(mean_AUPR), np.mean(mean_F1)])

evaluation.append([np.std(mean_acc), np.std(mean_precision), np.std(mean_sensitivity), np.std(mean_specificity),
                   np.std(mean_MCC), np.std(mean_AUC), np.std(mean_AUPR), np.std(mean_F1)])

evaluation.append(["total_time:{0}".format(time.time() - original_time)])
evaluation = pd.DataFrame(evaluation)
evaluation.to_csv(r"{0}/evaluation.csv".format(file_dir), header=False, index=False)
# ---------------------------------------------以上是第一阶段只把序列按照氨基酸字典编码成数字的--------------------------------------------------------------
