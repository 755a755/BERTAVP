import os, sys, re
# import keras.losses
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, Input, Dropout, Embedding, Flatten, MaxPooling1D, Conv1D, SimpleRNN, LSTM, GRU, \
    Multiply, GlobalMaxPooling1D
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, fbeta_score, recall_score, precision_score, \
    f1_score, accuracy_score
import time
from keras import backend as K
from protein_encoding import PC_6
from transformers import TFAutoModelForSequenceClassification, BertTokenizer, TFBertModel,PretrainedConfig
from sklearn.preprocessing import MinMaxScaler,StandardScaler


if tf.config.list_physical_devices('GPU') != []:
    visible_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices([visible_devices[1]],'GPU')

def focal_loss(alpha,gamma=2.):
    def focal_loss_fixed(y_true, y_pred):

        log_pred = K.log(y_pred)
        log_pred = tf.cast(log_pred, tf.float32)
        mul = y_true * log_pred
        log = alpha * K.pow(1.0 - y_pred, gamma) * tf.cast(mul, dtype=tf.float32)
        sum = K.sum(log,axis=1)
        loss = -K.mean(sum)

        return  loss
    return focal_loss_fixed


def to_one_hot(labels, dimension):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


class myCallback(tf.keras.callbacks.Callback):
    data = []

    def on_epoch_end(self, epoch, logs=None):
        acc_1 = logs.get('outputs_1_accuracy')
        acc_2 = logs.get('outputs_2_accuracy')
        loss = logs.get('loss')
        val_acc_1 = logs.get('val_outputs_1_accuracy')
        val_acc_2 = logs.get('val_outputs_2_accuracy')
        val_loss = logs.get('val_loss')
        self.data.append([epoch, acc_1, acc_2, loss, val_acc_1, val_acc_2, val_loss])

    def on_train_end(self, logs=None):
        pass

    def get_data(self):
        return self.data

    def reset_data(self, logs=None):
        self.data = []


def calculate_performace(label, pred_y):
    for i in range(len(pred_y)):
        max_value = max(pred_y[i])
        for j in range(len(pred_y[i])):
            if max_value == pred_y[i][j]:
                pred_y[i][j] = 1
            else:
                pred_y[i][j] = 0
    MacroP = precision_score(label, pred_y, average='macro')
    MacroR = recall_score(label, pred_y, average='macro')
    MacroF = f1_score(label, pred_y, average='macro')
    Accuracy = accuracy_score(label, pred_y)
    return Accuracy, MacroP, MacroR, MacroF


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




def build_model():

    inputs_1 = Input(shape=(123,),dtype=tf.int32)
    inputs_mask = Input(shape=(123,),dtype=tf.int32)
    inputs_3 = Input(shape=(2453, 1))  # inputs3 for CNN branch
    bert = TFBertModel.from_pretrained("../bert-base-uncased")
    x_1 = bert(inputs_1,attention_mask=inputs_mask)[0]
    conv1d_1 = layers.Conv1D(filters=64, strides=1, padding='SAME', kernel_size=2)(x_1)
    conv1d_2 = layers.Conv1D(filters=64, strides=1, padding='SAME', kernel_size=5)(x_1)
    conv1d_3 = layers.Conv1D(filters=64, strides=1, padding='SAME', kernel_size=8)(x_1)
    max_pooling1d_1 = layers.MaxPool1D(pool_size=2, name='max_pooling1d_1')(conv1d_1)
    max_pooling1d_2 = layers.MaxPool1D(pool_size=2, name='max_pooling1d_2')(conv1d_2)
    max_pooling1d_3 = layers.MaxPool1D(pool_size=2, name='max_pooling1d_3')(conv1d_3)
    concatenate = layers.Concatenate()([max_pooling1d_1, max_pooling1d_2, max_pooling1d_3])

    x = layers.Bidirectional(LSTM(units=128, return_sequences=True))(concatenate)
    x = Dropout(0.1)(x)
    x = Flatten()(x)
    channel1 = Dense(256, activation='relu')(x)
    #------------------Bert-------------------------------------

    Conv1 = layers.Conv1D(filters=512, strides=1, padding='SAME', kernel_size=2, activation='relu')(inputs_3)
    Pool1 = layers.MaxPool1D(2, name='Pool1')(Conv1)
    Conv2 = layers.Conv1D(filters=256, strides=1, padding='SAME', kernel_size=2, activation='relu')(Pool1)
    Pool2 = layers.MaxPool1D(2, name='Pool2')(Conv2)
    Conv3 = layers.Conv1D(filters=256, strides=1, padding='SAME', kernel_size=2, activation='relu')(Pool2)
    Pool3 = layers.MaxPool1D(2, name='Pool3')(Conv3)
    x_3 = Flatten()(Pool3)
    x_3 = Dense(512, activation='relu')(x_3)
    x_3 = Dropout(0.1)(x_3)
    channel2 = Dense(256, activation='relu')(x_3)
    #------------------CNN-----------------------------------------
    concat = layers.Concatenate()([channel1, channel2])

    outputs_1 = Dense(256, activation='relu')(concat)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = Dense(128, activation='relu')(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = Dense(64,activation='relu')(outputs_1)
    outputs_1 = Dropout(0.1)(outputs_1)
    outputs_1 = Dense(7, activation='softmax',name='outputs_1')(outputs_1)
    # ------------------Dense------------------------------------------
    outputs_2 = Dense(256, activation='relu')(concat)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = Dense(128, activation='relu')(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = Dense(64, activation='relu')(outputs_2)
    outputs_2 = Dropout(0.1)(outputs_2)
    outputs_2 = Dense(9, activation='softmax',name='outputs_2')(outputs_2)
    # ------------------Dense------------------------------------------
    model = keras.Model([inputs_1,inputs_mask,inputs_3], [outputs_1,outputs_2])
    model.compile(loss=losses, optimizer=tf.optimizers.Adam(0.00001), metrics=["accuracy"])
    return model


alpha_family = [0.16,0.079,0.14,0.34,0.14,0.04,0.01]
alpha_virus = [0.17,0.04,0.02,0.2,0.08,0.164,0.1,0.08,0.01]
tokenizer = BertTokenizer.from_pretrained("./data/vocab.txt")
losses = {'outputs_1': focal_loss(alpha_family), 'outputs_2':  focal_loss(alpha_virus)}
# losses = {'outputs_1': 'sparse_categorical_crossentropy', 'outputs_2': 'sparse_categorical_crossentropy'}
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
protein_dict = dict((c, i) for i, c in enumerate(amino_acids))

# label_family 2328
label_family = pd.read_csv("./data/dataset/main dataset/second/label_Family.csv", header=None)
label_family = np.array(pd.DataFrame(label_family))

# label_virus  2116
label_virus = pd.read_csv("./data/dataset/main dataset/second/label_Virus.csv", header=None)
label_virus = np.array(pd.DataFrame(label_virus))


encoder = LabelEncoder()
label_family = encoder.fit_transform(label_family)
label_virus = encoder.fit_transform(label_virus)
label_family = label_family.reshape((2347, 1))
label_virus = label_virus.reshape((2347, 1))


seq_length = 121

file = open('./data/dataset/main dataset/second/secondStage.faa', encoding="utf-8")
all_line = file.readlines()
fasta = []
for i in range(len(all_line)):
    if i % 2 == 1:
        fasta.append(all_line[i][0:-1])

list2 = [i.ljust(121, 'O') for i in fasta]
temp = ' '
list3 = []
for y in fasta:
    for i in range(len(y)):
        temp = temp + y[i] + ' '
    list3.append(temp)
    temp = ' '

token = tokenizer(list3, return_tensors='tf', padding=True, truncation=True)
input_data = np.array(token["input_ids"])
input_mask = np.array(token["attention_mask"])
# --------------------------------load ACC+CKSAAP+PAAC+PHYC---------------------------------------------------------------------
input_data3 = pd.read_csv("./data/feature/second_AAC+CKSAAP+PAAC+PHYC.csv")  # Amino acid features
input_data3 = pd.DataFrame(input_data3)
input_data3 = np.array(input_data3.values)
transfer=MinMaxScaler(feature_range=[-1,1])
input_data3=transfer.fit_transform(input_data3)
# --------------------------------load ACC+CKSAAP+PAAC+PHYC---------------------------------------------------------------------

callback = myCallback()
original_time = time.time()

#
evaluation = []
evaluation.append(['Original time: {0}'.format(original_time)])
data = []
end_time = 0
file_dir = './model_secondstage{0},{1}'.format(time.strftime("%Y-%m-%d", time.localtime()), original_time)
os.makedirs(file_dir)
mean_Accuracy_1 = []
mean_MacroP_1 = []
mean_MacroR_1 = []
mean_MacroF_1 = []

mean_Accuracy_2 = []
mean_MacroP_2 = []
mean_MacroR_2 = []
mean_MacroF_2 = []


for i in range(5):
    model = build_model()
    evaluation.append((['model_trainable_weights is'], ['{0}'.format(len(model.trainable_weights))]))
    seed = i + 520

    train_set,test_set,train_set2, test_set2, train_set3, test_set3,label_train_family, label_test_family,label_train_virus, label_test_virus\
    = train_test_split(input_data, input_mask, input_data3, label_family, label_virus,test_size=0.2,
                                                                    train_size=0.8, random_state=seed, shuffle=True)

    fold_time = 0
    kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=0o425)
    start_time = time.time()
    for train, test in kfold.split(train_set, label_train_family):
        evaluation.append(['model{0}_{1} and start time:{2}'.format(i, fold_time, start_time)])

        label_train_family_1= to_one_hot(label_train_family, dimension=7)

        label_train_virus_1 = to_one_hot(label_train_virus, dimension=9)

        model.fit([train_set[train], train_set2[train], train_set3[train]], [label_train_family_1[train],label_train_virus_1[train]],
                    validation_data=([train_set[test], train_set2[test], train_set3[test]], [label_train_family_1[test], label_train_virus_1[test]]), epochs=50, batch_size=64,
                    callbacks=[callback])

        model.save_weights('{2}/model{0}_{1}.h5'.format(i, fold_time, file_dir))
        evaluation.append(['epoch', 'acc_1', 'acc_2', 'loss', 'val_acc_1', 'val_acc_2', 'val_loss'])
        data = callback.get_data()
        for y in range(len(data)):
            evaluation.append(data[y])
        callback.reset_data()
        fold_time = fold_time + 1

    end_time = time.time()
    evaluation.append(['model{0} end time:{1},and model cost :{2}'.format(i, end_time, end_time - start_time)])
    #---------------------test-----------------------------------


    # family data and label
    x_test_family_1= []
    x_test_family_2= []
    x_test_family_3 = []
    y_test_family = []
    for x, y in enumerate(label_test_family):
        if y != 6:
            x_test_family_1.append(test_set[x])
            x_test_family_2.append(test_set2[x])
            x_test_family_3.append(test_set3[x])
            y_test_family.append(label_test_family[x])
    # features
    x_test_family_1 = np.array(x_test_family_1)
    x_test_family_2 = np.array(x_test_family_2)
    x_test_family_3 = np.array(x_test_family_3)
    y_test_family = np.array(y_test_family)
    y_test_family = to_one_hot(y_test_family, dimension=6)



    x_test_virus_1 = []
    x_test_virus_2 = []
    x_test_virus_3 = []
    y_test_virus = []
    for x, y in enumerate(label_test_virus):
        if y != 8:
            x_test_virus_1.append(test_set[x])
            x_test_virus_2.append(test_set2[x])
            x_test_virus_3.append(test_set3[x])
            y_test_virus.append(label_test_virus[x])
    # features
    x_test_virus_1 = np.array(x_test_virus_1)
    x_test_virus_2 = np.array(x_test_virus_2)
    x_test_virus_3 = np.array(x_test_virus_3)
    y_test_virus = np.array(y_test_virus)
    y_test_virus = to_one_hot(y_test_virus, dimension=8)

    y_family = np.array(label_test_family)
    y_family = to_one_hot(y_family, dimension=7)
    # label_virus
    y_virus = np.array(label_test_virus)
    y_virus = to_one_hot(y_virus, dimension=9)

    # ---------------------------------Task 1 Prediction-------------------------------------------
    preds_family = model.predict([x_test_family_1,x_test_family_2,x_test_family_3])
    preds_family = preds_family[0][:, :6]
    print("******************************task1********************************")
    Accuracy, MacroP, MacroR, MacroF = calculate_performace(y_test_family, preds_family)
    print('model%d,Accuracy=%f,MacroP=%f,MacroR=%f,MacroF=%f' % (i, Accuracy, MacroP, MacroR, MacroF))
    evaluation.append(['task1 prediction'])
    evaluation.append(['Accuracy', 'MacroP', 'MacroR', 'MacroF'])
    evaluation.append([str(Accuracy), str(MacroP), str(MacroR), str(MacroF)])
    evaluation.append('\n')
    mean_Accuracy_1.append(Accuracy)
    mean_MacroP_1.append(MacroP)
    mean_MacroR_1.append(MacroR)
    mean_MacroF_1.append(MacroF)
    # ---------------------------------Task 2 Prediction-------------------------------------------
    preds_virus = model.predict([x_test_virus_1,x_test_virus_2,x_test_virus_3])
    preds_virus = preds_virus[1][:, :8]
    Accuracy, MacroP, MacroR, MacroF = calculate_performace(y_test_virus, preds_virus)
    print("******************************task2********************************")
    print('Accuracy=%f,MacroP=%f,MacroR=%f,MacroF=%f' % (Accuracy, MacroP, MacroR, MacroF))
    evaluation.append(['task2 prediction'])
    evaluation.append(['Accuracy', 'MacroP', 'MacroR', 'MacroF'])
    evaluation.append([str(Accuracy), str(MacroP), str(MacroR), str(MacroF)])
    evaluation.append('\n')
    mean_Accuracy_2.append(Accuracy)
    mean_MacroP_2.append(MacroP)
    mean_MacroR_2.append(MacroR)
    mean_MacroF_2.append(MacroF)

evaluation.append(['task1 mean'])
evaluation.append(['mean_Accuracy', 'mean_MacroP', 'mean_MacroR', 'mean_MacroF'])
evaluation.append(
    [np.mean(mean_Accuracy_1), np.mean(mean_MacroP_1), np.mean(mean_MacroR_1), np.mean(mean_MacroF_1)])

evaluation.append(
    [np.std(mean_Accuracy_1), np.std(mean_MacroP_1), np.std(mean_MacroR_1),np.std(mean_MacroF_1)])

evaluation.append(['task2 mean'])
evaluation.append(['mean_Accuracy', 'mean_MacroP', 'mean_MacroR', 'mean_MacroF'])
evaluation.append(
    [np.mean(mean_Accuracy_2), np.mean(mean_MacroP_2), np.mean(mean_MacroR_2), np.mean(mean_MacroF_2)])

evaluation.append(
    [np.std(mean_Accuracy_2), np.std(mean_MacroP_2), np.std(mean_MacroR_2),np.std(mean_MacroF_2)])


evaluation.append(["total_time:{0}".format(time.time() - original_time)])
evaluation = pd.DataFrame(evaluation)
evaluation.to_csv(r"{0}/evaluation.csv".format(file_dir), header=False, index=False)

