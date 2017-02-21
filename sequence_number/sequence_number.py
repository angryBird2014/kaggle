import numpy as np
import tensorflow as tf
import pandas as pd
import random
class preprocess():

    def loadTrainDataFromCSV(self,filename):
        train_data = pd.read_csv(filename)
        print(train_data.count())
        return train_data['Sequence']


    def extractTrainData_label(self,data,Test = False):


        train_data =[]
        lable = []

        print(len(data))
        for index,value in data.iteritems():
            values = value.split(',')
            values_int = list(map(int,values))
            values_int = np.array(values_int)
            if Test is False:
                train_data.append(values_int[:-1])
                lable.append(values_int[-1])
            else:
                train_data.append(values_int)

        return train_data,lable

    def preDefineParameters(self,train_data,lable):
        data_size = len(train_data)
        sequence_max_length = 0
        sequence_length = []
        for row in train_data:
            sequence_length.append(len(row))
            if len(row) > sequence_max_length:
                sequence_max_length = len(row)
        return data_size,sequence_max_length,sequence_length

def constructGraph(batch_size,sequence_length):
    data = tf.placeholder(tf.float32,[None,sequence_max_length,1])
    target = tf.placeholder(tf.float32,[None,1])
    num_units = 128
    cell = tf.nn.rnn_cell.LSTMCell(num_units)
    out,state = tf.nn.dynamic_rnn(cell,data,sequence_length=sequence_length,dtype=tf.float32)
    out = tf.transpose(out,[1,0,2])  #[max_time,batch_size,num_units]
    last_step_index = out.get_shape()[0]-1
    final_stat = tf.gather(out,last_step_index) #final_stat [batch_size,num_units]
    weights = tf.Variable(tf.truncated_normal([num_units,int(target.get_shape()[1])]))
    bias = tf.Variable(tf.random_normal([batch_size,int(target.get_shape()[1])]))
    prediction = tf.nn.softmax(tf.matmul(final_stat,weights)+bias)
    loss = -tf.reduce_sum(target*tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
    train = tf.train.AdamOptimizer().minimize(loss)
    val = tf.equal(tf.cast(prediction,dtype=tf.float32),target)
#    accurate = tf.reduce_mean(val)
#    return train,accurate
    return train,loss,data,target



if __name__ == '__main__':
    pre = preprocess()
    train_data = pre.loadTrainDataFromCSV('train.csv')

    train_data, target = pre.extractTrainData_label(train_data)
    num_examples = int(len(train_data) * 0.8)
    train_data,target = train_data[:num_examples],target[:num_examples]
    test_data,test_target = train_data[num_examples:],target[num_examples:]
    batch_size = 1000
    epoch = 10000
    prediction = None
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        index = 0
        for i in range(epoch):
            DataTrain,lable = train_data[index:index+batch_size],target[index:index+batch_size]
#            DataTrain,lable = tf.convert_to_tensor(DataTrain),tf.convert_to_tensor(lable)
            index += batch_size
            data_train_new = []
            data_size, sequence_max_length, sequence_length = pre.preDefineParameters(train_data,lable)
            for element in DataTrain:
                if len(element) < sequence_max_length:
                    element = np.append(element,[-1]*(sequence_max_length-element))
                    data_train_new.append(data_train_new)
#            print(sequence_length)
            train, loss, data, target = constructGraph(batch_size, sequence_length)
            _, loss = sess.run([train, loss], feed_dict={data: data_train_new, target: lable})
            print("epoch", str(i), "loss", str(loss))
