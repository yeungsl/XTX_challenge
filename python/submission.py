import subprocess, sys, os
from core import Submission

sys.stdout = open(os.devnull, 'w')  # do NOT remove this code, place logic & imports below this line

"""
PYTHON submission

Implement the model below

##################################################### OVERVIEW ######################################################

1. Use get_next_data_as_string() OR get_next_data_as_list() OR get_next_data_as_numpy_array() to recieve the next row of data
2. Use the predict method to write the prediction logic, and return a float representing your prediction
3. Submit a prediction using self.submit_prediction(...)

################################################# OVERVIEW OF DATA ##################################################

1. get_next_data_as_string() accepts no input and returns a String representing a row of data extracted from data.csv
     Example output: '1619.5,1620.0,1621.0,,,,,,,,,,,,,1.0,10.0,24.0,,,,,,,,,,,,,1615.0,1614.0,1613.0,1612.0,1611.0,
     1610.0,1607.0,1606.0,1605.0,1604.0,1603.0,1602.0,1601.5,1601.0,1600.0,7.0,10.0,1.0,10.0,20.0,3.0,20.0,27.0,11.0,
     14.0,35.0,10.0,1.0,10.0,13.0'

2. get_next_data_as_list() accepts no input and returns a List representing a row of data extracted from data.csv,
   missing data is represented as NaN (math.nan)
     Example output: [1619.5, 1620.0, 1621.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1.0, 10.0,
     24.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, 1615.0, 1614.0, 1613.0, 1612.0, 1611.0, 1610.0,
     1607.0, 1606.0, 1605.0, 1604.0, 1603.0, 1602.0, 1601.5, 1601.0, 1600.0, 7.0, 10.0, 1.0, 10.0, 20.0, 3.0, 20.0,
     27.0, 11.0, 14.0, 35.0, 10.0, 1.0, 10.0, 13.0]

3. get_next_data_as_numpy_array() accepts no input and returns a Numpy Array representing a row of data extracted from
   data.csv, missing data is represented as NaN (math.nan)
   Example output: [1.6195e+03 1.6200e+03 1.6210e+03 nan nan nan nan nan nan nan nan nan nan nan nan 1.0000e+00
    1.0000e+01 2.4000e+01 nan nan nan nan nan nan nan nan nan nan nan nan 1.6150e+03 1.6140e+03 1.6130e+03 1.6120e+03
     1.6110e+03 1.6100e+03 1.6070e+03 1.6060e+03 1.6050e+03 1.6040e+03 1.6030e+03 1.6020e+03 1.6015e+03 1.6010e+03
      1.6000e+03 7.0000e+00 1.0000e+01 1.0000e+00 1.0000e+01 2.0000e+01 3.0000e+00 2.0000e+01 2.7000e+01 1.1000e+01
       1.4000e+01 3.5000e+01 1.0000e+01 1.0000e+00 1.0000e+01 1.3000e+01]

##################################################### IMPORTANT ######################################################

1. One of the methods get_next_data_as_string(), get_next_data_as_list(), or get_next_data_as_numpy_array() MUST be used and
   _prediction(pred) MUST be used to submit below in the solution implementation for the submission to work correctly.
2. get_next_data_as_string(), get_next_data_as_list(), or get_next_data_as_numpy_array() CANNOT be called more then once in a
   row without calling self.submit_prediction(pred).
3. In order to debug by printing do NOT call the default method `print(...)`, rather call self.debug_print(...)

"""
import pandas as pd
import numpy as np
import tensorflow as tf
import constants

# class MySubmission is the class that you will need to implement
class MySubmission(Submission):

    """
    get_prediction(data) expects a row of data from data.csv as input and should return a float that represents a
       prediction for the supplied row of data
    """
    def get_prediction(self, data):
        x = [float(x) if x else 0 for x in data.split(',')]
        bidSize0 = x[45]
        askSize0 = x[15]
        return 0.0025 * (bidSize0 - askSize0)

    """
    run_submission() will iteratively fetch the next row of data in the format 
       specified (get_next_data_as_string, get_next_data_as_list, get_next_data_as_numpy_array)
       for every prediction submitted to self.submit_prediction()
    """
    def run_submission(self):

        self.debug_print("Use the print function `self.debug_print(...)` for debugging purposes, do NOT use the default `print(...)`")
        self.build_and_train_LSTM()

        while(True):
            """
            NOTE: Only one of (get_next_data_as_string, get_next_data_as_list, get_next_data_as_numpy_array) can be used
            to get the row of data, please refer to the `OVERVIEW OF DATA` section above.

            Uncomment the one that will be used, and comment the others.
            """

            # data = self.get_next_data_as_list()
            # data = self.get_next_data_as_numpy_array()
            data = self.get_next_data_as_string()

            prediction = self.get_prediction(data)

            """
            submit_prediction(prediction) MUST be used to submit your prediction for the current row of data
            """
            self.submit_prediction(prediction)

    """
    import_training_data() imports training data set
        remember to change the data.csv to data_training.csv once everything runs
    """
    def import_data(self):
        training_set = pd.read_csv("../data-training.csv")
        training_set = training_set.fillna(0)
        return training_set
      
                

    """
    window_data(window_size, data_set) split the data sets into windows
    """
    def window_data(self, window_size, data_set):
        X = []
        Y = []
        for i in range(window_size, len(data_set)):
            X.append(data_set[i-window_size:i,])
            Y.append(data_set[i,])
        return np.array(X, dtype=np.float32),np.array(Y, dtype=np.float32)


    """
    splitting_data_set() split the data set into trainning set(80%) and testing set(20%)
    """
    def splitting_data_set(self):
        dataset = self.import_data()
        raw=np.array(dataset.iloc[:,1:].values, dtype=np.float32)
        self.debug_print(raw.shape)
        split_index = round(raw.shape[0]* constants.SPLIT)
        self.X_train, self.Y_train = self.window_data(constants.TIME_STEP_PRED, raw[:split_index,])
        self.X_test, self.Y_test = self.window_data(constants.TIME_STEP_PRED, raw[split_index+1:,])
        self.debug_print(self.X_train.shape)
        self.debug_print(self.X_test.shape)
        self.debug_print(self.Y_train.shape)
        self.debug_print(self.Y_test.shape)

    """
    build_and_train_LSTM() trains the LSTM with the splitted training set
    """

    def build_and_train_LSTM(self):
        self.splitting_data_set()

        LSTM_graph = tf.Graph()
        with LSTM_graph.as_default():
                inputs = tf.placeholder(dtype=tf.float32, shape=(None, constants.TIME_STEP_PRED, constants.HIDDEN_UNITS), name="input_placeholder")
                pred = tf.placeholder(dtype=tf.float32, shape=(None, constants.HIDDEN_UNITS), name="prediction_placeholder")
                
                weights={
                    'in':tf.Variable(tf.random_normal(shape=(constants.HIDDEN_UNITS, constants.RNN_UNITS))),
                    'out':tf.Variable(tf.random_normal(shape=(constants.RNN_UNITS, constants.HIDDEN_UNITS)))
                }
                
                biases={
                    'in':tf.Variable(tf.constant(0.1, shape=(constants.RNN_UNITS,))),
                    'out':tf.Variable(tf.constant(0.1, shape=(constants.HIDDEN_UNITS,)))
                }
                
                reshaped_inputs = tf.reshape(inputs, [-1, constants.HIDDEN_UNITS])
                self.debug_print(reshaped_inputs)
                input_rnn = tf.matmul(reshaped_inputs, weights['in']) + biases['in']
                input_rnn = tf.reshape(input_rnn, [-1, constants.TIME_STEP_PRED, constants.RNN_UNITS])
                self.debug_print(input_rnn.shape)
                
                lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(constants.RNN_UNITS)
                lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(constants.RNN_UNITS)
                drop_cell = [tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=1, output_keep_prob=1-constants.DROPOUT, state_keep_prob=1-constants.DROPOUT)
                            for lstm in[lstm_cell1, lstm_cell2]]
                cell=tf.contrib.rnn.MultiRNNCell(cells=drop_cell)
                
                init_state=cell.zero_state(batch_size=constants.BATCH_SIZE, dtype=tf.float32)
                self.debug_print(init_state)

                with tf.variable_scope('scope', reuse=tf.AUTO_REUSE):
                    output_rnn, states= tf.nn.dynamic_rnn(cell=cell, inputs=input_rnn, initial_state=init_state, dtype=tf.float32)
                self.debug_print(output_rnn)
                outputs = output_rnn[:,-1,:]
                self.debug_print(outputs.shape)
                h = tf.matmul(outputs, weights['out']) + biases['out']
                self.debug_print("predictor {}".format(h.shape))
                mse=tf.losses.mean_squared_error(labels=pred,predictions=h)
                optimizer=tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(loss=mse)
                init=tf.global_variables_initializer()
        
        with tf.Session(graph=LSTM_graph) as sess:
            sess.run(init)
            for epoch in range(1, constants.EPOCH+1):
                results = np.zeros(shape=(len(self.Y_test),constants.HIDDEN_UNITS))
                train_losses=[]
                test_losses=[]
                for j in range(len(self.Y_train)//constants.BATCH_SIZE):
                    _, train_loss=sess.run(
                                    fetches=(optimizer, mse),
                                    feed_dict={
                                        inputs:self.X_train[j*constants.BATCH_SIZE:(j+1)*constants.BATCH_SIZE,:,:],
                                        pred:self.Y_train[j*constants.BATCH_SIZE:(j+1)*constants.BATCH_SIZE,:]
                                    })
                    train_losses.append(train_loss)


                for j in range(len(self.Y_test)//constants.BATCH_SIZE):
                    result, test_loss=sess.run(
                                        fetches=(h,mse), 
                                        feed_dict={
                                            inputs:self.X_test[j*constants.BATCH_SIZE:(j+1)*constants.BATCH_SIZE,:,:],
                                            pred:self.Y_test[j*constants.BATCH_SIZE:(j+1)*constants.BATCH_SIZE,:]
                                        })
                    results[j*constants.BATCH_SIZE:(j+1)*constants.BATCH_SIZE] = result
                    test_losses.append(test_loss)

                if epoch % 10 == 0:
                    print("epoch: {}/{}".format(epoch, constants.EPOCH))
                    print("average training loss: ", sum(train_losses)/len(train_losses))
                    print("average testing loss: ", sum(test_losses)/len(test_losses))


if __name__ == "__main__":
    MySubmission()
