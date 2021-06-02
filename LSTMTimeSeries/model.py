import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, LSTM, TimeDistributed, RepeatVector
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
epsilon = 1e-10


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)

class seq2seqLSTM():

    def __init__(self,lags,n_future,n_features = 6):
        encoder_inputs = Input(shape=(lags, n_features))
        encoder = LSTM(100, return_state=True)
        encoder_outputs = encoder(encoder_inputs)
        decoder_inputs = RepeatVector(n_future)(encoder_outputs[0])
        encoder_states = encoder_outputs[1:]
        decoder = LSTM(100, return_sequences=True)(decoder_inputs,initial_state=encoder_states)
        decoder_outputs = TimeDistributed(tf.keras.layers.Dense(n_features))(decoder)
        self.model = Model(encoder_inputs, decoder_outputs)
        self.reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
        self.model.compile(optimizer='adam', loss='mae')



    def summary(self):
        print(self.model.summary())

    def fit(self,train_scaled_values,train_scaled_labels,test_values, test_labels,epochs):
        self.model_fit = self.model.fit(train_scaled_values, train_scaled_labels, epochs=epochs, batch_size=16,
                                        verbose=0, callbacks=[self.reduce_lr],
                                        validation_data=(test_values, test_labels))

    def vis_train_history(self):
        training_loss = self.model_fit.history["loss"]
        test_loss = self.model_fit.history["val_loss"]
        epoch_counter = range(1,len(training_loss)+1)
        plt.plot(epoch_counter,training_loss,'--r')
        plt.plot(epoch_counter, test_loss, 'b-')
        plt.legend(["Training Loss", "Test Loss"])
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


    def predict(self,test_values):
        pred_e1d1 = self.model.predict(test_values)
        return pred_e1d1



def mean_directional_accuracy(actual, predicted):
    return np.mean((np.sign(actual[1:] - actual[:-1]) == np.sign(predicted[1:] - predicted[:-1])).astype(int))


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))
