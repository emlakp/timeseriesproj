from data_preproccessing import *
from model import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:f}'.format})




def experiment():
   data_path = str(input("Enter the path to data\n"))
   data = read_data(data_path)
   train_scaled, test_scaled, scaler = split(data, 0.9, "standard")
   n_past = int(input("Enter the number of lagged observations in the model\n"))
   n_future = int(input("Enter the number of predictions\n"))
   train_values, train_labels = gen_windows(train_scaled, n_past, n_future)
   model = seq2seqLSTM(n_past, n_future, 6)
   model.summary()
   test_values, test_labels = gen_windows(test_scaled, n_past, n_future)
   model.fit(train_values, train_labels, test_values, test_labels, 60)
   print('n_past = ',n_past)
   print("n_future=",n_future)
   predictions = model.predict(test_values)
   model.vis_train_history()
   predictions = scaler.inverse_transform(predictions)
   test_labels_inversed = scaler.inverse_transform(test_labels)

   predicts = []
   i = 0
   while i < len(predictions[:, :, 0]):
      predicts = predicts + predictions[i, :, 0].tolist()
      i += n_future

   actual = []
   j = 0
   while j < len(test_labels_inversed[:, :, 0]):
      actual = actual + test_labels_inversed[j, :, 0].tolist()
      j += n_future
   actual = np.array(actual)
   predicts = np.array(predicts)
   plt.figure(figsize=(20, 8))
   plt.grid(True)
   plt.plot(actual, color='red', label="Test values")
   plt.plot(predicts, color='blue', label="Predicted Values")
   plt.legend()
   plt.show()

   print("Mean directional accuracy is:",
         mean_directional_accuracy(actual, predicts))
   print("Root Mean squared error is:",
         mean_squared_error(actual, predicts,squared=False))


experiment()