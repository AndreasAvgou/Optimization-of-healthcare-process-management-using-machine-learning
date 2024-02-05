from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

class RNN(BaseEstimator, RegressorMixin):
    def __init__(self, input_shape, epochs=30, batch_size=64):
        self.input_shape = input_shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = self.create_rnn_model()
        self.scaler = StandardScaler()

    def create_rnn_model(self):
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def fit(self, X, y, validation_data=None):
        X, y = check_X_y(X, y)
        X_scaled = self.scaler.fit_transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
        self.model.fit(X_reshaped, y, epochs=self.epochs, batch_size=self.batch_size, validation_data=validation_data, callbacks=callbacks)
        return self

    def predict(self, X):
        X = check_array(X)
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
        return self.model.predict(X_reshaped)
