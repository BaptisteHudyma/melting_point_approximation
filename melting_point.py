import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot

import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.constraints import min_max_norm

from keras.models import load_model


def soft_acc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))

class Model:
    def __init__(self):
        self.load_dataset()
        
        #separate test/train
        indexes = list(range(len(self.outputs)))
        np.random.shuffle(indexes)
        inTrain, inVal, outTrain, self.outVal = train_test_split(indexes, indexes, test_size=0.20, random_state=42)

        #train data
        self.trainInput = [self.inputs[i] for i in inTrain]
        self.trainOutput = [self.outputs[i] for i in outTrain]
        #test data
        self.valSet = [self.inputs[i] for i in inVal]
        self.valOut = [self.outputs[i] for i in self.outVal]


    def load_dataset(self):
        meltingDatas = pd.read_csv("transformVec.csv")
        rawOutputs = meltingDatas.mp.tolist()
        self.labels = meltingDatas.smiles.tolist()
        
        
        self.standardDev = float(np.std(rawOutputs))
        self.mean = float(np.mean(rawOutputs))
        
        self.outputs = list()
        self.inputs = list()
        for i in range(len(rawOutputs)):
            self.outputs.append( (rawOutputs[i] - self.mean) / self.standardDev)
        for i in range(len(self.outputs)):
            self.inputs.append([])
            for j in range(512):
                self.inputs[i].append(float(meltingDatas[str(j)][i]))
        
        #normalize input temperatures
        print((-158 - self.mean) / self.standardDev, min(self.outputs), "\n\n\n")
    
    def output2temperature(self, out):
        return out * self.standardDev + self.mean

    def train(self):
        #model setup
        model = Sequential()
        model.add(Dense(64, input_dim=512, kernel_initializer='normal', activation='tanh')) 
        model.add(Dropout(rate=0.2))
        model.add(Dense(32, activation='linear'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(8, activation='linear'))
        model.add(Dense(1, activation='linear'))# activation='linear'))
        
        model.compile(
                loss='mse', 
                optimizer= 'adam', 
                metrics=[soft_acc]
                )
        
        model.summary()
        print(" ")
        
        #callbacks
        mc = ModelCheckpoint('models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        
        history = model.fit(
                self.trainInput, 
                self.trainOutput, 
                epochs=500, 
                batch_size=50,
                validation_data=(self.valSet, self.valOut),
                callbacks=[es, mc]
        )
        
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()
        
        model.save("models/model.h5")
        print("Saved model to disk")


    def test(self):
        
        print("")
        print("Train Len :", len(self.trainInput))
        print("Val Len :", len(self.valSet))
        
        def soft_acc(y_true, y_pred):
            return K.mean(K.equal(K.round(y_true), K.round(y_pred)))
        
        #callbacks
        bestModel = load_model('models/best_model.h5', custom_objects={"soft_acc": soft_acc})
        
        labelVal = [self.labels[i] for i in self.outVal]
        predictions = bestModel.predict(self.valSet)
        # summarize the first 5 cases
        equartMoyen = 0
        for i in range(len(self.valSet)):
            predictedTemperature = round(self.output2temperature(self.valOut[i]), 2)
            realTemperature = round(self.output2temperature(predictions[i])[0], 2)
            equart = round(predictedTemperature - realTemperature, 2)
        
            equartMoyen += abs(equart)/len(self.valSet)
            print(realTemperature, "Celsius (expected", predictedTemperature, ") diff is", equart, "Celsius =>", labelVal[i])
        
        print("Écart moyen ", round(equartMoyen, 2), " degres Celsius")
        print("\n")
        eval = bestModel.evaluate(self.trainInput, self.trainOutput)
        print("Training loss/acc :", eval)
        eval = bestModel.evaluate(self.valSet, self.valOut)
        print("Tests loss/acc:", eval)
    
    
model = Model()
#model.train()
model.test()
    
    








