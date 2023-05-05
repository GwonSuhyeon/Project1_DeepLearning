

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import time

from keras import models
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Activation, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
from keras import initializers
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score



Predict = []
Real = []




class A_PositionalEncoder(nn.Module):

  def __init__( self, inputData : int, dimension : int, log_space : bool = False ):

    super().__init__()
    self.inputData = inputData
    self.dimension = dimension
    self.log_space = log_space

    self.outputData = self.inputData * ( 1 +  ( 2 * self.dimension ) )

    self.embed_fns = [lambda x: x]

    if self.log_space:
      freq_bands = 2.**torch.linspace(0., self.dimension - 1, self.dimension)
    else:
      freq_bands = torch.linspace(2.**0., 2.**(self.dimension - 1), self.dimension)

    for freq in freq_bands:
      self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
      self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
  
  # Positional Encoder에 입력이 들어왔을 때 호출되는 순전파 함수
  def forward(
    self,
    x
  ) -> torch.Tensor:
  
    return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)



class A_FileManager():

    def __init__( self ):

        super().__init__()

        self.TrainFileCnt = 0
        self.ValidFileCnt = 0

        self.TrainFileName = ""
        self.ValidFileName = ""

        self.RootFilePath = ""

        self.RawDataColumns = 5

        return


    def A_Get_TrainFile_Cnt( self ):

        with open( "", "r" ) as file:
            fileCnt = file.readline()

        self.TrainFileCnt = fileCnt

        return


    def A_Get_ValidFile_Cnt( self ):

        with open( "", "r" ) as file:
            fileCnt = file.readline()

        self.ValidFileCnt = fileCnt

        return


    def A_Read_TrainDataFile( self ):

        data = np.empty( ( 0, self.RawDataColumns ), float )

        for cnt in range( 1, int( self.TrainFileCnt ) + 1 ):

            fileName = self.TrainFileName + str( cnt ) + ".csv"

            data = np.concatenate( ( data, np.loadtxt( self.RootFilePath + fileName, delimiter = ",", dtype = np.float64 ) ), axis = 0 )

        return data


    def A_Read_ValidDataFile( self ):

        #data = pd.DataFrame()
        data = np.empty( ( 0, self.RawDataColumns ), float )

        for cnt in range( 1, int( self.ValidFileCnt ) + 1 ):

            fileName = self.ValidFileName + str( cnt ) + ".csv"

            data = np.concatenate( ( data, np.loadtxt( self.RootFilePath + fileName, delimiter = ",", dtype = np.float64 ) ), axis = 0 )

        return data


class A_Model( nn.Module ):

    def __init__( self, input : int, output : int ):

        super().__init__()

        self.input : int = input
        self.hiddenLayers : int = 10
        self.neurons : int = 512
        self.output : int = output
        self.activation = nn.functional.relu

        self.layers = nn.ModuleList( [ nn.Linear( self.input, self.neurons, bias = True ) ] + 
                                     [ nn.Linear( self.neurons + self.input , self.neurons, bias = True ) if hiddens in (int(self.hiddenLayers / 2), )
                                      else nn.Linear( self.neurons, self.neurons, bias = True ) for hiddens in range( self.hiddenLayers ) ] )

        self.outputLayer = nn.Linear( self.neurons, self.output, bias = True )

        for layer in self.layers:  
          torch.nn.init.kaiming_uniform_(layer.weight.data)
          torch.nn.init.zeros_(layer.bias.data)

        torch.nn.init.kaiming_uniform_(self.outputLayer.weight.data)
        torch.nn.init.zeros_(self.outputLayer.bias.data)

        return


    def forward( self, x : torch.Tensor ) -> torch.Tensor:

        x_input = x

        for i, layer in enumerate( self.layers ):

            x = self.activation( layer( x ) )

            if i in (int(self.hiddenLayers / 2), ):
                x = torch.cat([x, x_input], dim = -1)

            #x = self.activation( layer( x ) )

        #x = torch.sum( x, dim = -1 ).reshape( -1, 1 )

        x = self.outputLayer( x )

        return x



def train_loop(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)

    for batch, (X, y) in enumerate(dataloader):

        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ( batch * len(X) ) % 100 == 0:
        #if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):

    size = len(dataloader.dataset)

    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    y_total = 0
    acc = 0

    with torch.no_grad():
        for X, y in dataloader:

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            #correct = 0
            #y_total = 0
            #correct = torch.sum(pred, dim = 0).item() + 1e-10
            #y_total = torch.sum(y, dim = 0).item()
            #acc += ( correct / y_total )

            #Predict.append(pred)
            #Real.append(y)
            Predict.append(pred.numpy())
            Real.append(y.numpy())

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    #acc /= num_batches
    #print(f"Test Error: \n Accuracy: {(100*acc)}%, Avg loss: {test_loss:>8f} \n")





class TensorData(Dataset):

    def __init__(self, x_data : torch.Tensor, y_data : torch.Tensor):

        self.x_data : torch.Tensor = x_data
        self.y_data : torch.Tensor = y_data
        self.len = self.y_data.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len





'''

실행코드

'''


FileManager = A_FileManager()

FileManager.A_Get_TrainFile_Cnt()
FileManager.A_Get_ValidFile_Cnt()

print("Train File Cnt")
print(FileManager.TrainFileCnt)
print("Valid File Cnt")
print(FileManager.ValidFileCnt)
print("")

TrainData = FileManager.A_Read_TrainDataFile()
ValidData = FileManager.A_Read_ValidDataFile()

print("Raw Train Data")
print(TrainData)
print(TrainData.shape)
print("Raw Valid Data")
print(ValidData)
print(ValidData.shape)
print("")



TotalTrainData = TrainData.shape[0]
TotalValidData = ValidData.shape[0]





x_train = TrainData[..., :4]
y_train = TrainData[..., -1:]

x_valid = ValidData[..., :4]
y_valid = ValidData[..., -1:]


print("Split x_train, y_train")
print(x_train.shape)
print(y_train.shape)

print("Split x_valid, y_valid")
print(x_valid.shape)
print(y_valid.shape)
print("")

#Train_Tensor = [ x_train, y_train ]
#print(Train_Tensor)


#x_train = x_train.reshape( TotalTrainData, -1 )
#y_train = y_train.reshape( TotalTrainData, -1 )
#y_train = torch.sum( y_train, dim = -1 ).reshape( -1, 1 )
#print(x_train)

#x_valid = x_valid.reshape( TotalValidData, -1 )
#y_valid = y_valid.reshape( TotalValidData, -1 )
#y_valid = torch.sum( y_valid, dim = -1 ).reshape( -1, 1 )

#print("Origin x_train, y_train")
#print(x_train.shape)
#print(y_train.shape)

#print("Origin x_valid, y_valid")
#print(x_valid.shape)
#print(y_valid.shape)
#print("")





PE_Dimension = 20

Encoder = A_PositionalEncoder(4, PE_Dimension)
Encoder_y = A_PositionalEncoder(1, 0)



'''
Positional Encoding 사용
'''

# Train data encode inputs
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

Train_EncodedOutputs = Encoder(x_train)
Train_y_EncoderOutputs = Encoder_y(y_train)

print("Train Encoded Outputs")
print(Train_EncodedOutputs[:1])
print(Train_EncodedOutputs.shape)
print(Train_y_EncoderOutputs[:1])
print(Train_y_EncoderOutputs.shape)

#Train_EncodedOutputs = Train_EncodedOutputs.reshape( -1, 4 )
#Train_y_EncoderOutputs = Train_y_EncoderOutputs.reshape( -1, 1 )

print("Train Encoded Outputs Reshape")
print(Train_EncodedOutputs)
print(Train_EncodedOutputs.shape)
print(Train_y_EncoderOutputs)
print(Train_y_EncoderOutputs.shape)
print("")


# Valid data encode inputs
x_valid = torch.Tensor(x_valid)
y_valid = torch.Tensor(y_valid)

Valid_EncodedOutputs = Encoder(x_valid)
Valid_y_EncoderOutputs = Encoder_y(y_valid)

print("Valid Encoded Outputs")
print(Valid_EncodedOutputs[:1])
print(Valid_EncodedOutputs.shape)
print(Valid_y_EncoderOutputs[:1])
print(Valid_y_EncoderOutputs.shape)

#Valid_EncodedOutputs = Valid_EncodedOutputs.reshape( -1, 4 )
#Valid_y_EncoderOutputs = Valid_y_EncoderOutputs.reshape( -1, 1 )

print("Valid Encoded Outputs Reshape")
print(Valid_EncodedOutputs)
print(Valid_EncodedOutputs.shape)
print(Valid_y_EncoderOutputs)
print(Valid_y_EncoderOutputs.shape)
print("")


x_train = Train_EncodedOutputs
y_train = Train_y_EncoderOutputs
x_valid = Valid_EncodedOutputs
y_valid = Valid_y_EncoderOutputs




print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)

Valid_Data_Size = y_valid.shape[0]



learning_rate = 1e-5
batch_size = 64
#batch_size = 1 +  ( 2 * PE_Dimension )
epochs = 300


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")




input = 4 * ( 1 +  ( 2 * PE_Dimension ) )
#input = 4
#output = 1 * ( 1 + ( 2 * PE_Dimension ) )
output = 1


model = A_Model( input, output )
model.to(device)

print(model)



Train_Tensor = TensorData(x_train, y_train)
Valid_Tensor = TensorData(x_valid, y_valid)



train_dataloader = DataLoader(Train_Tensor, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(Valid_Tensor, batch_size=batch_size, shuffle=True)



loss_fn = nn.MSELoss()
#loss_fn = torch.nn.functional.mse_loss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)





for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    #test_loop(test_dataloader, model, loss_fn) 

test_loop(test_dataloader, model, loss_fn)

print("Done!")




result = pd.DataFrame( index = range( 0, Valid_Data_Size ), columns = [ 'predict', 'real' ] )


for i in range(len(Predict)):
    Predict[i] = Predict[i].reshape(-1)
    Real[i] = Real[i].reshape(-1)

for k in range(1, len(Predict)):
    #pos = len(result)
    #result['predict'].loc[pos] = Predict[k].reshape(-1, 1)
    Predict[0] = np.hstack([Predict[0], Predict[k]])
    Real[0] = np.hstack([Real[0], Real[k]])

result['predict'] = Predict[0].reshape(-1, 1)
result['real'] = Real[0].reshape(-1, 1)


print(result)



currentTime = time.strftime( '%y%m%d_%H%M%S', time.localtime( time.time() ) )
result.to_csv( "" + currentTime + ".csv", index = False )




#Predict_np = Predict[0].numpy()
#Real_np = Real[0].numpy()
Predict_np = Predict[0]
Real_np = Real[0]

print(len(Predict_np))

acc = 0.
data_size = Valid_Data_Size
for i in range(data_size):

    real = float(Real_np[i])
    if real != 0.0:
        acc += ( float(Predict_np[i]) / real )
    else:
        data_size -= 1

acc *= 100
acc /= data_size

print(f"Accuracy : {acc}")





#print("Weights")
#for layer in model.layers:
#    print(layer.weight.data)

#print(model.outputLayer.weight.data)


