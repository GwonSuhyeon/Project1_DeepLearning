

from keras import models
from keras.models import Sequential, Model
from keras.layers import Input, Dense, BatchNormalization, Activation, LSTM, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


a1 = 10
a2 = 10


def Get_TrainDataFile_Cnt():

    with open( "", "r" ) as file:
        fileCnt = file.readline()


    return int( fileCnt )


def Get_RealDataFile_Cnt():

    with open( "", "r" ) as file:
        fileCnt = file.readline()


    return int( fileCnt )


def Read_Data_File( fileCnt, csvFile, fileType, ColumnCnt, dataSetType ):

    Model_InputSize = 0


    if dataSetType == "A1":
        Model_InputSize = ColumnCnt * g
    elif dataSetType == "A2":
        Model_InputSize = ColumnCnt * g + h * 2
    elif dataSetType == "A3":
        Model_InputSize = ColumnCnt * g + 2
    elif dataSetType == "A4":
        Model_InputSize = ColumnCnt * g + h * 2
    elif dataSetType == "A5":
        Model_InputSize = ColumnCnt
    elif dataSetType == "A6":
        Model_InputSize = ( int )( ( ColumnCnt * g ) / 2 )
    elif dataSetType == "A7":
        Model_InputSize = ColumnCnt * g
    elif dataSetType == "A8":
        Model_InputSize = ColumnCnt
    elif dataSetType == "A9":
        Model_InputSize = 1
    elif dataSetType == "A10":
        Model_InputSize = g
    elif dataSetType == "A11":
        Model_InputSize = ColumnCnt
    elif dataSetType == "A12":
        Model_InputSize = ColumnCnt * g


    xData = np.empty( ( 0, Model_InputSize ), float )
    yData = np.empty( ( 0, 1 ), float )
    


    for cnt in range( 1, fileCnt + 1 ):
        
        fileName = csvFile + "_" + str( cnt ) + ".csv"

        data = pd.read_csv( "" + fileName, encoding = 'UTF-8' )


        if fileType == "A1":
            D1 = np.array( data[ [ 'a1', 'a2', 'a3', 'a4' ] ] )
            D2 = np.array( data [ [ 'a5' ] ] )
        elif fileType == "A2":
            D1 = np.array( data[ [ 'a6', 'a7', 'a8', 'a9', 'a10' ] ] )
            D2 = np.array( data [ [ 'a11' ] ] )

        


        if dataSetType == "A1":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1 ):

                a = np.array( [] )
                for k in range( i, i + g ):
                    a = np.append( a, D1[ k ] )

                xData = np.vstack( ( xData, a ) )
                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )


        elif dataSetType == "A2":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g - h ) + 1 ):

                a = np.array( [] )
                for k in range( i, i + g ):
                    a = np.append( a, D1[ k ] )

                for j in range( i + g, i + g + h ):
                    a = np.append( a, D1[ j ][ 0 ] )
                    a = np.append( a, D1[ j ][ 1 ] )


                xData = np.vstack( ( xData, a ) )
                yData = np.vstack( ( yData, D2[ i + g + h - 1 ] ) )


        elif dataSetType == "A3":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g * 2 ) + 1 ):
        
                a = np.array( [] )
                for k in range( 0, g ):
                    a = np.append( a, D1[ i + k ][ 0 ] )
                    a = np.append( a, D1[ i + k ][ 1 ] )

                    a = np.append( a, D1[ i + g + k ][ 2 ] )
                    a = np.append( a, D1[ i + g + k ][ 3 ] )


                a = np.append( a, D1[ i + g - 1 ][ 0 ] )
                a = np.append( a, D1[ i + g - 1 ][ 1 ] )

                xData = np.vstack( ( xData, a ) )
                yData = np.vstack( ( yData, D2[ i + g ] ) )


        elif dataSetType == "A4":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g - h ) + 1 ):

                a = np.array( [] )
                for k in range( i, i + g ):
                    a = np.append( a, D1[ k ] )

                for j in range( i + 30, i + 30 + h ):
                    a = np.append( a, D1[ j ][ 2 ] )
                    a = np.append( a, D1[ j ][ 3 ] )


                xData = np.vstack( ( xData, a ) )
                yData = np.vstack( ( yData, D2[ i + 30 + h - 1 ] ) )


        elif dataSetType == "A5":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1 ):

                a = np.array( [] )
                a1 = 0
                a2 = 0
                a3 = 0
                a4 = 0
                for k in range( i, i + g ):
                    a1 += D1[ k ][ 0 ]
                    a2 += D1[ k ][ 1 ]
                    a3 += D1[ k ][ 2 ]
                    a4 += D1[ k ][ 3 ]


                a = np.append( a, a1 )
                a = np.append( a, a2 )
                a = np.append( a, a3 )
                a = np.append( a, a4 )

                xData = np.vstack( ( xData, a ) )
                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )


        elif dataSetType == "A6":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1 ):

                a = np.array( [] )
                for k in range( i, i + g ):
                    a = np.append( a, D1[ k ][ 0 ] )
                    a = np.append( a, D1[ k ][ 2 ] )

                xData = np.vstack( ( xData, a ) )
                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )
                

        elif dataSetType == "A7":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1, g ):
                
                a = np.array( [] )
                for k in range( i, i + g ):
                    a = np.append( a, D1[ k ] )

                xData = np.vstack( ( xData, a ) )
                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )


        elif dataSetType == "A8":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1, g ):

                for k in range( i, i + g ):
                    a = np.array( [] )
                    a = np.append( a, D1[ k ] )

                    xData = np.vstack( ( xData, a ) )
                
                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )


        elif dataSetType == "A9":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1, 3 ):
                
                for k in range( i, i + g ):
                    xData = np.vstack( ( xData, D1[ k ][ 0 ] ) )
                    xData = np.vstack( ( xData, D1[ k ][ 1 ] ) )
                    xData = np.vstack( ( xData, D1[ k ][ 2 ] ) )
                    xData = np.vstack( ( xData, D1[ k ][ 3 ] ) )

                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )


        elif dataSetType == "A10":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1 ):

                a1 = np.array( [] )
                a2 = np.array( [] )
                a3 = np.array( [] )
                a4 = np.array( [] )
                
                for k in range( i, i + g ):
                    a1 = np.append( a1, D1[ k ][ 0 ] )
                    a2 = np.append( a2, D1[ k ][ 1 ] )
                    a3 = np.append( a3, D1[ k ][ 2 ] )
                    a4 = np.append( a4, D1[ k ][ 3 ] )

                xData = np.vstack( ( xData, a1 ) )
                xData = np.vstack( ( xData, a2 ) )
                xData = np.vstack( ( xData, a3 ) )
                xData = np.vstack( ( xData, a4 ) )
                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )


        elif dataSetType == "A11":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1 ):

                for k in range( i, i + g ):
                    a = np.array( [] )
                    a = np.append( a, D1[ k ] )

                    xData = np.vstack( ( xData, a ) )
                
                yData = np.vstack( ( yData, D2[ i + g - 1 ] ) )


        elif dataSetType == "A12":

            for i in range( 0, ( ( int )( D1.size / ColumnCnt ) - g ) + 1 ):

                a = np.array( [] )
                for k in range( i, i + g ):
                    a = np.append( a, D1[ k ] )

                xData = np.vstack( ( xData, a ) )

                B = np.array( [] )
                if D2[ i + g - 1 ] == 0:
                    B = np.append( B, 1 )
                    B = np.append( B, 0 )
                    B = np.append( B, 0 )

                if D2[ i + g - 1 ] == 1:
                    B = np.append( B, 0 )
                    B = np.append( B, 1 )
                    B = np.append( B, 0 )

                if D2[ i + g - 1 ] == 2:
                    B = np.append( B, 0 )
                    B = np.append( B, 0 )
                    B = np.append( B, 1 )

                yData = np.vstack( ( yData, B ) )




    return xData, yData, Model_InputSize


def Build_Model( modelType, ColumnCnt, Model_InputSize ):

    if modelType == "A_Model":

        model = Sequential()

        model.add( Dense( 32, input_shape = ( Model_InputSize, ), kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 64, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 128, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 64, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 32, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 1, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'linear' ) )


        optimizing = optimizers.Adam( lr = 0.001 )
        model.compile( optimizer = optimizing, loss = 'mse', metrics = [ 'mae' ] )

    elif modelType == "B_Model":

        model = Sequential()

        model.add( Dense( 32, input_shape = ( InputSize, ), kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 64, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )
        model.add( Dense( 64, kernel_initializer = 'he_uniform' ) )
        #model.add( BatchNormalization() )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 128, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )
        model.add( Dense( 128, kernel_initializer = 'he_uniform' ) )
        #model.add( BatchNormalization() )
        model.add( Activation( 'relu' ) )

        '''
        model.add( Dense( 256, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )
        model.add( Dense( 256, kernel_initializer = 'he_uniform' ) )
        #model.add( BatchNormalization() )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 128, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )
        model.add( Dense( 128, kernel_initializer = 'he_uniform' ) )
        #model.add( BatchNormalization() )
        model.add( Activation( 'relu' ) )
        '''

        model.add( Dense( 64, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'relu' ) )
        model.add( Dense( 64, kernel_initializer = 'he_uniform' ) )
        #model.add( BatchNormalization() )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 32, kernel_initializer = 'he_uniform' ) )
        #model.add( BatchNormalization() )
        model.add( Activation( 'relu' ) )

        model.add( Dense( 3, kernel_initializer = 'he_uniform' ) )
        model.add( Activation( 'softmax' ) )


        optimizing = optimizers.Adam( lr = 0.001 )
        model.compile( optimizer = optimizing, loss = 'categorical_crossentropy', metrics = [ 'accuracy' ] )

    

    return model


def LSTM_Model_Build():

    model = Sequential()

    model.add( LSTM( 16, input_shape = ( 4, 10 ), kernel_initializer = 'he_uniform' ) )
    model.add( Activation( 'relu' ) )

    model.add( Dense( 5, kernel_initializer = 'he_uniform' ) )
    model.add( Dense( 1, kernel_initializer = 'he_uniform' ) )

    optimizing = optimizers.Adam( lr = 0.001 )
    model.compile( optimizer = optimizing, loss = 'mse', metrics = [ 'mae' ] )


    return model


def CNN_Model_Build():

    model = Sequential()

    
    model.add( Conv1D( filters = 16, kernel_size = 2, strides = 1, input_shape = ( 10, 4 ), kernel_initializer = 'he_uniform', padding = 'same' ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv1D( filters = 32, kernel_size = 2, strides = 1, kernel_initializer = 'he_uniform', padding = 'same' ) )
    model.add( Activation( 'relu' ) )

    #model.add( MaxPooling1D( pool_size = 2 ) )
    

    '''
    model.add( Conv2D( filters = 16, kernel_size = ( 3, 3 ), strides = 1, input_shape = ( 10, 4, 1 ), kernel_initializer = 'he_uniform', padding = 'same' ) )
    model.add( Activation( 'relu' ) )

    model.add( Conv2D( filters = 32, kernel_size = ( 3, 3 ), strides = 1, kernel_initializer = 'he_uniform', padding = 'same' ) )
    model.add( Activation( 'relu' ) )

    #model.add( MaxPooling2D( pool_size = ( 2, 2 ) ) )
    '''

    model.add( Flatten() )

    model.add( Dense( 32, kernel_initializer = 'he_uniform' ) )
    model.add( Activation( 'relu' ) )

    model.add( Dense( 16, kernel_initializer = 'he_uniform' ) )
    model.add( Activation( 'relu' ) )

    model.add( Dense( 1, kernel_initializer = 'he_uniform' ) )

    optimizing = optimizers.Adam( lr = 0.001 )
    model.compile( optimizer = optimizing, loss = 'mse', metrics = [ 'mae' ] )


    return model


def plot_A_history(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(14, 15))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, loss, 'b--', label='train_loss')
    ax1.plot(epochs, val_loss, 'r:', label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss')
    ax1.grid()
    ax1.legend()
    
    acc = history_dict['mae']
    val_acc = history_dict['val_mae']
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, acc, 'b--', label='train_mae')
    ax2.plot(epochs, val_acc, 'r:', label='val_mae')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('mae')
    ax2.grid()
    ax2.legend()
    
    plt.show()

    return


def plot_B_history(history_dict):
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    
    epochs = range(1, len(loss) + 1)
    fig = plt.figure(figsize=(14, 15))
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, loss, 'b--', label='train_loss')
    ax1.plot(epochs, val_loss, 'r:', label='val_loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('loss')
    ax1.grid()
    ax1.legend()
    
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, acc, 'b--', label='train_accuracy')
    ax2.plot(epochs, val_acc, 'r:', label='val_accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('accuracy')
    ax2.grid()
    ax2.legend()
    
    plt.show()

    return


def plot_predicted( testY, predicted ):
    
    #testY_1d = testY.reshape( ( -1, ) )
    testY_1d = testY.flatten()
    predicted_1d = predicted.flatten()

    samples = range(1, testY.size + 1 )
    fig = plt.figure(figsize=(14, 15))
    
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(samples, testY_1d, 'b--', label='testY')
    ax1.plot(samples, predicted_1d, 'r:', label='predicted')
    ax1.set_xlabel('samples')
    ax1.set_ylabel('value')
    ax1.grid()
    ax1.legend()
    
    plt.show()

    return



def Show_Displot( data ):

    #plt.figure( figsize = ( 1, 1 ) )

    fig, ax= plt.subplots()

    fig.set_size_inches( 1, 1 )

    sns.displot( data )

    plt.show()


    return


def Show_Regplot( data ):

    figure, axes = plt.subplots( nrows = 2, ncols = 2 )
    #plt.tight_layout()
    figure.set_size_inches( 10, 8 )

    sns.regplot( x = 'a', y = 'e', data = data, ax = axes[ 0, 0 ], scatter_kws = { 'alpha' : 0.2 }, line_kws = { 'color' : 'blue' } )
    sns.regplot( x = 'b', y = 'e', data = data, ax = axes[ 0, 1 ], scatter_kws = { 'alpha' : 0.2 }, line_kws = { 'color' : 'blue' } )
    sns.regplot( x = 'c', y = 'e', data = data, ax = axes[ 1, 0 ], scatter_kws = { 'alpha' : 0.2 }, line_kws = { 'color' : 'blue' } )
    sns.regplot( x = 'd', y = 'e', data = data, ax = axes[ 1, 1 ], scatter_kws = { 'alpha' : 0.2 }, line_kws = { 'color' : 'blue' } )

    plt.show()


    return


def Show_Heatmap( data ):

    corrMat = data[ [ 'a', 'b', 'c', 'd', 'e' ] ].corr()

    fig, ax= plt.subplots()

    fig.set_size_inches( 8, 8 )

    sns.heatmap( corrMat, annot=True )

    plt.show()


    return


def Data_Scaler( data ):

    scaler = Normalizer()
    data[ 'a' ] = scaler.fit_transform( np.array( data[ 'a' ] ).reshape( -1, 1 ) )

    scaler = Normalizer()
    data[ 'b' ] = scaler.fit_transform( np.array( data[ 'b' ] ).reshape( -1, 1 ) )

    scaler = Normalizer()
    data[ 'c' ] = scaler.fit_transform( np.array( data[ 'c' ] ).reshape( -1, 1 ) )

    scaler = Normalizer()
    data[ 'd' ] = scaler.fit_transform( np.array( data[ 'd' ] ).reshape( -1, 1 ) )

    scaler = Normalizer()
    data[ 'e' ] = scaler.fit_transform( np.array( data[ 'e' ] ).reshape( -1, 1 ) )



    return data


def Verify_To_RealData( model, scalerX, scalerY ):

    fileCnt = Get_RealDataFile_Cnt()

    testX, testY, _ = Read_Data_File( fileCnt, "", "A", 4, "A8" )
    print( testX.shape, testY.shape )

    #_, testX, _, testY = train_test_split( testX, testY, test_size = 0.1 )
    #print( testX.shape, testY.shape )

    #스케일러 적용
    testX = scalerX.transform( testX )
    testY = scalerY.transform( testY )

    predicted = model.predict( testX )
    #print( "Predict : ", predicted )


    result = model.evaluate( testX, testY )
    print( "MAE : ", result[ 1 ] )

    print( "RMSE : ", np.sqrt( mean_squared_error( testY, predicted ) ) )
    print( "R2 : ", r2_score( testY, predicted ) )


    testY = scalerY.inverse_transform( testY )
    predicted = scalerY.inverse_transform( predicted )


    currentTime = time.strftime( '%y%m%d_%H%M%S', time.localtime( time.time() ) )


    As = pd.DataFrame( index = range( 0, len( testY ) ), columns = [ 'testY', 'predict' ] )
    As[ 'testY' ] = testY.reshape( -1, 1 )
    As[ 'predict' ] = predicted.reshape( -1, 1 )
    As.to_csv( "" + currentTime + ".csv", index = False )


    return


def Verify_LSTM_To_RealData( model, scalerX, scalerY ):

    fileCnt = Get_RealDataFile_Cnt()

    testX, testY, _ = Read_Data_File( fileCnt, "", "A", 4, "A10" )
    print( testX.shape, testY.shape )

    #_, testX, _, testY = train_test_split( testX, testY, test_size = 0.1 )
    #print( testX.shape, testY.shape )

    
    #스케일러 적용
    testX = scalerX.transform( testX )
    testY = scalerY.transform( testY )
    

    # A8
    #testX = testX.reshape( ( int )( len( testX ) / g ), g, 4 )

    # A9
    #testX = testX.reshape( ( int )( len( testX ) / ( g * 4 ) ), g * 4, 1 )

    # A10
    testX = testX.reshape( ( int )( len( testX ) / 4 ), 4, g )


    predicted = model.predict( testX )
    #print( "Predict : ", predicted )


    result = model.evaluate( testX, testY )
    print( "MAE : ", result[ 1 ] )

    print( "RMSE : ", np.sqrt( mean_squared_error( testY, predicted ) ) )
    print( "R2 : ", r2_score( testY, predicted ) )

    
    testY = scalerY.inverse_transform( testY )
    predicted = scalerY.inverse_transform( predicted )
    

    currentTime = time.strftime( '%y%m%d_%H%M%S', time.localtime( time.time() ) )


    As = pd.DataFrame( index = range( 0, len( testY ) ), columns = [ 'testY', 'predict' ] )
    As[ 'testY' ] = testY.reshape( -1, 1 )
    As[ 'predict' ] = predicted.reshape( -1, 1 )
    As.to_csv( "" + currentTime + ".csv", index = False )


    return


def Verify_CNN_To_RealData( model, scalerX, scalerY ):

    fileCnt = Get_RealDataFile_Cnt()

    testX, testY, _ = Read_Data_File( fileCnt, "", "A", 4, "A11" )
    print( testX.shape, testY.shape )

    #_, testX, _, testY = train_test_split( testX, testY, test_size = 0.1 )
    #print( testX.shape, testY.shape )

    
    #스케일러 적용
    testX = scalerX.transform( testX )
    testY = scalerY.transform( testY )
    

    testX = testX.reshape( ( int )( len( testX ) / g ), g, 4 )


    predicted = model.predict( testX )
    #print( "Predict : ", predicted )


    result = model.evaluate( testX, testY )
    print( "MAE : ", result[ 1 ] )

    print( "RMSE : ", np.sqrt( mean_squared_error( testY, predicted ) ) )
    print( "R2 : ", r2_score( testY, predicted ) )

    
    testY = scalerY.inverse_transform( testY )
    predicted = scalerY.inverse_transform( predicted )
    

    currentTime = time.strftime( '%y%m%d_%H%M%S', time.localtime( time.time() ) )


    As = pd.DataFrame( index = range( 0, len( testY ) ), columns = [ 'testY', 'predict' ] )
    As[ 'testY' ] = testY.reshape( -1, 1 )
    As[ 'predict' ] = predicted.reshape( -1, 1 )
    As.to_csv( "" + currentTime + ".csv", index = False )


    return


def Model1():

    fileCnt = Get_TrainDataFile_Cnt()


    D1, D2, Model_InputSize = Read_Data_File( fileCnt, "", "A", 4, "A7" )
    print( D1.shape, D2.shape )

    
    scalerX = StandardScaler()
    scaled_D1 = scalerX.fit_transform( D1 )

    scalerY = StandardScaler()
    scaled_D2 = scalerY.fit_transform( D2 )


    #Show_Displot( scaled_D2 )
    
    
    trainX, testX, trainY, testY = train_test_split( scaled_D1, scaled_D2, test_size = 0.3 )
    print( trainX.shape, trainY.shape, testX.shape, testY.shape )


    print( testX )
    print( testY )

    
    # 단일 모델

    model = Build_Model( "A_Model", 4, Model_InputSize )


    model.summary()

    history = model.fit( trainX, trainY, epochs = 100, batch_size = 128, validation_split = 0.3, shuffle = True )

    plot_A_history( history.history )

    predicted = model.predict( testX )
    print( "Predict : ", predicted )

    result = model.evaluate( testX, testY )
    print( "MAE : ", result[ 1 ] )

    print( "RMSE : ", np.sqrt( mean_squared_error( testY, predicted ) ) )
    print( "R2 : ", r2_score( testY, predicted ) )


    #print( testY.shape, predicted.shape )
    #plot_predicted( testY, predicted )


    testY = scalerY.inverse_transform( testY )
    predicted = scalerY.inverse_transform( predicted )


    currentTime = time.strftime( '%y%m%d_%H%M%S', time.localtime( time.time() ) )


    As = pd.DataFrame( index = range( 0, len( testY ) ), columns = [ 'testY', 'predict' ] )
    As[ 'testY' ] = testY.reshape( -1, 1 )
    As[ 'predict' ] = predicted.reshape( -1, 1 )
    As.to_csv( "" + currentTime + ".csv", index = False )
    

    Verify_To_RealData( model, scalerX, scalerY )
    

    return


def LSTM_Model2():

    fileCnt = Get_TrainDataFile_Cnt()


    D1, D2, Model_InputSize = Read_Data_File( fileCnt, "", "A", 4, "A10" )
    print( D1.shape, D2.shape )

    
    
    scalerX = StandardScaler()
    scaled_D1 = scalerX.fit_transform( D1 )
    
    scalerY = StandardScaler()
    scaled_D2 = scalerY.fit_transform( D2 )
    
    
    

    # A8
    #scaled_D1 = scaled_D1.reshape( ( int )( len( scaled_D1 ) / g ), g, 4 )

    # A9
    #scaled_D1 = scaled_D1.reshape( ( int )( len( scaled_D1 ) / ( g * 4 ) ), g * 4, 1 )

    # A10
    scaled_D1 = scaled_D1.reshape( ( int )( len( scaled_D1 ) / 4 ), 4, g )


    print( scaled_D1.shape )



    #Show_Displot( scaled_D2 )
    
    
    
    trainX, testX, trainY, testY = train_test_split( scaled_D1, scaled_D2, test_size = 0.2, shuffle = True )
    print( trainX.shape, trainY.shape, testX.shape, testY.shape )


    print( testX )
    print( testY )


    model = LSTM_Model_Build()

    model.summary()

    #earlyStop = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
    #history = model.fit( trainX, trainY, epochs = 1000, batch_size = 32, validation_split = 0.3, callbacks = [ earlyStop ], shuffle = True )
    history = model.fit( trainX, trainY, epochs = 300, batch_size = 128, validation_split = 0.2, shuffle = True )

    plot_A_history( history.history )

    predicted = model.predict( testX )
    print( "Predict : ", predicted )

    result = model.evaluate( testX, testY )
    print( "MAE : ", result[ 1 ] )

    print( "RMSE : ", np.sqrt( mean_squared_error( testY, predicted ) ) )
    print( "R2 : ", r2_score( testY, predicted ) )


    #print( testY.shape, predicted.shape )
    #plot_predicted( testY, predicted )

    
    testY = scalerY.inverse_transform( testY )
    predicted = scalerY.inverse_transform( predicted )
    

    currentTime = time.strftime( '%y%m%d_%H%M%S', time.localtime( time.time() ) )


    As = pd.DataFrame( index = range( 0, len( testY ) ), columns = [ 'testY', 'predict' ] )
    As[ 'testY' ] = testY.reshape( -1, 1 )
    As[ 'predict' ] = predicted.reshape( -1, 1 )
    As.to_csv( "" + currentTime + ".csv", index = False )
    

    Verify_LSTM_To_RealData( model, scalerX, scalerY )
    

    return


def CNN_Model3():

    fileCnt = Get_TrainDataFile_Cnt()


    D1, D2, Model_InputSize = Read_Data_File( fileCnt, "", "A", 4, "A11" )
    print( D1.shape, D2.shape )


    
    scalerX = StandardScaler()
    scaled_D1 = scalerX.fit_transform( D1 )
    
    scalerY = StandardScaler()
    scaled_D2 = scalerY.fit_transform( D2 )
    

    scaled_D1 = scaled_D1.reshape( ( int )( len( scaled_D1 ) / g ), g, 4 )


    #Show_Displot( scaled_D2 )
    
    
    
    trainX, testX, trainY, testY = train_test_split( scaled_D1, scaled_D2, test_size = 0.2, shuffle = True )
    print( trainX.shape, trainY.shape, testX.shape, testY.shape )


    print( testX )
    print( testY )


    model = CNN_Model_Build()

    model.summary()

    #earlyStop = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'auto')
    #history = model.fit( trainX, trainY, epochs = 1000, batch_size = 32, validation_split = 0.3, callbacks = [ earlyStop ], shuffle = True )
    history = model.fit( trainX, trainY, epochs = 300, batch_size = 128, validation_split = 0.2, shuffle = True )

    plot_A_history( history.history )

    predicted = model.predict( testX )
    print( "Predict : ", predicted )

    result = model.evaluate( testX, testY )
    print( "MAE : ", result[ 1 ] )

    print( "RMSE : ", np.sqrt( mean_squared_error( testY, predicted ) ) )
    print( "R2 : ", r2_score( testY, predicted ) )


    #print( testY.shape, predicted.shape )
    #plot_predicted( testY, predicted )

    
    testY = scalerY.inverse_transform( testY )
    predicted = scalerY.inverse_transform( predicted )
    

    currentTime = time.strftime( '%y%m%d_%H%M%S', time.localtime( time.time() ) )


    As = pd.DataFrame( index = range( 0, len( testY ) ), columns = [ 'testY', 'predict' ] )
    As[ 'testY' ] = testY.reshape( -1, 1 )
    As[ 'predict' ] = predicted.reshape( -1, 1 )
    As.to_csv( "" + currentTime + ".csv", index = False )
    

    Verify_CNN_To_RealData( model, scalerX, scalerY )


    return


def Model4():

    fileCnt = Get_TrainDataFile_Cnt()

    D1, D2 = Read_Data_File( fileCnt, "", "B", 5 )
    print( D1.shape, D2.shape )

    print( D1 )
    print( D2 )


    #scalerX = Normalizer()
    #scaled_D1 = scalerX.fit_transform( D1 )

    oneHot_D2 = to_categorical( D2 )


    trainX, testX, trainY, testY = train_test_split( D1, oneHot_D2, test_size = 0.3 )
    print( trainX.shape, trainY.shape, testX.shape, testY.shape )



    # 단일 모델

    model = Build_Model( "B_Model", 5 )
    model.summary()

    history = model.fit( trainX, trainY, epochs = 100, batch_size = 128, validation_split = 0.3 )

    plot_B_history( history.history )

    predicted = model.predict( testX )
    print( "Predict : ", predicted )


    B_testY = np.empty( ( 0, 1 ), int )
    for i in range( 0, len( testY ) ):
        max = testY[ i ][ 0 ]
        B = 0
        for k in range( 1, len( testY[ i ] ) ):
            if testY[ i ][ k ] > max:
                max = testY[ i ][ k ]
                B = k

        B_testY = np.vstack( ( B_testY, B ) )


    B_predicted = np.empty( ( 0, 1 ), int )
    for i in range( 0, len( predicted ) ):
        max = predicted[ i ][ 0 ]
        B = 0
        for k in range( 1, len( predicted[ i ] ) ):
            if predicted[ i ][ k ] > max:
                max = predicted[ i ][ k ]
                B = k

        B_predicted = np.vstack( ( B_predicted, B ) )



    result = model.evaluate( testX, testY )
    print( "Accuracy : ", result[ 1 ] )

    #print( testY.shape, predicted.shape )
    #plot_predicted( testY, predicted )


    As = pd.DataFrame( index = range( 0, len( testY ) ), columns = [ 'testY', 'predict' ] )
    As[ 'testY' ] = B_testY.reshape( -1, 1 )
    As[ 'predict' ] = B_predicted.reshape( -1, 1 )
    As.to_csv( "", index = False )



    return



#Model1()
LSTM_Model2()
#CNN_Model3()

#Model4()




# 앙상블 모델
'''
model1 = KerasClassifier( build_fn = Build_Model, epochs = 100 )
model1._estimator_type="classifier"
model2 = KerasClassifier( build_fn = Build_Model, epochs = 100 )
model2._estimator_type="classifier"
model3 = KerasClassifier( build_fn = Build_Model, epochs = 100 )
model3._estimator_type="classifier"

ensembleModel = VotingClassifier( estimators = [ ( 'model1', model1 ), ( 'model2', model2 ), ( 'model3', model3 ) ], voting = 'soft' )
ensembleModel.fit( trainX, trainY )

predicted = ensembleModel.predict( testX )
print( "Accuracy : ", accuracy_score( predicted, testY ) )
'''
