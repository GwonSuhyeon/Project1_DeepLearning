

import pandas as pd


a = 2


fileAddrList = []
receiveDataList = []

a1 = []
a2 = []


def Read_DataFile( filePath ):

    fileAddrList.clear()

    with open( filePath, "r" ) as file:
        for line in file:
            fileAddrList.append( line.strip() )
        

    for i in range( 0, len( fileAddrList ) ):
        print( fileAddrList[ i ] )

    return


def Read_ReceiveFile( fileNumber ):
    
    receiveDataList.clear()

    with open( fileAddrList[ fileNumber ], "r" ) as file:
        for line in file:
            receiveDataList.append( line.strip().split() )

            idx = len( receiveDataList ) - 1
            
            receiveDataList[ idx ][ 0 ] = '%0.1f' % ( float( receiveDataList[ idx ][ 0 ] ) / float( 10 ) - a )
            receiveDataList[ idx ][ 1 ] = '%0.1f' % ( float( receiveDataList[ idx ][ 1 ] ) / float( 10 ) - b )
            receiveDataList[ idx ][ 3 ] = '%0.1f' % ( float( receiveDataList[ idx ][ 3 ] ) / float( 10 ) + c + d )
            receiveDataList[ idx ][ 4 ] = '%0.1f' % ( float( receiveDataList[ idx ][ 4 ] ) / float( 10 ) + e + f )
            
            receiveDataList[ idx ][ 6 ] = int( receiveDataList[ idx ][ 6 ] ) + 2

    return


def Extract_ReceiveFile():

    a1.clear()
    a2.clear()


    for i in range( 0, len( receiveDataList ) ):
        list = [ receiveDataList[ i ][ 0 ], receiveDataList[ i ][ 1 ], receiveDataList[ i ][ 3 ], receiveDataList[ i ][ 4 ], receiveDataList[ i ][ 6 ], receiveDataList[ i ][ 7 ] ]
        a1.append( list )


    for k in range( 0, len( receiveDataList ) - a ):
        list = [ receiveDataList[ k ][ 0 ], receiveDataList[ k ][ 1 ], receiveDataList[ k ][ 3 ], receiveDataList[ k ][ 4 ], receiveDataList[ k + a ][ 6 ] ]
        a2.append( list )

    return


def Make_Refined_DataFile( fileNumber, fileType ):

    if fileType == "Train":
        n1 = "" + "_" + str( fileNumber ) + ".csv"
        n2 = "" + "_" + str( fileNumber ) + ".csv"
    elif fileType == "Real":
        n1 = "" + "_" + str( fileNumber ) + ".csv"
        n2 = "" + "_" + str( fileNumber ) + ".csv"


    r1 = [ 'a', 'b', 'c', 'd', 'e', 'f' ]
    c1 = [ 'a', 'b', 'c', 'd', 'e' ]

    
    d1 = pd.DataFrame( a1, columns = r1 )
    d1.to_csv( "" + n1, index = False )
    print( d1 )
    
    d2 = pd.DataFrame( a2, columns = c1 )
    d2.to_csv( "" + n2, index = False )
    print( d2 )


    return



def Make_TrainData_Info_File( dataFileCnt ):

    with open( "", "w" ) as file:
        file.write( "%d" % dataFileCnt )


    return


def Make_RealData_Info_File( dataFileCnt ):

    with open( "", "w" ) as file:
        file.write( "%d" % dataFileCnt )


    return



Read_DataFile( "" )
for i in range( 0, len( fileAddrList ) ):
    Read_ReceiveFile( i )
    Extract_ReceiveFile()

    Make_Refined_DataFile( i + 1, "Train" )
Make_TrainData_Info_File( len( fileAddrList ) )




Read_DataFile( "" )
for i in range( 0, len( fileAddrList ) ):
    Read_ReceiveFile( i )
    Extract_ReceiveFile()

    Make_Refined_DataFile( i + 1, "Real" )
Make_RealData_Info_File( len( fileAddrList ) )

