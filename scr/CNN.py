# Python 3.6.13
# 
# tensorflow  1.2.0
# 
# Ubuntu 16.04

import numpy as np
import os
import tensorflow as tf
import numpy as np
import random 

class DataReaderRL:

    def __init__ ( self ):
        self.a = 0


    def get_filelist ( self, rootpath ):

        pathlist = list()

        country = os.listdir( rootpath )
        for i in range(  len(country) ):
            country[i] = rootpath + str( country[i] ) + '/'

        datelist = list()
        for i in range( len(country) ):
            datelist = os.listdir( country[i] ) 

            for j in range( len (datelist) ):
                pathlist.append( country[i] + datelist[j] + '/')


        pathlist.sort()

        #for i in range( len(pathlist) ):
         #   print pathlist[i]
        print('numof all data : ', len( pathlist ))
        return pathlist

 
    def readRaw_generate_X  (self, filepath, height, width ):

        # Generate height by wdith   input chart image
        

        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.split( '\nF\n'  )
        DataX   =   list()
        N       =   len( rawdata ) - 1
        Days    =   len( rawdata[0].split( '\nE\n' ) )
        print(Days, N)

        for c in range( N ) :
            state_seq  = rawdata[c].split( '\nE\n' )

            # matrix seq for company c
            matrix_seq = list()
            for t in range ( Days ):
                matrix  = np.zeros( ( height, width ) )
                rows    = state_seq[t].split('\n')
                #print(Days, t)
                # input matrix on day t
                for r in range ( height ):
                    row  = rows[r].split( ' ' )     
                    for w in range( width ):
                        matrix[r][w] = int( row[w] )

                matrix_seq.append( matrix )

            DataX.append ( matrix_seq )


        return DataX 
                                          
    def readRaw_generate_Y    ( self, filepath, N, Days ):

        # Generate input price change L_c^t
        f       =   open ( filepath, 'r' )
        rawdata =   f.read()
        rawdata =   rawdata.split( '\n'  )
        DataY   = list()

        if  (len(rawdata) - 1) != (N*Days) :
            print('number of input data is invalid')

        cnt     = 0
        for c in range ( N ) :
            return_seq = list()

            for t in range(Days):
                return_seq.append( float( rawdata [cnt] ) )
                cnt = cnt + 1

            DataY.append ( return_seq )

 
        return DataY 

class ConstructCNN:

    def __init__        ( self, Height, Width, FSize, PSize, PStride, NumAction ):

        self.H          = Height
        self.W          = Width
        self.FSize      = FSize
        self.PSize      = PSize
        self.PStride    = PStride
        self.NumAction  = NumAction
    


    def QValue   ( self, state, isTrain ):


        X           = tf.reshape            ( state, [ -1, self.H, self.W, 1] )
        M,V         = tf.nn.moments         ( X, [0,1,2,3] )
        X           = self.normalize_input  ( X, M, V )

        # CNN Layer
        Layer1, M1,V1    = self.stackedLayer ( 'L1', X,      self.FSize, self.PSize, self.PStride,  1,   16,  2, isTrain )
        Layer2, M2,V2    = self.stackedLayer ( 'L2', Layer1, self.FSize, self.PSize, self.PStride,  16,  32,  2, isTrain )
 
        # Fully Connected Network
        # L6    :  Batch x inputSize
        L6          = tf.contrib.layers.flatten( Layer2 )
        FC1         = self.FCLayer      ( "FC_1", L6, int( L6.get_shape()[-1]), 32, isTrain )
        FC2         = self.FinalLayer   ( "FC_2", FC1, 32, self.NumAction, isTrain )

        # FC2 : Batch * 3
        rho         = FC2
        eta         = tf.one_hot(  tf.argmax( rho, 1 ), self.NumAction, on_value = 1, off_value = 0, dtype = tf.int32 )

        return rho, eta 


    def optimize_Q          ( self, Q, A, Y, batchsize, learning_rate ):

        # Q : Batch * numaction
        # A : Batch * numaction 

        # update BatchNorm var first, and then update Loss, Opt
        updates     = tf.get_collection ( tf.GraphKeys.UPDATE_OPS )
        trablevars  = tf.trainable_variables()

        with tf.control_dependencies ( updates ):

            Loss    = tf.reduce_sum             ( tf.square(  Y - (Q*A) ) ) /   batchsize 
            opt     = tf.train.AdamOptimizer    ( learning_rate )

            grads   = opt.compute_gradients     ( Loss )
            minz    = opt.minimize              ( Loss )
        
        return  Loss, grads, updates, trablevars, minz 


    def FinalLayer             ( self, Name, Lin, inputSize, LayerSize, isTrain ):
        with tf.variable_scope(Name, reuse=True  ):

            # inputSize, LayerSize
            W   = tf.get_variable( Name, [ inputSize, LayerSize],  initializer = tf.contrib.layers.xavier_initializer()  )
            B   = tf.get_variable( Name + "_B" ,  initializer = tf.truncated_normal( [1,LayerSize],stddev = 0.01) )
            Out = tf.matmul (Lin, W) + B
            return Out


    def FCLayer             ( self, Name, Lin, inputSize, LayerSize, isTrain ):
        with tf.variable_scope(Name, reuse=True  ):

            # inputSize, LayerSize
            W   = tf.get_variable( Name, [ inputSize, LayerSize],  initializer = tf.contrib.layers.xavier_initializer()  )
            B   = tf.get_variable( Name + "_B" ,  initializer = tf.truncated_normal( [1,LayerSize],stddev = 0.01) )

            Out = tf.matmul (Lin, W) + B
            BN  = tf.contrib.layers.batch_norm( Out, scale = True, is_training = isTrain, scope = Name )
            return tf.nn.relu( BN )


    def stackedLayer   ( self, Name, Lin, Fsize, poolsize,  poolstride,  inSize, outSize,  numLayer, isTrain ):
        with tf.variable_scope(Name, reuse=True  ):
            
            L       = self.convLayer                (   Name+'_0' , Lin, Fsize, inSize,  outSize )
            BN      = tf.contrib.layers.batch_norm  (   L, scale = True, 
                                                    is_training=isTrain, scope=Name+'_0' )
            A       = tf.nn.relu                    (   BN )

            for i in range ( 1, numLayer ):
                L   = self.convLayer                (   Name+ '_' +str(i), A, Fsize, outSize, outSize )
                BN  = tf.contrib.layers.batch_norm  (   L, scale = True, 
                                                    is_training = isTrain, scope=Name+'_'+str(i) )

                A   = tf.nn.relu                    (   BN )
        
            Mlast, Vlast    = tf.nn.moments( BN, [ 0,1,2,3] )  
            Lout            = tf.nn.max_pool    (  A,   [1,poolsize,poolsize,1], [1,poolstride,poolstride,1], 'VALID' )
            return Lout, Mlast, Vlast


    def convLayer           ( self, Name, Lin, Fsize,Channel, Osize ):
        with tf.variable_scope(Name, reuse=True  ):

            # BHWC        
            W   = tf.get_variable   ( Name, [Fsize,Fsize,Channel,Osize], initializer = tf.contrib.layers.xavier_initializer() )
            L   = tf.nn.conv2d      ( Lin, W,   [1,1,1,1], 'SAME' )
            return L


    def normalize_input     ( self, X, M, V ):
        return ( X - M ) / tf.sqrt ( V )

# no need tensorflow
class exRep:

    def __init__( self, M, width, height ) :
        self.M          = M
        self.W          = width
        self.H          = height

        self.curS       = list()    # listof Matrix
        self.curA       = list()    # listof lenth 3 onehot vector
        self.curR       = list()    # listof Scalar
        self.nxtS       = list()    # listof Matrix

        # No Terminal State


    def remember ( self, curS, curA, curR, nxtS ):

        # remember current experience
        self.curS.append   ( curS )
        self.curA.append   ( curA )
        self.curR.append   ( round( curR, 4)  )
        self.nxtS.append    ( nxtS )

        # delete oldest experience
        if( len( self.curS ) > self.M ):
            del self.curS[0]
            del self.curA[0]
            del self.curR[0]
            del self.nxtS[0]


    def get_Batch   ( self, sessT, QA_Tuple, state_PH, isTrain_PH, Beta, numActions, Gamma ):

        curSs   = np.zeros( (Beta, self.H, self.W ) )
        curAs   = np.zeros( (Beta, numActions ) )
        Targets = np.zeros( (Beta, numActions ) )

        # get batchsize Beta random index from Memory Size List
        rIdxs   = random.sample( range( len( self.curS ) ), Beta )


        for k in range ( Beta ):

            input_kth   = self.curS[ rIdxs[k] ]
            action_kth  = self.curA[ rIdxs[k] ]

            QAValues    = sessT.run( QA_Tuple, feed_dict={ state_PH:self.nxtS[rIdxs[k]].reshape(1,self.H,self.W),isTrain_PH:False } )
            nxtQs       = QAValues[0]

            target_kth  = np.zeros( numActions)
            target_kth[ np.argmax(action_kth)] = self.curR[ rIdxs[k] ]  + Gamma * nxtQs[0][np.argmax(nxtQs[0])]

            curSs[k]    = input_kth
            curAs[k]    = action_kth
            Targets[k]  = target_kth
    
        return curSs, curAs, Targets

gpu_config = tf.ConfigProto()  
gpu_config.gpu_options.allow_growth = True # only use required resource(memory)
gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.5 # restrict to 50%


class trainModel:

    def __init__   ( self,  epsilon_init, epsilon_min, maxiter, Beta, B,C,  learning_rate, P  ):

        self.DataX          = list() 
        self.DataY          = list()

        self.epsilon        = epsilon_init 
        self.epsilon_min    = epsilon_min

        self.maxiter        = maxiter
        self.Beta           = Beta
        self.learning_rate  = learning_rate 
        self.P              = P 
        self.B              = B
        self.C              = C

    def set_Data    ( self, DataX, DataY ):
        self.DataX = DataX
        self.DataY = DataY

        print('X Data:  Comp#, Days# ', len( self.DataX ), len( self.DataX[0] ))
        print('Y Data:  Comp#, Days# ', len( self.DataY ), len( self.DataY[0] ))

    def trainModel ( self, H,W, FSize, PSize, PStride, NumAction, M, Gamma  ):

        # place holder
        state       = tf.placeholder ( tf.float32, [None,H,W] )
        isTrain     = tf.placeholder ( tf.bool, [] )

        Action      = tf.placeholder ( tf.float32,  [ None,NumAction ] )
        Target      = tf.placeholder ( tf.float32,[ None,NumAction ] )

        # construct Graph
        C           = ConstructCNN( H,W, FSize, PSize, PStride, NumAction )
        rho_eta     = C.QValue    ( state, isTrain  )
        Loss_Tuple  = C.optimize_Q( rho_eta[0], Action, Target, self.Beta, self.learning_rate )

        sess        = tf.Session ( config = gpu_config )    # maintains network parameter theta
        sessT       = tf.Session ( config = gpu_config )    # maintains target networ parameter theta^*
        sess.run ( tf.global_variables_initializer () )

        # saver
        saver       = tf.train.Saver( max_to_keep = 20 )

        # copy inital
        saver.save      ( sess, '/DeepQ/' )
        saver.restore   ( sess, '/DeepQ/' )
        saver.restore   ( sessT, '/DeepQ/' )

        # current experience
        preS    = np.empty( (1,H,W), dtype = np.float32 )
        preA    = np.empty( ( NumAction ), dtype = np.int32 )

        curS    = np.empty( (1,H,W), dtype = np.float32 )
        curA    = np.empty( (NumAction), dtype = np.int32 )
        curR    = 0
        nxtS    = np.empty( (H,W), dtype = np.float32 )

        memory  = exRep( M, W, H )  # memory buffer
        b       = 1                     # iteration counter

        while True:

            #1.0 get random valid index c, t
            c       = random.randrange( 0, len( self.DataX ) )
            t       = random.randrange( 1, len( self.DataX[c] ) -1  )

            #1.1 get preS
            preS    = self.DataX[c][t-1]
            
            #1.2 get preA by applying epsilon greedy policy to preS
            if( self.randf(0,1) <= self.epsilon):
                preA        = self.get_randaction   ( NumAction ) 
            else:                    
                QAValues    = sess.run              ( rho_eta, feed_dict={ state: preS.reshape(1,H,W), isTrain:False } )
                preA        = QAValues[1].reshape   ( NumAction )

            #1.3 get curS
            curS    = self.DataX[c][t]

            #1.4 get curA by applying epsilon greedy policy to curS
            if( self.randf(0,1) <= self.epsilon):
                curA        = self.get_randaction   ( NumAction ) 
            else:                    
                QAValues    = sess.run              ( rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False } )
                curA        = QAValues[1].reshape   ( NumAction )

            #1.5 get current reward and next state
            curR    = self.get_reward( preA, curA, self.DataY[c][t], self.P )
            nxtS    = self.DataX[c][t+1]

            #1.6 remember experience : tuple of curS, curA, curR, nxtS   
            memory.remember( curS, curA, curR, nxtS )

            #1.7: set epsilon                       
            if ( self.epsilon > self.epsilon_min ):
                self.epsilon = self.epsilon * 0.999999  

            #2: update network parameter theta  every  B iteration
            if ( len( memory.curS ) >= M ) and( b % self.B == 0 ) :

                #2.1:  update Target network parameter theta^*
                if( b % ( self.C * self.B ) == 0 )  : 
                    saver.save      ( sess, '/DeepQ/'  )
                    saver.restore   ( sessT, '/DeepQ/' )

                #2.2: sample Beta size batch from memory buffer and take gradient step with repect to network parameter theta 
                S,A,Y   = memory.get_Batch  ( sessT, rho_eta, state, isTrain,  self.Beta, NumAction, Gamma )
                Opts    = sess.run          ( Loss_Tuple, feed_dict = { state:S, isTrain:True, Action:A, Target:Y }  )

                #2.3: print Loss 
                if( b % ( 100 * self.B  ) == 0 ):
                    print('Loss: ' ,b, Opts[0]) 

            #3: update iteration counter
            b   = b + 1

            #4: save model 
            if( b >= self.maxiter ):
                saver.save( sess, "/DeepQ/" )
                print('Finish! ')
                return 0


    def validate_Neutralized_Portfolio       ( self, DataX, DataY, sess, rho_eta, state, isTrain, NumAction, H,W  ):
       
        # list
        N           = len( DataX )
        Days        = len( DataX[0] )
        curA        = np.zeros(( N, NumAction ))

        # alpha
        preAlpha_n  = np.zeros( N )
        curAlpha_n  = np.zeros( N )
        posChange   = 0

        # reward
        curR        = np.zeros( N )
        avgDailyR   = np.zeros( Days )


        # cumulative asset:  initialize cumAsset to 1.0
        cumAsset    = 1

        for t in range ( Days - 1 ):
    
            for c in range ( N ):
           
                #1: choose action from current state 
                curS        = DataX[c][t]
                QAValues    = sess.run  ( rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False } )
                curA[c]     = np.round  ( QAValues[1].reshape( ( NumAction) ) )
            
            # set Neutralized portfolio for day t
            curAlpha_n  = self.get_NeutralizedPortfolio ( curA,  N  )

            for c in range ( N ) :

                #1: get daily reward sum 
                curR[c]                     = np.round(  curAlpha_n[c] * DataY[c][t], 8)
                avgDailyR[t]                = np.round(  avgDailyR[t] + curR[c], 8 )

                #2: pos change sum
                posChange                   = np.round(  posChange +  abs( curAlpha_n[c] - preAlpha_n[c] ), 8)
                preAlpha_n[c]               = curAlpha_n[c]


        # calculate cumulative return
        for t in range( Days ):
            cumAsset = round ( cumAsset + ( cumAsset * avgDailyR[t] * 0.01  ), 8 )

        print('cumAsset ',  cumAsset)
        return N, posChange, cumAsset, avgDailyR


    def validate_TopBottomK_Portfolio       ( self, DataX, DataY, sess, rho_eta, state, isTrain, NumAction, H,W, K  ):

        # list
        N           = len( DataX )
        Days        = len( DataX[0] )

        # alpha
        preAlpha_s  = np.zeros( N )
        curAlpha_s  = np.zeros( N )
        posChange   = 0

        # reward
        curR        = np.zeros( N )
        avgDailyR   = np.zeros( Days )
      
        # cumulative asset: initialize curAsset to 1.0
        cumAsset    = 1

        # action value for Signals and Threshold for Top/Bottom K 
        curActValue = np.zeros( (N, NumAction ) )
        LongSignals = np.zeros( N )

        UprTH       = 0
        LwrTH       = 0

        for t in range ( Days - 1 ):

            for c in range ( N ):
           
                #1: choose action from current state 
                curS            = DataX[c][t]
                QAValues        = sess.run  ( rho_eta, feed_dict={ state: curS.reshape(1,H,W), isTrain:False } )
                curActValue[c]  = np.round  ( QAValues[0].reshape( ( NumAction) ), 4 )
                LongSignals[c]  = curActValue[c][0] - curActValue[c][2]

            # set Top/Bottom portfolio for day t
            UprTH, LwrTH        = self.givenLongSignals_getKTH  ( LongSignals, K, t  ) 
            curAlpha_s          = self.get_TopBottomPortfolio   ( UprTH, LwrTH, LongSignals, N )       

            for c in range ( N ):

                #1: get daily reward sum
                curR[c]                     = np.round(  curAlpha_s[c] * DataY[c][t], 8)
                avgDailyR[t]                = np.round(  avgDailyR[t] + curR[c], 8 )

                #2: pos change sum
                posChange                   = np.round(  posChange +  abs( curAlpha_s[c] - preAlpha_s[c] ), 8)
                preAlpha_s[c]               = curAlpha_s[c]


        # calculate cumulative return
        for t in range( Days ):
            cumAsset = round (cumAsset + ( cumAsset * avgDailyR[t] * 0.01  ), 8 )

        print('cumAsset ',  cumAsset)
        return N, posChange, cumAsset


    def TestModel_ConstructGraph    ( self, H,W, FSize, PSize, PStride,  NumAction  ):

        # place holder
        state       = tf.placeholder ( tf.float32, [None,H,W] )
        isTrain     = tf.placeholder ( tf.bool, [] )

        #print tf.shape( isTrain)
        #print(tf.__version__)

        # construct Graph
        C           = ConstructCNN( H,W, FSize, PSize, PStride, NumAction )
        rho_eta     = C.QValue    ( state, isTrain  )

        sess        = tf.Session ( config = gpu_config )
        saver       = tf.train.Saver()

        return sess, saver, state, isTrain, rho_eta

    def Test_TopBottomK_Portfolio   ( self, sess, saver, state, isTrain, rho_eta,  H,W, NumAction, TopK  ):

        saver.restore( sess, '/DeepQ/' )
        Outcome     = self.validate_TopBottomK_Portfolio (  self.DataX, self.DataY, sess, rho_eta, state, isTrain, NumAction, H,W, TopK  )

        print('NumComp#: ',  Outcome[0],  'Transactions: ', Outcome[1]/2, 'cumulative asset',Outcome[2])
        self.writeResult_daily( 'TestResult.txt', Outcome,  len ( self.DataX[0] ) -1  )


    def Test_Neutralized_Portfolio  ( self, sess, saver, state, isTrain,  rho_eta,  H,W, NumAction  ):

        saver.restore( sess, '/DeepQ/' )
        Outcome      = self.validate_Neutralized_Portfolio (  self.DataX, self.DataY, sess, rho_eta, state, isTrain, NumAction, H,W  )

        print('NumComp#: ',  Outcome[0],  'Transactions: ', Outcome[1]/2, 'cumulative asset',Outcome[2])
        self.writeResult_daily ( 'TestResult.txt', Outcome, len( self.DataX[0] ) -1  )
        self.write_avgDailyR('avgDailyR.txt', Outcome[3])

 
    def get_NeutralizedPortfolio         ( self, curA, N ):         
        
        alpha       = np.zeros( N )
        avg         = 0
        
        # get average
        for c in range ( N ):
            alpha[c]    = 1 - np.argmax( curA[c] )
            avg         = avg + alpha[c]
            
        avg     = np.round( avg / N, 4 )

        #set alpha
        sum_a       = 0
        for c in range ( N ):
            alpha[c]= np.round( alpha[c] - avg, 4 )
            sum_a   = np.round( sum_a + abs(alpha[c]), 4 )

        #set alpha
        if sum_a == 0 :
            return alpha

        for c in range ( N ):
            alpha[c] =np.round(  alpha[c] / sum_a, 8 )

        alpha[0] = 1 - np.argmax( curA[0] )

        return alpha


    def givenLongSignals_getKTH       ( self, LongSignals, K, t  ):
        
        Num         =  int( len(LongSignals) * K)
        SortedLongS =  np.sort( LongSignals )

        return SortedLongS[len(LongSignals) - Num], SortedLongS[Num-1]


    def get_TopBottomPortfolio              ( self, UprTH, LwrTH, LongSignals, N ):

        alpha   = np.zeros( N )
        sum_a   = 0

        for c in range ( N ):
            if LongSignals[c] >= UprTH:
                alpha[c] = 1
                sum_a = sum_a + 1
            elif LongSignals[c] <= LwrTH:
                alpha[c] = -1
                sum_a = sum_a+1
            else:
                alpha[c] = 0

        if sum_a == 0: 
            return alpha

        for c in range ( N ) :
            alpha[c] = np.round( alpha[c] / float(sum_a), 8 )

        return alpha
        

    def randf           ( self,  s, e):
        return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;

    def get_randaction  ( self,  numofaction ) :
        actvec      =  np.zeros( (numofaction), dtype = np.int32 )
        idx         =  random.randrange(0,numofaction)
        actvec[idx] = 1
        return actvec


    def get_reward    ( self, preA, curA, inputY, P ):
        
        # 1,0,-1 is assined to pre_act, cur_act 
        # for action long, neutral, short respectively
        pre_act = 1- np.argmax( preA ) 
        cur_act = 1- np.argmax( curA ) 

        return  (cur_act * inputY) - P * abs( cur_act - pre_act ) 


    def writeResult_daily    ( self,  filename,  outcome, numDays ):
        f = open( filename, 'a' )

        f.write( 'Comp#,'       + str( outcome[0]) + ',' )
        f.write( 'Days#'        + str( numDays-1 ) + ',' )
        f.write( 'TR#,'         + str( round( outcome[1]/2, 4) ) + ',' )
        f.write( 'FinalAsset,'  + str( round( outcome[2], 4 )) )
        
        f.write("\n")
        f.close()

    def write_avgDailyR    ( self,  filename,  outcome):
        f = open( filename, 'w' )

        for dailyR in outcome:
            f.write(str(round(dailyR,8)))
            f.write("\n")
        f.close()

FSize           = 5     
PSize           = 2
PStride         = 2
NumAction       = 3



# hyper parameters described in the paper 
#################################################################################
maxiter         = 5000000       # maxmimum iteration number         
learning_rate   = 0.00001       # learning rate
epsilon_min     = 0.1           # minimum epsilon

W               = 32            # input matrix size
M               = 1000          # memory buffer capacity
B               = 10            # parameter theta  update interval               
C               = 1000          # parameter theta^* update interval ( TargetQ )
Gamma           = 0.99          # discount factor
P               = 0.04             # transaction panalty while training.  0.05 (%) for training, 0 for testing
Beta            = 32            # batch size
#################################################################################

# initialize
DRead           = DataReaderRL()
Model           = trainModel( 1.0, epsilon_min, maxiter, Beta, B , C, learning_rate, P  )



######## Test Model ###########
'''
# folder list for testing 
folderlist                          =  DRead.get_filelist(  'Sample_Testing/')
sess,saver, state, isTrain, rho_eta = Model.TestModel_ConstructGraph( W,W,FSize,PSize,PStride,NumAction )

for i in range ( 0, len( folderlist) ):

    print(folderlist[i])
   
    filepathX       =   folderlist[i] + 'inputX.txt'
    filepathY       =   folderlist[i] + 'inputY.txt' 

    XData           =   DRead.readRaw_generate_X( filepathX, W, W )
    YData           =   DRead.readRaw_generate_Y( filepathY, len(XData), len(XData[0]) )   

    Model.set_Data                          ( XData, YData )
    Model.Test_Neutralized_Portfolio        ( sess, saver, state, isTrain, rho_eta, W, W, NumAction )
    Model.Test_TopBottomK_Portfolio         ( sess, saver, state, isTrain, rho_eta, W, W, NumAction,  0.2 )
'''

sess,saver, state, isTrain, rho_eta = Model.TestModel_ConstructGraph( W,W,FSize,PSize,PStride,NumAction )

filepathX       =   'input_test_X.txt'
filepathY       =   'input_test_Y.txt' 

XData           =   DRead.readRaw_generate_X( filepathX, W, W )
YData           =   DRead.readRaw_generate_Y( filepathY, len(XData), len(XData[0]) )   

Model.set_Data                          ( XData, YData )
Model.Test_Neutralized_Portfolio        ( sess, saver, state, isTrain, rho_eta, W, W, NumAction )
Model.Test_TopBottomK_Portfolio         ( sess, saver, state, isTrain, rho_eta, W, W, NumAction,  0.2 )

###################################

########## Train Model ############

'''
# folder path for training

filepathX       =   'input_train_X.txt'
filepathY       =   'input_train_Y.txt'


XData           = DRead.readRaw_generate_X( filepathX, W, W )                       # input chart
YData           = DRead.readRaw_generate_Y( filepathY, len(XData), len(XData[0]) )  # L_c^t  
Model.set_Data      ( XData, YData)
Model.trainModel    ( W,W, FSize, PSize, PStride, NumAction, M, Gamma )
'''
####################################


