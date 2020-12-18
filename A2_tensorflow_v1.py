import tensorflow as tf
#import numpy as np
import matplotlib.pyplot as plt
import time
import os
from scipy import stats

import autograd.numpy as np
from autograd import grad
#from tabulate import tabulate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"]="0" #for training on gpu

graph_directory = 'C:\\Users\\Ayesha\\Documents\\UofT\\Winter 2019\\ECE 421 - Intro to ML\\Assignments\\A2 Files\\Graphs\\'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

def calculate_accuracy(pred,y):
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return accuracy * 100
    


def build_cnn(learning_rate,reg,dropout=False, keep_prob=1):
    
   
    
    n_classes = 10
    epsilon = 1e-5
    K = 1000
    #both placeholders are of type float
    x = tf.placeholder(dtype=tf.float32,shape=[None,28,28,1],name="x")
    y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes],name="y")
    y_pred = tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name='y_pred')
    
    l2_regularizer = tf.contrib.layers.l2_regularizer(reg)
   
    
    
    
    #weights initialization
    weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer(), trainable= True, regularizer=l2_regularizer), 
    'wd1': tf.get_variable('W1', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer(), trainable= True, regularizer=l2_regularizer), 
    'out': tf.get_variable('W2', shape=(784,n_classes), initializer=tf.contrib.layers.xavier_initializer(), trainable= True, regularizer=l2_regularizer)
    }    
    
    #bias initialization
    biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer(), trainable= True, regularizer=l2_regularizer),
    'bd1': tf.get_variable('B1', shape=(784), initializer=tf.contrib.layers.xavier_initializer(), trainable= True, regularizer=l2_regularizer),
    'out': tf.get_variable('B2', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer(), trainable= True, regularizer=l2_regularizer)
    }
    
    #outputs a 27 x 27 matrix
    conv1 = conv2d(x,weights['wc1'],biases['bc1'],strides=1)
    
    mean, variance = tf.nn.moments(conv1,axes=[0, 1, 2])
    #TODO: should offset and scale be specified?
    batch_norm1 = tf.nn.batch_normalization(conv1, mean, variance,None,None,epsilon)
    
    #outputs a 14 x 14 matrix
    max_pool1 = maxpool2d(batch_norm1,k=2)
    flatten1 = tf.reshape(max_pool1, [-1, weights['wd1'].get_shape().as_list()[0]])
    
    
    fc1 = tf.add(tf.matmul(flatten1, weights['wd1']), biases['bd1'])
    
    
    #implement conditional dropout 
    
    if(dropout):
        fc1 = tf.nn.dropout(fc1,keep_prob)
    
    relu1 = tf.nn.relu(fc1)
    
    y_pred = tf.add(tf.matmul(relu1, weights['out']), biases['out'])
    
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))  + tf.losses.get_regularization_loss()
    
    accuracy = calculate_accuracy(y_pred,y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    return  x, y, weights, biases, y_pred, optimizer, accuracy, cost
    
    
    

def train_cnn(model_directory, epochs, minibatch_size, alpha, regularization_coeff=0.0, dropout=False, keep_prob=1):
    
    epoch_values = list(range(1,epochs+1))
    
    train_losses = []
    train_accuracies = []
    test_losses= []
    test_accuracies = []
    validation_losses= []
    validation_accuracies = []
    
    
    tf.set_random_seed(421)
    
    
    #data preprocessing
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    
    trainData = trainData.reshape(-1, 28, 28, 1)
    testData = testData.reshape(-1, 28, 28, 1)
    validData = validData.reshape(-1, 28, 28, 1)
    
    
    train_N = trainData.shape[0]
    
    no_of_batches = int(train_N/minibatch_size)
    
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    
    x, y, weights, biases, pred_op, train_op, accuracy_op, cost_op = build_cnn(alpha,regularization_coeff,dropout,keep_prob)
    
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.save(sess, model_directory,global_step=1000)


    
    for epoch_no in epoch_values:
        
        
        
        for batch_no in range(0,no_of_batches):
            
#            print("running epoch {}, batch {}".format(epoch_no,batch_no))
            
            startIndex = minibatch_size*(batch_no)
            endIndex = minibatch_size*(batch_no+1)
            tData_batch = trainData[startIndex:endIndex]
            tTarget_batch = trainTarget[startIndex:endIndex]
            
            #optimization of cost 
            sess.run([pred_op,cost_op,train_op], feed_dict={x:tData_batch,y:tTarget_batch})
            
        train_accuracy, train_loss = sess.run([accuracy_op, cost_op], feed_dict={x:tData_batch,y:tTarget_batch})
        validation_accuracy, validation_loss = sess.run([accuracy_op, cost_op], feed_dict={x:validData,y:validTarget})
        test_accuracy, test_loss = sess.run([accuracy_op, cost_op], feed_dict={x:testData,y:testTarget})
        
        #capture the accuracy and losses after each epoch
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)
        
        if epoch_no%10 == 0:
            print("epoch ",epoch_no)
            print("accuracy:\n training\t\t {}\n validation\t\t {}\ntest \t\t{}".format(train_accuracy,validation_accuracy,test_accuracy))
            print("loss:\n training\t\t {}\n validation\t\t {}\ntest \t\t{}".format(train_loss,validation_loss,test_loss))
        
        
        #shuffle training data after each epoch
        trainData, trainTarget = shuffle(trainData, trainTarget)
        
    #end for loop

    
    #close session
    sess.close()
    
    
    
    #print final accuracies
    print ({'Train Accuracy': train_accuracies[-1], 'Validation Accuracy': validation_accuracies[-1],'Test Accuracy': test_accuracies[-1]})
    
     #print final accuracies
    print ({'Train Loss': train_losses[-1], 'Validation Loss': validation_losses[-1],'Test Loss': test_losses[-1]})
    

    
    return {'epoch_values':epoch_values,
            'train_accuracies':train_accuracies,
            'test_accuracies':test_accuracies,
            'validation_accuracies':validation_accuracies,
            'train_losses':train_losses,
            'test_losses':test_losses,
            'validation_losses':validation_losses}

def plot_curves(data,regularization_coeff=0.0,dropout=False,keep_prob=1):
    dropout_str = "No Dropout" if (keep_prob==1 or not dropout) else "Dropout Keep Probability= {}".format(keep_prob)
    reg_str = "No Regularization" if (regularization_coeff==0) else "Lambda= {}".format(regularization_coeff)
    
    #plot accuracies
    plt.figure(1)
    plt.plot(data['epoch_values'],data['train_accuracies'],"r-",label='train')
    plt.plot(data['epoch_values'],data['test_accuracies'],"b--",label='test')
    plt.plot(data['epoch_values'],data['validation_accuracies'],"g:",label='validation')
    plt.legend()
    plt.title('Accuracy Curves with Lamda= {}, {}'.format(regularization_coeff,dropout_str))
    
    #plot losses
    plt.figure(2)
    plt.plot(data['epoch_values'],data['train_losses'],"r-",label='train')
    plt.plot(data['epoch_values'],data['test_losses'],"b--",label='test')
    plt.plot(data['epoch_values'],data['validation_losses'],"g:",label='validation')
    plt.legend()
    plt.title('Loss Curves with {}, {}'.format(reg_str,dropout_str))
    
    plt.show()


    
#Question 2.2 Model Training 
# SGD batch size of 32 and 50 epochs, alpha = 1e-4
# No regularization and no dropout
epochs = 50
batch_size = 32
alpha = 1e-4

loss_accuracy_data = train_cnn('saved-model\\model-1',epochs, batch_size,alpha)
plot_curves(loss_accuracy_data)


#Question 2.3 Hyperparameter Investigation - L2 Normalization
epochs =50
batch_size = 32
alpha = 1e-4

lambdas = [0.01,0.1, 0.5]

reg_loss_accuracy_data = []

for reg in lambdas:
    model_name = 'saved-model\\model-reg-{}'.format(reg).replace(".","_")
    
    single_loss_accuracy_data = train_cnn(model_name, epochs, batch_size, alpha, reg)
    plot_curves(single_loss_accuracy_data,reg)
    
    reg_loss_accuracy_data.append({'lambda':reg,'stats':single_loss_accuracy_data})


#Question 2.3 Dropout 

epochs = 50
batch_size = 32
alpha = 1e-4
reg= 0.0 #no regularization
keep_probabilities = [0.9, 0.75, 0.5]

dropout_loss_accuracy_data = []

for prob in keep_probabilities:
    model_name = 'saved-model\\model-prob-{}'.format(reg).replace(".","_")
    
    single_loss_accuracy_data = train_cnn(model_name, epochs, batch_size, alpha, reg, dropout=True, keep_prob=prob)
    plot_curves(single_loss_accuracy_data,reg,dropout=True,keep_prob=prob)
    
    dropout_loss_accuracy_data.append({'keep_prob':prob,'stats':single_loss_accuracy_data})







    


