import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#graph_directory = 'C:\\Users\\megha\\Documents\\UOFT\\UOFT Winter2019\\Machine Learning- ECE421\\Assignment2\\Xavier_Graphs\\'
graph_directory = 'C:\\Users\\megha\\Documents\\UOFT\\UOFT Winter2019\\Machine Learning- ECE421\\Assignment2\\Graphs\\'

#Early Stopping
#max_checks_without_progress = 20
#checks_without_progress = 0



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


def relu(x):
    return np.where(x>0,x,0)
    # TODO
    
def grad_relu(x):
    
    return np.where(x>0,1,0)
   
def gradCE(target, prediction):
    
    s = softmax(prediction)
    grad = np.sum(np.matmul(target.T,s),axis=0)
    grad = (1/target.shape[0]) * grad
    
    return grad
        
        

def softmax(x):
    # TODO
#    print("before softmax",x[0])
    tmp = x - x.max(axis=1).reshape(-1,1)
    exp_x = np.exp(tmp)
    sum_of_exp = np.sum(exp_x,axis=1).reshape(-1,1)
    softmax_x = exp_x/sum_of_exp
    
#    print("before softmax",softmax_x[0])
    
    return softmax_x


def compute(X, W, b):
    # TODO
    print(X.shape,W.shape)
    return np.matmul(X,W) + b

def averageCE(target, prediction):
    
    ce_per_class = np.multiply(target,np.log(prediction))
    ce_per_point = np.sum(ce_per_class,axis=1)
    average_ce = -1*np.mean(ce_per_point)

    
    return average_ce


def calculate_accuracy(target,prediction):

    correct_guesses = np.where(np.argmax(prediction,axis=1) == np.argmax(target,axis=1),1,0)
    return np.sum(correct_guesses)*100/target.shape[0]
    
def calculate_loss_and_accuracy(data,target,W_hidden,bias_hidden, W_output, bias_output):
    
    S_hidden=compute(data,W_hidden,bias_hidden)
    #S_hidden = np.matmul(data,W_hidden) + bias_hidden
    X_hidden = relu(S_hidden)
    S_output=compute(X_hidden,W_output,bias_output)
    #S_output = np.matmul(X_hidden,W_output) + bias_output
    X_output = softmax(S_output)
        
    #    print("X_output",np.sum(X_output,axis=1).shape) 
    CE_loss = averageCE(target,X_output)
    print("CE loss",CE_loss)
    
    accuracy = calculate_accuracy(target,X_output)
    
    return CE_loss, accuracy

def train_neural_network(K,epochs,trainData, validData, testData, trainTarget, validTarget, testTarget):
   
    train_losses = []
    train_accuracies = []
    test_losses= []
    test_accuracies = []
    validation_losses= []
    validation_accuracies = []
    epoch_values = list(range(1,epochs+1))

    #best_loss = np.infty #For early stopping
    
    train_N = trainData.shape[0]
    test_N = testData.shape[0]
    valid_N = validData.shape[0]
    
    X_input = trainData.reshape(train_N,784)
    X_test_input = testData.reshape(test_N,784)
    X_valid_input = validData.reshape(valid_N,784)
    
    F = 784
#    dimension_input = F+1
#    dimension_hidden = K+1
    dimension_output = 10
    
    
    gamma = 0.9 # or 0.9
    alpha = 0.2
    
    #Weight Initialization
    #Xavier Initiliazation of weights with zero-mean Gaussian with variance 2/(units_in + units_out)
    W_hidden = np.random.randn(F,K)*np.sqrt(2/(F+K))
    W_output = np.random.randn(K,dimension_output)*np.sqrt(2/(K+dimension_output))
    
    
    #Bias Initialization, biases intialized to zero
#    bias_hidden = np.zeros((1,K))
#    bias_output = np.zeros((1,dimension_output))
    #Bias xavier initialization
    bias_hidden = np.random.randn(1,K)*np.sqrt(2/(1+K))
    bias_output = np.random.randn(1,dimension_output)*np.sqrt(2/(1+dimension_output))

    
    #Momentum Initialization
    v_Wh = np.full((F,K), 1e-5)
    v_Wo = np.full((K,dimension_output), 1e-5)
    v_bh = np.full((1,K), 1e-5)
    v_bo = np.full((1,dimension_output), 1e-5)
    
  
    
    for epoch in epoch_values :
        
#        if epoch%25 == 0:
#            alpha =  0.5*alpha
        
        
        print("----------Starting epoch",epoch)
        
        #Compute a forward pass
        S_hidden=compute(X_input,W_hidden,bias_hidden)
        #S_hidden = np.matmul(X_input,W_hidden) + bias_hidden
        X_hidden = relu(S_hidden)
        S_output=compute(X_hidden,W_output,bias_output)
        #S_output = np.matmul(X_hidden,W_output) + bias_output
        X_output = softmax(S_output)

        
        
        ############## back propagation, calculating delta and gradients
    
        
        delta_2 = X_output - trainTarget # (10000,10)
        delta_1 = np.matmul(delta_2,W_output.T)*grad_relu(S_hidden) #(10000,1000)
        
        dWo = (1/train_N)*np.matmul(X_hidden.T,delta_2) #(10000,10), (10000,1000) => (10,1000)
        dbo = (1/train_N)*np.sum(delta_2,axis=0)
        dWh = (1/train_N)*np.matmul(X_input.T,delta_1) # (10000,1000), (10000,784)
        dbh = (1/train_N)*np.sum(delta_1,axis=0)
        
        
        
        
        ############## calculate momentum
        
        v_Wh = gamma*v_Wh + alpha*dWh
        v_Wo = gamma*v_Wo + alpha*dWo
        v_bh = gamma*v_bh + alpha*dbh
        v_bo = gamma*v_bo + alpha*dbo
        
        ############## update weights and biases
        
        W_hidden -= v_Wh
        W_output -= v_Wo
        bias_hidden -= v_bh
        bias_output -= v_bo
        
        
        #    print("X_output",np.sum(X_output,axis=1).shape) 
        loss = averageCE(trainTarget,X_output)
        train_accuracy = calculate_accuracy(trainTarget,X_output)
        train_losses.append(loss)
        train_accuracies.append(train_accuracy)
    
        #Calculating validation error and accuracy
        valid_loss, valid_accuracy = calculate_loss_and_accuracy(X_valid_input,validTarget,W_hidden,bias_hidden, W_output, bias_output)
        validation_losses.append(valid_loss)
        validation_accuracies.append(valid_accuracy)
        #for early stopping
#        if valid_loss<best_loss:
#            best_loss = valid_loss
#            checks_without_progress = 0
#        else:
#            checks_without_progress += 1
#        if checks_without_progress > max_checks_without_progress:
#            print(epoch)
#            print("Early stopping!")
#            break
            
            
        #Calculating test error and accuracy
        test_loss, test_accuracy = calculate_loss_and_accuracy(X_test_input,testTarget,W_hidden,bias_hidden, W_output, bias_output)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        print("train loss: ",loss,"\ntrain accuracy: ",train_accuracy, "\nvalid loss: ",valid_loss,"\nvalid accuracy: ",valid_accuracy,"\ntest loss: ",test_loss,"\ntest accuracy: ",test_accuracy)
        
        print("----------End of epoch",epoch)
    
    plt.figure(1)
    plt.plot(epoch_values,train_losses,label="training loss")
    plt.plot(epoch_values,test_losses,label="test loss")
    plt.plot(epoch_values,validation_losses,label="validation loss")
    plt.xlabel("epochs")
    plt.ylabel("losses")
    plt.legend()
    plt.grid(b=True,which='both')
    plt.title("Loss Curve for alpha = {}, K = {}".format(alpha,K))
    plt.savefig(graph_directory+"losscurve_alpha_{}_k_{}.png".format(alpha,K))
    
    
    plt.figure(2)
    plt.plot(epoch_values,test_losses,label="test loss")
    plt.xlabel("epochs")
    plt.ylabel("losses")
    plt.grid(b=True,which='both')
    plt.title("Test Loss for alpha = {}, K = {}".format(alpha,K))
    plt.savefig(graph_directory+"testloss_alpha_{}_k_{}.png".format(alpha,K))
    
    plt.figure(3)
    plt.plot(epoch_values,train_accuracies,label="training accuracy")
    plt.plot(epoch_values,test_accuracies,label="test accuracy")
    plt.plot(epoch_values,validation_accuracies,label="validation accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(b=True,which='both')
    plt.title("Accuracy Curve for alpha = {}, K = {}".format(alpha,K))
    plt.savefig(graph_directory+"accuracycurve_alpha_{}_k_{}.png".format(alpha,K))
    
    plt.figure(4)
    plt.plot(epoch_values,test_accuracies,label="test accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.grid(b=True,which='both')
    #plt.set_ylim([80,110])
    plt.title("Test Accuracy for alpha = {}, K = {}".format(alpha,K))
    plt.savefig(graph_directory+"testcurve_alpha_{}_k_{}.png".format(alpha,K))
    
    
    plt.show()
    #print('ValidationAccuracies:',validation_accuracies)
    print("Final Test Accuracy: ",test_accuracies[-1])
    print("Final Validation Accuracy: ",validation_accuracies[-1])
    print("Final Training Accuracy: ",train_accuracies[-1])
    
    print("Final Test Loss: ",test_losses[-1])
    print("Final Validation Loss: ",validation_losses[-1])
    print("Final Training Loss: ",train_losses[-1])
        
    print("validation_accuracies",validation_accuracies)  
    return W_hidden,bias_hidden, W_output, bias_output
    

        
    


    
    
    
   
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

print("trainData",trainData.shape)
print("trainTarget",trainTarget.shape)


#Part 1.3
K=1000
epochs = 1000
train_neural_network(K,epochs,trainData, validData, testData, trainTarget, validTarget, testTarget)

#Part 1.4 HyperParameter Investigation
hidden_units = [100,500,2000]
epochs = 200
for k in hidden_units:
    train_neural_network(k,epochs,trainData, validData, testData, trainTarget, validTarget, testTarget)


#143 --Early stopping
