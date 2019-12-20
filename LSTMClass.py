import numpy as np

class LSTMClass(object):
    def __init__(self, inputSize, hiddenLayerSize):
        self.inputSize = inputSize ## input size
        self.hiddenLayerSize = hiddenLayerSize ## hidden layer input size
        self.Wlstm = np.random.randn(1 + inputSize + hiddenLayerSize, 4 * hiddenLayerSize) / np.sqrt(inputSize + hiddenLayerSize) ## weights of lstm cell
        self.Wlstm[0,:] = 0
        self.WpeepConnection = np.ones((3, hiddenLayerSize)) ## weights of peephole connection
        self.states = np.zeros((self.hiddenLayerSize)) ## states
        self.hiddenLayerStates = np.zeros((self.hiddenLayerSize)) ## hidden layer states
        self.updatedStates = np.zeros((self.hiddenLayerSize)) ## updated state
        self.updatedHiddenLayerStates = np.zeros((self.hiddenLayerSize)) ## updated state of hidden layer
        self.updatedWlstm = np.zeros_like(self.Wlstm) ## updated weights of lstm cell
        self.updatedWpeepConnection = np.zeros_like(self.WpeepConnection) ## updated weights of peephole connection
        
    def resetStates(self):
        self.states = np.zeros((self.hiddenLayerSize))
        self.hiddenLayerStates = np.zeros((self.hiddenLayerSize))
        self.updatedStates = np.zeros((self.hiddenLayerSize))
        self.updatedHiddenLayerStates = np.zeros((self.hiddenLayerSize))
        print ("Network states RESET")
        
    def forward(self, X):
        """
        X : input
        n = length of sequence
        inputSize = input size (input dimension)
        """
        n = X.shape[0]

        # Forward Pass 
        inputHiddenBias = self.Wlstm.shape[0] # input layer and hidden layer bias
        self.hiddenLayerInput = np.zeros((n, inputHiddenBias))
        self.hiddenLayerOutput = np.zeros((n, self.hiddenLayerSize))
    
        self.mofaGate = np.zeros((n, self.hiddenLayerSize * 4)) # 4 gates of lstm: modulation, output, forget, input previous
        self.mofaGate_forward = np.zeros((n, self.hiddenLayerSize * 4)) # 4 gates of lstm: modulation, output, forget, input forward 
        self.currentState = np.zeros((n, self.hiddenLayerSize)) # current state

        for t in range(n):
            previousHiddenLayerStates = self.hiddenLayerOutput[t-1,:] if (t > 0) else self.hiddenLayerStates
            previousState = self.currentState[t-1,:] if (t>0) else self.states
            self.hiddenLayerInput[t,0] = 1 # this is for the bias
            self.hiddenLayerInput[t,1:1+self.inputSize] = X[t, :]
            self.hiddenLayerInput[t,1+self.inputSize:] = previousHiddenLayerStates
            # Computing gate values 
            self.mofaGate[t,:] = self.hiddenLayerInput[t,:].dot(self.Wlstm)
            # Adding peephole weights connections
            self.mofaGate[t,:self.hiddenLayerSize] = self.mofaGate[t,:self.hiddenLayerSize] + np.multiply(previousState, self.WpeepConnection[0,:])   # input gate - adding peephole connections
            self.mofaGate[t,self.hiddenLayerSize:2*self.hiddenLayerSize] = self.mofaGate[t,self.hiddenLayerSize:2*self.hiddenLayerSize] + np.multiply(previousState, self.WpeepConnection[1,:])       
            # forget gate - adding peephole connections
            
            # forget and input gates output due to peep hole connection 
            self.mofaGate_forward[t,0:2*self.hiddenLayerSize] = 1.0 / (1.0 + np.exp(-self.mofaGate[t,0:2*self.hiddenLayerSize]))
            self.mofaGate_forward[t,3*self.hiddenLayerSize:] = np.tanh(self.mofaGate[t,3*self.hiddenLayerSize:]) 
            # tanh for modulation gate
            
            # Computing the lstm cell values            
            self.currentState[t,:] = self.mofaGate_forward[t,self.hiddenLayerSize:2*self.hiddenLayerSize]*previousState + self.mofaGate_forward[t,:self.hiddenLayerSize]*self.mofaGate_forward[t,3*self.hiddenLayerSize:]

            # Computing the output gate with peephole connections
            self.mofaGate[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize] = self.mofaGate[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize] + np.multiply(self.currentState[t,:], self.WpeepConnection[2,:]) 
            # output gate - adding peephole connections            
            
            self.mofaGate_forward[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize] = 1.0 / (1.0 + np.exp(-self.mofaGate[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize]))
            self.hiddenLayerOutput[t,:] = self.mofaGate_forward[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize]*np.tanh(self.currentState[t,:])
        
        self.states = self.currentState[t,:]
        self.hiddenLayerStates = self.hiddenLayerOutput[t,:]
        
        
    def backward(self, dHout_temp):               
        # backprop through the LSTM now
        self.backWardMofaGate = np.zeros_like(self.mofaGate) ## duplicating gates weights
        self.backWardMofaGate_forward = np.zeros_like(self.mofaGate_forward) ## duplicating gates weights forward
        self.backWardWlstm = np.zeros_like(self.Wlstm) ## duplicating weights of lstm cell
        self.backWardWpeepConnection = np.zeros_like(self.WpeepConnection) ## duplicating weights of peephole connection
        self.backWardCurrentState = np.zeros_like(self.currentState) ## duplicating current state
        self.backWardHiddenLayerOutput = dHout_temp.copy() ## backward hidden layer output
        self.backWardHiddenLayerInput = np.zeros_like(self.hiddenLayerInput) ## backward hidden layer input
        self.backWardHiddenLayerStates = np.zeros((self.hiddenLayerSize)) ## backward hidden layer state
        
        n = self.hiddenLayerInput.shape[0]
        
        if self.updatedStates is not None: self.backWardCurrentState[n-1] += self.updatedStates.copy()
        if self.updatedHiddenLayerStates is not None: self.backWardHiddenLayerOutput[n-1] += self.updatedHiddenLayerStates.copy()
        
#        print(dHout.shape, C.shape)
        for t in reversed(range(n)):
            self.backWardMofaGate_forward[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize] = self.backWardHiddenLayerOutput[t,:]*np.tanh(self.currentState[t,:]) # backprop in to output gate
            # backprop through the tanh non-linearity to get in to the cell, then will continue through it
            self.backWardCurrentState[t,:] += (self.backWardHiddenLayerOutput[t,:] * self.mofaGate_forward[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize]) * (1 - np.tanh(self.currentState[t,:]**2))
                     
            if (t>0):
                self.backWardMofaGate_forward[t,self.hiddenLayerSize:2*self.hiddenLayerSize] = self.backWardCurrentState[t,:]*self.currentState[t-1,:] # backprop in to the forget gate
                self.backWardCurrentState[t-1,:] += self.mofaGate_forward[t,self.hiddenLayerSize:2*self.hiddenLayerSize] * self.backWardCurrentState[t,:] # backprop through time for C (The recurrent connection to C from itself)
            else:
                self.backWardMofaGate_forward[t,self.hiddenLayerSize:2*self.hiddenLayerSize] = self.backWardCurrentState[t,:]*self.states # backprop in to forget gate
                self.backWardState = self.mofaGate_forward[t,self.hiddenLayerSize:2*self.hiddenLayerSize] * self.backWardCurrentState[t,:]
            
            self.backWardMofaGate_forward[t,:self.hiddenLayerSize] = self.backWardCurrentState[t,:]*self.mofaGate_forward[t,3*self.hiddenLayerSize:] #backprop in to the input gate
            self.backWardMofaGate_forward[t,3*self.hiddenLayerSize:] = self.backWardCurrentState[t,:]*self.mofaGate_forward[t,:self.hiddenLayerSize] #backprop in to the a gate                    

            # backprop through the activation functions
            # for input, forget and output gates - derivative of the sigmoid function
            # for a - derivative of the tanh function                
            
            self.backWardMofaGate[t,3*self.hiddenLayerSize:] =  self.backWardMofaGate_forward[t,3*self.hiddenLayerSize:] * (1 - self.mofaGate_forward[t,3*self.hiddenLayerSize:]**2)              
            y = self.mofaGate_forward[t,:3*self.hiddenLayerSize]
            self.backWardMofaGate[t,:3*self.hiddenLayerSize] = (y*(1-y)) * self.backWardMofaGate_forward[t,:3*self.hiddenLayerSize] 
        
            # backprop the input matrix multiplication            
            self.backWardWlstm += np.dot(self.hiddenLayerInput[t:t+1,:].T, self.backWardMofaGate[t:t+1,:])
            self.backWardHiddenLayerInput[t,:] = self.backWardMofaGate[t,:].dot(self.Wlstm.T) 
            
            # backprop the peephole connections
            if t>0:
                self.backWardWpeepConnection[0,:] += np.multiply(self.backWardMofaGate[t,:self.hiddenLayerSize], self.currentState[t-1,:])
                self.backWardWpeepConnection[1,:] += np.multiply(self.backWardMofaGate[t,self.hiddenLayerSize:2*self.hiddenLayerSize], self.currentState[t-1,:])  
                self.backWardWpeepConnection[2,:] += np.multiply(self.backWardMofaGate[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize], self.currentState[t,:]) 
            else:
                self.backWardWpeepConnection[0,:] += np.multiply(self.backWardMofaGate[t,:self.hiddenLayerSize], self.states)
                self.backWardWpeepConnection[1,:] += np.multiply(self.backWardMofaGate[t,self.hiddenLayerSize:2*self.hiddenLayerSize], self.states)  
                self.backWardWpeepConnection[2,:] += np.multiply(self.backWardMofaGate[t,2*self.hiddenLayerSize:3*self.hiddenLayerSize], self.currentState[t,:])
                    
            if (t>0):
                self.backWardHiddenLayerOutput[t-1,:] += self.backWardHiddenLayerInput[t,1+self.inputSize:]
            else:
                self.backWardHiddenLayerStates += self.backWardHiddenLayerInput[t,1+self.inputSize:] 
                
    def getHiddenOutput(self):      
        return self.hiddenLayerOutput


    def trainNetwork(self, learning_rate):
        for param, dparam, mem in zip([self.Wlstm, self.WpeepConnection],
                                  [self.backWardWlstm, self.backWardWpeepConnection ],
                                  [self.updatedWlstm, self.updatedWpeepConnection]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)