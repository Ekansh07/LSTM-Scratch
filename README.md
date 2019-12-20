# Implementing LSTM Algorithm For Stock Price Prediction

### Files Included:
* **LSTM.py**:
    * Implements the execution of LSTMClass and pre-processing of the data.
	* Used labelEncoder, SimpleImputer, MinMaxScaler for Pre-processing of the data.
	* labelEncoder: converts categorical to numerical attributes
	* SimpleImputer: Handles "NULL" values
	* MinMaxScaler: Standardizes and scales the attributes
	* matplotlib: plotting various graphs
	
* **LSTMClass.py**:
    * Implements the LSTM Cell.
	* Forward Pass, Backward Pass and Training the network for LSTM cell.
	* Handling weights of the different gates in LSTM cell
	
## Link For Dataset
* https://www.kaggle.com/dgawlik/nyse/
* use "prices-split-adjusted.csv" file

## Requirements
* Python 3
* Command Line Interface
* Pandas * https://pandas.pydata.org/*
* Scikit Learn
* Numpy
* sklearn
* matplotlib

## Steps to run the program
* Open the CLI and run the command, *python LSTM.py stock_name hidden_layer_size learning_rate*
* Replace the *stock_name* in the above command with the name of the stock from the dataset for which you want to predict the open prices.
* Replace the *hidden_layer_size* in the above command with the size of the hidden layer
* Replace the *learning_rate* in the above command with desired learning rate for the algorithm


### Example Run commands
* **MSFT Stock**: _python3 LSTM.py "MSFT" 200 0.01
* **ADSK Stock**: _python3 LSTM.py "ADSK" 2000 0.1
* **FB Stock**: _python3 LSTM.py "FB" 500 0.001
