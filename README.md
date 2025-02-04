# Stock-Market-AI-GUI
Stock Market Prediction &amp; Trading Bot using AI with a Web Interface

## Stock Market Prediction using an LSTM Network
Long Short Term Memory networks – usually just called “LSTMs” – are a special kind of RNN, capable of learning long-term dependencies. They were introduced by Hochreiter & Schmidhuber (1997), and were refined and popularized by many people in following work. They work tremendously well on a large variety of problems, and are now widely used. LSTMs are explicitly designed to avoid the vanishing gradient problem. 

<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/lstm_model.png" align="middle" />  </p>

All recurrent neural networks have the form of a chain of repeating modules of neural network. In standard RNNs, this repeating module will have a very simple structure, such as a single tanh layer. LSTMs also have this chain like structure, but the repeating module has a different structure. Instead of having a single neural network layer, there are four, interacting in a very special way.

For more info check out this [article](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Stock Market Agent using Evolution Strategy Agent

Even though the name sounds fancy but under the hood, it’s perhaps the simplest algorithm you can devise for exploring a landscape. Consider an agent in an environment (like Pong) that’s implemented via a neural network. It takes pixels in the input layer and outputs probabilities of actions available to it (move the paddle up, down or do nothing).

<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/evolve_agent.png" align="middle" />  </p>

Our task in reinforcement learning is to find the parameters (weights and biases) of the neural network (weights and biases) that make the agent win more often and hence get more rewards. 

For more info check out this [article](https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f)

## Requirements
* Python 3.6.2 (https://www.python.org/downloads/release/python-362/)
* Django (https://www.djangoproject.com/)
* Numpy (https://pypi.org/project/numpy/)
* Tensorflow (https://pypi.org/project/tensorflow/)
* Keras (https://pypi.org/project/Keras/)
* Seaborn (https://pypi.org/project/seaborn/)
* Yahoo-Finance (https://pypi.org/project/yahoo-finance/)
* Pandas (https://pypi.org/project/pandas/)
* Matplotlib (https://pypi.org/project/matplotlib/)

## Usage

First start the django server using the following line,
```
python manage.py runserver
```
### Main Page
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/Main.PNG" align="middle" />  </p>

The main page gives you three options to choose from:

### 1. Stock Info
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/Info.PNG" align="middle" />  </p>
Just Input the Symbol of the Stock and the Duration for which to get the data and the data is fetched using the yahoo-finance library and graphed using matplotlib and mpld3.
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/Info-done.png" align="middle" />  </p>
The details are shown in the table and the closing prices are graphed. Hover your mouse over the points will give you a tooltip with the date and the closing price for that day.

### 2. Prediction
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/Prediction.png" align="middle" />  </p>
For the prediction you have to input the Symbol for the Stock, the Period of Data to train with, The Number of Simulations to run, and the Number of Future Days to predict for.
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/Prediction-done.png" align="middle" />  </p>
The closing prices of the simulations that are deemed acceptable is graphed using matplotlib and mpld3. Hover your mouse over the points will give you a tooltip with the date and the closing price for that day.

### 3.Trading Agent
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/Agent.png" align="middle" />  </p>
For the trading agent you have to input the Symbol for the Stock, the Period of Data to trade on, The Initial Fund, and the Number of Days to Skip in between selling or buying.
<p align="center">
<img src="https://github.com/crypto-code/Stock-Market-AI-GUI/blob/master/assets/Agent-done.png" align="middle" />  </p>
The closing prices is graphed and the selling and buying days are marked with their respective markers using matplotlib and mpld3. Hover your mouse over the marker to get the date and the action done on that day.


### 4. Docker usage:
```
docker build -t smag:latest . 
./runOnceDocker.sh 
python manage runserver
```

# G00D LUCK

For doubts email me at:
atinsaki@gmail.com
