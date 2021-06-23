# Deep Learning Approach And Bio-Inspired Computing For Survival Prediction
In the recent industrial era, where science and technology have been growing with an exponential pattern, commercial civilian transportation has turned into a paramount industry. The modes of travel for civilians have never been this elabo- rate as they are to date. Transportation has become much more standardised and has resulted in greater movement of high density clusters of population. But with this magnanimous scale, there also comes a boon as a result of fiddling with the laws of nature.

This research is aimed at achieving a detailed data analysis and understanding the effect or parameters key to the survival of individuals on board a civilian transportation vessel. Considering the case of Titanic, a British cruise, the analysis will be conducted. Using highly optimised algorithms and search functions, we would like to predict the survivability rates. The bio-inspired algorithms will be implemented to optimise the performance of classifiers. Towards the end, accuracies of different algorithms based on features fed to them will be compared in a tabular form.

## Aim, Expected Output and Motivation 
The preliminary objective of our project was forecasting survivability rates of diverse groups of population using a bio-inspired machine learning fabric to optimise the perforce of the classifiers and to have indigenously manifisteted the problem of optimising disaster management predictions

The aim of our project is to implement highly efficient and accurate machine learning and deep learning models to ensure that disaster management and response authorities have the greater edge towards managing a potential catastrophe. 

With this project we wanted to the solve the problem of helping disaster management authorities in providing accurate predictions about the survival chances of each passenger in case of a natural or anthropogenic catastrophe 

## Technologies Used 
1. Python v3.8 along with google colab and jupyter notebook 
2. Tensorflow v2.5
3. Sci-Kit Learn 
4. PySwarm 
5. Pandas 
6. MatPlotLib

## Data Preprocessing 
This is one of the major steps which can make or break the entire model. We need to ensure that at every step in the training and testing phase, the model is provided with clean and consistent data. This includes that all the data fed in must be numerical and not categorical, there must be no missing and null values and that strongly correlated attributes are highlighted.

All the steps in the data preprocessing phase ensure that the data is highly consistent throughout. Emphasis is largely made on eliminating any occurring null values. The model does not work well under the influence of null values. Having null values increases the chances of bugs and errors in the data model which is highly non trivial. Also this might be highly non trivial, but null values take up more space than other placeholders
The next part in this process is data visualisation. Knowing before hand, the at- tributes which are strongly co-related to each other ensures that the model can be emphasised to use them in the appropriate priority. We can setup a priority in training as to train highly co-related values with much emphasis cause these values have a higher chance of boosting accuracy figures
<p> </p>
<p> </p>
<img width="450" alt="Screenshot 2021-06-23 at 6 41 31 PM" src="https://user-images.githubusercontent.com/81710149/123103099-50ecc380-d453-11eb-9547-b3266bd9f251.png">
<p> </p>
<img width="450" alt="Screenshot 2021-06-23 at 6 42 09 PM" src="https://user-images.githubusercontent.com/81710149/123102575-d623a880-d452-11eb-9a93-fe11439d2801.png">

We have split the data into 7:3 ratio in which the former part constitutes the training data and the latter the testing data. This ratio is appropriate the provide enough data in the testing phase such that the model neither overfits not does it under fit. This division is a sweet spot for general machine learning algorithms and works like a charm for our proposed as well as base paper implemented algorithms.

# Custom Artificial Neural Network 
Initially we will start by implementing our custom neural network. We have implemented a four layer sequential neural net- work, with the fourth layer being the output layer and the first layer being the input layer.

The first layern comprises of 39 Dense neurons. Dense, in neural networks means that all the nodes in this layer will be connected to each node in the next layer. There is a 1 to all relation between each node of first layer to every other node in the next layer and vice versa. The next layer is also a dense layer, but with only 27 neurons.

The third layer consists of 19 neurons which are again dense in nature. Which means that these 19 neurons will be connected to each of the previous 27 neurons. Hence in total there are 19 ∗ 27 = 513 connections between the 2nd and third layer.

Now the final layer consists of only one neuron as it is the output layer. Even this layer is a dense layer which means it is connected to all the previous 19 neurons of the third layer. Hence there are a total of 19 ∗ 1 = 19 connections between the last and third layer. In total our model has 1026 + 513 + 19 = 1558 connections.

Moving on to the activation functions, the first layer uses an activation function called Rectified Linear Unit or simply called RELU. Relu, is preferred over the widely using sigmoid function because it avoids the problems of vanishing gradi- ents. Relu takes an input and returns zero, if the value is less than zero, or return the input itself if the value is greater than zero.

The second layer uses what is called Exponential Linear Unit as a primary activation method. It is used to avoid the problem of static relu. Elu scales the input by a certain degree to avoid static relu problems

The third layer uses the softmax activation function, which turns the entire input into probability density. Softmax is used to parse an input into a probability density than can be later parsed into a sigmoid distribution.

We compile the model using the Adam optimiser which is used in the place of traditional stochastic gradient descent. Adam has been derived from the phrase Adaptive Motion Estimation. Adam does not have any sort of constraints on hyper parameter tuning as well as they are quite intuitive in nature. It is also not computationally dense and can be used of large datasets without any hassle.

```Python
class node:
  def __init__(self, data):
    pass 

````




