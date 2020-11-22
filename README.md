<h1><strong>Efficiency Analysis Trees (EAT)</strong></h1>

<p style="justify">EAT is a new methodology based on regression trees for estimating production frontiers satisfying fundamental postulates of microeconomics, such as free disposability. This new approach, baptized as Efficiency Analysis Trees (EAT), shares some similarities with the Free Disposal Hull (FDH) technique. However, and in contrast to FDH, EAT overcomes the problem of overfitting by using cross-validation to prune back the deep tree obtained in the first stage. Finally, the performance of EAT is measured via Monte Carlo simulations, showing that the new approach reduces the mean squared error associated with the estimation of the true frontier by between 13% and 70% in comparison with the standard FDH.</p>

For more info see: https://doi.org/10.1016/j.eswa.2020.113783

<h2>Installation</h2>
To facilitate installation on a personal computer, we recommend installing git (see: https://git-scm.com/downloads) and the Anaconda distribution (see: https://www.anaconda.com/products/individual). The steps to follow are based on these two installations.


Step1. Open the Anaconda Prompt console, place it in the desired directory for installation and enter the instruction: 
```
git clone https://github.com/MiriamEsteve/EATpy.git
```

Step 2. Place us in the folder created by EATpy, using EATpy cd, and execute the instruction:
```
python setup.py install
```

<h2>Import libraries</h2>
All the libraries in the repository are imported since they will be used in all the examples presented.

```python
import eat
import graphviz
```

<h2>Generate simulated data </h2>
EATpy repository includes a simulated data generator module. It is used as an example of the use of the repository. For do that, the seed of the generator and the size of the dataset are stablished. 

```python
dataset = eat.Data(1, 50).data
```
<h2>Create the EAT model</h2>
The creation of the EAT model consist on specify the inputs and outputs columns name, the ending rule and the number of folder for Cross-Validation process. Once this is done, the model is created and fitted to build it.

First, the name of the columns of the inputs and outputs in the dataset are indicated. If these ones don't exist in the dataset, the EAT model returns an error. 
```python
x = ["x1", "x2"]
y = ["y1", "y2"]
```

Second, the ending rule and the number of fold in Cross-Validation process are specified.
```python
numStop = 5
fold = 5
```
Third, the creation and fit of the EAT model are done.
```python
model = eat.EAT(dataset, x, y, numStop, fold)
model.fit()
```

<h2>Draw tree EAT</h2>
The drawing of the EAT tree is done using the external graphviz library. For this purpose, the first instruction generates the dot_data that graphviz needs to draw the EAT tree. In addition, it saves it as an image in the working directory. 

```python
dot_data = model.export_graphviz('EAT')
graph = graphviz.Source(dot_data, filename="tree", format="png")
graph.view()
```

<h2>Predictions</h2>
The prediction of the EAT model can be with one dataset or with a single register of the dataset. To do this, you need the data set or single register you want to predict and the names of the input columns. In order to indicate the names of the inputs in the dataset to be predicted. As a general rule, these names will be the same as those in the initial dataset.

In this example, the first 10 register are selected from the initial dataset and the name of the inputs are the same as it. Then, the model EAT realize the prediction and return the dataset with the predictions. These ones are named by "p_" at the beginning of the output name.

```python
x_p = ["x1", "x2"]
data_pred = dataset.loc[:10, x_p]
data_prediction = model.predict(data_pred, x_p)
```

<h2>Efficiency Scores</h2>
The repository has three ways to calculate the efficiency score of EAT model. The first one is the model BBC output oriented. The second one is the model BBC input oriented. The last one is the model DDF. 

To do that, the model EAT of scores is carried out.
```python
mdl_scores = eat.Scores(dataset, x, y, model.tree)
```

Then, the three models exposed before are called.
```python
#Fit BBC output oriented of EAT
mdl_scores.BBC_output_EAT()

#Fit BBC input oriented of EAT
mdl_scores.BBC_input_EAT()

#Fit DDF of EAT
mdl_scores.DDF_EAT()
```
