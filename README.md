<h1><strong>Efficiency Analysis Trees (EAT)</strong></h1>

<p style="justify">EAT is a new methodology based on regression trees for estimating production frontiers satisfying fundamental postulates of microeconomics, such as free disposability. This new approach, baptized as Efficiency Analysis Trees (EAT), shares some similarities with the Free Disposal Hull (FDH) technique. However, and in contrast to FDH, EAT overcomes the problem of overfitting by using cross-validation to prune back the deep tree obtained in the first stage. Finally, the performance of EAT is measured via Monte Carlo simulations, showing that the new approach reduces the mean squared error associated with the estimation of the true frontier by between 13% and 70% in comparison with the standard FDH.</p>

For more info see: https://doi.org/10.1016/j.eswa.2020.113783

<h2>Import libreries</h2>
All the libraries in the repository are imported since they will be used in all the examples presented.

```python
import data
import EAT as fEAT
import scores
import graphviz
```

<h2>Generate simulated data </h2>
EATpy repository includes a simulated data generator module. It is used as an example of the use of the repository. For do that, the seed of the generator and the size of the sample are stablished. 

```python
dataset = data.Data(1, 50).data
```
<h2>Create the EAT model</h2>
The creation of the EAT model consist on specify the inputs and outputs columns name, the ending rule and the number of folder for Cross-Validation process. Once this is done, the model is created and fitted to build it.

First, the name of the columns of the inputs and outputs in the sample are indicated.
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
model = fEAT.EAT(dataset, x, y, numStop, fold)
model.fit()
```
