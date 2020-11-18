<h1><strong>Efficiency Analysis Trees (EAT)</strong></h1>

<p style="justify">EAT is a new methodology based on regression trees for estimating production frontiers satisfying fundamental postulates of microeconomics, such as free disposability. This new approach, baptized as Efficiency Analysis Trees (EAT), shares some similarities with the Free Disposal Hull (FDH) technique. However, and in contrast to FDH, EAT overcomes the problem of overfitting by using cross-validation to prune back the deep tree obtained in the first stage. Finally, the performance of EAT is measured via Monte Carlo simulations, showing that the new approach reduces the mean squared error associated with the estimation of the true frontier by between 13% and 70% in comparison with the standard FDH.</p>

For more info see: https://doi.org/10.1016/j.eswa.2020.113783

<h2>Generate simulated data </h2>
EATpy repository includes a simulated data generator module. It is used as an example of the use of the repository. First, we stablish the seed of the generator and the size of the sample.

```python
dataset = data.Data(1, 50).data
```
