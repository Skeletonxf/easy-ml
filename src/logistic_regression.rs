/*!
Logistic regression example

Logistic regression can be used for classification. By performing linear regression on a logit function
a a linear classifier can be obtained that retains probabilistic semantics.

Given some data on a binary classification problem (ie a
[Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)),
transforming the probabilities with a logit function:

<pre>log(p / (1 - p))</pre>

puts them in the -infinity to infinity range. If we assume a simple linear model over two inputs
x1 and x2 then:

<pre>log(p / (1 - p)) = w0 + w1*x1 + w2*x2</pre>

For more complex data [basis functions](../linear_regression/index.html)
can be used on the inputs to model non linearity. Once we have a model we can try to learn
the weights with gradient descent to minimise error on the data. Once we have fixed weights
we can estimate the probability of new data by taking the inverse of the logit function
(the sigmoid function):

<pre>1 / (1 + e^(-(w0 + w1*x1 + w2*x2)))</pre>

which maps back into the 0 - 1 range and produces a probability for the unseen data. We can then
choose a cutoff at say 0.5 and we have a classifier that ouputs True for any unseen data estimated
to have a probability >= 0.5 and False otherwise.

# Logistic regression example
*/
