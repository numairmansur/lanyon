---
layout: post
title: Basic Usage
---

## RoBO in a few lines of code

RoBO offers a simple interface such that you can use it as a optimizer for black box function without knowing what’s going on inside. In order to do that you first have to define the objective function and the bounds of the configuration space:

{% highlight python %}
import numpy as np
from robo.fmin import fmin

def objective_function(x):
    return  np.sin(3 * x) * 4 * (x - 1) * (x + 2)

X_lower = np.array([0])
X_upper = np.array([6])
{% endhighlight %}

The you can start RoBO with the following command and it will return the best configuration / function value it found:

{% highlight python %}
x_best, fval = fmin(objective_function, X_lower, X_upper)
{% endhighlight %}

## Bayesian optimization with RoBO

RoBO is a flexible modular framework for Bayesian optimization. It distinguishes between different components that are necessary for Bayesian optimization and treats all of those components as modules which allows us to easily switch between different modules and add new-modules:

- [Task:](http://robo-fork.readthedocs.io/en/latest/modules.html#task) This module contains the necessary information that RoBO needs to optimize the objective function (for example an interface for the objective function the input bounds and the dimensionality of the objective function).
- [Moels:](http://robo-fork.readthedocs.io/en/latest/modules.html#models) This is the regression method to model the current believe of the objective function.
- [Acquisition functions:](http://robo-fork.readthedocs.io/en/latest/modules.html#acquisitionfunctions) This module represents the acquisition function which acts as a surrogate that determines which configuration will be evaluated in the next step.
- [Maximizers:](http://robo-fork.readthedocs.io/en/latest/modules.html#maximizers) This module is used to optimize the acquisition function to pick the next configuration.

### Defining an objective function

RoBo can optimize any function X \rightarrow Y with X as an N\times D numpy array and Y as an N\times K numpy array. Thereby N is the number of points you want to evaluate at, D is the dimension of the input X and K the number of output dimensions (mostly K = 1). In order to optimize any function you have to define a task object that implements the interface BaseTask. This class should contain the objective function and the bounds of the input space.

{% highlight python %}
import numpy as np

    from robo.task.base_task import BaseTask

class ExampleTask(BaseTask):

        def __init__(self):
            self.X_lower = np.array([0])
            self.X_upper = np.array([6])
            self.n_dims = 1

        def objective_function(self, x):
            return np.sin(3 * x) * 4 * (x - 1) * (x + 2)

    task = ExampleTask()
{% endhighlight %}


### Building a model

The first step to optimize this objective function is to define a model that captures the current believe of potential functions. The probably most used method in Bayesian optimization for modeling the objective function are Gaussian processes. RoBO uses the well-known [GPy](http://sheffieldml.github.io/GPy/) library as implementation for Gaussian processes. The following code snippet shows how to use a GPy model via RoBO:

{% highlight python %}
import GPy

from robo.models.GPyModel import GPyModel

kernel = GPy.kern.Matern52(input_dim=task_ndims)
model = GPyModel(kernel, optimize=True, noise_variance = 1e-4, num_restarts=10)
{% endhighlight %}

RoBO offers a wrapper interface GPyModel to access the Gaussian processes in [GPy](http://sheffieldml.github.io/GPy/). We have to specify a kernel from GPy library as covariance function when we initialize the model. For further details on those kernels visit GPy. We can either use fix kernel hyperparameter or optimize them by optimizing the marginal likelihood. This is achieved by setting the optimize flag to True.

### Creating the Acquisition Function
After we defined a model we can define an acquisition function as a surrogate function that is used to pick the next point to evaluate. RoBO offers the following acquisition functions in the acquisition package:

In order to use an acquisition function (in this case Expected Improvement) you have to pass it the models as well as the bounds of the input space:

{% highlight python %}
from robo.acquisition.EI import EI
from robo.recommendation.incumbent import compute_incumbent

acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower, compute_incumbent=compute_incumbent, par=0.1)
{% endhighlight %}

Expected Improvement as well as Probability of Improvement need as additional input the current best configuration (i.e. incumbent). There are different ways to determine the incumbent. You can easily plug in any method by giving Expected Improvement a function handle (via compute_incumbent). This function is supposed to return a configuration and expects the model as input. In the case of EI and PI you additionally have to specify the parameter “par” which controls the balance between exploration and exploitation of the acquisition function.


### Maximizing the acquisition function

The last component is the maximizer which will be used to optimize the acquisition function in order to get a new configuration to evaluate. RoBO offers different ways to optimize the acquisition functions such as:

* grid search
* DIRECT
* CMA-ES
* stochastic local search

Here we will use a simple grid search to determine the configuration with the highest acquisition value:

{% highlight python %}
from robo.maximizers.grid_search import GridSearch

maximizer = GridSearch(acquisition_func, task.X_lower, task.X_upper)
{% endhighlight %}

### Putting it all together

Now we have all the ingredients to optimize our objective function. We can put all the above described components in the BayesianOptimization class

{% highlight python %}
from robo.solver.bayesian_optimization import BayesianOptimization

bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                          model=model,
                          maximize_fkt=maximizer,
                          task=task)
{% endhighlight %}

Afterwards we can run it by:

{% highlight python %}
bo.run(num_iterations=10)
{% endhighlight %}

-----
