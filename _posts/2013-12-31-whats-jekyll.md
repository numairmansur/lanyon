---
layout: post
title: Modules
---

## Task

In order to optimize any function, RoBO expects a task object that is derived from the BaseTask class. If you want to optimize your own objective function you need to derive from this base class and implement at least the objective_function(self, x): method as well as the self.X_lower and self.X_upper. However you can add any additional information here. For example the well-known synthetic benchmark function Branin would look like:

{% highlight python %}
import numpy as np

from robo.task.base_task import BaseTask


class Branin(BaseTask):

    def __init__(self):
        self.X_lower = np.array([-5, 0])
        self.X_upper = np.array([10, 15])
        self.n_dims = 2
        self.opt = np.array([[-np.pi, 12.275], [np.pi, 2.275], [9.42478, 2.475]])
        self.fopt = 0.397887

    def objective_function(self, x):
        y = (x[:, 1] - (5.1 / (4 * np.pi ** 2)) * x[:, 0] ** 2 + 5 * x[:, 0] / np.pi - 6) ** 2
        y += 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10

        return y[:, np.newaxis]
{% endhighlight %}

In this case we can also set the known global optimas and the best function value. This allows to plot the distance between the best found function value and the global optimum. However, of course for real world benchmark we do not have this information so you can just drop them.

Note that the method objective_function(self, x) expects a 2 dimensional numpy array and also returns a two dimension numpy array. Furthermore bounds are also specified as numpy arrays:

{% highlight python %}
self.X_lower = np.array([-5, 0])
self.X_upper = np.array([10, 15])
{% endhighlight %}

## Models

The model class contains the regression model that is used to model the objective function. To use any kind of regression model in RoBO it has to implement the interface from them BaseModel class. Also each model has its own hyperparameters (for instance the type of kernel for GaussianProcesses). Here is an example how to use GPs in RoBO:

{% highlight python %}
import GPy
from robo.models.GPyModel import GPyModel

kernel = GPy.kern.Matern52(input_dim=task.n_dims)
model = GPyModel(kernel, optimize=True, noise_variance=1e-4, num_restarts=10)
model.train(X, Y)
mean, var = model.predict(X_test)
{% endhighlight %}

## Acquisition functions

The role of an acquisition function in Bayesian optimization is to compute how useful it is to evaluate a candidate x. In each iteration RoBO maximizes the acquisition function in order to pick a new configuration which will be then evaluated. The following acquisition functions are currently implemented in RoBO and each of them has its own properties.

* Expected Improvement
* Log Expected Improvement
* Probability of Improvement
* Entropy
* EntropyMC
* Upper Confidence Bound

Each acquisition function expects at least a model and a the input bounds of the task as input, for example:

{% highlight python %}
acquisition_func = EI(model, X_upper=task.X_upper, X_lower=task.X_lower)
{% endhighlight %}

Furthermore, every acquisition functions has its own individual parameters that control its computations. To compute now the for a specific x its acquisition value you can call. The input point x has to be a 1\times D numpy array:

{% highlight python %}
val = acquisition_func(x)
{% endhighlight %}

If you marginalize over the hyperparameter of a Gaussian Process via the GPyMCMC module this command will compute the sum over the acquisition value computed based on every single GP

Some acquisition functions allow to compute gradient, you can compute them by:

{% highlight python %}
val, grad = acquisition_func(x, derivative=True)
{% endhighlight %}

If you updated your model with new data you also have to update you acquisition function by:

{% highlight python %}
acquisition_func.update(model)
{% endhighlight %}

## Maximizers

The role of the maximizers is to optimize the acquisition function in order to find a new configuration which will be evaluated in the next iteration. All maximizer have to implement the BaseMaximizer interface. Ever maximizer has its own parameter (see here for more information) but all expect at least an acquisition function object as well as the bounds of the input space:

{% highlight python %}
maximizer = CMAES(acquisition_func, task.X_lower, task.X_upper)
{% endhighlight %}

Afterwards you can easily optimize the acquisition function by:

{% highlight python %}
x_new = maximizer.maximize()
{% endhighlight %}

## Solver

The solver module represents the actual Bayesian optimizer. The standard module is BayesianOptimization which implements the vanilla BO procedure.

{% highlight python %}
bo = BayesianOptimization(acquisition_fkt=acquisition_func,
                  model=model,
                  maximize_fkt=maximizer,
                  task=task,
                  save_dir=os.path.join(save_dir, acq_method + "_" + max_method, "run_" + str(run_id)),
                  num_save=1)

bo.run(num_iterations)
{% endhighlight %}

If you just want to perform one single iteration based on some given data to get a new configuration you can call:

{% highlight python %}
new_x = bo.choose_next(X, Y)
{% endhighlight %}

It also offers functions to save the output and measure the time of each function evaluation and the optimization overhead. If you develop a new BO strategy it might be a good idea to derive from this class and uses those functionalities to be compatible with RoBO’s tools.
