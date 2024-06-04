{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "515013df",
   "metadata": {},
   "outputs": [],
   "source": [
    "from carga_datos import *\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cfc9a512",
   "metadata": {},
   "source": [
    "This notebook contains the implementation of a logistic regression model with mini-batch, without using scikit-learn.\n",
    "\n",
    "For training the logistic regressor I choose the maximization of the likehood method.\n",
    "\n",
    "The log-likehood formula is:\n",
    "$$ LL\n",
    "= \\sum_{i=1}^{m} [y^{(i)} \\log(\\sigma(x^{(i)})) + (1 - y^{(i)}) \\log(1 - \\sigma(x^{(i)}))] $$\n",
    "\n",
    "Component of the gradient vector:\n",
    "$$ \\nabla LL / \\nabla w_j = \\sum_{i=1}^{m} (y^{(i)} - \\sigma(x^{(i)})) \\cdot x_{(j)}^{(i)} $$\n",
    "\n",
    "So for a mini-batch gradient ascent, the weight update is:\n",
    "\n",
    "$$ w_i \\leftarrow w_i + \\alpha \\sum_{1 \\leq k \\leq P} (y^{(k)} - \\sigma(x^{(k)})) \\cdot x_{(i)}^{(k)}  $$\n",
    "\n",
    "With matrix representation the gradient can be written as:\n",
    "\n",
    "$$ \\nabla LL = \\frac{1}{m} X^T (Y - \\overline{Y})   $$\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 118,
   "id": "6931bcea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# sigma function\n",
    "def sigmoid(z):\n",
    "    return 1 / (1 + np.exp(-z))\n",
    "\n",
    "# random initialization of weights\n",
    "def initialize_weights(input_size):\n",
    "    weights = np.random.randn(input_size, 1)\n",
    "    return weights\n",
    "\n",
    "# likehood\n",
    "def compute_likelihood(X, y, weights,h):\n",
    "    m = len(y)\n",
    "    h = sigmoid(X.dot(weights))\n",
    "    likelihood = np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))\n",
    "    return likelihood"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 121,
   "id": "5acfa158",
   "metadata": {},
   "outputs": [],
   "source": [
    "def logistic_regression(X, y, learning_rate=0.01, epochs=2, batch_size=32):\n",
    "    m, n = X.shape\n",
    "    weights = initialize_weights(n)\n",
    "    likelihoods = []\n",
    "\n",
    "    for epoch in range(epochs):\n",
    "        \n",
    "        indexes = np.random.choice(m, size=batch_size, replace=False)\n",
    "        \n",
    "        # extracting the batch randomly from X and y\n",
    "        X_batch = X[indexes]\n",
    "        y_batch = y[indexes]\n",
    "        \n",
    "        # make a prediction with the sigmoid function applied to X_batch * weights\n",
    "        predictions = sigmoid(X_batch.dot(weights))\n",
    "        \n",
    "        # estimating Y - Y^\n",
    "        y_batch = y_batch.reshape(-1, 1)\n",
    "        error = y_batch - predictions\n",
    "        \n",
    "        # calculating the gradient vector\n",
    "        gradient = X_batch.T.dot(error) / batch_size\n",
    "        \n",
    "        # updating weights\n",
    "        weights += learning_rate * gradient\n",
    "        \n",
    "        \n",
    "        likelihood = compute_likelihood(X_batch, y_batch, weights)\n",
    "        \n",
    "        likelihoods.append(likelihood)\n",
    "\n",
    "    return weights, likelihoods"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 122,
   "id": "46336d34",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "weights:  [[ 1.22250523]\n",
      " [-0.2893147 ]\n",
      " [-0.22682207]\n",
      " [ 0.91055268]\n",
      " [-0.23536598]\n",
      " [-1.03650926]\n",
      " [ 0.15567041]\n",
      " [ 0.56059577]\n",
      " [ 0.51105333]\n",
      " [ 0.34021912]\n",
      " [ 0.16325595]\n",
      " [-0.78333095]\n",
      " [-0.09263188]\n",
      " [ 0.67266871]\n",
      " [-1.13393053]\n",
      " [-0.36388806]\n",
      " [-0.28313088]\n",
      " [-0.93873372]\n",
      " [ 1.19351594]\n",
      " [ 0.33772769]\n",
      " [-1.23996313]\n",
      " [ 1.89178356]\n",
      " [-1.94955163]\n",
      " [ 0.31630734]\n",
      " [-0.8376587 ]\n",
      " [ 0.45647886]\n",
      " [ 0.49783314]\n",
      " [-2.01100026]\n",
      " [ 0.06026964]\n",
      " [ 1.59291718]]\n"
     ]
    },
    {
     "ename": "TypeError",
     "evalue": "compute_likelihood() missing 1 required positional argument: 'h'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[122], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m logistic_regression(X_cancer,y_cancer)\n",
      "Cell \u001b[0;32mIn[121], line 28\u001b[0m, in \u001b[0;36mlogistic_regression\u001b[0;34m(X, y, learning_rate, epochs, batch_size)\u001b[0m\n\u001b[1;32m     24\u001b[0m     \u001b[38;5;66;03m# updating weights\u001b[39;00m\n\u001b[1;32m     25\u001b[0m     weights \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m learning_rate \u001b[38;5;241m*\u001b[39m gradient\n\u001b[0;32m---> 28\u001b[0m     likelihood \u001b[38;5;241m=\u001b[39m compute_likelihood(X_batch, y_batch, weights)\n\u001b[1;32m     30\u001b[0m     likelihoods\u001b[38;5;241m.\u001b[39mappend(likelihood)\n\u001b[1;32m     32\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m weights, likelihoods\n",
      "\u001b[0;31mTypeError\u001b[0m: compute_likelihood() missing 1 required positional argument: 'h'"
     ]
    }
   ],
   "source": [
    "logistic_regression(X_cancer,y_cancer)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
