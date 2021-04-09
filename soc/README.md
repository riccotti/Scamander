# Diverse Coherent Explanations

This code is an implementation of the paper "Efficient Search for Diverse
Coherent Explanations". 
https://arxiv.org/pdf/1901.04909.pdf 

To run without modification, it requires a copy of the
FICO data taken from:
https://community.fico.com/s/explainable-machine-learning-challenge?&tabset-3158a=2

It also uses the gurobi solver http://www.gurobi.com/ for the MIP solver. 

The code explicitly targets this data set and has made a couple of simple
assumptions as to the form the dataset takes. Each variable is assumed to take a
range of continuous values and a set of discrete values; as simplifying
assumptions we assume that all strictly negative values are the discrete values,
while the continuous values are the non-negative ones.

If this is not the case for your dataset, the code can be adapted to match
assumptions, but it probably easier to manipulate the data so that it follows
these assumptions.

The code counterfactual.py is a library that implements an object based
interface at over the code. 

example.py is a commented demo that learns a logistic
regression classifier over the dataset using the encoding described in the
original paper.



