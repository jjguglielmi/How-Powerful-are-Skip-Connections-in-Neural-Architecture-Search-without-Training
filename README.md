# How Powerful are Skip Connections in Neural Architecture Search without Training
Group 1 / Project 8 -> S.Ferrua, J.Guglielmi, A.Orazalin

Machine Learning and Deep Learning course in master's degree at Politecnico di Torino.

In this README, we provide:
- [Summary of our work](#Summary)
  - [Our-variation](#Our-variation-in-REA-'-s-mutation)
- [Architecture pretrained with NASWOT metric](#Architectures-for-datasets-with-metrics)
- [How to run the code](#Running-the-Code)


**If you have any questions, please open an issue or email us: **
- S.Ferrua : s292946@studenti.polito.it
- J.Guglielmi : s303105@studenti.polito.it
- A. Orazalin: s298255@studenti.polito.it

## Summary
**Abstract.** Neural Architecture Search is a technique that aspire to discover the best architecture for a neural network based on a specific need. Briefly, NAS is a gradient-based method using a controller network that produces an architecture suggestion specified by a mutable-length string. This string is then trained by a child network on real data, generating a signal fed into the controller, which after that, will give a higher probability to architectures to reach higher accuracies. Our research, tries to demonstrate how the application of some variations on the search algorithm, might provide better outcomes. 
The first phase we accomplish on NAS, consists of computing ahead of time the training-free metrics for all the possible architectures, scoring each of them at initialization. 
After that, we implement the NASWOT algorithm, a similar random search for picking randomly a candidate from the search space and iteratively finding the best network. 
To do a level up gaining higher scores, we look through existing aging research, finding the Regularized Evolutionary Algorithm, which can discriminate between the best architectures based on a genotype structure at each iteration. But REA needs a way to evaluate each architecture, so we provide a zero-cost proxy, the synaptic flow score, known as SynFlow, which was originally thought as a pruner. 
Based on this research, we concentrate on the mutation of the REA’s parent architecture, because after a few considerations, we noticed that *skip connections* have an importance during the aging evolution.
We purpose a series of combinations between REA + NASWOT/SynFlow without or with our modification.

### Our variation in REA's mutation
To find the architecture that satisfies us, using search algorithms, we used skip connection as a filter to find such an architecture. 
We have decided to introduce a filter while we construct a random architecture, because by plotting histograms based on the whole set of 15K architectures with relatives scores calculated for a batch size = 32, we find out that better accuracies are reached with skips included in the process. It is clear that, even if the score at initialization for all the architectures are done with a batch size lower than the one used for the Regularized Evolutionary Algorithm (we use a batch size = 128), as the batch size rises, the metric improves accordingly. 


_Figure 1: Difference between networks with and without skip connections (Cifar10)._


![alt text](https://github.com/jjguglielmi/How-Powerful-are-Skip-Connections-in-Neural-Architecture-Search-without-Training/blob/main/images/cifar10/histSkip_noSkipCifar10.png)


After that, we try to understand how many skip connections appear in those architectures which have the best accuracies, and those with one, reach optimal results and have an intense density.


_Figure 2: Distribution of architectures based on the number of skips (Cifar10)._


![alt text](https://github.com/jjguglielmi/How-Powerful-are-Skip-Connections-in-Neural-Architecture-Search-without-Training/blob/main/images/cifar10/histHighAccCifar10.png)


So, we pick only those elements with one skip connection. 
Then we generate a population where we have 50 architectures that include only one skip connection. Next, we do the evolution in cycles, filling a sample with a fixed number of candidates picked from the population. At this point, the higher accuracy model is picked from the population and is renamed as “parent”, because the next step will be the mutation of the parent architecture, generating the “child” one. 


_Table 1 : SKIP-POSITION and TEST ACCURACY on first 5 higher accuracies (Cifar10)._


| Position of the skip  |     Test Accuracy     |   
| --------------------- | --------------------- |
|         0 0 1         |         89.16         |
|         0 0 1         |         89.12         |
|         0 0 1         |         89.09         |
|         0 0 1         |         89.04         |
|         0 0 1         |         89.03         |


For the caption `0 0 1`:
 -  The first 0 represents the absence of skip connections on the edge that connects the input node to the first middle one
 -  The second 0 represents the absence of skip connections on the edges that connect the input node to the second middle one and that connect the first middle node to the second middle one
 -  The 1 represents the presence of the skip connection either:
  - On the edge that connects the input node to the output one
  - Or on the edge that connects the first hidden node to the output one

The model that reaches the best accuracy, has 1 skip in the edges that come to the output node. But where is it located precisely? In our case study, we have found that, selecting the first 10 architectures in terms of accuracy, all of them have the skip connection located on the edge that connects the input node to the output node. So, our operation mutation found consists of the satisfaction of the quoted condition. If the skip connection is not there, then the mutation occurs with the following possible options: 
1) If the skip connection is on the edge that connects the input node to the first node, then we invert this operation with the one that connects the input node and the output node. 
2) If the skip connection is on the edge that connects the input node to the second node, then we invert the skip operation with the one located on the edge that connect the input node to the output node, and we do an operation inversion between: edge that connects the nodes in the middle and the edge that attaches the first node to the output node. 
3) If the skip connection is on the edge that connects the first node to the second one, first of all we overturn this with the one on the edge that brings together the input node with the second node. After that it is executed the point 2. 
4) If the skip connection is on the edge that connects the first node with the output node, simply we interchange this operation with the one on the edge that connects the input node with the output node. 
5) If the skip connection is on the edge that connects the second node with the output node, we do nothing more than change this operation with the one on the edge that connects the input node with the output node.


## Architectures for datasets with metrics
You can find all the csv files with the all 15625 architectures applied to each dataset with a batch_size of 32, representing the respective dataset, the structure of the architecture, the metric NASWOT, the correspondent test-accuracy and at the end the execution time. All the plots reported in the file plots.py, are based on the same batch size.
## How to run the code
- Install [PyTorch](https://pytorch.org/) for your system (v1.5.0 or later).
- Install the package: `pip install .` (add `-e` for editable mode) -- note that all dependencies other than pytorch will be automatically installed.
  - Reproducing results on NATS Benchmark
  All you need is the file called Project8_Group1_MLDL.ipynb. There you can find all our purposes.
  
  
