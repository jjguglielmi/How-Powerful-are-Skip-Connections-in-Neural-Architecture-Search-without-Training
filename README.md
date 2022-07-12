# How Powerful are Skip Connections in Neural Architecture Search without Training
Group 1 / Project 8 -> S.Ferrua, J.Guglielmi, A.Orazalin

Machine Learning and Deep Learning course in master's degree at Politecnico di Torino.

In this project, you'll find the application of NAS algorithm on NATS-Bench. 

Neural Architecture Search is a technique that aims at discovering the best architecture for a neural network based on a specific need. Briefly, NAS is a gradient-based method using a controller network that produces an architecture suggestion specified by a mutable-length string. This string is then trained by a child network on real data, generating a signal fed into the controller, which after that, will give higher probability to architectures to reach higher accuracies. 
In the very last few years, researchers have employed their efforts to implement algorithms that bring improvements in the quality of the NAS. Our research, which is based on this concept, tries to demonstrate how the application of some variations on the search algorithm, might provide better outcomes.
The first phase we accomplish on NAS consist of computing ahead of time the training-free metrics for all the possible architectures, scoring each of them at initialization. After that, we implement the NASWOT algorithm, a similar random search for picking randomly a candidate from the search space and iteratively find the best network. To do a level up gaining higher scores, we look through existing aging research, finding the Regularized Evolutionary Algorithm, which can discriminate between the best architectures based on a genotype structure at each iteration. But REA needs a way to evaluate each architecture, so we provide a zero-cost proxy, the synaptic flow score, known as synflow, which was originally thought as a pruner.
Based on these research, we concentrate ourselves on the mutation of the REA’s parent architecture, because after few studies, we noticed that skip connections have an importance during the aging evolution.

## How to run the code
