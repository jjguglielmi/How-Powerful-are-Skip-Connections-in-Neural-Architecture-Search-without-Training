# How Powerful are Skip Connections in Neural Architecture Search without Training
Group 1 / Project 8 -> S.Ferrua, J.Guglielmi, A.Orazalin

Machine Learning and Deep Learning course in master's degree at Politecnico di Torino.

In this README, we provide:
- [Summary of our work](#Summary)
- [Architecture pretrained with NASWOT metric](#Architectures-for-datasets-with-metrics)
- [How to run the code](#Running-the-Code)


*If you have any questions, please open an issue or email us:*
- S.Ferrua : s292946@studenti.polito.it
- J.Guglielmi : s303105@studenti.polito.it
- A. Orazalin: s298255@studenti.polito.it

## Summary
**Abstract.** Neural Architecture Search is a technique that
aspires to discover the best architecture for a neural network
based on a specific need. Briefly, NAS uses a controller network
that produces an architecture suggestion specified by a mutablelength string. The neural network is then trained by a child
network on real data, generating a signal fed into the controller,
which will subsequently give a higher probability of architectures
reaching greater accuracies. In the last few years, researchers
have employed their efforts to implement algorithms that bring
improvements in the quality of the NAS. This research, which is
based on this concept, aims to demonstrate how the application
of certain variations on the search algorithm, might provide
better outcomes. The first phase accomplished on NAS, consists
of computing ahead of time the training-free metrics for all
the possible architectures, scoring each of them at initialization.
Subsequently, the NASWOT algorithm is implemented, a similar
random search for picking randomly a candidate from the
search space and iteratively finding the best network. To do
a level up gaining higher scores, existing aging research is
examined, finding the Regularized Evolutionary Algorithm,
which can discriminate between the best architectures based
on a genotype structure at each iteration. However REA needs
a way to evaluate each architecture, so a zero-cost proxy is
provided, the synaptic flow score, known as SynFlow, which
was originally considered a pruner. Based on this research, the
focus moved on the mutation of the REA’s parent architecture,
because following certain considerations, it was noted that skip
connections have an importance during the aging evolution.


Skip connections are a module in which an algorithm can
skip multiple layers of architecture in convolutional neural
networks. These connections are used to connect features
derived from previous layers to further layers. The method
can also avoid the problem with a small element size, when in
previous layers there is multiplication on very small elements.
Skip connections are known for reducing the length of the
information propagation path after capturing long-term dependencies. In fact, during the previous experiments described, it
is found that architectures with skip connections scored higher,
as is evident in figure below, despite the fact that there are more of
those architectures without skip connections. To plot the graph
in figure below, 120 iterations conducted for Cifar10 were considered
with 4 different sample sizes, each of these executed 30 times.
The top 10 accuracy models were extracted from figure below.


![alt text](https://github.com/jjguglielmi/How-Powerful-are-Skip-Connections-in-Neural-Architecture-Search-without-Training/blob/main/images/histSkip_noSkipScorescifar10.png)




### Our variation in REA's mutation
To find the architecture that satisfies us, using search algorithms, we used skip connection as a filter to find such an architecture. 
We have decided to introduce a filter while we construct a random architecture, because by plotting histograms based on the conducted experiments, we find out that better accuracies are reached either with skips included in the process or not included at all. 

## Architectures for datasets with metrics
You can find all the csv files (from the `step1_and_2.py`)with the all 15625 architectures applied to each dataset with a batch_size of 128, representing the respective dataset, the structure of the architecture, the metric NASWOT, the correspondent test-accuracy and at the end the execution time. All the plots reported in the file `FinalProject.ipynb`, are based on the same batch size.
## How to run the code
- Install [PyTorch](https://pytorch.org/) for your system (v1.5.0 or later).
- Install the package: `pip install .` (add `-e` for editable mode) -- note that all dependencies other than pytorch will be automatically installed.
  - Reproducing results on NATS Benchmark
  
  All you need is the file called Project8_Group1_MLDL.ipynb. There you can find all our purposes.
  
 
