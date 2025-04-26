# Model Enumeration of Two-Variable Logic with Quadratic Delay Complexity




## 1. Code and Implementation
We use Python to implement the algorithm mentioned in the paper. The code is located in the directory `code` of this repository.

On Linux, the environment required to run the code can be installed by executing the following command:

`conda env create -f environment.yml`

You can run the example described in Section 3 of the paper by executing the following command:

`python enum_fo2.py -i sentence/nonisolated_graph.wfomcs -n 3 -p`

where `-i sentence/nonisolated_graph.wfomcs` represents the input $FO^2$ sentence, `-n 3` represents the domain size is 3, and `-p` represents printing the enumerated model to the terminal.

The following is the terminal output after the code is executed.

```
The compatible 1-types:
 0: ~E(X,X)
The compatible 2-types:
 (~E(X,X), ~E(X,X)):
   0: E(X,Y)^E(Y,X)
   1: ~E(X,Y)^~E(Y,X)

Model 1:
[[0 0 0]
 [0 0 0]
 [0 0 0]]
Model 2:
[[0 1 0]
 [1 0 0]
 [0 0 0]]
Model 3:
[[0 0 1]
 [0 0 0]
 [1 0 0]]
Model 4:
[[0 0 0]
 [0 0 1]
 [0 1 0]]

domain size: 3,
preprocess time(s): 0.001738457940518856,
enumeration time(s): 0.0018407920142635703,
num of meta configurations: 11,
num of enumerated models: 4
```
In the output, first output the valid 1-types and 2-types in the input sentence and their corresponding numbers. 
Then, start outputting the enumerated models. Each model is represented by an $n\times n$ matrix, where the $(i,i)$ -th value of the matrix represents the 1-type of the $i$ -th element, and the $(i,j)$ -th ( $i\neq j$ ) value represents the 2-types between the $i$ -th and $j$ -th elements.
Finally, output information such as the preprocessing time, enumeration time, and the number of enumerated models.

In the implementation, we define `meta configurations` as satisfying that **there is no configuration that is satisfiable and less than it (on the partial order defined in the paper)**.
Obviously, according to Theorem 3 in the paper, the number of meta configurations of the input sentence is independent of the domain size, and each satisfiable configuration can be derived by at least a meta configuration.
Therefore, we first find all meta configurations in preprocessing, and then when we solve a configuration decision problem, we only need to check whether there is a meta configuration that can be used to derive the current configuration, thereby speeding up the runtime of the enumeration. 

## 2. Experimental Results

We conducted experiments to evaluate the correctness and performance of our enumeration algorithm.
The experiments were performed on a computer with an 8-core Intel i7-9700K 3.60GHz processor and 32 GB of RAM.

We compared our algorithm with **Glucose**, which is a well-known SAT solver and has been integrated into **PySAT** (https://pysathq.github.io/docs/html/index.html).

We conducted experiments on two examples.
The first is the non-isolated graph described in the paper, and the sentence is as follows:

$$
\forall x: \neg E(x,x) \land \\
\forall x\forall y: E(x,y) \rightarrow E(y,x) \land \\
\forall x\exists y: (E(x,y)).
$$

The second is the friend-smoke example, and the sentence is as follows:

$$
\forall x: \neg fr(x,x) \land \\
\forall x\forall y: fr(x,y) \rightarrow fr(y,x) \land \\
\forall x\forall y: fr(x,y) \land sm(x) \rightarrow sm(y) \land \\
\forall x\exists y: fr(x,y)
$$

which means that the smoker's friends also smoke, and everyone has at least one friend.

We first verified the correctness of our algorithm by comparing the models output by the two algorithms and the results given by the model counting algorithm when the domain size are {3,4,5,6} (when the domain size is larger, there are more than 1M models and it is difficult to enumerate them all).

Then, we compared the runtime of the two algorithms to enumerate one million models with different domain sizes. The results are shown in the figure below.
Apparently, our algorithm is efficient and stable, significantly better than Glucose.
We may notice an interesting pattern in the performance of Glucose which becomes better in enumerating the first 1M solutions as the domain size increases. This paradoxical behavior has a simple explanation. 
For larger domains, finding the first 1M models is easier because there are more “easy-to-find” models.
Therefore, the All-SAT solver takes more time when the domain size is smaller, because it is difficult to find model when there are few models to be enumerated.

![image](https://anonymous.4open.science/r/enum_fo2/results/runtime.jpg)


Next, we compared the delay for the two algorithms to enumerate every 10K models when the domain size is 10, that is, the interval between the algorithm outputting the $(1\times 10K)$ -th model, the $(2\times 10K)$ -th model, the $(3\times 10K)$ -th model, ..., and the $(100\times 10K)$ -th model.
The results are shown in the figure below.
This well demonstrates the stability and advantages of our algorithm's quadratic delay.
As we analyzed earlier, the All-SAT solver becomes less efficient as it enumerates more and more models. This is because it constantly adds the enumerated models as 'block clauses' to the logic formula, making the formula more complicated.

![image](https://anonymous.4open.science/r/enum_fo2/results/delay.jpg)
