# Hopfield Network
The aim of this repository is to gain a self-understanding of the Hopfield Network

### Theory
A **Hopfield Network** is a type of recurrent artificial neural network that serves as a content-addressable ("associative") memory system with binary threshold nodes. It is used primarily for solving optimization problems and storing patterns.

#### Structure
- The network consists of **N fully connected neurons**, meaning every neuron is connected to every other neuron (but not to itself).
- Each neuron has a binary state, typically represented as $s_i \in \{-1, +1\}$ where $ s_i = +1 $ represents the neuron being active and $ s_i = -1 $ represents the neuron being inactive.
- The state of the network is defined as a vector $ \mathbf{s} = [s_1, s_2, \dots, s_N] $.

#### Weights
- Each pair of neurons $ i $ and $ j $ are connected by a symmetric weight $w_{ij}$. The weights are usually learned based on a Hebbian learning rule to store specific patterns.
- The weight matrix $ W $ is symmetric ($ w_{ij} = w_{ji} $) and has zero diagonal entries ($ w_{ii} = 0 $), i.e., no self-connections.

#### Energy Function
The Hopfield network uses an energy function $ E(\mathbf{s}) $, which helps it converge to stable states (local minima) that correspond to the stored patterns. The energy function is defined as:

$$
E(\mathbf{s}) = -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} w_{ij} s_i s_j + \sum_{i=1}^{N} \theta_i s_i
$$

where:
- $ w_{ij} $ is the weight between neurons $ i $ and $ j $,
- $ s_i $ is the state of neuron $ i $,
- $ \theta_i $ is the threshold of neuron $ i $.

The goal of the network is to minimize this energy function through iterative updates of the neurons. In this implementation, we set $\theta_i = 0$

#### Dynamics
The network operates asynchronously by updating one neuron at a time based on the input from its neighbors. The update rule for neuron $ i $ is:

$$
s_i^{\text{new}} = \text{sign}\left( \sum_{j} w_{ij} s_j - \theta_i \right)
$$

This rule means neuron $ i $ updates its state to be the sign of the total input it receives from other neurons, weighted by the connection strengths.

#### Memory Storage and Retrieval
Hopfield networks can store patterns as stable states. To store a pattern $ \mathbf{s}^\mu $, the weights are trained using Hebbian learning:

$$
w_{ij} = \frac{1}{N} \sum_{\mu} s_i^\mu s_j^\mu
$$

- $ \mu $ indexes the different patterns to be stored.
- $ s_i^\mu $ is the state of neuron $ i $ in pattern $ \mu $.

When a partial or noisy version of a stored pattern is presented, the network converges to the closest stored pattern, effectively acting as an associative memory.

#### Limitations
- **Capacity**: The maximum number of patterns $ P $ that can be stored in a Hopfield network is limited. Empirically, it can store around $ 0.15N $ patterns, where $ N $ is the number of neurons.
- **Spurious states**: The network can sometimes converge to local minima that do not correspond to any stored pattern (spurious memories).


#### References
- J. J. Hopfield, *"Neural Networks and Physical Systems with Emergent Collective Computational Abilities,"* Proceedings of the National Academy of Sciences, 1982.


