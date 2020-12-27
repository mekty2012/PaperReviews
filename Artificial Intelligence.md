# Artificial Intelligence


### Neural Compositional Denotational Semantics for Question Answering

Topic : Deep Learning, Natural Language Processing, Computational Linguistics

<https://arxiv.org/pdf/1808.09942.pdf>

One of main challengge in computational linguistics is query and answering, given knowledge graph.
In specific, knowledge graph is given as n items, properties of item, relations between item.
Then the goal is to give correct answer for natural language question.
This paper neural compositional denotational semantics, which is combination of ML-based NLP and
computational linguistics based NLP.
First, every span of text are assigned to type, like E(entity), T(Truth), R(Relation), phi(vacuous).
And their embedding is defined by their type, as entity to n-length vector, relation to n times n matrix.
For each syntax rule, like E(Large) + E(Red) -> E(Large Red), we define rule to compose each embedding
of types. 
Then this model is trained, where trained vector is embedding of each word, and probability of type for each word.
In result, this model shows 100% correctness for short questions, and 84.6% correctness for complex question.

### An Inductive Synthesis Framework for Verifiable Reinforcement Learning

Topic : Reinforcement Learning, Program Verification

<https://herowanzhu.github.io/herowanzhu.github.io/pldi2019.pdf>

There are consecutive researches on verification of neural network, which is complex due to nonlinearity and
stochasticity of neural network.
This paper focuses on verifying reinforcement learning on low number of variable.
First it generates deterministic, relatively simple function that approximates neural network,
and find some invariant that assures this simple function is safe.
Now to lift safety proof to neural network, if neural network violates invariant, we use function's action
instead of neural network.
In finding deterministic function and constraint, it first finds some initial point not-currently-covered, and tries to find
neighborhood of that point which is consistent under deterministic function by reducing radius.
Then if whole initial space is covered with such covers, conditional combination of deterministic function becomes resulting function.

### Code2Vec: Learning Distributed Representations of Code, Code2Seq: Generating Sequences From Structured Representations of Code

Topic : Embedding, Programming Language

<https://arxiv.org/pdf/1803.09473.pdf> <https://arxiv.org/pdf/1808.01400.pdf>

These two consecutive models is new neural network model on programming language, which is highly structured.
The basic principle is viewing code as a AST(Abstract Syntax Tree), and input as path between terminal nodes.
In Code2Vec, each unique terminal and paths are embedded to d-dimensional vector. Now for each path and its two ends,
embed it to d-dimensional vector with FC layer, and finally multiplying attention value gives resulting representation.
In Code2Seq, each terminal is decomposed to subterminals, as ArrayList to Array + List, and representation is summed up.
The path is encoded by LSTM to single d-dimensional vector.
Now concatenate two ends and path, embed to d-dimensional vector with FC-layer, and multiplying attention, finally decoder network
generates target sequence.

### Learning Nonlinear Loop Invariants with Gated Continuous Logic Networks

Topic : Logic Learning, Programming Language

<https://arxiv.org/pdf/2003.07959.pdf>

One of main undecidability in Program Logic is Loop Invariant, because we need to search for many formulas especially when invariant is nonlinear.
This paper proposes differentiable programming approach on loop invariant finding, with gated continuous logic network.
The basic idea relies on fuzzy logic, which is viewing a logic as a differentiable function satisfying some conditions, like T-norm and T-conorm.
To successfully find conjunction/disjunctions, the paper uses gated T-norm, which is continuous variation of conjunction and T-conorm.
And to find tight bound the formula uses new activation function which penalizes when term is too large (not tight).
Since convergence is not assured due to less test datas, the formula generalizes tests by using different initial value.
Also to encourage simplicity of formula and learning procedure, we use term dropout which is exhibiting some terms before training.
Final structure forms as CNF form, with two layer representing equality and inequality.
Then from learned fuzzy-logic formula, we extract discrete formula, and after rounding it, test with theorem prover. If there is counterexample,
add it as data and train again.
The proposed model resulted 26/27 correctness, with 97.5% convergence rate on quadratic problem.
