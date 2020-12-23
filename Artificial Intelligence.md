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
