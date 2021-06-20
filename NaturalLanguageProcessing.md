# Natural Language Processing


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
