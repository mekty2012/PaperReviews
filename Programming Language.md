# Programming Language Theory
Programming Language Theory is part of computer science that studies of syntax and semantics of programming language.

### A Convinient Category for Higher-Order Probability Theory

Topic : Probabilistic Programming, Denotational Semantics

<https://arxiv.org/abs/1701.02547>

Probabilistic programming is new paradigm of programming that enables inference on conditional distribution.
However, due to the problem that there is no natural measure on function space R -> R such that execution function is measurable, 
measurable space fails to correctly formalize denotational semantics of probabilistic programming language.
In this paper, the author proposes quasi-borel-space, which is space with set R -> X satisfying some conditions.
Then unlike Meas category, QBS category is cartesian closed, therefore can be used to formalize higher order probabilistic programming langauge.
Also the paper proposes probability monad which is a strong monad.
Finally with this formalization, the paper proves De Finetti's theorem.

### A Promising Semantics for Relaxed-Memory Concurrency

Topic : Concurrent Programming, Operational Semantics

<https://sf.snu.ac.kr/publications/promising.pdf>

Modern compilers optimize programs while preserving sequential correctness.
This optimization preserves semantics of program in sequential execution, however not in concurrent execution.
However with this importance, there were no formalization of semantics of concurrent programming considering optimization.
The main problem this paper treats is reordering instruction.
To correctly formalize ordering on instruction, the paper uses view, where per-message view, per-thread view, global view are used.
And for smarter compilers, they analyze code and replace variable with constant, allowing more reordering.
To formalize this problem, the paper uses 'promise' which is a phantom message that should be assured that it is correct.
