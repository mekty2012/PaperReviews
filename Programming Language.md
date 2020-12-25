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

### Incorrectness Logic

Topic : Program Logic

<https://dl.acm.org/doi/abs/10.1145/3371078>

Hoare logic aims to prove correctness of program, by showing that every state satisfying precondition satisfies postcondition after execution.
However, conversely proving faultiness of program is harder in Hoare logic, since we need to prove that correctness is not provable.
Incorrectness logic is variant of Hoare logic, that is used to prove that there every state satisfying postcondition is reachable from some state satisfying precondition.
The paper proposes incorrectness logic over simple imperative language with labelled error statement, and nondeterminism.
The conditional execution and loop is modelled by kleene star and choice execution.
Rules proposed in paper is sound and complete, and author proved some bugs which is not captured by static analyzer.

### A Probabilistic Separation Logic

Topic : Program Logic, Probabilistic Programming

<https://arxiv.org/pdf/1907.10708.pdf>

One of importance notion in probability theory is independence, however probabilistic logic before are not enough to prove
independence easily. This paper, adopts separation logic's idea to probabilistic programming, designs program logic on 
simple imperative probabilistic programming language that is available to reason on independence.
In specific, the heap separation in separation logic becomes independence relation in this logic.
With this logic, the author proves four cryptographic examples are safe due to independence of public inforamtion and private information.
This logic is sound, however not complete due to randomized conditional and baseline theory on operators.
The decidability result is not yet known, however author states that like separation logic, PSL may be helped by small model property.
