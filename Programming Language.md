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

### Aiming Low Is Harder

Topic : Probabilistic Programming

<https://arxiv.org/pdf/1904.01117.pdf>

Computing exact expected value in probabilistic programming is infeasible, therefore approximating it with upper bound and lower bound is helpful.
Proving upper bound is relatively simple since semantics of loop gives least fixed point, therefore showing inductiveness is enough.
Conversely, proving lower bound is complex since inductiveness only gives least fixed point over lower bound, which may not be value.
This paper uses well-known results in probability theory, called martingale and optional stopping time to give inductive rule for proving this.
In martingale theory, notion called uniform integrability gives bound on intermediate value, and convergence result on (sub/super) martingales.
To use this notion, this paper transforms some conditions on uniform integrability to weakest preexpectation transformer for probabilistic programming.
Similar to above work, this paper also gives rule on proving lower bound of expected runtime with erf transformer.

### A Simple Differentiable Programming Language

Topic : Differentiable Programming

<https://arxiv.org/abs/1911.04523>

Currently, one of mostly well used DSL(Domain Specific Language) is differentiable programming langauge, like torch and tensorflow. 
This paper gives formal semantics for such differentiable programming language, both operational and denotational. 
The paper implements operational semantics based on trace semantics, which means that the variable stores its computation path, so that
we can use the trace to compute derivative. In specific, the control commands are partially evaluated while computations are not evaluated.
For the denotational semantics, the paper defines continuity, differentiability, and smoothness on partial functions, and define complete
partial order on differentiable functions.
Finally, the paper proves that given semantics is adequate, that both operational and denotational semantics coincide and complete, that denotational semantics
is defined when operational semantics terminates.
This paper only treats vectors and simple imperative language, and recent paper proposed differentiability on probability, higher data type with higher order functions.

### RustBelt: Securing the Foundations of the Rust Programming Language

Topic : Concurrent Programming

<https://people.mpi-sws.org/~dreyer/papers/rustbelt/paper.pdf>

Rust Programming Language is a secure system programming language, where other system programming languages are not secure.
The core idea of Rust is ownership type, where each pointer has owner variable so that shared mutable state cannot exists.
However there were no proof of correctness that the type system ensures such behavior. This paper proves this by establishing lifetime logic.
Also, one of main feature of Rust is unsafe Rust, where we can use shared mutable state, but in a controlled manner.
The idea of verifying such unsafe type is allowing type to pick its two predicate shr and own, so that those gives what it means to
own that pointer, or share that pointer. For example Mutex type's shr is defined as mutable inner structure.

### Paradoxes of Probabilistic Programming

Topic : Probabilistic Programming

<https://arxiv.org/pdf/2101.03391.pdf>

Probabilistic Programming Languages are well used in machine learning tasks, however they sometimes require to 'carefully' designed to avoid some issues.
This paper analyzes three types of paradoxes of probabilistic programming, and proposes two new semantics for resolving this paradoxes.
The first paradox is difference on unit, that the two conditionals chosen randomly, is evaluated on different types of value, gives different expectation value
when we change the unit.
The second paradox is on different number of observes, that continuous observe is always measure zero event, so branch with least number of observe dominates all others.
The last paradox is on parameter transformation, that transforming random variables like normal to lognormal changes the expectation value.
The main reason these paradoxes arises is that continuous observation is done on measure zero event, where discrete observation doesn't.
So first proposal changes continuous observation to interval observation, instead of point estimation, which will give exact expected value when we let interval 
converges to point.
The second proposal adopts the idea of infinitesimal, instead of limit. Meaning that each continuous observation gives infinitesimal weight, which will be summed and divided.
With this new semantics, the paper proves that semantics are well matched, and limiting behavior is correct.

### Towards Verified Stochastic Variational Inference for Probabilistic Programs

Topic : Static Analysis, Probabilistic Programming

<https://arxiv.org/abs/1907.08827>

Stochastic Variational Inference is one of most popular probabilistic inference algorithm that finds approximation of posterior inference.
Due to its use of probabilistic expressions and differentiation, the correctness of SVI highly depends on some of assumptions on implementation.
This paper designs conditions for SVI to be defined, and implements static analyzer that checks the Pyro program.
First, SVI mainly consists two programs, model and guide. The model is defined using sampling and conditioning, where guide only uses sample.
The conditions for SVI to converges, is that guide's sampling must be absolutely continuous to model's sampling (model-guide match),
parameter differentiability, and exchange of differential with integration.
Each of conditions are weakened, for example last one with continuous differentiability, and the weakened conditions are checked with static analysis.
As a result, they found two model-guide mismatch, and verified 31 examples. The analysis took less than a second, so it was scalable enough.

### A unifying type-theory for higher-order (amortized) cost analysis

Topic : Type theory, Amortized analysis

<https://www.cs.cmu.edu/~janh/assets/pdf/RajaniGDH20.pdf>

Cost analysis is topic of static verification that tries to find upper bound of the cost of program, which includes elapsed time, memory use.
In this paper, cost analysis is performed with help of type theory, so that typed program gives upper bound of cost.
The main challenge of this paper is that this cost analysis includes amortized cost analysis, meaning it focuses complexity of algorithm, instead of operation.
In specific, this paper uses potential method, which attaches potential to each type that allows saving credit for future use.
The semantics of type is defined by 3-tuple (p, T, v) where p is potential, T is number of execution of rule, v is value.
With refinement type for length of list, the paper successfully proves three examples bound, FIFO list implemented with two queue, church encoding, list fold.
There are two already existing works, RAML and dlPCF, and this paper shows that RAML can be embedded to lambda-amor without sub-exp, and
dlPCF can be embedded to lambda-amor with sub-exp.

### λPSI: Exact Inference for Higher-Order Probabilistic Programs

Topic : Probabilistic Programming

<https://files.sri.inf.ethz.ch/website/papers/pldi20-lpsi.pdf>

Most of probabilistic inference for probabilistic programs are approximation, for example monte carlo algorithms and importance sampling.
However as they are approximation, they suffer from approximation error and nondeterminism. This leads to need of exact inference.
Before this paper, PSI is an exact inference algorithm for simple imperative probabilsitic programs.
This paper extends PSI system, so that we can handle conditioning on function, and to do so, it explicitly uses lebesgue measure.
First the denotation of program is computed recursively, using iversion bracket, integration, disintegration.
For example, sampling is denoted by sum of dirac delta or integration of variable, where observing is denoted by integrating
iversion bracket or integrating cobserve and disintegrate it.
Now the denotation is simplified by symbolic computation, including dirac delta substitution, linearization and guard simplification, 
symbolic (dis-)integration.
There exists some programs that can't be simplified enough, especially when having product of two gaussian, but lambda PSI successfully
exactly solved 30 instances of 31 instances.

### A Language for Probabilistically Oblivious Computation

Topic : Security Programming, Probabilistic Programming

<https://arxiv.org/abs/1711.09305>

Oblivious computation prevents leak of secured data, by adding randomization to public data. 
This paper presents lambda_obliv, which is a language for oblivious computation, where oblivious computation is ensured by type system.
The core idea is using affine type, where each secure random bit can be observed only once in public use. 
So in type system, each random bit are labelled whether they are secured or public, so that casting secured random bit to public bit only occurs once.
However even with this rule, the independence between secured bits and observable trace is not ensured, due to probabilstic correlation.
To address this problem, the type system uses probability region, which is a join lattice structure over subsets of random bits.
Two set of random bits rho' is less than rho, if every random bits in rho are independent to rho'. 
Now by allowing xor operation to ordered random bits, we acheive absence of probabilistic correlation.
The exact specification of oblivious computation is given by PMTO, which argues that if two typed terms are observably equivalent,
they both continue progress, and the observation during the execution is also equal.

### Denotational Validation of Higher-Order Bayesian Inference

Topic : Probabilistic Programming

<https://arxiv.org/pdf/1711.03219.pdf>

This paper gives formal framework for validating higher order bayesian inferences.
To support continuous inference which suffer measure problem on higher order functions, this paper uses Quasi Borel Space for the semantics.
An inference representation contains 6 elements, three elements, functor T, object functions return\_X^T as X->TX, morphism functions >>=\_{X,U}^T as (X->Y)->(TX->TY) forming monadic inference, two QBS-morphisms sample^T and score^T, and the meaning morphism which should preserves return and composition, and sample is uniform distribution on [0, 1], score is mapped to rescaled measure.
Then the inference transformation is a function indexed by Set, that t_X : TX->SX that meaning morphism is preserved.
With this representation, this paper gives validation of two inference algorithm, sequential monte carlo and trace chain monte carlo algorithms are verified.
Sequential Monte Carlo algorithm is defined as composition of transformations, first spawn, then loop of resample and advance, which is proven that it forms an inference transformation.

### Trace types and denotational semantics for sound programmable inference in probabilistic languages

Topic : Probabilstic Programming

<https://dl.acm.org/doi/10.1145/3371087>

Some probabilistic programming language like Gen supports user to write own probabilistic inference algorithm, which enhances language's expressiveness.
However if user writes wrong inference, the resulting distribution may be unsound.
This paper develops type system for such probabilistic language, called trace type, that can check soundness of three types of inference engines, IS, MCMC, VI. 
The type of probailistic program has two types, trace type and return type. The trace type is defined as list of tuples, that each tuple consists label for probabilistic variable and its type which considers support of distribution.
To handle branching, fixed-length loop, arbitrary length loop, the distribution also have types sum type, vector type, list type respectively.
Importance sampling requires two conditions, posterior distribution must be absolutely continuous w.r.t. model, and two distribution must shares its base meausre.
These two conditions are ensured by trace type equality.
Similarly, kernel type of MCMC method ensures that modified variables are fully contained in trace types. 
Finally variational inference operations is also ensured that KL divergence is well defined (not infinite).

### Relatively Complete Verification of Probabilistic Programs

Topic : Probabilistic Programming, Automatic Verification

<https://arxiv.org/abs/2010.14548>

There are various probabilistic program verification methods, which can be distinguished to two types, extensional and intensional.
Extensional methods allows to reason arbitrary assertions, however this makes hard to verified automatically.
Intensional methods instead defines assertion language, and restrict them so that we can easily verify.
In specific, this paper aims to find some language, that if f is contained in that language its weakest preexpectation is also contained.
The target language is simple imperative language with probabilistic choice with rational probability.
Some of features of expression is carefully selected, for example we can't multiply two expectation since it can make undefined expectation for 0 times infinity.
And to model existential quantifier and universal quantifier, we take supremum and infimum.
Now most of language is well modelled, except while loop. To handle while loop, it uses Kozen duality and unfold loop, so that weakest preexpectation be sum of all execution paths.
Then the sum itself is encoded with language, by introducing Godel encoding for states and finite sequence of program states. 
Now given an oracle for comparing inequalities over assertion language, this system makes relative complete verification.

### Automatic Differentiation in PCF

Topic : Differentiable Programming

<https://arxiv.org/abs/2011.03335>

This paper develops semantics of automatic differentiation over PCF with reals, which is purely functional programming language with real numbers.
With this semantics, paper proves that automatic differentiation over this language is almost sound, that automatic differentiation and real derivative is different for measure zero set.
The goal of this paper is to prove that automatic differentiation is sound almost everywhere, taking any reduction strategy. 
First automatic differentiation is implemented on trace term, where trace term contains no conditional statement and fixed point.
Then set of stable points is defined, which is the points that epsilon ball shares same trace term. And in stable set, it is proven that automatic differentiation is correct. 
The second step is proving that unstable points, which may be unsound points, are measure zero. To prove this, paper defines quasivariety which is a subset of union of zero sets of not identically zero basic functions, named after algebraic variety from algebraic geometry.
Then by showing that set of unstable points is quasivariety and proving quasivariety is measure zero proves unsoundness is limited to measure zero.

### λS : Computable Semantics for Differentiable Programming with Higher-Order Functions and Datatypes

Topic : Differentiable Programming

<https://arxiv.org/abs/2007.08017>

This paper develops semantics for differentiable programming, by primitives coming from computable analysis.
First, due to undefined derivative problems as in max/min or ReLU, this paper uses Clarke derivative which is defined as convex hull of limit points of derivatives.
For example Clarke derivative of ReLU at 0 is \[0, 1\], since we have both 0, 1 as limit point, and taking convex hull gives \[0, 1\].
Then with Clarke derivative, semantics of functions are defined with infinite tower of derivatives.
The uniqueness of this paper is that it includes some higher order primitives, like intergral01, firstRoot and cutRoot, max01 and argmax01.
All these functions are defined with help of computable analysis, and their derivative is defined by infinitesimal perturbation and implicit function theorem.
For the implementation, paper uses arbitrary precision where if the precision can't be achieved, program may not terminate. The explicit implementation relys on
computable analysis.
With support of higher order primitives, paper gives library for probability distribution, surface, parametric surface.

### A Pre-Expectation Calculus for Probabilistic Sensitivity

Topic : Probabilistic Programming

<https://arxiv.org/abs/1901.06540>

Sensitivity of program is measure that how program's output change on its output's change.
In probabilistic programs, such a measure gives good bound for convergence rate of algorithms.
This paper gives a rule-based approach to prove bound for relative weakest pre extpectation.
Bbecause related pre expectation uses Kantorovich distance which uses infimum over probabilistic couplings, 
related pre expection is not composable, as optimal solution is not resulted by choosing locally optimal solutions.
So instead, paper definds bound of pre expectation that is composable. if-else statement allows bound only if its conditional
is synchronized, and while loop is defined as least fixed point. 
Two problem exists in this definition, first sampling statement's pre expectation is defined as Kantorovich distance, which uses infimum
so can not be computed, and similarly lfp can not be computed.
The rule uses approximation for both two problems, sampling rule that bounds Kantorovich distance by giving one sample of coupling,
and loop invariant gives bound for least fixed point. 
With these rules, the paper proves algorithmic stability of SGD and TD(0), and card shuffle algorithms.
Also by defining TV distance with supremum and defining lower invariant, the paper gives method for proving lower bound of wpe.
