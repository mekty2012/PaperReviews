# Software Engineering
Software Engineering is part of computer engineering that treats process of software development and maintenance.

### Evolutionary Improvement of Programs

Topic : Search Based Software Engineering

<https://dl.acm.org/doi/10.1109/TEVC.2010.2083669>

Genetic Programming is part of SBSE that evolve program with genetic algorithm to optimize some constraint.
This paper tries to optimize the code in terms of execution time, while preserving the semantic correctness.
The genetic algorithm initialized with given program, and genetic algorithm tries to evolve the program
with fitness function of elapsed time and test case correctness. 
To enhance the performance, the author co-evolved the test case population, and used multi objective optimization.
For the result, the program successfully optimized, that modern compilers fails to optimize.

### Evolving Human Competitive Spectra-Based Fault Localisation Techniques

Topic : Search Based Software Engineering

<https://link.springer.com/chapter/10.1007/978-3-642-33119-0_18>

Spectra-based Fault Localisation is part of software debugging techniques.
First, spectra is summary of test case execution, that stores number of correct execution, buggy execution,
correct non-execution, buggy non-execution, per each line of code.
Then SBFL tries to find some formula that orders each line of code by possibility of faultyness.
This paper tries to tackle this problem by genetically evolve function with fitness function as ranking of faulty line.
The result of evolution was human-competitive, in a sense that there were some formula that exceeds human designed formulas.
For the follow up study, some of formula was proven that it is optimal.
