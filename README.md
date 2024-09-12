# Collection of Sparsity Generating Techniques 

### Introduction
Sparse neural networks (SNNs) are neural networks which have some degree of "zero-weight" edges, are missing some neurons, or both. Due to the missing information, the weight matrices describing the connections between layers (e.g. of a Multilayer Perceptron) will contain zeros. These NN architectures present many opportunities for optimization since we do not need to explicity store the zero-weights or "dead neurons", but also present severe implementation challenges. Due to these challenges many modern frameworks avoid sparsity concerns alltogether. As models continue to grow, and their training consumes more and more energy, the status quo of using dense linear algbra kernels and including the zeros in the computations will simply not do. Furthermore, the pruning/compression techniques produce (in general) rectangular weight matrices with a seemingly random sparseity pattern. 
In an effort to study these seemingly random sparsity patterns, this repo will serve as a collection of techniques to generate SNNs. After generation, we can save the sparse weight matrices and study them in isolation. This should aid in insights and understanding of performance of central kernels in nerual network training and inference. Contributions are welcome.

### Usage
`generators/`
    - Contains seperate Python scripts for each generation method. Generators should write to `matrices/` (See `example.py`)
`matrices/` 
    - Meant to be somewhere to collect sparse matrices currently under investigation 
`scripts/` 
    - Contains useful scripts. Notably `mm2sparsityPattern.py` visualizes the sparsity pattern of a given .mtx file
`figures/`
    - Meant to be somewhere to collect the outputs from `mm2sparsityPattern.py`
