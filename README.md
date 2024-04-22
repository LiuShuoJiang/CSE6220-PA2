# Georgia Tech CSE 6220 Lab 2

## Overview

This program performs matrix transposition using MPI (Message Passing Interface) in parallel. It implements three different algorithms for the all-to-all communication primitive: arbitrary permutations, hypercubic permutations, and `MPI_Alltoall`. The program takes an input matrix, transposes it using the specified algorithm, and writes the transposed matrix to an output file.

## Program Usage

The program takes the following command line arguments:

1. Input file: The file containing the input matrix.
2. Output file: The file where the transposed matrix will be written.
3. Algorithm choice: `'a'` for arbitrary permutations, `'h'` for hypercubic permutations, or `'m'` for `MPI_Alltoall`.
4. Matrix size (`n`): The size of the square matrix (`n x n`).

Before running, please compile the program with `make` command.

Sample command line input:

```shell
mpirun -np 8 ./transpose matrix8.txt transpose.txt m 8
```

## Program Workflow

1. The program starts by reading the command line arguments and initializing MPI.
2. Process 0 reads the input matrix from the specified file and scatters the rows among all processes using `MPI_Scatter`.
3. Each process performs the matrix transposition using the specified algorithm:
    - Arbitrary permutations (`HPC_Alltoall_A`): Each process communicates with every other process in an arbitrary order.
    - Hypercubic permutations (`HPC_Alltoall_H`): Processes communicate with their neighbors in a hypercube topology.
    - `MPI_Alltoall`: The built-in MPI function for all-to-all communication is used.
4. After the transposition, each process performs local data manipulation to rearrange the transposed matrix elements.
5. The transposed matrix is gathered to process 0 using `MPI_Gather`.
6. Process 0 writes the transposed matrix to the specified output file.
7. The program prints the time taken for the transposition (excluding I/O) in milliseconds.
8. MPI is finalized, and the program exits.

## Implementation Details

- The program is written in C++ and uses MPI for parallel communication.
- The arbitrary permutations algorithm (`HPC_Alltoall_A`) is implemented using point-to-point communication, where each process sends and receives data from every other process.
- The hypercubic permutations algorithm (`HPC_Alltoall_H`) is implemented using the hypercube topology, where processes communicate with their neighbors in each dimension of the hypercube.
- The program assumes that the number of processes divides the matrix size evenly and that the number of processes is a power of 2.

## Local Machine Used for Results

Windows Subsystem for Linux (WSL) running Ubuntu 22.04
