#include <fstream>
#include <iostream>
#include <mpi/mpi.h>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <cmath>


std::vector<int> readMatrix(const std::string &filename, int n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + filename);
    }

    std::vector<int> matrix(n * n);
    std::string line;
    int row = 0;

    while (getline(file, line) && row < n) {
        std::istringstream iss(line);
        int value, col = 0;

        while (iss >> value && col < n) {
            matrix[row * n + col] = value;
            col++;
        }

        if (col != n) {
            std::cerr << "Incorrect number of elements in row " << row << " in file: " << filename << std::endl;
            exit(1);
        }

        row++;
    }

    if (row != n) {
        std::cerr << "Incorrect number of rows in file: " << filename << std::endl;
        exit(1);
    }

    file.close();
    return matrix;
}


void writeMatrix(const std::string &filename, const std::vector<int> &matrix, int n) {
    std::ofstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i * n + j] << (j < n - 1 ? ' ' : '\n');
        }
    }

    file.close();
}


void HPC_Alltoall_H(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int *sbuf = (int*)sendbuf;
    int *rbuf = (int*)recvbuf;

    if (size & (size - 1)) {
        throw std::runtime_error("HPC_Alltoall_H requires the number of processes to be a power of 2.");
    }

    memcpy(rbuf + rank * recvcount, sbuf + rank * sendcount, sendcount * sizeof(int));

    int logP = static_cast<int>(std::log2(size));
    int totalCount = size * sendcount;
    int messageCount = totalCount / 2;

    for (int i = logP - 1; i >= 0; i--) {
        std::vector<int> tempBuf(totalCount, 0);
        std::vector<int> messageBuf(messageCount, 0);
        int partner = rank ^ (1 << i);
        if ((rank & (1 << i)) == 0) {
            MPI_Sendrecv(sbuf + messageCount, messageCount, sendtype, partner, 0,
                         messageBuf.data(), messageCount, recvtype, partner, 0,
                         comm, MPI_STATUS_IGNORE);
//            MPI_Send(sbuf + messageCount, messageCount, sendtype, partner, 0, comm);
//            MPI_Recv(messageBuf.data(), messageCount, recvtype, partner, 0, comm, MPI_STATUS_IGNORE);
            for (int k = 0; k < size / 2; k++) {
                for (int m = 0; m < sendcount; m++) {
                    tempBuf[2 * k * sendcount + m] = sbuf[k * sendcount + m];
                    tempBuf[(2 * k + 1) * sendcount + m] = messageBuf[k * sendcount + m];
                }
            }
        } else {
            MPI_Sendrecv(sbuf, messageCount, sendtype, partner, 0,
                         messageBuf.data(), messageCount, recvtype, partner, 0,
                         comm, MPI_STATUS_IGNORE);
//            MPI_Recv(messageBuf.data(), messageCount, recvtype, partner, 0, comm, MPI_STATUS_IGNORE);
//            MPI_Send(sbuf, messageCount, sendtype, partner, 0, comm);
            for (int k = 0; k < size / 2; k++) {
                for (int m = 0; m < sendcount; m++) {
                    tempBuf[2 * k * sendcount + m] = messageBuf[k * sendcount + m];
                    tempBuf[(2 * k + 1) * sendcount + m] = sbuf[(k + size / 2) * sendcount + m];
                }
            }
        }
        memcpy(sbuf, tempBuf.data(), totalCount * sizeof(int));
        memcpy(rbuf, sbuf, totalCount * sizeof(int));
    }
}


void HPC_Alltoall_A(const void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for (int i = 0; i < size; ++i) {
        int sendTo = (rank + i) % size;
        int recvFrom = (rank - i + size) % size;
        MPI_Sendrecv((char*)sendbuf + sendTo * sendcount * sizeof(int), sendcount, sendtype, sendTo, 0,
                     (char*)recvbuf + recvFrom * recvcount * sizeof(int), recvcount, recvtype, recvFrom, 0,
                     comm, MPI_STATUS_IGNORE);
    }
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) {
            std::cout << "Usage: " << argv[0] << " <input file> <output file> <algorithm: a/h/m> <matrix size n>" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    char algo = argv[3][0];
    int n = atoi(argv[4]);

    if (n % size != 0 || (size & (size - 1)) != 0) {
        if (rank == 0) {
            std::cerr << "Number of processors must divide n and be a power of 2" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rowsPerProc = n / size;

    std::vector<int> matrix(n * n);
    std::vector<int> localMatrix(rowsPerProc * n);

    if (rank == 0) {
        try {
            matrix = readMatrix(inputFile, n);
            MPI_Scatter(matrix.data(), rowsPerProc * n, MPI_INT, localMatrix.data(), rowsPerProc * n, MPI_INT, 0, MPI_COMM_WORLD);
        } catch (const std::exception &e) {
            std::cerr << "Error in process 0 when reading: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }  else {
        MPI_Scatter(nullptr, rowsPerProc * n, MPI_INT, localMatrix.data(), rowsPerProc * n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    double startTime = MPI_Wtime();

    std::vector<int> sendBuffer(rowsPerProc * n);
    std::vector<int> receiveBuffer(rowsPerProc * n);

    for (int index = 0, k = 0; k < size; k++) {
        for (int i = 0; i < rowsPerProc; i++) {
            for (int j = 0; j < rowsPerProc; j++) {
                sendBuffer[index++] = localMatrix[i * n + j + k * rowsPerProc];
            }
        }
    }

    switch (algo) {
        case 'a':
            HPC_Alltoall_A(sendBuffer.data(), rowsPerProc * rowsPerProc, MPI_INT, receiveBuffer.data(), rowsPerProc * rowsPerProc, MPI_INT, MPI_COMM_WORLD);
            break;
        case 'h':
            HPC_Alltoall_H(sendBuffer.data(), rowsPerProc * rowsPerProc, MPI_INT, receiveBuffer.data(), rowsPerProc * rowsPerProc, MPI_INT, MPI_COMM_WORLD);
            break;
        case 'm':
            MPI_Alltoall(sendBuffer.data(), rowsPerProc * rowsPerProc, MPI_INT, receiveBuffer.data(), rowsPerProc * rowsPerProc, MPI_INT, MPI_COMM_WORLD);
            break;
        default:
            if (rank == 0) {
                std::cout << "Invalid algorithm choice. Valid options are 'a', 'h', or 'm'." << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> transposedMatrix(rowsPerProc * n);
    for (int index = 0, k = 0; k < size; k++) {
        for (int i = 0; i < rowsPerProc; i++) {
            for (int j = 0; j < rowsPerProc; j++) {
                transposedMatrix[j * n + i + k * rowsPerProc] = receiveBuffer[index++];
            }
        }
    }

    double endTime = MPI_Wtime();

    if (rank == 0) {
        std::vector<int> finalMatrix(n * n);
        MPI_Gather(transposedMatrix.data(), rowsPerProc * n, MPI_INT, finalMatrix.data(), rowsPerProc * n, MPI_INT, 0, MPI_COMM_WORLD);
        writeMatrix(outputFile, finalMatrix, n);
        printf("Time taken: %.6f ms\n", (endTime - startTime) * 1000);
    } else {
        MPI_Gather(transposedMatrix.data(), rowsPerProc * n, MPI_INT, nullptr, rowsPerProc * n, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
