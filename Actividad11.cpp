#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>

#define N 10
#define FAILURE_ITERATION 5

struct CheckpointState {
    int iteration;
    int partial_sum;
};

std::string get_checkpoint_filename(int rank) {
    std::ostringstream oss;
    oss << "checkpoint_" << rank << ".dat";
    return oss.str();
}

void save_checkpoint(int rank, const CheckpointState& state) {
    std::ofstream ofs(get_checkpoint_filename(rank), std::ios::binary);
    if (ofs) {
        ofs.write(reinterpret_cast<const char*>(&state), sizeof(CheckpointState));
    }
}

bool load_checkpoint(int rank, CheckpointState& state) {
    std::ifstream ifs(get_checkpoint_filename(rank), std::ios::binary);
    if (ifs) {
        ifs.read(reinterpret_cast<char*>(&state), sizeof(CheckpointState));
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CheckpointState state;
    bool resumed = load_checkpoint(rank, state);
    if (!resumed) {
        state.iteration = 0;
        state.partial_sum = 0;
    }

    std::vector<int> vec(N, rank + 1);

    for (int i = state.iteration; i < N; ++i) {
        state.partial_sum += vec[i];
        state.iteration = i + 1;

        if (state.iteration % 2 == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            save_checkpoint(rank, state);
        }

        // 🟢 AHORA el fallo va después del checkpoint
        if (state.iteration == FAILURE_ITERATION && argc < 2) {
            std::cout << "Proceso " << rank << " simula fallo en iteración " << state.iteration << std::endl;
            MPI_Finalize();
            exit(1);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Simula cómputo
    }

    std::cout << "Proceso " << rank << " terminó. Resultado parcial: " << state.partial_sum << std::endl;

    MPI_Finalize();
    return 0;
}
