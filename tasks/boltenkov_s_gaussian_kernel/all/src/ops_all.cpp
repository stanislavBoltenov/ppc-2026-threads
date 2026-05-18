#include "boltenkov_s_gaussian_kernel/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <vector>

#include "util/include/util.hpp"

namespace boltenkov_s_gaussian_kernel {

BoltenkovSGaussianKernelALL::BoltenkovSGaussianKernelALL(const InType& in)
    : kernel_{{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}} {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = InType();
  }
  GetOutput() = std::vector<std::vector<int>>();
}

bool BoltenkovSGaussianKernelALL::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::size_t n = std::get<0>(GetInput());
    std::size_t m = std::get<1>(GetInput());
    const auto& data = std::get<2>(GetInput());
    if (data.size() != n) {
      return false;
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (data[i].size() != m) {
        return false;
      }
    }
    return true;
  }
  return true;
}

bool BoltenkovSGaussianKernelALL::PreProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::size_t n = std::get<0>(GetInput());
    std::size_t m = std::get<1>(GetInput());
    GetOutput().resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      GetOutput()[i].resize(m);
    }
  }
  return true;
}

void BoltenkovSGaussianKernelALL::BcastSizes(int& n, int& m, int rank) {
  if (rank == 0) {
    n = static_cast<int>(std::get<0>(GetInput()));
    m = static_cast<int>(std::get<1>(GetInput()));
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void BoltenkovSGaussianKernelALL::ScatterRows(const std::vector<std::vector<int>>& global_data,
                                              std::vector<std::vector<int>>& local_halo, int& local_start_row,
                                              int& local_rows, int m, int rank, int size) {
  int n = static_cast<int>(global_data.size());
  int rows_per_proc = (n + size - 1) / size;

  local_start_row = rank * rows_per_proc;
  int local_end_row = std::min(local_start_row + rows_per_proc, n) - 1;
  local_rows = (local_start_row < n) ? (local_end_row - local_start_row + 1) : 0;

  if (rank == 0) {
    for (int p = 1; p < size; ++p) {
      int p_start = p * rows_per_proc;
      int p_end = std::min(p_start + rows_per_proc, n) - 1;
      int p_rows = (p_start < n) ? (p_end - p_start + 1) : 0;
      if (p_rows == 0) {
        int zero = 0;
        MPI_Send(&zero, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
        continue;
      }
      int p_halo_first = std::max(0, p_start - 1);
      int p_halo_last = std::min(n - 1, p_end + 1);
      int p_halo_rows = p_halo_last - p_halo_first + 1;
      MPI_Send(&p_halo_rows, 1, MPI_INT, p, 0, MPI_COMM_WORLD);
      MPI_Send(&p_halo_first, 1, MPI_INT, p, 1, MPI_COMM_WORLD);
      MPI_Send(&p_start, 1, MPI_INT, p, 2, MPI_COMM_WORLD);
      MPI_Send(&p_rows, 1, MPI_INT, p, 3, MPI_COMM_WORLD);
      for (int i = 0; i < p_halo_rows; ++i) {
        const auto& row = global_data[p_halo_first + i];
        MPI_Send(const_cast<int*>(row.data()), m, MPI_INT, p, 4 + i, MPI_COMM_WORLD);
      }
    }

    if (local_rows > 0) {
      int halo_first = std::max(0, local_start_row - 1);
      int halo_last = std::min(n - 1, local_end_row + 1);
      int halo_rows = halo_last - halo_first + 1;
      local_halo.resize(halo_rows, std::vector<int>(m));
      for (int i = 0; i < halo_rows; ++i) {
        local_halo[i] = global_data[halo_first + i];
      }
    }
  } else {
    int recv_halo_rows = 0;
    MPI_Recv(&recv_halo_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (recv_halo_rows == 0) {
      local_rows = 0;
      return;
    }
    int recv_halo_first = 0;
    MPI_Recv(&recv_halo_first, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&local_start_row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&local_rows, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_halo.resize(recv_halo_rows, std::vector<int>(m));
    for (int i = 0; i < recv_halo_rows; ++i) {
      MPI_Recv(local_halo[i].data(), m, MPI_INT, 0, 4 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void BoltenkovSGaussianKernelALL::GatherResults(const std::vector<std::vector<int>>& local_res, int local_start_row,
                                                int local_rows, int m, int rank, int size) {
  if (rank == 0) {
    for (int i = 0; i < local_rows; ++i) {
      GetOutput()[local_start_row + i] = local_res[i];
    }
    for (int p = 1; p < size; ++p) {
      int p_rows = 0;
      MPI_Recv(&p_rows, 1, MPI_INT, p, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (p_rows == 0) {
        continue;
      }
      int p_start = 0;
      MPI_Recv(&p_start, 1, MPI_INT, p, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::vector<std::vector<int>> p_res(p_rows, std::vector<int>(m));
      for (int i = 0; i < p_rows; ++i) {
        MPI_Recv(p_res[i].data(), m, MPI_INT, p, 12 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      for (int i = 0; i < p_rows; ++i) {
        GetOutput()[p_start + i] = p_res[i];
      }
    }
  } else {
    if (local_rows > 0) {
      MPI_Send(&local_rows, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
      MPI_Send(&local_start_row, 1, MPI_INT, 0, 11, MPI_COMM_WORLD);
      for (int i = 0; i < local_rows; ++i) {
        MPI_Send(const_cast<int*>(local_res[i].data()), m, MPI_INT, 0, 12 + i, MPI_COMM_WORLD);
      }
    } else {
      int zero = 0;
      MPI_Send(&zero, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
    }
  }
}

bool BoltenkovSGaussianKernelALL::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0, m = 0;
  BcastSizes(n, m, rank);

  const auto& global_data = (rank == 0) ? std::get<2>(GetInput()) : std::vector<std::vector<int>>();
  std::vector<std::vector<int>> local_halo;
  int local_start_row = 0, local_rows = 0;
  ScatterRows(global_data, local_halo, local_start_row, local_rows, m, rank, size);

  if (local_rows == 0) {
    GatherResults(std::vector<std::vector<int>>(), 0, 0, m, rank, size);
    return true;
  }

  int halo_first = std::max(0, local_start_row - 1);
  int halo_rows = static_cast<int>(local_halo.size());
  std::vector<std::vector<int>> tmp(local_rows + 2, std::vector<int>(m + 2, 0));

  for (int i = 0; i < local_rows + 2; ++i) {
    int global_row = local_start_row - 1 + i;
    if (global_row >= halo_first && global_row < halo_first + halo_rows) {
      std::copy(local_halo[global_row - halo_first].begin(), local_halo[global_row - halo_first].end(),
                tmp[i].begin() + 1);
    }
  }

  std::vector<std::vector<int>> local_res(local_rows, std::vector<int>(m, 0));
  auto kernel = kernel_;
  int shift = shift_;

#pragma omp parallel for num_threads(ppc::util::GetNumThreads()) default(none) \
    shared(tmp, local_res, local_rows, m, kernel, shift)
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 1; j <= m; ++j) {
      int val = (tmp[i][j - 1] * kernel[0][0]) + (tmp[i][j] * kernel[0][1]) + (tmp[i][j + 1] * kernel[0][2]) +
                (tmp[i + 1][j - 1] * kernel[1][0]) + (tmp[i + 1][j] * kernel[1][1]) +
                (tmp[i + 1][j + 1] * kernel[1][2]) + (tmp[i + 2][j - 1] * kernel[2][0]) +
                (tmp[i + 2][j] * kernel[2][1]) + (tmp[i + 2][j + 1] * kernel[2][2]);
      local_res[i][j - 1] = val >> shift;
    }
  }

  GatherResults(local_res, local_start_row, local_rows, m, rank, size);
  return true;
}

bool BoltenkovSGaussianKernelALL::PostProcessingImpl() {
  return true;
}

}  // namespace boltenkov_s_gaussian_kernel
