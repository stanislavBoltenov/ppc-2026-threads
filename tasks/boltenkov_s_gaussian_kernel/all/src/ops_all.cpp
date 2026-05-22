#include "boltenkov_s_gaussian_kernel/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <limits>
#include <vector>

#include "boltenkov_s_gaussian_kernel/common/include/common.hpp"
#include "util/include/util.hpp"

namespace boltenkov_s_gaussian_kernel {

BoltenkovSGaussianKernelALL::BoltenkovSGaussianKernelALL(const InType &in)
    : kernel_{{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}}} {
  SetTypeOfTask(GetStaticTypeOfTask());
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    GetInput() = in;
  } else {
    GetInput() = InType();
  }
  GetOutput() = std::vector<std::vector<int>>();
}

bool BoltenkovSGaussianKernelALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    auto [n, m, data] = GetInput();
    if (data.size() != n) {
      return false;
    }
    for (const auto &row : data) {
      if (row.size() != m) {
        return false;
      }
    }
    return true;
  }
  return true;
}

bool BoltenkovSGaussianKernelALL::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t n = 0, m = 0;

  if (rank == 0) {
    n = std::get<0>(GetInput());
    m = std::get<1>(GetInput());
  }

  if (n > static_cast<std::size_t>(INT_MAX - 2) || m > static_cast<std::size_t>(INT_MAX - 2)) {
    return false;
  }

  int n_val = static_cast<int>(n);
  int m_val = static_cast<int>(m);

  MPI_Bcast(&n_val, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n_val <= 0 || m_val <= 0) {
    return false;
  }

  GetOutput().resize(static_cast<std::size_t>(n_val));
  for (auto &row : GetOutput()) {
    row.resize(static_cast<std::size_t>(m_val));
  }
  return true;
}

void BoltenkovSGaussianKernelALL::BcastSizes(int &n, int &m, int rank) {
  if (rank == 0) {
    auto n_sz = std::get<0>(GetInput());
    auto m_sz = std::get<1>(GetInput());
    n = (n_sz > INT_MAX - 2) ? 0 : static_cast<int>(n_sz);
    m = (m_sz > INT_MAX - 2) ? 0 : static_cast<int>(m_sz);
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

void BoltenkovSGaussianKernelALL::SendRowsToOneProcess(int proc, int rows_per_proc, int n, int m,
                                                       const std::vector<std::vector<int>> &global_data) {
  if (rows_per_proc <= 0 || n <= 0 || m <= 0) {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
    return;
  }

  long long p_start_ll = static_cast<long long>(proc) * rows_per_proc;
  long long p_end_ll = std::min(p_start_ll + rows_per_proc, static_cast<long long>(n)) - 1;

  if (p_start_ll > INT_MAX || p_end_ll > INT_MAX || p_start_ll < 0) {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
    return;
  }

  int p_start = static_cast<int>(p_start_ll);
  int p_end = static_cast<int>(p_end_ll);
  int p_rows = (p_start < n) ? (p_end - p_start + 1) : 0;

  if (p_rows <= 0) {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
    return;
  }

  int p_halo_first = std::max(0, p_start - 1);
  int p_halo_last = std::min(n - 1, p_end + 1);
  int p_halo_rows = p_halo_last - p_halo_first + 1;

  MPI_Send(&p_halo_rows, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
  MPI_Send(&p_halo_first, 1, MPI_INT, proc, 1, MPI_COMM_WORLD);
  MPI_Send(&p_start, 1, MPI_INT, proc, 2, MPI_COMM_WORLD);
  MPI_Send(&p_rows, 1, MPI_INT, proc, 3, MPI_COMM_WORLD);

  for (int i = 0; i < p_halo_rows; ++i) {
    const auto &row = global_data[static_cast<std::size_t>(p_halo_first + i)];
    MPI_Send(row.data(), m, MPI_INT, proc, 4 + i, MPI_COMM_WORLD);
  }
}

void BoltenkovSGaussianKernelALL::FillLocalHaloForRoot(int local_start_row, int local_end_row, int n, int m,
                                                       const std::vector<std::vector<int>> &global_data,
                                                       std::vector<std::vector<int>> &local_halo) {
  if (local_start_row < 0 || local_end_row < 0 || m <= 0) {
    return;
  }

  int halo_first = std::max(0, local_start_row - 1);
  int halo_last = std::min(n - 1, local_end_row + 1);
  int halo_rows = halo_last - halo_first + 1;

  if (halo_rows > 0) {
    local_halo.resize(static_cast<std::size_t>(halo_rows), std::vector<int>(static_cast<std::size_t>(m)));
    for (int i = 0; i < halo_rows; ++i) {
      local_halo[static_cast<std::size_t>(i)] = global_data[static_cast<std::size_t>(halo_first + i)];
    }
  }
}

void BoltenkovSGaussianKernelALL::ReceiveRowsOnWorker(int m, int &local_start_row, int &local_rows,
                                                      std::vector<std::vector<int>> &local_halo) {
  int recv_halo_rows = 0;
  MPI_Recv(&recv_halo_rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (recv_halo_rows <= 0 || m <= 0) {
    local_rows = 0;
    return;
  }

  int recv_halo_first = 0;
  MPI_Recv(&recv_halo_first, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&local_start_row, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&local_rows, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (recv_halo_rows > 0 && m > 0) {
    local_halo.resize(static_cast<std::size_t>(recv_halo_rows), std::vector<int>(static_cast<std::size_t>(m)));
    for (int i = 0; i < recv_halo_rows; ++i) {
      MPI_Recv(local_halo[static_cast<std::size_t>(i)].data(), m, MPI_INT, 0, 4 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

void BoltenkovSGaussianKernelALL::ScatterRows(std::vector<std::vector<int>> &global_data,
                                              std::vector<std::vector<int>> &local_halo, int &local_start_row,
                                              int &local_rows, int m, int rank, int size) {
  int n = static_cast<int>(global_data.size());
  if (n <= 0 || m <= 0) {
    local_rows = 0;
    return;
  }

  long long rows_per_proc_ll = (static_cast<long long>(n) + size - 1) / size;
  if (rows_per_proc_ll > INT_MAX) {
    local_rows = 0;
    return;
  }
  int rows_per_proc = static_cast<int>(rows_per_proc_ll);

  long long local_start_ll = static_cast<long long>(rank) * rows_per_proc;
  if (local_start_ll > INT_MAX) {
    local_rows = 0;
    return;
  }
  local_start_row = static_cast<int>(local_start_ll);

  long long local_end_ll = std::min(local_start_ll + rows_per_proc_ll, static_cast<long long>(n)) - 1;
  local_rows = (local_start_row < n) ? (static_cast<int>(local_end_ll) - local_start_row + 1) : 0;

  if (rank == 0) {
    for (int proc = 1; proc < size; ++proc) {
      SendRowsToOneProcess(proc, rows_per_proc, n, m, global_data);
    }
    if (local_rows > 0) {
      FillLocalHaloForRoot(local_start_row, local_start_row + local_rows - 1, n, m, global_data, local_halo);
    }
  } else {
    ReceiveRowsOnWorker(m, local_start_row, local_rows, local_halo);
  }
}

void BoltenkovSGaussianKernelALL::GatherResultsRoot(std::vector<std::vector<int>> &local_res, int local_start_row,
                                                    int local_rows, int m, int size) {
  // Копируем свои результаты
  for (int i = 0; i < local_rows; ++i) {
    GetOutput()[static_cast<std::size_t>(local_start_row + i)] = local_res[static_cast<std::size_t>(i)];
  }

  for (int proc = 1; proc < size; ++proc) {
    int p_rows = 0;
    MPI_Recv(&p_rows, 1, MPI_INT, proc, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (p_rows <= 0) {
      continue;
    }

    int p_start = 0;
    MPI_Recv(&p_start, 1, MPI_INT, proc, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (m <= 0) {
      continue;
    }

    std::vector<std::vector<int>> p_res(static_cast<std::size_t>(p_rows),
                                        std::vector<int>(static_cast<std::size_t>(m)));

    for (int i = 0; i < p_rows; ++i) {
      MPI_Recv(p_res[static_cast<std::size_t>(i)].data(), m, MPI_INT, proc, 12 + i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    for (int i = 0; i < p_rows; ++i) {
      GetOutput()[static_cast<std::size_t>(p_start + i)] = std::move(p_res[static_cast<std::size_t>(i)]);
    }
  }
}

void BoltenkovSGaussianKernelALL::GatherResultsOthers(std::vector<std::vector<int>> &local_res, int local_start_row,
                                                      int local_rows, int m) {
  if (local_rows > 0) {
    MPI_Send(&local_rows, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
    MPI_Send(&local_start_row, 1, MPI_INT, 0, 11, MPI_COMM_WORLD);
    for (int i = 0; i < local_rows; ++i) {
      MPI_Send(local_res[static_cast<std::size_t>(i)].data(), m, MPI_INT, 0, 12 + i, MPI_COMM_WORLD);
    }
  } else {
    int zero = 0;
    MPI_Send(&zero, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
  }
}

void BoltenkovSGaussianKernelALL::GatherResults(std::vector<std::vector<int>> &local_res, int local_start_row,
                                                int local_rows, int m, int rank, int size) {
  if (rank == 0) {
    GatherResultsRoot(local_res, local_start_row, local_rows, m, size);
  } else {
    GatherResultsOthers(local_res, local_start_row, local_rows, m);
  }
}

std::vector<std::vector<int>> BoltenkovSGaussianKernelALL::ApplyGaussianFilter(
    const std::vector<std::vector<int>> &local_halo, int local_start_row, int local_rows, int m) {
  if (local_rows <= 0 || m <= 0) {
    return {};
  }

  const int halo_first = std::max(0, local_start_row - 1);
  const std::size_t halo_rows = local_halo.size();

  const std::size_t tmp_rows = static_cast<std::size_t>(local_rows) + 2;
  const std::size_t tmp_cols = static_cast<std::size_t>(m) + 2;

  std::vector<std::vector<int>> tmp(tmp_rows, std::vector<int>(tmp_cols, 0));

  for (int i = 0; i < local_rows + 2; ++i) {
    int global_row = local_start_row - 1 + i;
    if (global_row >= halo_first && static_cast<std::size_t>(global_row - halo_first) < halo_rows) {
      const auto &src = local_halo[static_cast<std::size_t>(global_row - halo_first)];
      for (int j = 0; j < m; ++j) {
        tmp[static_cast<std::size_t>(i)][static_cast<std::size_t>(j + 1)] = src[static_cast<std::size_t>(j)];
      }
    }
  }

  std::vector<std::vector<int>> local_res(static_cast<std::size_t>(local_rows),
                                          std::vector<int>(static_cast<std::size_t>(m), 0));

  const auto &kernel = kernel_;
  const int shift = shift_;

#pragma omp parallel for num_threads(ppc::util::GetNumThreads()) default(none) \
    shared(tmp, local_res, local_rows, m, kernel, shift)
  for (int i = 0; i < local_rows; ++i) {
    for (int j = 1; j <= m; ++j) {
      int val = (tmp[i][j - 1] * kernel[0][0]) + (tmp[i][j] * kernel[0][1]) + (tmp[i][j + 1] * kernel[0][2]) +
                (tmp[i + 1][j - 1] * kernel[1][0]) + (tmp[i + 1][j] * kernel[1][1]) +
                (tmp[i + 1][j + 1] * kernel[1][2]) + (tmp[i + 2][j - 1] * kernel[2][0]) +
                (tmp[i + 2][j] * kernel[2][1]) + (tmp[i + 2][j + 1] * kernel[2][2]);

      local_res[static_cast<std::size_t>(i)][static_cast<std::size_t>(j - 1)] = val >> shift;
    }
  }

  return local_res;
}

bool BoltenkovSGaussianKernelALL::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n = 0, m = 0;
  BcastSizes(n, m, rank);

  if (n <= 0 || m <= 0) {
    return false;
  }

  std::vector<std::vector<int>> global_data;
  if (rank == 0) {
    global_data = std::get<2>(GetInput());
  }

  std::vector<std::vector<int>> local_halo;
  int local_start_row = 0;
  int local_rows = 0;

  ScatterRows(global_data, local_halo, local_start_row, local_rows, m, rank, size);

  std::vector<std::vector<int>> local_res;
  if (local_rows > 0) {
    local_res = ApplyGaussianFilter(local_halo, local_start_row, local_rows, m);
  }

  GatherResults(local_res, local_start_row, local_rows, m, rank, size);

  // Финальный broadcast результата всем процессам
  for (int i = 0; i < n; ++i) {
    MPI_Bcast(GetOutput()[static_cast<std::size_t>(i)].data(), m, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool BoltenkovSGaussianKernelALL::PostProcessingImpl() {
  return true;
}

}  // namespace boltenkov_s_gaussian_kernel
