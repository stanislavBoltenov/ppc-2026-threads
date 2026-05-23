#include "boltenkov_s_gaussian_kernel/all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstddef>
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
  GetOutput().clear();
}

bool BoltenkovSGaussianKernelALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0) {
    return true;
  }

  const auto &[n, m, data] = GetInput();
  if (n == 0 || m == 0) {
    return false;
  }
  if (n > static_cast<std::size_t>(INT_MAX) || m > static_cast<std::size_t>(INT_MAX)) {
    return false;
  }
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

bool BoltenkovSGaussianKernelALL::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::size_t n = 0, m = 0;
  if (rank == 0) {
    n = std::get<0>(GetInput());
    m = std::get<1>(GetInput());
    if (n == 0 || m == 0) {
      return false;
    }
    if (n > static_cast<std::size_t>(INT_MAX) || m > static_cast<std::size_t>(INT_MAX)) {
      return false;
    }
  }

  int n_int = static_cast<int>(n);
  int m_int = static_cast<int>(m);
  MPI_Bcast(&n_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m_int, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n_int == 0 || m_int == 0) {
    return false;
  }

  auto &out = GetOutput();
  out.resize(static_cast<std::size_t>(n_int));
  for (std::size_t i = 0; i < static_cast<std::size_t>(n_int); ++i) {
    out[i].resize(static_cast<std::size_t>(m_int), 0);
  }
  return true;
}

static void scatter_rows(int rank, int size, const std::vector<std::vector<int>> &global_data,
                         std::vector<std::vector<int>> &local_halo, std::size_t &local_start_row,
                         std::size_t &local_rows, std::size_t m) {
  std::size_t n = global_data.size();
  if (n == 0 || m == 0) {
    local_rows = 0;
    return;
  }

  std::size_t base = n / static_cast<std::size_t>(size);
  std::size_t rem = n % static_cast<std::size_t>(size);
  local_rows = base + (static_cast<std::size_t>(rank) < rem ? 1 : 0);
  local_start_row = static_cast<std::size_t>(rank) * base + std::min(static_cast<std::size_t>(rank), rem);

  if (local_rows == 0) {
    return;
  }

  std::size_t halo_first = (local_start_row == 0) ? 0 : local_start_row - 1;
  std::size_t halo_last = (local_start_row + local_rows == n) ? n - 1 : local_start_row + local_rows;
  std::size_t halo_rows = halo_last - halo_first + 1;

  local_halo.resize(halo_rows, std::vector<int>(m, 0));

  if (rank == 0) {
    for (std::size_t i = 0; i < halo_rows; ++i) {
      local_halo[i] = global_data[halo_first + i];
    }

    for (int proc = 1; proc < size; ++proc) {
      std::size_t p_rows = base + (static_cast<std::size_t>(proc) < rem ? 1 : 0);
      if (p_rows == 0) {
        int zero = 0;
        MPI_Send(&zero, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
        continue;
      }

      std::size_t p_start = static_cast<std::size_t>(proc) * base + std::min(static_cast<std::size_t>(proc), rem);
      std::size_t p_halo_first = (p_start == 0) ? 0 : p_start - 1;
      std::size_t p_halo_last = (p_start + p_rows == n) ? n - 1 : p_start + p_rows;
      std::size_t p_halo_rows = p_halo_last - p_halo_first + 1;

      int p_rows_int = static_cast<int>(p_rows);
      int p_start_int = static_cast<int>(p_start);
      int p_halo_rows_int = static_cast<int>(p_halo_rows);
      MPI_Send(&p_rows_int, 1, MPI_INT, proc, 0, MPI_COMM_WORLD);
      MPI_Send(&p_start_int, 1, MPI_INT, proc, 1, MPI_COMM_WORLD);
      MPI_Send(&p_halo_rows_int, 1, MPI_INT, proc, 2, MPI_COMM_WORLD);

      for (std::size_t i = 0; i < p_halo_rows; ++i) {
        const auto &row = global_data[p_halo_first + i];
        MPI_Send(const_cast<int *>(row.data()), static_cast<int>(m), MPI_INT, proc, 10 + static_cast<int>(i),
                 MPI_COMM_WORLD);
      }
    }
  } else {
    int p_rows_int = 0, p_start_int = 0, p_halo_rows_int = 0;
    MPI_Recv(&p_rows_int, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (p_rows_int <= 0) {
      local_rows = 0;
      return;
    }
    MPI_Recv(&p_start_int, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&p_halo_rows_int, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (p_halo_rows_int <= 0) {
      local_rows = 0;
      return;
    }

    local_rows = static_cast<std::size_t>(p_rows_int);
    local_start_row = static_cast<std::size_t>(p_start_int);
    std::size_t halo_rows = static_cast<std::size_t>(p_halo_rows_int);

    local_halo.resize(halo_rows, std::vector<int>(m, 0));
    for (std::size_t i = 0; i < halo_rows; ++i) {
      MPI_Recv(local_halo[i].data(), static_cast<int>(m), MPI_INT, 0, 10 + static_cast<int>(i), MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }
  }
}

static std::vector<std::vector<int>> apply_gaussian_filter(const std::vector<std::vector<int>> &local_halo,
                                                           std::size_t local_start_row, std::size_t local_rows,
                                                           std::size_t m, const int kernel[3][3], int shift) {
  if (local_rows == 0 || m == 0) {
    return {};
  }

  std::vector<std::vector<int>> result(local_rows, std::vector<int>(m, 0));
  std::size_t halo_offset = (local_start_row == 0) ? 0 : 1;

#pragma omp parallel for num_threads(ppc::util::GetNumThreads()) default(none) \
    shared(local_halo, result, local_rows, m, kernel, shift, halo_offset)
  for (std::size_t i = 0; i < local_rows; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      int sum = 0;
      for (int ki = -1; ki <= 1; ++ki) {
        for (int kj = -1; kj <= 1; ++kj) {
          std::size_t row_idx = halo_offset + i + static_cast<std::size_t>(ki);
          if (row_idx >= local_halo.size()) {
            continue;
          }
          std::size_t col_idx = j + static_cast<std::size_t>(kj);
          if (col_idx >= m) {
            continue;
          }
          sum += local_halo[row_idx][col_idx] * kernel[ki + 1][kj + 1];
        }
      }
      result[i][j] = sum >> shift;
    }
  }
  return result;
}

static void gather_results(int rank, int size, const std::vector<std::vector<int>> &local_res,
                           std::size_t local_start_row, std::size_t local_rows, std::size_t m,
                           std::vector<std::vector<int>> &output) {
  if (rank == 0) {
    for (std::size_t i = 0; i < local_rows; ++i) {
      output[local_start_row + i] = local_res[i];
    }

    for (int proc = 1; proc < size; ++proc) {
      int p_rows_int = 0;
      MPI_Recv(&p_rows_int, 1, MPI_INT, proc, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      if (p_rows_int <= 0) {
        continue;
      }
      std::size_t p_rows = static_cast<std::size_t>(p_rows_int);

      int p_start_int = 0;
      MPI_Recv(&p_start_int, 1, MPI_INT, proc, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      std::size_t p_start = static_cast<std::size_t>(p_start_int);

      for (std::size_t i = 0; i < p_rows; ++i) {
        MPI_Recv(output[p_start + i].data(), static_cast<int>(m), MPI_INT, proc, 200 + static_cast<int>(i),
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    }
  } else {
    if (local_rows > 0) {
      int p_rows_int = static_cast<int>(local_rows);
      int p_start_int = static_cast<int>(local_start_row);
      MPI_Send(&p_rows_int, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
      MPI_Send(&p_start_int, 1, MPI_INT, 0, 101, MPI_COMM_WORLD);
      for (std::size_t i = 0; i < local_rows; ++i) {
        MPI_Send(const_cast<int *>(local_res[i].data()), static_cast<int>(m), MPI_INT, 0, 200 + static_cast<int>(i),
                 MPI_COMM_WORLD);
      }
    } else {
      int zero = 0;
      MPI_Send(&zero, 1, MPI_INT, 0, 100, MPI_COMM_WORLD);
    }
  }
}

bool BoltenkovSGaussianKernelALL::RunImpl() {
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::size_t n = GetOutput().size();
  std::size_t m = (n > 0) ? GetOutput()[0].size() : 0;
  if (n == 0 || m == 0) {
    return false;
  }

  std::vector<std::vector<int>> global_data;
  if (rank == 0) {
    global_data = std::get<2>(GetInput());
    if (global_data.size() != n || (n > 0 && global_data[0].size() != m)) {
      return false;
    }
  }

  std::vector<std::vector<int>> local_halo;
  std::size_t local_start_row = 0, local_rows = 0;
  scatter_rows(rank, size, global_data, local_halo, local_start_row, local_rows, m);

  std::vector<std::vector<int>> local_res;
  if (local_rows > 0) {
    local_res = apply_gaussian_filter(local_halo, local_start_row, local_rows, m, kernel_, shift_);
  }

  gather_results(rank, size, local_res, local_start_row, local_rows, m, GetOutput());

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BoltenkovSGaussianKernelALL::PostProcessingImpl() {
  return true;
}

}  // namespace boltenkov_s_gaussian_kernel
