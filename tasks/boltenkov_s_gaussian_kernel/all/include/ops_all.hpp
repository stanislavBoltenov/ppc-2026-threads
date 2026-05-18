#pragma once

#include <vector>

#include "boltenkov_s_gaussian_kernel/common/include/common.hpp"
#include "task/include/task.hpp"

namespace boltenkov_s_gaussian_kernel {

class BoltenkovSGaussianKernelALL : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kALL;
  }

  explicit BoltenkovSGaussianKernelALL(const InType& in);

 private:
  std::vector<std::vector<int>> kernel_;
  int shift_ = 4;

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  void BcastSizes(int& n, int& m, int rank);
  void ScatterRows(const std::vector<std::vector<int>>& global_data, std::vector<std::vector<int>>& local_halo,
                   int& local_start_row, int& local_rows, int m, int rank, int size);
  void GatherResults(const std::vector<std::vector<int>>& local_res, int local_start_row, int local_rows, int m,
                     int rank, int size);
};

}  // namespace boltenkov_s_gaussian_kernel
