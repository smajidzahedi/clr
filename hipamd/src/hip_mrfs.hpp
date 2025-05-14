#pragma once

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"

namespace hip {

struct MemcpyTask {
  void* dst;
  const void* src;
  size_t sizeBytes;
  hipMemcpyKind kind;
  hipStream_t stream;
  int deviceId;
};

class QuotaManager {
 private:
  static constexpr size_t TIME_WINDOW_MS = 100;
  size_t processQuota = 22.0 * 1024 * 1024 * 1024;
  size_t chunkSize = processQuota * TIME_WINDOW_MS / 1000;
  int deviceCount;
  std::vector<size_t> deviceQuota;
  std::vector<size_t> deviceUsage;
  std::vector<std::vector<std::unique_ptr<MemcpyTask>>> taskQueues;
  std::vector<std::unique_ptr<std::mutex>> taskQueueMutexes;
  std::vector<std::thread> workers;
  void workerFunction(int deviceId);
  void reallocateDeviceQuota();

 protected:
  QuotaManager() {
    hipError_t _ = hipGetDeviceCount(&deviceCount);
    deviceQuota.resize(deviceCount);
    deviceUsage.resize(deviceCount);
    taskQueues.resize(deviceCount);
    taskQueueMutexes.resize(deviceCount);
    workers.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      deviceQuota[i] = 0;
      deviceUsage[i] = 0;
      taskQueues[i].clear();
      taskQueueMutexes[i] = std::make_unique<std::mutex>();
      workers[i] = std::thread(&QuotaManager::workerFunction, this, i);
    }
  }

 public:
  QuotaManager(QuotaManager& other) = delete;
  void operator=(const QuotaManager&) = delete;
  static QuotaManager& GetInstance() {
    static QuotaManager manager;
    return manager;
  }

  void setProcessQuota(double quota);
  void addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream,
               int deviceId);
  void addChunkedTasks(void* dst, const void* src, size_t totalSize, hipMemcpyKind kind,
                       hipStream_t stream, int deviceId);
  void waitForDevice(int deviceId);
  void waitForStream(hipStream_t stream);
  size_t getChunkSize();
};

hipError_t hipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hip::Stream& stream);
void interceptedHipDeviceSynchronize();
hipError_t interceptedHipStreamSynchronize(hipStream_t stream);

}  // namespace hip
