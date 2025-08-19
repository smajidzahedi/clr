#pragma once
#include <condition_variable>
#include <hip/hip_runtime.h>
#include <chrono>
#include <thread>
#include "hip_internal.hpp"
#include <numa.h>
#include <numaif.h>
#include <atomic>

namespace hip {
#define MRFS_MIN_CHUNK_SIZE 4096
#define MRFS_M_CHANCE 4
#define MRFS_TASK_QUEUE_SIZE (128)  // TODO: Fine tune this!
#define MRFS_REALLOCATE_PERIOD_NS (10000000) // Every 10 ms

static uint64_t getCurrentTime() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  uint64_t t = ts.tv_sec * 1000 * 1000 * 1000 + ts.tv_nsec;
  return t;
}

struct MemcpyTask {
  void* dst;
  const void* src;
  size_t sizeBytes;
  hipMemcpyKind kind;
  hipStream_t stream;
  int deviceId;
  int numaId;
};

class TaskBuffer {
 private:
  std::unique_ptr<MemcpyTask> buffer_[MRFS_TASK_QUEUE_SIZE];

  mutable std::mutex mutex_;
  std::condition_variable on_empty_;
  std::condition_variable on_not_empty_;

  int head_;
  int tail_;
  int count_;
  int *count_per_numa_;
  int capacity_;

 public:

  TaskBuffer(int numaCount): head_(0), tail_(0), count_(0), capacity_(MRFS_TASK_QUEUE_SIZE) {
    count_per_numa_ = new int[numaCount];
    for (int i = 0; i < numaCount; i++) {
      count_per_numa_[i] = 0;
    }
  }

  ~TaskBuffer(){
    delete [] count_per_numa_;
  }

  bool push_back(std::unique_ptr<MemcpyTask> task) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ == capacity_)
      return false;

    ++count_;
    ++count_per_numa_[task->numaId];

    buffer_[tail_] = std::move(task);
    tail_ = (tail_ + 1) % capacity_;
    on_empty_.notify_one();
    return true;
  }

  void pop_front(int numaId) {
    std::unique_lock<std::mutex> lock(mutex_);
    on_empty_.wait(lock, [this]() { return count_ > 0; });
    head_ = (head_ + 1) % capacity_;
    --count_;
    --count_per_numa_[numaId];
    if (count_ == 0) {
      on_not_empty_.notify_all();
    }
  }

  std::unique_ptr<MemcpyTask> get_front() {
    std::unique_lock<std::mutex> lock(mutex_);
    on_empty_.wait(lock, [this]() { return count_ > 0; });
    std::unique_ptr<MemcpyTask> task = std::move(buffer_[head_]);
    buffer_[head_] = nullptr;
    return task;
  }

  int size(int numaId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_per_numa_[numaId];
  }

  int size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return count_;
  }

  void wait_till_empty() {
    std::unique_lock<std::mutex> lock(mutex_);
    on_not_empty_.wait(lock, [this]() { return count_ == 0; });
  }
};

class QuotaManager {
 private:
  int deviceCount;
  int numaCount;
  std::atomic<size_t> *processNumaQuotaBPS;
  std::atomic<int> *numaActiveDevices;
  TaskBuffer **buffers;

  std::vector<std::thread> workers;
  void workerFunction(int deviceId);

 protected:
  QuotaManager() {
    hipError_t _ = hipGetDeviceCount(&deviceCount);

    if (numa_available() < 0) {
      numaCount = 1;
    } else {
      numaCount = numa_max_node() + 1;
    }

    processNumaQuotaBPS = new std::atomic<size_t>[numaCount];
    numaActiveDevices = new std::atomic<int>[numaCount];

    for (int i = 0; i < numaCount; i++) {
      processNumaQuotaBPS[i].store(deviceCount * 8ULL * 1024 * 1024 * 1024);
      numaActiveDevices[i].store(0);
    }

    buffers = new TaskBuffer*[deviceCount];
    workers.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      buffers[i] = new TaskBuffer(numaCount);
      workers[i] = std::thread(&QuotaManager::workerFunction, this, i);
    }
  }

  ~QuotaManager() {
    delete [] processNumaQuotaBPS;
    delete [] numaActiveDevices;
    for (int i = 0; i < deviceCount; i++) {
      delete buffers[i];
    }
    delete [] buffers;
  }

 public:
  QuotaManager(QuotaManager& other) = delete;
  void operator=(const QuotaManager&) = delete;
  static QuotaManager& getInstance() {
    static QuotaManager instance;
    return instance;
  }

  void setProcessQuota(size_t quota); // Sets the same quota for all numa nodes
  void setProcessNumaQuota(size_t quota, int numaId);
  bool addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream,
               int deviceId);
  void waitForDevice(int deviceId);
};

hipError_t ihipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hipStream_t stream);
void ihipDeviceSynchronize_mrfs();
hipError_t ihipStreamSynchronize_mrfs(hipStream_t stream);
hipError_t ihipSetProcessQuota_mrfs(size_t quota);

}  // namespace hip
