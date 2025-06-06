#pragma once
#include <condition_variable>
#include <hip/hip_runtime.h>
#include <chrono>
#include <thread>
#include "hip_internal.hpp"

namespace hip {

#define TASK_QUEUE_SIZE (1024 * 4)  // TODO: Fine tune this!
#define TIME_WINDOW_MS (100)
#define TIME_WINDOW_NS (TIME_WINDOW_MS * 1000 * 1000)
#define REALLOCATE_PERIOD_MS (5 * TIME_WINDOW_MS)

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
};

class TaskBuffer {
 private:
  std::unique_ptr<MemcpyTask> buffer_[TASK_QUEUE_SIZE];

  // TODO: check compiler. For C++11 we don't need to initialize these!
  mutable std::mutex mutex_;
  std::condition_variable on_empty_;
  std::condition_variable on_not_empty_;

  int head_;
  int tail_;
  int count_;
  int capacity_;

 public:

  TaskBuffer(): head_(0), tail_(0), count_(0), capacity_(TASK_QUEUE_SIZE) {}

  bool push_back(std::vector<std::unique_ptr<MemcpyTask>> tasks) {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t num_tasks = tasks.size();

    if (count_ + num_tasks > capacity_)
      return false;

    for (size_t i = 0; i < num_tasks; ++i) {
      buffer_[tail_] = std::move(tasks[i]);
      tail_ = (tail_ + 1) % capacity_;
      ++count_;
    }

    on_empty_.notify_all();
    return true;
  }

  std::unique_ptr<MemcpyTask> pop_front() {
    std::unique_lock<std::mutex> lock(mutex_);
    on_empty_.wait(lock, [this]() { return count_ > 0; });
    std::unique_ptr<MemcpyTask> task = std::move(buffer_[head_]);
    buffer_[head_] = nullptr;
    head_ = (head_ + 1) % capacity_;
    --count_;
    if (count_ == 0)
      on_not_empty_.notify_all();
    return task;
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
  std::atomic<size_t> processQuotaBPS;
  std::atomic<size_t> chunkSize;
  int deviceCount;

  std::vector<std::unique_ptr<std::atomic<int>>> deviceQuota;
  std::vector<int> deviceUsage;
  std::vector<uint64_t> lastUpdate;
  std::vector<std::unique_ptr<TaskBuffer>> buffers;

  std::vector<std::thread> workers;
  std::thread quotaManager;
  void workerFunction(int deviceId);
  void reallocateDeviceQuota();

  static QuotaManager* instance;

 protected:
  QuotaManager() {
    hipError_t _ = hipGetDeviceCount(&deviceCount);
    processQuotaBPS = 22.0 * 1024 * 1024 * 1024;
    chunkSize = (processQuotaBPS * TIME_WINDOW_MS) / 1000; //TODO figure out why CLion is giving warning!

    deviceQuota.reserve(deviceCount);
    deviceUsage.resize(deviceCount);
    lastUpdate.resize(deviceCount);
    buffers.reserve(deviceCount);


    workers.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      deviceQuota.emplace_back(std::make_unique<std::atomic<int>>());
      deviceUsage[i] = 0;
      lastUpdate[i] = 0;
      buffers.emplace_back(std::make_unique<TaskBuffer>());
      workers[i] = std::thread(&QuotaManager::workerFunction, this, i);
      quotaManager = std::thread(&QuotaManager::reallocateDeviceQuota, this);
    }
  }

 public:
  QuotaManager(QuotaManager& other) = delete;
  void operator=(const QuotaManager&) = delete;
  static QuotaManager* GetInstance() {
    if (instance == nullptr)
      instance = new QuotaManager;
    return instance;
  }

  void setProcessQuota(double quota);
  void addChunkedTasks(void* dst, const void* src, size_t totalSize, hipMemcpyKind kind,
                       hipStream_t stream, int deviceId);
  void waitForDevice(int deviceId);
  void enforceBW(uint64_t, size_t, int);
};

hipError_t hipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hip::Stream& stream);
void interceptedHipDeviceSynchronize();
hipError_t interceptedHipStreamSynchronize(hipStream_t stream);

}  // namespace hip
