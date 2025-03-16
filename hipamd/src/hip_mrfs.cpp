#include "hip_mrfs.hpp"
#include <iostream>
#include <hip/hip_runtime.h>
#include <thread>
#include <mutex>
#include <vector>
#include <map>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include "thread/thread.hpp"

#define MRFS_DEBUG 0

#if MRFS_DEBUG
#define MRFS_LOG(tag, message)                                                                     \
  do {                                                                                             \
    std::cerr << "[" << tag << "] " << message << std::endl;                                       \
  } while (0)
#else
#define MRFS_LOG(tag, message)                                                                     \
  do {                                                                                             \
  } while (0)
#endif

namespace hip {

// Forward declarations
class Device;
class Stream;
extern hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind);
extern Device* getCurrentDevice();

// Process quota management
class QuotaManager {
  private:
  std::map<int, size_t> _processQuotas;
  std::map<int, size_t> _processUsage;
  std::mutex _mutex;
  std::thread _resetThread;
  std::atomic<bool> _stopResetThread{false};

  void quotaResetThreadFunction() {
    MRFS_LOG("QUOTA", "Quota reset thread started");
    while (!_stopResetThread) {
      // Sleep for 1 second
      std::this_thread::sleep_for(std::chrono::seconds(1));

      // Reset all process usage
      std::lock_guard<std::mutex> lock(_mutex);
      for (auto& usage : _processUsage) {
        size_t oldUsage = usage.second;
        usage.second = 0;
        MRFS_LOG("QUOTA",
                 "Reset quota usage for process " << usage.first << " from " << oldUsage
                                                  << " to 0 bytes");
      }
    }
  }

  QuotaManager() {
    // Start the reset thread
    _resetThread = std::thread(&QuotaManager::quotaResetThreadFunction, this);
    MRFS_LOG("QUOTA", "QuotaManager initialized with reset thread");
  }

  // Destructor to clean up thread
  ~QuotaManager() {
    _stopResetThread = true;
    if (_resetThread.joinable()) {
      _resetThread.join();
    }
  }

  public:
  static QuotaManager& instance() {
    static QuotaManager manager;
    return manager;
  }

  // Set quota for a process
  void setQuota(int processId, size_t quotaBytes) {
    std::lock_guard<std::mutex> lock(_mutex);
    _processQuotas[processId] = quotaBytes;
    MRFS_LOG("QUOTA", "Set quota for process " << processId << " to " << quotaBytes << " bytes");
  }

  // Check if operation is within quota
  bool checkQuota(int processId, size_t sizeBytes) {
    std::lock_guard<std::mutex> lock(_mutex);

    // If no quota is set, allow the operation
    if (_processQuotas.find(processId) == _processQuotas.end()) {
      return true;
    }

    // Initialize usage if not present - ie first time process requesting bandwidth
    if (_processUsage.find(processId) == _processUsage.end()) {
      _processUsage[processId] = 0;
    }

    // Check if operation would exceed quota
    if (_processUsage[processId] + sizeBytes <= _processQuotas[processId]) {
      _processUsage[processId] += sizeBytes;
      MRFS_LOG("QUOTA",
               "Process " << processId << " using " << sizeBytes << " bytes, total usage: "
                          << _processUsage[processId] << "/" << _processQuotas[processId]);
      return true;
    }

    MRFS_LOG("QUOTA",
             "Process " << processId << " quota exceeded: " << _processUsage[processId] << " + "
                        << sizeBytes << " > " << _processQuotas[processId]);
    return false;
  }

  // Release quota when task completes
  void releaseQuota(int processId, size_t sizeBytes) {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_processUsage.find(processId) != _processUsage.end()) {
      if (_processUsage[processId] >= sizeBytes) {
        _processUsage[processId] -= sizeBytes;
      } else {
        _processUsage[processId] = 0;
      }
      MRFS_LOG("QUOTA",
               "Process " << processId << " released " << sizeBytes
                          << " bytes, remaining usage: " << _processUsage[processId]);
    }
  }
};

// Struct used to keep track of memory copy operations
struct MemcpyTask {
  void* dst;
  const void* src;
  size_t sizeBytes;
  hipMemcpyKind kind;
  hip::Stream* stream;
  int deviceId;
  int processId;
  std::atomic<bool> completed{false};
};

// Thread pool for paralell memory operations
class MemcpyThreadPool {
  private:
  std::vector<std::thread> _workers;
  std::vector<std::shared_ptr<MemcpyTask>> _tasks;
  std::map<hip::Stream*, std::vector<std::shared_ptr<MemcpyTask>>> _streamTasks;
  std::mutex _mutex;
  std::condition_variable _condition;
  std::atomic<bool> _stop{false};
  std::atomic<int> _taskCount{0};

  void workerFunction() {
    amd::Thread::init();
    MRFS_LOG("WORKER", "Thread started");

    while (!_stop) {
      std::shared_ptr<MemcpyTask> task;

      // Get a task
      {
        std::unique_lock<std::mutex> lock(_mutex);
        _condition.wait(lock, [this] { return _stop || !_tasks.empty(); });

        if (_stop && _tasks.empty()) break;

        if (!_tasks.empty()) {
          task = _tasks.back();
          _tasks.pop_back();
          MRFS_LOG("WORKER",
                   "Got task: " << task->sizeBytes << " bytes, src=" << task->src
                                << ", dst=" << task->dst);
        }
      }

      // Process the task
      if (task) {
        (void)hipSetDevice(task->deviceId);
        MRFS_LOG("WORKER", "Executing task on device " << task->deviceId);
        (void)ihipMemcpy(task->dst, task->src, task->sizeBytes, task->kind, *(task->stream), true);
        MRFS_LOG("WORKER", "Task completed: " << task->sizeBytes << " bytes");

        // Wait for device to finish task
        hipError_t status = hipStreamSynchronize_spt(reinterpret_cast<hipStream_t>(task->stream));
        if (status != hipSuccess) {
          MRFS_LOG("WORKER", "Stream synchronize failed with error " << status);
        }

        // TODO: uncomment following line if quota should be released after task is completed
        // ^ (double check with Maizi)
        QuotaManager::instance().releaseQuota(task->processId, task->sizeBytes);

        task->completed.store(true);
      }
    }

    MRFS_LOG("WORKER", "Thread exiting");
  }

  public:
  // Initialize with specified number of threads
  MemcpyThreadPool(int numThreads) {
    MRFS_LOG("POOL", "Creating thread pool with " << numThreads << " threads");
    for (int i = 0; i < numThreads; ++i) {
      _workers.emplace_back(&MemcpyThreadPool::workerFunction, this);
    }
  }

  // Clean up
  ~MemcpyThreadPool() {
    MRFS_LOG("POOL", "Shutting down thread pool");
    {
      std::lock_guard<std::mutex> lock(_mutex);
      _stop = true;
    }
    _condition.notify_all();

    for (auto& worker : _workers) {
      if (worker.joinable()) worker.join();
    }
    MRFS_LOG("POOL", "Thread pool shutdown complete");
  }

  // Add a new memory copy task
  bool addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
               hip::Stream* stream, int deviceId, int processId) {
    // Define retry parameters
    // TODO: this should be defined based on 1s / retry delay (ie retry to ensure 1s has passed
    // since last fail)
    const int maxRetries = 20;
    // TODO: may want to change with cond var to speed up delay
    const auto retryDelay = std::chrono::milliseconds(50);
    int retryCount = 0;

    while (retryCount < maxRetries) {
      // Check with scheduler for quota before issuing load
      if (QuotaManager::instance().checkQuota(processId, sizeBytes)) {
        auto task = std::make_shared<MemcpyTask>();
        task->dst = dst;
        task->src = src;
        task->sizeBytes = sizeBytes;
        task->kind = kind;
        task->stream = stream;
        task->deviceId = deviceId;
        task->processId = processId;

        int taskId = ++_taskCount;
        MRFS_LOG("POOL",
                 "Adding task " << taskId << ": " << sizeBytes << " bytes, device=" << deviceId
                                << ", stream=" << stream << ", process=" << processId);

        {
          std::lock_guard<std::mutex> lock(_mutex);
          _tasks.push_back(task);
          _streamTasks[stream].push_back(task);
        }

        _condition.notify_one();
        return true;
      }

      // Quota exceeded, log and wait before retry
      retryCount++;
      MRFS_LOG("POOL",
               "Process " << processId << " quota exceeded, retry " << retryCount << "/"
                          << maxRetries << ", waiting " << retryDelay.count() << "ms");

      std::this_thread::sleep_for(retryDelay);
    }

    // If we've exhausted all retries --> then log and return failure
    MRFS_LOG("POOL",
             "Task rejected: quota exceeded for process " << processId << " after " << maxRetries
                                                          << " retries");
    return false;
  }

  // Wait for stream tasks to complete
  void waitForStream(hip::Stream* stream) {
    MRFS_LOG("SYNC", "Waiting for stream " << stream);
    std::vector<std::shared_ptr<MemcpyTask>> streamTasks;

    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (_streamTasks.find(stream) != _streamTasks.end()) {
        streamTasks = _streamTasks[stream];
        MRFS_LOG("SYNC", "Found " << streamTasks.size() << " tasks for stream " << stream);
      } else {
        MRFS_LOG("SYNC", "No tasks found for stream " << stream);
      }
    }

    for (auto& task : streamTasks) {
      while (!task->completed.load()) {
        // MRFS_LOG("SYNC", "Waiting for task to complete: " << task->sizeBytes << " bytes");
        std::this_thread::yield();
      }
    }

    // Clean up completed tasks
    {
      std::lock_guard<std::mutex> lock(_mutex);
      if (_streamTasks.find(stream) != _streamTasks.end()) {
        MRFS_LOG(
            "SYNC",
            "Clearing " << _streamTasks[stream].size() << " completed tasks for stream " << stream);
        _streamTasks[stream].clear();
      }
    }

    MRFS_LOG("SYNC", "Stream " << stream << " sync complete");
  }

  // Wait for all tasks to complete
  void waitForAll() {
    MRFS_LOG("SYNC", "Waiting for all tasks to complete");
    std::vector<std::shared_ptr<MemcpyTask>> allTasks;

    {
      std::lock_guard<std::mutex> lock(_mutex);
      // Get all tasks from all streams
      for (auto& pair : _streamTasks) {
        allTasks.insert(allTasks.end(), pair.second.begin(), pair.second.end());
      }
      // Also include any tasks not yet assigned to a worker
      allTasks.insert(allTasks.end(), _tasks.begin(), _tasks.end());

      MRFS_LOG("SYNC", "Found " << allTasks.size() << " total tasks across all streams");
    }

    // Wait for all tasks to complete
    for (auto& task : allTasks) {
      while (!task->completed.load()) {
        // MRFS_LOG("SYNC",
        //          "Waiting for task to complete: " << task->sizeBytes
        //                                           << " bytes, stream=" << task->stream);
        std::this_thread::yield();
      }
    }

    // Clean up
    {
      std::lock_guard<std::mutex> lock(_mutex);
      MRFS_LOG("SYNC", "Clearing all completed tasks");
      _streamTasks.clear();
    }

    MRFS_LOG("SYNC", "All tasks complete");
  }
};

// Singleton to manage the thread pool
class ThreadManager {
  private:
  std::unique_ptr<MemcpyThreadPool> _pool;

  ThreadManager() {
    int numDevices = 1;
    (void)hipGetDeviceCount(&numDevices);
    MRFS_LOG("MANAGER", "Initializing thread manager with " << numDevices << " devices");
    _pool = std::make_unique<MemcpyThreadPool>(numDevices * 2);
  }

  public:
  static ThreadManager& instance() {
    static ThreadManager manager;
    return manager;
  }

  MemcpyThreadPool* pool() { return _pool.get(); }
};


hipError_t hipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hip::Stream& stream) {
  MRFS_LOG("HIPAPI",
           "hipMemcpyAsync_mrfs: " << sizeBytes << " bytes, src=" << src << ", dst=" << dst);

  if (sizeBytes == 0) {
    MRFS_LOG("HIPAPI", "Empty copy, returning success");
    return hipSuccess;
  }

  hipError_t status = ihipMemcpy_validate(dst, src, sizeBytes, kind);
  if (status != hipSuccess) {
    MRFS_LOG("HIPAPI", "Validation failed with error " << status);
    return status;
  }

  if (src == dst && kind == hipMemcpyDefault) {
    MRFS_LOG("HIPAPI", "Same src/dst with default kind, skipping");
    return hipSuccess;
  }

  // Get current process ID
  int processId = getpid();

  size_t quotaSize = 1ULL * 1024 * 1024 * 1024;  // Set initial quota to some arbitrary bandwidth

  QuotaManager::instance().setQuota(processId, quotaSize);

  // Add to thread pool with quota check and retry logic
  MRFS_LOG("HIPAPI", "Submitting to thread pool for process " << processId);
  bool accepted = ThreadManager::instance().pool()->addTask(
      dst, src, sizeBytes, kind, &stream, getCurrentDevice()->deviceId(), processId);

  if (!accepted) {
    MRFS_LOG("HIPAPI", "Task rejected due to quota constraints after all retries");
    return hipErrorOutOfMemory;  // Or another appropriate error code
  }

  return hipSuccess;
}

// Device synchronization
void interceptedHipDeviceSynchronize() {
  MRFS_LOG("HIPAPI", "interceptedHipDeviceSynchronize called");
  // Wait for all memory operations to complete
  ThreadManager::instance().pool()->waitForAll();

  // Call the original sync function
  constexpr bool kDoWaitForCpu = false;
  hip::getCurrentDevice()->SyncAllStreams(kDoWaitForCpu);
}

// Stream synchronization
hipError_t interceptedHipStreamSynchronize(hipStream_t streamHandle) {
  MRFS_LOG("HIPAPI", "interceptedHipStreamSynchronize called for stream " << streamHandle);

  hip::Stream* stream = reinterpret_cast<hip::Stream*>(streamHandle);
  // Wait for stream operations to complete
  ThreadManager::instance().pool()->waitForStream(stream);

  // // Call the original stream sync function
  // HIP_RETURN(hipStreamSynchronize_common(stream));
  return hipSuccess;
}

}  // namespace hip