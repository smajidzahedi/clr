#include "hip_mrfs.hpp"
#include <iostream>
#include <hip/hip_runtime.h>
#include <thread>
#include <mutex>
#include <vector>
#include <map>
#include <atomic>
#include <condition_variable>
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
extern hipError_t ihipDeviceSynchronize();
extern hipError_t ihipStreamSynchronize(hipStream_t stream);
extern hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind);
// extern hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes,
//                            hipMemcpyKind kind, hip::Stream& stream, bool isAsync);
extern Device* getCurrentDevice();

// Struct used to keep track of memory copy operations
struct MemcpyTask {
  void* dst;
  const void* src;
  size_t sizeBytes;
  hipMemcpyKind kind;
  hip::Stream* stream;
  int deviceId;
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
  void addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
               hip::Stream* stream, int deviceId) {
    auto task = std::make_shared<MemcpyTask>();
    task->dst = dst;
    task->src = src;
    task->sizeBytes = sizeBytes;
    task->kind = kind;
    task->stream = stream;
    task->deviceId = deviceId;

    int taskId = ++_taskCount;
    MRFS_LOG("POOL",
             "Adding task " << taskId << ": " << sizeBytes << " bytes, device=" << deviceId
                            << ", stream=" << stream);

    {
      std::lock_guard<std::mutex> lock(_mutex);
      _tasks.push_back(task);
      _streamTasks[stream].push_back(task);
    }

    _condition.notify_one();
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
        MRFS_LOG("SYNC", "Waiting for task to complete: " << task->sizeBytes << " bytes");
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
        MRFS_LOG("SYNC",
                 "Waiting for task to complete: " << task->sizeBytes
                                                  << " bytes, stream=" << task->stream);
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

  // Add to thread pool
  MRFS_LOG("HIPAPI", "Submitting to thread pool");
  ThreadManager::instance().pool()->addTask(dst, src, sizeBytes, kind, &stream,
                                            getCurrentDevice()->deviceId());

  return hipSuccess;
}

// Device synchronization
inline hipError_t interceptedHipDeviceSynchronize() {
  MRFS_LOG("HIPAPI", "interceptedHipDeviceSynchronize called");
  // Wait for all memory operations to complete
  ThreadManager::instance().pool()->waitForAll();

  MRFS_LOG("HIPAPI", "Calling original ihipDeviceSynchronize");
  hipError_t result = ihipDeviceSynchronize();

  MRFS_LOG("HIPAPI", "ihipDeviceSynchronize returned " << result);
  return result;
}

// Stream synchronization
inline hipError_t interceptedHipStreamSynchronize(hipStream_t streamHandle) {
  MRFS_LOG("HIPAPI", "interceptedHipStreamSynchronize called for stream " << streamHandle);

  hip::Stream* stream = reinterpret_cast<hip::Stream*>(streamHandle);
  // Wait for stream operations to complete
  ThreadManager::instance().pool()->waitForStream(stream);

  MRFS_LOG("HIPAPI", "Calling original ihipStreamSynchronize");
  hipError_t result = ihipStreamSynchronize(streamHandle);

  MRFS_LOG("HIPAPI", "ihipStreamSynchronize returned " << result);
  return result;
}

}  // namespace hip
