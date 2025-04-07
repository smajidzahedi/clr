#include "hip_mrfs.hpp"
#include <iostream>
#include <hip/hip_runtime.h>
#include <thread>
#include <mutex>
#include <vector>
#include <map>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <unistd.h>
#include "thread/thread.hpp"
#include <time.h>

#define MRFS_DEBUG 0

#if MRFS_DEBUG
#define MRFS_LOG(tag, message)                                                                     \
  do {                                                                                             \
    std::cout << "[" << tag << "] " << message << std::endl;                                       \
  } while (0)
#else
#define MRFS_LOG(tag, message)                                                                     \
  do {                                                                                             \
  } while (0)
#endif

namespace hip {

// Default to 1GB chunk size
constexpr size_t DEFAULT_MEMCPY_CHUNK_SIZE = 128ULL * 1024 * 1024;
static size_t gMemcpyChunkSize = DEFAULT_MEMCPY_CHUNK_SIZE;

// Default process bandwidth quota (10 GB/s)
constexpr double DEFAULT_PROCESS_BANDWIDTH_QUOTA = 20.0 * 1024 * 1024 * 1024;

size_t getMemcpyChunkSize() { return gMemcpyChunkSize; }

void setMemcpyChunkSize(size_t newChunkSize) {
  if (newChunkSize > 0) {
    gMemcpyChunkSize = newChunkSize;
  }
}

// Forward declarations
class Device;
class Stream;
extern hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind);
extern Device* getCurrentDevice();

struct QueueId {
  int deviceId;
  pid_t processId;

  // For using QueueId as a map key
  bool operator<(const QueueId& other) const {
    if (deviceId != other.deviceId) return deviceId < other.deviceId;
    return processId < other.processId;
  }

  bool operator==(const QueueId& other) const {
    return deviceId == other.deviceId && processId == other.processId;
  }
};

// Struct used to keep track of memory copy operations
struct MemcpyTask {
  void* dst;
  const void* src;
  size_t sizeBytes;
  hipMemcpyKind kind;
  hip::Stream* stream;
  QueueId queueId;
  std::atomic<bool> completed{false};
};

// quota management
class QuotaManager {
  private:
  // Store bandwidth quotas in bytes per second for each (device, process) queue
  std::map<QueueId, double> _queueQuotas;
  std::map<QueueId, double> _queueUsage;
  std::mutex _mutex;
  std::thread _resetThread;
  std::atomic<bool> _stopResetThread{false};
  double _totalProcessQuota;
  std::map<pid_t, double> _processQuotas;
  std::map<pid_t, double> _processUsage;  // Track usage at process level

  void quotaResetThreadFunction() {
    MRFS_LOG("QUOTA", "Quota reset thread started");
    while (!_stopResetThread) {
      // Sleep for 1 second
      struct timespec ts;
      ts.tv_sec = 1;  // 1 second
      ts.tv_nsec = 0;

      nanosleep(&ts, NULL);

      // Reset all queue usage
      std::lock_guard<std::mutex> lock(_mutex);
      for (auto& usage : _queueUsage) {
        double oldUsage = usage.second;
        usage.second = 0.0;
        MRFS_LOG("QUOTA",
                 "Reset bandwidth for queue (device=" << usage.first.deviceId
                                                      << ", process=" << usage.first.processId
                                                      << ") from " << oldUsage << " B/s to 0 B/s");
      }

      // Reset process usage too
      for (auto& usage : _processUsage) {
        double oldUsage = usage.second;
        usage.second = 0.0;
        MRFS_LOG("QUOTA",
                 "Reset bandwidth for process " << usage.first << " from " << oldUsage
                                                << " B/s to 0 B/s");
      }
    }
  }

  QuotaManager() : _totalProcessQuota(DEFAULT_PROCESS_BANDWIDTH_QUOTA) {
    // Start the reset thread
    _resetThread = std::thread(&QuotaManager::quotaResetThreadFunction, this);
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

  // Set total quota for a process (in bytes per second)
  void setProcessQuota(pid_t processId, double bandwidthBytesPerSec) {
    std::lock_guard<std::mutex> lock(_mutex);
    _processQuotas[processId] = bandwidthBytesPerSec;
    _processUsage[processId] = 0.0;  // Initialize process usage

    // Reallocate quotas for all queues associated with this process
    reallocateQuotas(processId);

    MRFS_LOG("QUOTA",
             "Set total bandwidth quota for process " << processId << " to " << bandwidthBytesPerSec
                                                      << " B/s");
  }

  void reallocateQuotas(pid_t processId) {
    // Find all queues for this process
    std::vector<QueueId> processQueues;
    for (const auto& quota : _queueQuotas) {
      if (quota.first.processId == processId) {
        processQueues.push_back(quota.first);
      }
    }

    if (processQueues.empty()) {
      return;
    }

    // Calculate total process quota
    double processQuota = _processQuotas[processId];

    size_t totalChunks = static_cast<size_t>(processQuota / DEFAULT_MEMCPY_CHUNK_SIZE);
    size_t chunksPerQueue = totalChunks / processQueues.size();
    size_t remainingChunks = totalChunks % processQueues.size();

    // Allocate quota to each queue
    for (size_t i = 0; i < processQueues.size(); i++) {
      const auto& queueId = processQueues[i];

      double queueQuota = chunksPerQueue * DEFAULT_MEMCPY_CHUNK_SIZE;

      // Last queue gets any remaining chunks
      if (i == processQueues.size() - 1) {
        queueQuota += remainingChunks * DEFAULT_MEMCPY_CHUNK_SIZE;

        double remainingBytes = processQuota - (totalChunks * DEFAULT_MEMCPY_CHUNK_SIZE);
        queueQuota += remainingBytes;
      }

      _queueQuotas[queueId] = queueQuota;
      MRFS_LOG("QUOTA",
               "Allocated " << queueQuota << " B/s to queue (device=" << queueId.deviceId
                            << ", process=" << queueId.processId << "), " << (i + 1) << " of "
                            << processQueues.size() << " queues");
    }
  }

  void registerQueue(const QueueId& queueId) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (_queueQuotas.find(queueId) != _queueQuotas.end()) {
      return;
    }
    _queueUsage[queueId] = 0.0;

    // Get process quota
    if (_processQuotas.find(queueId.processId) == _processQuotas.end()) {
      _processQuotas[queueId.processId] = _totalProcessQuota;
      _processUsage[queueId.processId] = 0.0;  // Initialize process usage
      MRFS_LOG("QUOTA",
               "Set default process quota for process " << queueId.processId << " to "
                                                        << _totalProcessQuota << " B/s");
    }

    // Count how many queues exist for this process now including this one
    int queueCount = 0;
    for (const auto& quota : _queueQuotas) {
      if (quota.first.processId == queueId.processId) {
        queueCount++;
      }
    }
    queueCount++;

    // Split the quota evenly here
    double processQuota = _processQuotas[queueId.processId];
    double queueQuota = processQuota / queueCount;


    _queueQuotas[queueId] = queueQuota;
    MRFS_LOG("QUOTA",
             "Allocated " << queueQuota << " B/s to queue (device=" << queueId.deviceId
                          << ", process=" << queueId.processId << "), 1 of " << queueCount
                          << " queues");

    // Reallocate quotas for all other queues in this process
    for (auto& quota : _queueQuotas) {
      if (quota.first.processId == queueId.processId && !(quota.first == queueId)) {
        quota.second = queueQuota;
        MRFS_LOG("QUOTA",
                 "Reallocated " << queueQuota << " B/s to queue (device=" << quota.first.deviceId
                                << ", process=" << quota.first.processId << ")");
      }
    }
  }

  // Check if operation is within quota for a queue
  bool checkQuota(const QueueId& queueId, size_t sizeBytes) {
    std::lock_guard<std::mutex> lock(_mutex);

    // Register queue if not already registered
    if (_queueQuotas.find(queueId) == _queueQuotas.end()) {
      registerQueue(queueId);
    }

    // Initialize usage if not present - i.e., first time queue requesting bandwidth
    if (_queueUsage.find(queueId) == _queueUsage.end()) {
      _queueUsage[queueId] = 0.0;
    }

    pid_t processId = queueId.processId;
    if (_processUsage.find(processId) == _processUsage.end()) {
      _processUsage[processId] = 0.0;
    }

    if (_processUsage[processId] + sizeBytes <= _processQuotas[processId]) {
      MRFS_LOG("QUOTA",
               "Queue (device=" << queueId.deviceId << ", process=" << queueId.processId
                                << ") pre-check passed for " << sizeBytes
                                << " bytes, process usage: " << _processUsage[processId] << "/"
                                << _processQuotas[processId] << " B/s");
      return true;
    }

    MRFS_LOG("QUOTA",
             "Queue (device=" << queueId.deviceId << ", process=" << queueId.processId
                              << ") process quota exceeded: " << _processUsage[processId] << " + "
                              << sizeBytes << " > " << _processQuotas[processId] << " B/s");
    return false;
  }


  void updateUsage(const QueueId& queueId, size_t sizeBytes, double executionTimeSec) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (executionTimeSec > 0) {
      double actualBandwidth = static_cast<double>(sizeBytes) / executionTimeSec;

      // Initialize queue usage if needed
      if (_queueUsage.find(queueId) == _queueUsage.end()) {
        _queueUsage[queueId] = 0.0;
      }

      // Initialize process usage if needed
      pid_t processId = queueId.processId;
      if (_processUsage.find(processId) == _processUsage.end()) {
        _processUsage[processId] = 0.0;
      }

      // Update both queue and process usage
      _queueUsage[queueId] += sizeBytes;
      _processUsage[processId] += sizeBytes;

      MRFS_LOG("QUOTA",
               "Queue (device=" << queueId.deviceId << ", process=" << queueId.processId
                                << ") used " << actualBandwidth << " B/s (" << sizeBytes
                                << " bytes in " << executionTimeSec
                                << "s), process usage: " << _processUsage[processId] << "/"
                                << _processQuotas[processId] << " B/s");

      // Add rate limiting based on process quota, not queue quota
      double expectedTime = static_cast<double>(sizeBytes) / _processQuotas[processId];
      if (executionTimeSec < expectedTime) {
        double sleepTime = expectedTime - executionTimeSec;
        MRFS_LOG("QUOTA",
                 "Limiting rate by sleeping for " << sleepTime << "s to match process quota");
        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
      }
    }
  }
};

// Thread pool
class MemcpyThreadPool {
  private:
  // Queue per (device, process) pair
  std::map<QueueId, std::queue<std::shared_ptr<MemcpyTask>>> _queues;

  // One dedicated worker thread per queue
  std::map<QueueId, std::thread> _queueWorkers;

  // Track tasks by stream for synchronization
  std::map<hip::Stream*, std::vector<std::shared_ptr<MemcpyTask>>> _streamTasks;

  // Use pointers for condition_variables since they cannot be copied or moved
  std::map<QueueId, std::unique_ptr<std::condition_variable>> _queueConditions;
  std::map<QueueId, std::atomic<bool>> _queueStopFlags;

  std::mutex _queuesMutex;
  std::mutex _streamsMutex;

  std::vector<int> _deviceIds;

  void createQueueWorker(const QueueId& queueId) {
    _queueWorkers[queueId] = std::thread(&MemcpyThreadPool::queueWorkerFunction, this, queueId);
  }

  void queueWorkerFunction(const QueueId& queueId) {
    amd::Thread::init();

    while (!_queueStopFlags[queueId]) {
      std::shared_ptr<MemcpyTask> task;

      // Get a task from the queue
      {
        std::unique_lock<std::mutex> lock(_queuesMutex);

        auto& condition = *_queueConditions[queueId];
        auto& queue = _queues[queueId];

        // Wait until there's work or we're stopping
        condition.wait(
            lock, [this, &queue, &queueId] { return _queueStopFlags[queueId] || !queue.empty(); });

        if (_queueStopFlags[queueId] && queue.empty()) {
          break;  // No more tasks and stopping
        }

        if (!queue.empty()) {
          task = queue.front();
          queue.pop();
        }
      }

      // Process the task
      if (task) {
        (void)hipSetDevice(task->queueId.deviceId);

        struct timespec startTime;
        clock_gettime(CLOCK_MONOTONIC, &startTime);

        (void)ihipMemcpy(task->dst, task->src, task->sizeBytes, task->kind, *(task->stream), true);

        // Wait for device to finish task
        hipError_t status = hipStreamSynchronize_spt(reinterpret_cast<hipStream_t>(task->stream));

        struct timespec endTime;
        clock_gettime(CLOCK_MONOTONIC, &endTime);

        // Convert to seconds
        double executionTimeSec =
            (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1e9;

        MRFS_LOG("WORKER",
                 "Task completed: " << task->sizeBytes << " bytes in " << executionTimeSec
                                    << " seconds");

        // Update quota usage with actual measured bandwidth
        QuotaManager::instance().updateUsage(task->queueId, task->sizeBytes, executionTimeSec);

        task->completed.store(true);
      }
    }
  }

  // Ensure a queue and associated worker thread exists
  void ensureQueueExists(const QueueId& queueId) {
    // First check if queue exists - do this with the lock
    {
      std::lock_guard<std::mutex> lock(_queuesMutex);
      if (_queues.find(queueId) != _queues.end()) {
        // Queue already exists, nothing to do
        return;
      }
    }

    // Create the queue and necessary components
    {
      std::lock_guard<std::mutex> lock(_queuesMutex);

      // Initialize queue
      _queues[queueId] = std::queue<std::shared_ptr<MemcpyTask>>();

      // Create a new condition variable with a unique_ptr
      _queueConditions[queueId] = std::make_unique<std::condition_variable>();
      _queueStopFlags[queueId] = false;
    }

    // Register queue with QuotaManager - do this outside the lock
    QuotaManager::instance().registerQueue(queueId);

    // Start dedicated worker thread for this queue
    createQueueWorker(queueId);
  }

  public:
  // Initialize with available device IDs
  MemcpyThreadPool(const std::vector<int>& deviceIds) : _deviceIds(deviceIds) {
    // Queues will be created dynamically as needed
  }

  // Clean up
  ~MemcpyThreadPool() {
    // Signal all worker threads to stop
    {
      std::lock_guard<std::mutex> lock(_queuesMutex);
      for (auto& pair : _queueStopFlags) {
        pair.second = true;
      }
    }

    // Notify all condition variables
    for (auto& pair : _queueConditions) {
      pair.second->notify_all();
    }

    // Join all worker threads
    for (auto& pair : _queueWorkers) {
      if (pair.second.joinable()) {
        pair.second.join();
      }
    }
  }

  // Add a new memory copy task
  bool addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
               hip::Stream* stream, int deviceId) {
    // Create queue ID with current process ID
    QueueId queueId = {deviceId, getpid()};

    // Ensure queue exists with dedicated worker
    ensureQueueExists(queueId);

    // Define retry parameters
    const int maxRetries = 20;
    const auto retryDelay = std::chrono::milliseconds(100);
    int retryCount = 0;

    while (retryCount < maxRetries) {
      // Check with scheduler for quota before issuing load
      if (QuotaManager::instance().checkQuota(queueId, sizeBytes)) {
        auto task = std::make_shared<MemcpyTask>();
        task->dst = dst;
        task->src = src;
        task->sizeBytes = sizeBytes;
        task->kind = kind;
        task->stream = stream;
        task->queueId = queueId;

        {
          // Add to the queue
          std::lock_guard<std::mutex> queueLock(_queuesMutex);
          _queues[queueId].push(task);

          // Notify the worker thread
          if (_queueConditions.find(queueId) != _queueConditions.end()) {
            _queueConditions[queueId]->notify_one();
          }
        }

        {
          // Track by stream for synchronization
          std::lock_guard<std::mutex> streamLock(_streamsMutex);
          _streamTasks[stream].push_back(task);
        }

        return true;
      }

      // Quota exceeded, log and wait before retry
      retryCount++;
      MRFS_LOG("POOL",
               "Queue (device=" << queueId.deviceId << ", process=" << queueId.processId
                                << ") bandwidth quota exceeded, retry " << retryCount << "/"
                                << maxRetries << ", waiting " << retryDelay.count() << "ms");

      std::this_thread::sleep_for(retryDelay);
    }

    // If we've exhausted all retries, log and return failure
    MRFS_LOG("POOL",
             "Task rejected: bandwidth quota exceeded for queue (device="
                 << queueId.deviceId << ", process=" << queueId.processId << ") after "
                 << maxRetries << " retries");
    return false;
  }

  // Add multiple chunked memory copy tasks
  bool addChunkedTasks(void* dst, const void* src, size_t totalSize, hipMemcpyKind kind,
                       hip::Stream* stream, int deviceId) {
    const size_t chunkSize = getMemcpyChunkSize();
    const size_t numChunks = (totalSize + chunkSize - 1) / chunkSize;  // Ceiling division

    MRFS_LOG("POOL",
             "Chunking " << totalSize << " byte request into " << numChunks << " chunks of "
                         << chunkSize << " bytes or less");

    for (size_t i = 0; i < numChunks; i++) {
      // Calculate chunk boundaries
      size_t offset = i * chunkSize;
      size_t currentChunkSize = std::min(chunkSize, totalSize - offset);

      // Calculate pointers for this chunk
      void* chunkDst = static_cast<char*>(dst) + offset;
      const void* chunkSrc = static_cast<const char*>(src) + offset;

      // Add the chunk as a separate task
      bool success = addTask(chunkDst, chunkSrc, currentChunkSize, kind, stream, deviceId);

      if (!success) {
        MRFS_LOG(
            "POOL",
            "Failed to add chunk " << i + 1 << "/" << numChunks << ". Aborting remaining chunks.");
        return false;
      }
    }

    return true;
  }

  // Wait for stream tasks to complete
  void waitForStream(hip::Stream* stream) {
    std::vector<std::shared_ptr<MemcpyTask>> streamTasks;

    {
      std::lock_guard<std::mutex> lock(_streamsMutex);
      if (_streamTasks.find(stream) != _streamTasks.end()) {
        streamTasks = _streamTasks[stream];
      }
    }

    for (auto& task : streamTasks) {
      while (!task->completed.load()) {
        std::this_thread::yield();
      }
    }

    // Clean up completed tasks
    {
      std::lock_guard<std::mutex> lock(_streamsMutex);
      if (_streamTasks.find(stream) != _streamTasks.end()) {
        _streamTasks[stream].clear();
      }
    }
  }

  // Wait for all tasks to complete
  void waitForAll() {
    std::vector<std::shared_ptr<MemcpyTask>> allTasks;

    {
      std::lock_guard<std::mutex> lock(_streamsMutex);
      // Get all tasks from all streams
      for (auto& pair : _streamTasks) {
        allTasks.insert(allTasks.end(), pair.second.begin(), pair.second.end());
      }
    }

    // Wait for all tasks to complete
    for (auto& task : allTasks) {
      while (!task->completed.load()) {
        std::this_thread::yield();
      }
    }

    // Clean up
    {
      std::lock_guard<std::mutex> lock(_streamsMutex);
      _streamTasks.clear();
    }
  }
};

// Singleton to manage the thread pool per process
class ThreadManager {
  private:
  std::unique_ptr<MemcpyThreadPool> _pool;
  pid_t _processId;

  ThreadManager() {
    int numDevices = 1;
    (void)hipGetDeviceCount(&numDevices);

    // Get the list of available device IDs
    std::vector<int> deviceIds;
    for (int i = 0; i < numDevices; ++i) {
      deviceIds.push_back(i);
    }

    _processId = getpid();
    _pool = std::make_unique<MemcpyThreadPool>(deviceIds);

    // Set initial process quota
    QuotaManager::instance().setProcessQuota(_processId, DEFAULT_PROCESS_BANDWIDTH_QUOTA);
  }

  public:
  static ThreadManager& instance() {
    static ThreadManager manager;
    return manager;
  }

  MemcpyThreadPool* pool() { return _pool.get(); }

  // Set bandwidth quota for the current process
  void setProcessBandwidthQuota(double bandwidthBytesPerSec) {
    QuotaManager::instance().setProcessQuota(_processId, bandwidthBytesPerSec);
  }
};

hipError_t hipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                               hip::Stream& stream) {
  if (sizeBytes == 0) {
    return hipSuccess;
  }

  hipError_t status = ihipMemcpy_validate(dst, src, sizeBytes, kind);
  if (status != hipSuccess) {
    return status;
  }

  if (src == dst && kind == hipMemcpyDefault) {
    return hipSuccess;
  }

  ThreadManager::instance().setProcessBandwidthQuota(DEFAULT_PROCESS_BANDWIDTH_QUOTA);

  // Check if we need to chunk the request
  bool accepted = false;
  size_t chunkSize = getMemcpyChunkSize();
  int deviceId = getCurrentDevice()->deviceId();

  if (sizeBytes > chunkSize) {
    // Large request -> divide into chunks
    accepted = ThreadManager::instance().pool()->addChunkedTasks(dst, src, sizeBytes, kind, &stream,
                                                                 deviceId);
  } else {
    // Small request -> process normally
    accepted =
        ThreadManager::instance().pool()->addTask(dst, src, sizeBytes, kind, &stream, deviceId);
  }

  if (!accepted) {
    MRFS_LOG("HIPAPI", "Task rejected due to bandwidth quota constraints after all retries");
    return hipErrorOutOfMemory;  // Or another appropriate error code
  }

  return hipSuccess;
}

// Device synchronization
void interceptedHipDeviceSynchronize() {
  // Wait for all memory operations to complete
  ThreadManager::instance().pool()->waitForAll();

  // Call the original sync function
  constexpr bool kDoWaitForCpu = false;
  hip::getCurrentDevice()->SyncAllStreams(kDoWaitForCpu);
}

// Stream synchronization
hipError_t interceptedHipStreamSynchronize(hipStream_t streamHandle) {
  hip::Stream* stream = reinterpret_cast<hip::Stream*>(streamHandle);
  // Wait for stream operations to complete
  ThreadManager::instance().pool()->waitForStream(stream);

  return hipSuccess;
}

}  // namespace hip