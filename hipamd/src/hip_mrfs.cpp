#include "hip_mrfs.hpp"

namespace hip {

constexpr size_t TIME_WINDOW_MS = 100;
constexpr double PROCESS_BANDWIDTH_QUOTA_BS = 22.0 * 1024 * 1024 * 1024;
static size_t MEMCPY_CHUNK_SIZE_B = PROCESS_BANDWIDTH_QUOTA_BS * TIME_WINDOW_MS / 1000;

class Device;
class Stream;
extern Device* getCurrentDevice();
extern hipError_t hipStreamSynchronize_common(hipStream_t stream);
extern hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind);

void QuotaManager::workerFunction(int deviceId) {
  amd::Thread::init();
  while (true) {
    taskQueueMutexes[deviceId]->lock();
    while (taskQueues[deviceId].empty()) {
      taskQueueMutexes[deviceId]->unlock();
      std::this_thread::yield();
      taskQueueMutexes[deviceId]->lock();
    }
    auto& task = taskQueues[deviceId].front();
    taskQueueMutexes[deviceId]->unlock();

    struct timespec startTime;
    clock_gettime(CLOCK_MONOTONIC, &startTime);
    (void)hipSetDevice(deviceId);
    hip::Stream* hip_stream = hip::getStream(task->stream);
    (void)ihipMemcpy(task->dst, task->src, task->sizeBytes, task->kind, *hip_stream, true);
    struct timespec endTime;
    clock_gettime(CLOCK_MONOTONIC, &endTime);
    double executionTime =
        (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_nsec - startTime.tv_nsec) / 1e9;

    taskQueueMutexes[deviceId]->lock();
    taskQueues[deviceId].erase(taskQueues[deviceId].begin());
    taskQueueMutexes[deviceId]->unlock();
  }
}

void QuotaManager::reallocateDeviceQuota() {
  int activeDeviceCount = 0;
  for (const auto& taskQueue : taskQueues) {
    if (!taskQueue.empty()) {
      activeDeviceCount++;
    }
  }

  size_t quotaPerDevice = processQuota / activeDeviceCount;
  for (int i = 0; i < deviceCount; i++) {
    if (!taskQueues[i].empty()) {
      deviceQuota[i] = quotaPerDevice;
    }
  }
}

void QuotaManager::setProcessQuota(double quota) {
  processQuota = quota;
  chunkSize = processQuota * TIME_WINDOW_MS / 1000;
  reallocateDeviceQuota();
}

void QuotaManager::addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                           hipStream_t stream, int deviceId) {
  auto task = std::make_unique<MemcpyTask>();
  task->dst = dst;
  task->src = src;
  task->sizeBytes = sizeBytes;
  task->kind = kind;
  task->stream = stream;
  task->deviceId = deviceId;

  taskQueueMutexes[deviceId]->lock();
  taskQueues[deviceId].push_back(std::move(task));
  taskQueueMutexes[deviceId]->unlock();
}

void QuotaManager::addChunkedTasks(void* dst, const void* src, size_t totalSize, hipMemcpyKind kind,
                                   hipStream_t stream, int deviceId) {
  const int numChunks = (totalSize + chunkSize - 1) / chunkSize;
  for (int i = 0; i < numChunks; i++) {
    int offset = i * chunkSize;
    int currentChunkSize = std::min(chunkSize, totalSize - offset);

    void* chunkDst = static_cast<char*>(dst) + offset;
    const void* chunkSrc = static_cast<const char*>(src) + offset;

    addTask(chunkDst, chunkSrc, currentChunkSize, kind, stream, deviceId);
  }
}

void QuotaManager::waitForDevice(int deviceId) {
  taskQueueMutexes[deviceId]->lock();
  while (!taskQueues[deviceId].empty()) {
    taskQueueMutexes[deviceId]->unlock();
    std::this_thread::yield();
    taskQueueMutexes[deviceId]->lock();
  }
  taskQueueMutexes[deviceId]->unlock();
}

void QuotaManager::waitForStream(hipStream_t stream) {
  for (int i = 0; i < deviceCount; i++) {
    taskQueueMutexes[i]->lock();
  }

  for (int i = 0; i < deviceCount; i++) {
    for (auto& task : taskQueues[i]) {
      if (task->stream == stream) {
        for (int j = 0; j < deviceCount; j++) {
          taskQueueMutexes[j]->unlock();
        }
        std::this_thread::yield();
        for (int j = 0; j < deviceCount; j++) {
          taskQueueMutexes[j]->lock();
        }
      }
    }
  }

  for (int i = 0; i < deviceCount; i++) {
    taskQueueMutexes[i]->unlock();
  }
}

size_t QuotaManager::getChunkSize() {
  return chunkSize;
}

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

  size_t chunkSize = QuotaManager::GetInstance().getChunkSize();
  int deviceId = getCurrentDevice()->deviceId();

  hipStream_t s = reinterpret_cast<hipStream_t>(&stream);
  if (sizeBytes > chunkSize) {
    QuotaManager::GetInstance().addChunkedTasks(dst, src, sizeBytes, kind, s, deviceId);
  } else {
    QuotaManager::GetInstance().addTask(dst, src, sizeBytes, kind, s, deviceId);
  }

  return hipSuccess;
}

void interceptedHipDeviceSynchronize() {
  QuotaManager::GetInstance().waitForDevice(getCurrentDevice()->deviceId());
  constexpr bool kDoWaitForCpu = false;
  hip::getCurrentDevice()->SyncAllStreams(kDoWaitForCpu);
}

// Stream synchronization
hipError_t interceptedHipStreamSynchronize(hipStream_t stream) {
  QuotaManager::GetInstance().waitForStream(stream);
  HIP_RETURN(hipStreamSynchronize_common(stream));
}

}  // namespace hip
