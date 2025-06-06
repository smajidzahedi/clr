#include "hip_mrfs.hpp"

namespace hip {

class Device;
class Stream;
extern Device* getCurrentDevice();
extern hipError_t hipStreamSynchronize_common(hipStream_t stream);
extern hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind);

void QuotaManager::enforceBW(uint64_t current_time, size_t requested_bytes, int deviceID) {

  if (current_time < lastUpdate[deviceID]) { // Overflow has happened
    lastUpdate[deviceID] = 0;
  }

  if (current_time - lastUpdate[deviceID] > TIME_WINDOW_NS) {
    lastUpdate[deviceID] = current_time;
    deviceUsage[deviceID] = 0;
  }

  int num_waiting_window = 0;

  while (true) {
    int my_quota = deviceQuota[deviceID]->load(std::memory_order_relaxed);
    if (deviceUsage[deviceID] + requested_bytes <= my_quota)
      break;
    std::this_thread::sleep_for(std::chrono::nanoseconds(TIME_WINDOW_NS);
    deviceUsage[deviceID] = deviceUsage[deviceID] - my_quota;
    num_waiting_window++;
  }

  deviceUsage[deviceID] = deviceUsage[deviceID] + requested_bytes;
  lastUpdate[deviceID] = current_time + num_waiting_window * TIME_WINDOW_NS;
  return;
}

void QuotaManager::workerFunction(int deviceId) {
  amd::Thread::init();
  while (true) {
    auto& task = taskBuffers[deviceId].pop_front();
    enforceBW(getCurrentTime(), task->sizeBytes, deviceId);
    (void)hipSetDevice(deviceId);
    hip::Stream* hip_stream = hip::getStream(task->stream);
    (void)ihipMemcpy(task->dst, task->src, task->sizeBytes, task->kind, *hip_stream);
  }
}

void QuotaManager::reallocateDeviceQuota() {
  while(true) {
    int activeDeviceCount = 0;
    std::vector<bool> activeDevices;
    activeDevices.resize(deviceCount);
    for (int i = 0; i < deviceCount; i++) {
      if (taskBuffers[i].size() != 0) {
        activeDeviceCount++;
        activeDevices[i] = true;
      } else {
        activeDevices[i] = false;
      }
    }

    size_t quotaPerDeviceBPS = processQuota / activeDeviceCount;
    size_t quotaPerDevice = (quotaPerDeviceBPS * 1000) / TIME_WINDOW_MS;
    chunkSize.store(quotaPerDevice, std::memory_order_relaxed);

    for (int i = 0; i < deviceCount; i++) {
      if (activeDevices[i]) {
        deviceQuota[i]->store(quotaPerDevice, std::memory_order_relaxed);
      } else {
        deviceQuota[i]->store(0, std::memory_order_relaxed);
      }
    }

    std::this_thread::sleep_for(std::chrono::nanoseconds(REALLOCATE_PERIOD * 1000 * 1000));
  }
}

void QuotaManager::setProcessQuota(size_t newQuota) {
  processQuotaBPS.store(newQuota, std::memory_order_relaxed);
}

bool QuotaManager::addChunkedTasks(void* dst, const void* src, size_t totalSize, hipMemcpyKind kind,
                                   hipStream_t stream, int deviceId) {
  std::vector<std::unique_ptr<MemcpyTask>> tasks;
  size_t current_chunk_size = chunkSize.load(std::memory_order_relaxed);
  const int numChunks = totalSize / current_chunk_size + 1;
  tasks.reserve(numChunks);

  for (int i = 0; i < numChunks; i++) {
    int offset = i * current_chunk_size;
    int task_chunk_size = std::min(current_chunk_size, totalSize - offset);

    void* chunkDst = static_cast<char*>(dst) + offset;
    const void* chunkSrc = static_cast<const char*>(src) + offset;

    tasks.emplace_back(std::make_unique<MemcpyTask>());
    tasks[i]->dst = chunkDst;
    tasks[i]->src = chunkSrc;
    tasks[i]->sizeBytes = task_chunk_size;
    tasks[i]->kind = kind;
    tasks[i]->stream = stream;
    tasks[i]->deviceId = deviceId;
  }

  return buffers[deviceId]->push_back(tasks);
}

void QuotaManager::waitForDevice(int deviceId) {
  buffers[deviceId]->wait_till_empty();
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

  int deviceId = getCurrentDevice()->deviceId();

  hipStream_t s = reinterpret_cast<hipStream_t>(&stream);
  if (QuotaManager::GetInstance().addChunkedTasks(dst, src, sizeBytes, kind, s, deviceId))
    return hipSuccess;
  else
    return hipErrorOutOfMemory; //TODO: Find or define a better error code
}

void interceptedHipDeviceSynchronize() {
  auto current_device = hip::getCurrentDevice();
  QuotaManager::GetInstance().waitForDevice(current_device->deviceId());
  constexpr bool kDoWaitForCpu = false;
  current_device->SyncAllStreams(kDoWaitForCpu);
}

// Stream synchronization
hipError_t interceptedHipStreamSynchronize(hipStream_t stream) {
  QuotaManager::GetInstance().waitForDevice(hip::getCurrentDevice()->deviceId());
  HIP_RETURN(hipStreamSynchronize_common(stream));
}

}  // namespace hip
