#include "hip_mrfs.hpp"

namespace hip {

class Device;
class Stream;
extern Device* getCurrentDevice();
extern hipError_t hipStreamSynchronize_common(hipStream_t stream);
extern hipError_t ihipMemcpy_validate(void* dst, const void* src, size_t sizeBytes,
                                      hipMemcpyKind kind);

void QuotaManager::workerFunction(int deviceId) {
  amd::Thread::init();
  std::cerr << "LOG: Worker " << deviceId << " started" << std::endl;
  uint64_t startTime = 0;
  uint64_t endTime = 0;
  size_t chunkSize = MRFS_MIN_CHUNK_SIZE << 10;
  std::vector<int> currentActiveNumaDevices(numaCount, 1.0);
  std::vector<int> mChance(numaCount, MRFS_M_CHANCE);


  while (true) {
    auto task = buffers[deviceId]->get_front();
    int numaId = task->numaId;
    numaActiveDevices[numaId].fetch_add(1, std::memory_order_relaxed);
    size_t remainingBytes = task->sizeBytes;

    while(remainingBytes > 0) {
      startTime = getCurrentTime();

      size_t numaQuotaBPS = processNumaQuotaBPS[numaId].load(std::memory_order_relaxed);
      int numActiveNumaDevices = numaActiveDevices[numaId].load(std::memory_order_relaxed);
      if (mChance[numaId] == 0) {
        currentActiveNumaDevices[numaId] = numActiveNumaDevices;
        mChance[numaId] = MRFS_M_CHANCE;
      } else if (numActiveNumaDevices != currentActiveNumaDevices[numaId]) {
        mChance[numaId]--;
      } else {
        mChance[numaId] = MRFS_M_CHANCE;
      }
      size_t quotaBPS = numaQuotaBPS / currentActiveNumaDevices[numaId];

      size_t sizeBytes = std::min(chunkSize, remainingBytes);
      void* dst = static_cast<char*>(task->dst) + (task->sizeBytes - remainingBytes);
      const void* src = static_cast<const char*>(task->src) + (task->sizeBytes - remainingBytes);
      (void)hipSetDevice(deviceId);
      hip::Stream* hip_stream = hip::getStream(task->stream);
      (void)ihipMemcpy(dst, src, sizeBytes, task->kind, *hip_stream);

      endTime = getCurrentTime();
      size_t usedBPS = (sizeBytes * 1000000000) / (endTime - startTime);

      if (usedBPS < ((9 * quotaBPS) / 10)) {
        chunkSize += (chunkSize >> 1); // Multiply by 1.5 -> Going up slowly!
      } else if (usedBPS > ((11 * quotaBPS) / 10) && chunkSize > 2 * MRFS_MIN_CHUNK_SIZE) {
        chunkSize >>= 1; // Divide by 2 -> Coming back down fast!
      }

      remainingBytes -= sizeBytes;
    }

    buffers[deviceId]->pop_front(numaId);
    numaActiveDevices[numaId].fetch_sub(1, std::memory_order_relaxed);
  }
}

void QuotaManager::setProcessNumaQuota(size_t newQuotaBPS, int numaId) {
  processNumaQuotaBPS[numaId].store(newQuotaBPS, std::memory_order_relaxed);
}

void QuotaManager::setProcessQuota(size_t newQuotaBPS) {
  for (int i = 0; i < numaCount; i++) {
    setProcessNumaQuota(newQuotaBPS, i);
  }
}

bool QuotaManager::addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                           hipStream_t stream, int deviceId) {

  void *ptr_1, *ptr_2;
  switch (kind) {
    case hipMemcpyHostToDevice:
      ptr_1 = const_cast<void *>(src);
      ptr_2 = dst;
      break;
    case hipMemcpyDeviceToHost:
      ptr_1 = dst;
      ptr_2 = const_cast<void *>(src);
      break;
    default:
      return false; // TODO: we currently only support H2D and D2H
  }

  int numa_node = -1;
  get_mempolicy(&numa_node, NULL, 0, ptr_1, MPOL_F_NODE | MPOL_F_ADDR);

  if (numa_node < 0 || numa_node > numaCount) {
    get_mempolicy(&numa_node, NULL, 0, ptr_2, MPOL_F_NODE | MPOL_F_ADDR);
    if (numa_node < 0 || numa_node > numaCount) {
      std::cerr << "ERR: could not find numa node for src and des" << std::endl;
      return false;
    }
  }

  std::unique_ptr task = std::make_unique<MemcpyTask>();
  task->dst = dst;
  task->src = src;
  task->sizeBytes = sizeBytes;
  task->kind = kind;
  task->stream = stream;
  task->deviceId = deviceId;
  task->numaId = numa_node;
  std::atomic_thread_fence(std::memory_order_release);
  return buffers[deviceId]->push_back(std::move(task));
}

void QuotaManager::waitForDevice(int deviceId) {
  buffers[deviceId]->wait_till_empty();
}

hipError_t ihipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                                hipStream_t stream) {
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

  if (QuotaManager::getInstance().addTask(dst, src, sizeBytes, kind, stream, deviceId)) {
    return hipSuccess;
  } else {
    return hipErrorTbd;  // TODO: Find or define a better error code
  }
}

void ihipDeviceSynchronize_mrfs() {
  auto current_device = hip::getCurrentDevice();
  QuotaManager::getInstance().waitForDevice(current_device->deviceId());
  constexpr bool kDoWaitForCpu = false;
  current_device->SyncAllStreams(kDoWaitForCpu);
}

// Stream synchronization
hipError_t ihipStreamSynchronize_mrfs(hipStream_t stream) {
  QuotaManager::getInstance().waitForDevice(hip::getCurrentDevice()->deviceId());
  HIP_RETURN(hipStreamSynchronize_common(stream));
}

hipError_t ihipSetProcessQuota_mrfs(size_t quota) {
  QuotaManager::getInstance().setProcessQuota(quota);
  return hipSuccess;
}

}  // namespace hip
