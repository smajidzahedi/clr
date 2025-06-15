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
  std::cerr << "Worker " << deviceId << " started" << std::endl;
  uint64_t startTime = 0;
  uint64_t endTime = 0;
  size_t chunkSize = MRFS_MIN_CHUNK_SIZE << 10;


  while (true) {
    auto task = buffers[deviceId]->get_front();
    size_t remainingBytes = task->sizeBytes;
    while(remainingBytes > 0) {
      startTime = getCurrentTime();

      size_t quotaBPS = deviceQuotaBPS[deviceId]->load(std::memory_order_relaxed);

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
      }
      else if (usedBPS > ((11 * quotaBPS) / 10) && chunkSize > MRFS_MIN_CHUNK_SIZE) {
        chunkSize >>= 1; // Divide by 2 -> Coming back down fast!
      }
      remainingBytes -= sizeBytes;
    }

    buffers[deviceId]->pop_front();
  }
}

void QuotaManager::reallocatedeviceQuotaBPW() {
  size_t currentQuotaBPS = 0;
  std::vector<int> activeDevices(deviceCount, 0);
  int activeDeviceCount = 0;
  std::cerr << "Quota reallocator started" << std::endl;

  while(true) {
    bool flag = false;
    size_t newQuotaBPS = processQuotaBPS.load(std::memory_order_relaxed);

    // Check if there is new quota
    if (currentQuotaBPS != newQuotaBPS) {
      currentQuotaBPS = newQuotaBPS;
      flag = true;
    }

    // Check if devices have become inactive or active
    for (int i = 0; i < deviceCount; i++) {
      if (buffers[i]->size() != 0) {
        if (activeDevices[i] == 0) {
          activeDeviceCount++;
	  flag = true;
	}
        activeDevices[i] = MRFS_M_CHANCE;
      }
      else {
        if (activeDevices[i] == 1) {
          activeDeviceCount--;
          flag = true;
	}
        if (activeDevices[i] > 0) {
          activeDevices[i]--;
        }
      }
    }

    if (flag) {
      for (int i = 0; i < deviceCount; i++) {
        if (activeDevices[i] > 0) {
          size_t quotaPerDeviceBPS = currentQuotaBPS / activeDeviceCount;
          deviceQuotaBPS[i]->store(quotaPerDeviceBPS, std::memory_order_relaxed);
        } else {
          deviceQuotaBPS[i]->store(0, std::memory_order_relaxed);
        }
      }
    }

    std::this_thread::sleep_for(std::chrono::nanoseconds(MRFS_REALLOCATE_PERIOD_NS));
  }
}

void QuotaManager::setProcessQuota(size_t newQuotaBPS) {
  processQuotaBPS.store(newQuotaBPS, std::memory_order_relaxed);
}

bool QuotaManager::addTask(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                           hipStream_t stream, int deviceId) {
  std::unique_ptr task = std::make_unique<MemcpyTask>();
  task->dst = dst;
  task->src = src;
  task->sizeBytes = sizeBytes;
  task->kind = kind;
  task->stream = stream;
  task->deviceId = deviceId;
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

  if (QuotaManager::getInstance().addTask(dst, src, sizeBytes, kind, stream, deviceId))
    return hipSuccess;
  else
    return hipErrorOutOfMemory; //TODO: Find or define a better error code
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
