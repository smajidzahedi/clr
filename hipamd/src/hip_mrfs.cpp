#include "hip_mrfs.hpp"
#include <atomic>

namespace hip {

std::thread* hipMemCpyThread = nullptr;
std::mutex threadMutex;
std::atomic<int> deviceSyncCount{0};
std::atomic<int> streamSyncCount{0};
std::atomic<bool> asyncMemcpyInProgress{false};


hipError_t hipMemcpyAsync_mrfs(void* dst, const void* src, size_t sizeBytes, 
                             hipMemcpyKind kind, hip::Stream& stream) {
  hipError_t status;
  
  if (sizeBytes == 0) {
    return hipSuccess;
  }
  
  status = ihipMemcpy_validate(dst, src, sizeBytes, kind);
  if (status != hipSuccess) {
    return status;
  }
  
  if (src == dst && kind == hipMemcpyDefault) {
    return hipSuccess;
  }

  int deviceId = getCurrentDevice()->deviceId();
  
  auto memcpyWork = [deviceId, dst, src, sizeBytes, kind, &stream]() {

    amd::Thread::init();
    (void)hipSetDevice(deviceId);
    
    // fprintf(stderr, "[INTERCEPT] Starting async memcpy: %zu bytes, src=%p, dst=%p\n", 
    //         sizeBytes, src, dst);
    
    asyncMemcpyInProgress.store(true);
    
    hipError_t result = ihipMemcpy(dst, src, sizeBytes, kind, stream, true);
    
    // fprintf(stderr, "[INTERCEPT] Completed async memcpy: %zu bytes, result=%d\n", 
    //         sizeBytes, result);
    
    asyncMemcpyInProgress.store(false);
  };

  std::lock_guard<std::mutex> lock(threadMutex);
  
  // Clean up previous thread if it exists...
  if (hipMemCpyThread && hipMemCpyThread->joinable()) {
    // fprintf(stderr, "[INTERCEPT] Joining previous memcpy thread before starting new one\n");
    hipMemCpyThread->join();
    delete hipMemCpyThread;
  }

  // Create and start new thread
  // fprintf(stderr, "[INTERCEPT] Starting new async memcpy thread\n");
  hipMemCpyThread = new std::thread(memcpyWork);

  return hipSuccess;
}

// Intercepted hipDeviceSynchronize
hipError_t interceptedHipDeviceSynchronize() {
  int count = ++deviceSyncCount;
  
  fprintf(stderr, "[INTERCEPT] hipDeviceSynchronize called (count: %d)\n", count);
  
  // Check to see if async copy in progress
  if (asyncMemcpyInProgress.load()) {
    fprintf(stderr, "[INTERCEPT] hipDeviceSynchronize called while async copy in progress!\n");
  }
  
  std::lock_guard<std::mutex> lock(threadMutex);
  
  // Join any ongoing memory copy thread
  if (hipMemCpyThread && hipMemCpyThread->joinable()) {
    fprintf(stderr, "[INTERCEPT] Joining memcpy thread in hipDeviceSynchronize\n");
    hipMemCpyThread->join();
    delete hipMemCpyThread;
    hipMemCpyThread = nullptr;
  }
  
  fprintf(stderr, "[INTERCEPT] Calling original ihipDeviceSynchronize\n");
  hipError_t result = ihipDeviceSynchronize();
  
  fprintf(stderr, "[INTERCEPT] hipDeviceSynchronize completed with status: %d\n", result);
  
  return result;
}

// Intercepted hipStreamSynchronize
hipError_t interceptedHipStreamSynchronize(hipStream_t stream) {
  int count = ++streamSyncCount;
  
  fprintf(stderr, "[INTERCEPT] hipStreamSynchronize called (count: %d, stream: %p)\n", 
          count, stream);
  
  std::lock_guard<std::mutex> lock(threadMutex);
  
  // Join any ongoing memory copy thread if it's for this stream
  // TODO: add checking if the thread is working on this specific stream
  if (hipMemCpyThread && hipMemCpyThread->joinable()) {
    fprintf(stderr, "[INTERCEPT] Joining memcpy thread in hipStreamSynchronize\n");
    hipMemCpyThread->join();
    delete hipMemCpyThread;
    hipMemCpyThread = nullptr;
  }
  
  // Call original implementation
  fprintf(stderr, "[INTERCEPT] Calling original ihipStreamSynchronize\n");
  hipError_t result = ihipStreamSynchronize(stream);
  
  fprintf(stderr, "[INTERCEPT] hipStreamSynchronize completed with status: %d\n", result);
  
  return result;
}

} // namespace hip