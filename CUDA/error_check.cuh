inline void gpuAssert(cudaError_t code, const char *file, cosnt int line, const bool abort = true)
{
   if (code != cudaSuccess) 
   {
      std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) std::exit(code);
   }
}

#ifdef NDEBUG
#define gpuErrchk(ans)
#else
#define gpuErrchk(ans) gpuAssert((ans), __FILE__, __LINE__)
#endif


/// check return code of API calls:
// gpuErrchk( cudaMalloc(&a_d, size*sizeof(int)) );

/// check for errors in kernel launches
// kernel<<<1,1>>>(a);
// gpuErrchk( cudaPeekAtLastError() );
// gpuErrchk( cudaDeviceSynchronize() );
