#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/// check return code of API calls:
// gpuErrchk( cudaMalloc(&a_d, size*sizeof(int)) );

/// check for errors in kernel launches
// kernel<<<1,1>>>(a);
// gpuErrchk( cudaPeekAtLastError() );
// gpuErrchk( cudaDeviceSynchronize() );
