#include <iostream>
#include <type_traits>
#include <chrono>


#ifdef ENABLE_TIMING
#define START_TIMING(name) auto start_##name = std::chrono::steady_clock::now();
#define END_TIMING(name) std::cout << "Elapsed time (" << #name << "): " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start_##name).count() << " ms" << std::endl;
#else
#define START_TIMING(name)
#define END_TIMING(name)
#endif



// START_TIMING(work_name);
// ... work ...
// END_TIMING(work_name);
