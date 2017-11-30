    library(OpenCL)
    p = oclPlatforms()       #gets computing platform 
    d = oclDevices(p[[1]])   #sets GPU device we want to use
    mm_code <- c(
            "//pragma OPENCL EXTENSION cl_khr_fp64 : enable //hmmm, slidy!!
            __kernel void myMM(__global float* C, 
                        const int totalMN, const int M, const int N, const int K,
                        const __global float* A,
                        const __global float* B)
            {   // Thread identifiers
                const int MN = get_global_id(0); // element i.d. of C
                int acc = 0;
                int m1Row = MN / M; //current row M1
                int m2Col = MN % N; //current col M2
                for (int k=0; k<K; k++) {
                        acc += A[m1Row*M + k]*B[m2Col+k*N];}
                // Store the result
                C[MN] = acc;};")
