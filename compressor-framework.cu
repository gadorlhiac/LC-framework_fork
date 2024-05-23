/*
This file is part of the LC framework for synthesizing high-speed parallel lossless and error-bounded lossy data compression and decompression algorithms for CPUs and GPUs.

BSD 3-Clause License

Copyright (c) 2021-2024, Noushin Azami, Alex Fallin, Brandon Burtchell, Andrew Rodriguez, Benila Jerald, Yiqian Liu, and Martin Burtscher
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of this code is available at https://github.com/burtscher/LC-framework.

Sponsor: This code is based upon work supported by the U.S. Department of Energy, Office of Science, Office of Advanced Scientific Research (ASCR), under contract DE-SC0022223.
*/


#define NDEBUG

using byte = unsigned char;
static const int CS = 1024 * 16;  // chunk size (in bytes) [must be multiple of 8]
static const int TPB = 512;  // threads per block [must be power of 2 and at least 128]
#if defined(__AMDGCN_WAVEFRONT_SIZE) && (__AMDGCN_WAVEFRONT_SIZE == 64)
#define WS 64
#else
#define WS 32
#endif

#include <cuda_fp16.h>
#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <cuda.h>
#include "include/sum_reduction.h"
#include "include/max_scan.h"
#include "include/prefix_sum.h"
/*##include-beg##*/
/*##include-end##*/


// copy (len) bytes from shared memory (source) to global memory (destination)
// source must we word aligned
static inline __device__ void s2g(void* const __restrict__ destination, const void* const __restrict__ source, const int len)
{
  const int tid = threadIdx.x;
  const byte* const __restrict__ input = (byte*)source;
  byte* const __restrict__ output = (byte*)destination;
  if (len < 128) {
    if (tid < len) output[tid] = input[tid];
  } else {
    const int nonaligned = (int)(size_t)output;
    const int wordaligned = (nonaligned + 3) & ~3;
    const int linealigned = (nonaligned + 127) & ~127;
    const int bcnt = wordaligned - nonaligned;
    const int wcnt = (linealigned - wordaligned) / 4;
    const int* const __restrict__ in_w = (int*)input;
    if (bcnt == 0) {
      int* const __restrict__ out_w = (int*)output;
      if (tid < wcnt) out_w[tid] = in_w[tid];
      for (int i = tid + wcnt; i < len / 4; i += TPB) {
        out_w[i] = in_w[i];
      }
      if (tid < (len & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    } else {
      const int shift = bcnt * 8;
      const int rlen = len - bcnt;
      int* const __restrict__ out_w = (int*)&output[bcnt];
      if (tid < bcnt) output[tid] = input[tid];
      if (tid < wcnt) out_w[tid] = __funnelshift_r(in_w[tid], in_w[tid + 1], shift);
      for (int i = tid + wcnt; i < rlen / 4; i += TPB) {
        out_w[i] = __funnelshift_r(in_w[i], in_w[i + 1], shift);
      }
      if (tid < (rlen & 3)) {
        const int i = len - 1 - tid;
        output[i] = input[i];
      }
    }
  }
}


static __device__ int g_chunk_counter;


static __global__ void d_reset()
{
  g_chunk_counter = 0;
}


static inline __device__ void propagate_carry(const int value, const int chunkID, volatile int* const __restrict__ fullcarry, int* const __restrict__ s_fullc)
{
  if (threadIdx.x == TPB - 1) {  // last thread
    fullcarry[chunkID] = (chunkID == 0) ? value : -value;
  }

  if (chunkID != 0) {
    if (threadIdx.x + WS >= TPB) {  // last warp
      const int lane = threadIdx.x % WS;
      const int cidm1ml = chunkID - 1 - lane;
      int val = -1;
      __syncwarp();  // not optional
      do {
        if (cidm1ml >= 0) {
          val = fullcarry[cidm1ml];
        }
      } while ((__any_sync(~0, val == 0)) || (__all_sync(~0, val <= 0)));
#if defined(WS) && (WS == 64)
      const long long mask = __ballot_sync(~0, val > 0);
      const int pos = __ffsll(mask) - 1;
#else
      const int mask = __ballot_sync(~0, val > 0);
      const int pos = __ffs(mask) - 1;
#endif
      int partc = (lane < pos) ? -val : 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
      partc = __reduce_add_sync(~0, partc);
#else
      partc += __shfl_xor_sync(~0, partc, 1);
      partc += __shfl_xor_sync(~0, partc, 2);
      partc += __shfl_xor_sync(~0, partc, 4);
      partc += __shfl_xor_sync(~0, partc, 8);
      partc += __shfl_xor_sync(~0, partc, 16);
#endif
      if (lane == pos) {
        const int fullc = partc + val;
        fullcarry[chunkID] = fullc + value;
        *s_fullc = fullc;
      }
    }
  }
}


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 800)
static __global__ __launch_bounds__(TPB, 3)
#else
static __global__ __launch_bounds__(TPB, 2)
#endif
void d_encode(const byte* const __restrict__ input,
              const byte* const __restrict__ ped,      /// NEW -- Doesn't actually need to be byte
              const byte* const __restrict__ gain,     /// NEW -- Doesn't actually need to be byte
              const int insize,                        // Input size in bytes
              byte* const __restrict__ output, 
              int*  const __restrict__ outsize, 
              int*  const __restrict__ fullcarry)
{
  // allocate shared memory buffer
  __shared__ long long chunk [3 * (CS / sizeof(long long))];

  // split into 3 shared memory buffers
  byte* in = (byte*)&chunk[0 * (CS / sizeof(long long))];
  byte* out = (byte*)&chunk[1 * (CS / sizeof(long long))];
  byte* const temp = (byte*)&chunk[2 * (CS / sizeof(long long))];

  // initialize
  const int tid = threadIdx.x;
  const int last = 3 * (CS / sizeof(long long)) - 2 - WS;
  const int chunks = (insize + CS - 1) / CS;  // round up
  int* const head_out = (int*)output;
  unsigned short* const size_out = (unsigned short*)&head_out[1];
  byte* const data_out = (byte*)&size_out[chunks];

  // loop over chunks
  do {
    // assign work dynamically
    if (tid == 0) chunk[last] = atomicAdd(&g_chunk_counter, 1);
    __syncthreads();  // chunk[last] produced, chunk consumed

    // terminate if done
    const int chunkID = chunk[last];
    const int base = chunkID * CS;
    if (base >= insize) break;

    // load chunk
    const int osize = min(CS, insize - base);
    long long* const input_l = (long long*)&input[base];
    //// NEW
    long long* const ped_l = (long long*)&ped[base];
    long long* const gain_l = (long long*)&gain[base];
    //// END NEW
    long long* const out_l = (long long*)out;
    for (int i = tid; i < osize / 8; i += TPB) {
      //// NEW
      //long long temp_dat = input_l[i];
      //long long temp_ped = ped_l[i];
      //unsigned short* pix_vals = new (&temp_dat) short[4];
      //unsigned short* ped_vals = new (&temp_ped) short[4];
      unsigned short* pix_vals = new (input_l+i) unsigned short[4];
      unsigned short* ped_vals = new (ped_l+i)   unsigned short[4];
      unsigned short* gain_vals = new (gain_l+i)   unsigned short[4];
      half out_vals[4];
      #pragma unroll
      for (size_t j = 0; j<4; ++j) {
        out_vals[j] = pix_vals[j] - ped_vals[j];
        if (gain_vals[j] == 2) out_vals[j] /= gain_vals[j]; // Assuming only gain of 1 and 2...
        //pix_vals[j] -= ped_vals[j];
        //pix_vals[j] >>= gain_vals[j] - 1; // Assuming only gain of 1 and 2...
      }
      out_l[i] = *new (out_vals) long long;
      //out_l[i] = *new (pix_vals) long long;          
      //// END NEW - Line below COMMENTED OUT
      //out_l[i] = input_l[i]; // Transferring 8 bytes (?) 4 pixels
    }
    const int extra = osize % 8;
    //// COMMENTED OUT -- ADDRESSED in "NEW" below
    //if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];
    //// END COMMENTED OUT
    //if (tid == 0) printf("Post chunk load\n");
    /// NEW
    const int extra_pix = extra/2;
    if (tid < extra_pix) {
      // 3 extra pix: tid==0 -> tid*2 == 0 (covers 0, 1)
      //              tid==1 -> tid*2 == 2 (covers 2, 3)
      //              tid==2 -> tid*2 == 4 (covers 4, 5)
      int idx = tid*2;
      unsigned short pix_val = input[base + osize - extra + idx];
      unsigned short ped_val = ped[base + osize - extra + idx];
      unsigned short gain_val = gain[base + osize - extra + idx];
      //pix_val -= ped_val;
      //pix_val >>= gain_val - 1;
      //byte* pix_bytes = new (&pix_val) byte[2];
      half out_val = pix_val - ped_val;
      if (gain_val == 2) out_val /= gain_val;
      byte* pix_bytes = new (&out_val) byte[2];
      out[base + osize - extra + idx] = pix_bytes[0];
      out[base + osize - extra + idx + 1] = pix_bytes[1];
    }
    /// END NEW
    //if (tid == 0) printf("Post chunk extra bit load\n");

    // encode chunk
    __syncthreads();  // chunk produced, chunk[last] consumed
    int csize = osize;
    bool good = true;
    /*##comp-encoder-beg##*/
    if (good) {
      byte* tmp = in; in = out; out = tmp;
      good = d_CLOG_1(csize, in, out, temp);
    }
    __syncthreads();  // chunk transformed
    /*##comp-encoder-end##*/

    // handle carry
    if (!good || (csize >= osize)) csize = osize;
    propagate_carry(csize, chunkID, fullcarry, (int*)temp);

    // reload chunk if incompressible
    if (tid == 0) size_out[chunkID] = csize;
    if (csize == osize) {
      //// COMMENTED OUT -- ADDRESSED IN "NEW" below
      // store original data
      //long long* const out_l = (long long*)out;
      //for (int i = tid; i < osize / 8; i += TPB) {
      //  out_l[i] = input_l[i];
      //}
      //const int extra = osize % 8;
      //if (tid < extra) out[osize - extra + tid] = input[base + osize - extra + tid];
      //// END COMMENTED OUT
      //// NEW
      long long* const ped_l = (long long*)&ped[base];
      long long* const gain_l = (long long*)&gain[base];
      //// END NEW
      long long* const out_l = (long long*)out;
      for (int i = tid; i < osize / 8; i += TPB) {
        //// NEW
        //long long temp_dat = input_l[i];
        //long long temp_ped = ped_l[i];
        //unsigned short* pix_vals = new (&temp_dat) short[4];
        //unsigned short* ped_vals = new (&temp_ped) short[4];
        unsigned short* pix_vals = new (input_l+i) unsigned short[4];
        unsigned short* ped_vals = new (ped_l+i)   unsigned short[4];
        unsigned short* gain_vals = new (gain_l+i)   unsigned short[4];
        half out_vals[4];
        #pragma unroll
        for (size_t j = 0; j<4; ++j) {
          out_vals[j] = pix_vals[j] - ped_vals[j];
          if (gain_vals[j] == 2) out_vals[j] /= gain_vals[j]; // Assuming only gain of 1 and 2...
        //pix_vals[j] -= ped_vals[j];
        //pix_vals[j] >>= gain_vals[j] - 1; // Assuming only gain of 1 and 2...
        }
        out_l[i] = *new (out_vals) long long;
      //out_l[i] = *new (pix_vals) long long;          
      //// END NEW - Line below COMMENTED OUT
      //out_l[i] = input_l[i]; // Transferring 8 bytes (?) 4 pixels
      }
      const int extra = osize % 8;
      const int extra_pix = extra/2;
      if (tid < extra_pix) {
        // 3 extra pix: tid==0 -> tid*2 == 0 (covers 0, 1)
        //              tid==1 -> tid*2 == 2 (covers 2, 3)
        //              tid==2 -> tid*2 == 4 (covers 4, 5)
        int idx = tid*2;
        unsigned short pix_val = input[base + osize - extra + idx];
        unsigned short ped_val = ped[base + osize - extra + idx];
        unsigned short gain_val = gain[base + osize - extra + idx];
        //pix_val -= ped_val;
        //pix_val >>= gain_val - 1;
        //byte* pix_bytes = new (&pix_val) byte[2];
        half out_val = pix_val - ped_val;
        if (gain_val == 2) out_val /= gain_val;
        byte* pix_bytes = new (&out_val) byte[2];
        out[base + osize - extra + idx] = pix_bytes[0];
        out[base + osize - extra + idx + 1] = pix_bytes[1];
      }
    }
    __syncthreads();  // "out" done, temp produced

    // store chunk
    const int offs = (chunkID == 0) ? 0 : *((int*)temp);
    s2g(&data_out[offs], out, csize);

    // finalize if last chunk
    if ((tid == 0) && (base + CS >= insize)) {
      // output header
      head_out[0] = insize;
      // compute compressed size
      *outsize = &data_out[fullcarry[chunkID]] - output;
    }
  } while (true);
}


struct GPUTimer
{
  cudaEvent_t beg, end;
  GPUTimer() {cudaEventCreate(&beg); cudaEventCreate(&end);}
  ~GPUTimer() {cudaEventDestroy(beg); cudaEventDestroy(end);}
  void start() {cudaEventRecord(beg, 0);}
  double stop() {cudaEventRecord(end, 0); cudaEventSynchronize(end); float ms; cudaEventElapsedTime(&ms, beg, end); return 0.001 * ms;}
};


static void CheckCuda(const int line)
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d on line %d: %s\n\n", e, line, cudaGetErrorString(e));
    throw std::runtime_error("LC error");
  }
}


int main(int argc, char* argv [])
{
  /*##print-beg##*/
  /*##print-end##*/
  printf("Copyright 2024 Texas State University\n\n");

  // read input from file
  if (argc < 3) {printf("USAGE: %s input_file_name compressed_file_name [performance_analysis (y)]\n\n", argv[0]); return -1;}
  FILE* const fin = fopen(argv[1], "rb");
  fseek(fin, 0, SEEK_END);
  const int fsize = ftell(fin);  assert(fsize > 0);
  byte* const input = new byte [fsize];
  fseek(fin, 0, SEEK_SET);
  const int insize = fread(input, 1, fsize, fin);  assert(insize == fsize);
  fclose(fin);
  printf("original size: %d bytes\n", insize);
  printf("We will operate on: %d bytes -- After converting from f32 to u16.\n", insize/2);

  /********************************************************************/
  //// NEW -- Kludge for converting data from f32 to u16
  ////     -- Make fake pedestal data
  // Pixel count should be input size/4 (since f32) and insize is bytes
  int num_pix = insize / 4; // 4 byte pixels initially since f32?
  //printf("# Pixels: %d\n", num_pix);
  //printf("Segment pixels: %d\n", 384*352);
  // Convert from f32 to u16
  float* const input_f = new (input) float[num_pix];
  unsigned short* const input_u16 = new unsigned short[num_pix];
  std::transform(input_f,
                 input_f + num_pix, 
                 input_u16, 
                 [](float f) { return f < 0 ? 0 : std::round(f); });
  //printf("Post input cast\n");

  // Make fake pedestal and gain map and then
  // apply pedestal and gain map to u16 data
  unsigned short* ped_u16 = new unsigned short[num_pix];
  // Writing gain as u16 for the time being to avoid index issues
  unsigned short* gain_u16 = new unsigned short[num_pix];
  //printf("Post pedestal/gain creation\n");
  
  for (size_t i=0; i<num_pix; ++i) {
    ped_u16[i] = input_u16[i]/10;
    
    // Add a bit more variability..
    if (i % 2 == 0) {
      if (ped_u16[i] == 0) ped_u16[i] = 1;
      
      if (i % 8 == 0) gain_u16[i] = 2;
    }
    input_u16[i] *= gain_u16[i];
    input_u16[i] += ped_u16[i];
  }
  

  //printf("Post pedestal/gain assignment\n");
  //return 1;
  // New insize since we've gone from f32 -> u16
  int new_insize = insize/2;
  byte const * const new_input = new (input_u16) byte[new_insize];
  byte const * const ped = new (ped_u16) byte[new_insize];

  byte const * const gain = new (gain_u16) byte[new_insize];
  //// END NEW
  /********************************************************************/
  //printf("Post pedestal/gain cast\n");

  // Check if the third argument is "y" to enable performance analysis
  char* perf_str = argv[3];
  bool perf = false;
  if (perf_str != nullptr && strcmp(perf_str, "y") == 0) {
    perf = true;
  } else if (perf_str != nullptr && strcmp(perf_str, "y") != 0) {
    fprintf(stderr, "ERROR: Invalid argument. Use 'y' or nothing.\n");
    throw std::runtime_error("LC error");
  }

  // get GPU info
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {fprintf(stderr, "ERROR: no CUDA capable device detected\n\n"); throw std::runtime_error("LC error");}
  const int SMs = deviceProp.multiProcessorCount;
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int blocks = SMs * (mTpSM / TPB);
  const int chunks = (insize + CS - 1) / CS;  // round up
  CheckCuda(__LINE__);
  const int maxsize = 3 * sizeof(int) + chunks * sizeof(short) + chunks * CS;
  // Why is the maxsize this? 3 ints, n_chunks shorts, and then data in chunks
  // of chunk_size (CS)?

  // allocate GPU memory
  byte* dencoded;
  cudaMallocHost((void **)&dencoded, maxsize);
  //// COMMENTED OUT -- addressed in "NEW" below
  //byte* d_input;
  //cudaMalloc((void **)&d_input, insize);
  //cudaMemcpy(d_input, input, insize, cudaMemcpyHostToDevice);
  //// END COMMENTED SECTION
  byte* d_encoded;
  cudaMalloc((void **)&d_encoded, maxsize);
  int* d_encsize;
  cudaMalloc((void **)&d_encsize, sizeof(int));

  //// NEW
  //// Allocate GPU memory
  byte* d_input;
  cudaMalloc(reinterpret_cast<void**>(&d_input), new_insize);
  cudaMemcpy(d_input, new_input, new_insize, cudaMemcpyHostToDevice);
  byte* d_ped;
  cudaMalloc(reinterpret_cast<void**>(&d_ped), new_insize);
  cudaMemcpy(d_ped, ped, new_insize, cudaMemcpyHostToDevice);
  byte* d_gain;
  cudaMalloc(reinterpret_cast<void**>(&d_gain), new_insize);
  cudaMemcpy(d_gain, gain, new_insize, cudaMemcpyHostToDevice);
  //// END NEW

  CheckCuda(__LINE__);
  
  //// COMMENTED OUT -- addressed in "NEW" below
  //byte* dpreencdata;
  //cudaMalloc((void **)&dpreencdata, insize);
  //cudaMemcpy(dpreencdata, d_input, insize, cudaMemcpyDeviceToDevice);
  //int dpreencsize = insize;
  //// END COMMENTED SECTION
  ////NEW
  byte* dpreencdata;
  cudaMalloc((void **)&dpreencdata, new_insize);
  cudaMemcpy(dpreencdata, d_input, new_insize, cudaMemcpyDeviceToDevice);
  int dpreencsize = new_insize;
  //// END NEW

  if (perf) {
    /*##comp-warm-beg##*/
    int* d_fullcarry;
    cudaMalloc((void **)&d_fullcarry, chunks * sizeof(int));
    d_reset<<<1, 1>>>();
    cudaMemset(d_fullcarry, 0, chunks * sizeof(int));
    //d_encode<<<blocks, TPB>>>(dpreencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry);
    d_encode<<<blocks, TPB>>>(dpreencdata, d_ped, d_gain, dpreencsize, d_encoded, d_encsize, d_fullcarry);
    cudaFree(d_fullcarry);
    cudaDeviceSynchronize();
    CheckCuda(__LINE__);
    /*##comp-warm-end##*/
    /*##pre-beg##*/
    //// COMMENTED OUT -- addressed in "NEW" below
    //byte* d_preencdata;
    //cudaMalloc((void **)&d_preencdata, insize);
    //cudaMemcpy(d_preencdata, d_input, insize, cudaMemcpyDeviceToDevice);
    //int dpreencsize = insize;
    //// END COMMENTED SECTION
    ////NEW
    byte* d_preencdata;
    cudaMalloc((void **)&d_preencdata, new_insize);
    cudaMemcpy(d_preencdata, d_input, new_insize, cudaMemcpyDeviceToDevice);
    int dpreencsize = new_insize;
    //// END NEW
    /*##pre-warm-beg##*/
    /*##pre-warm-end##*/
    cudaFree(d_preencdata);
    /*##pre-end##*/
  }
  GPUTimer dtimer;
  dtimer.start();
  /*##pre-encoder-beg##*/
  /*##pre-encoder-end##*/
  int* d_fullcarry;
  cudaMalloc((void **)&d_fullcarry, chunks * sizeof(int));
  d_reset<<<1, 1>>>();
  cudaMemset(d_fullcarry, 0, chunks * sizeof(byte));
  //d_encode<<<blocks, TPB>>>(dpreencdata, dpreencsize, d_encoded, d_encsize, d_fullcarry);
  d_encode<<<blocks, TPB>>>(dpreencdata, d_ped, d_gain, dpreencsize, d_encoded, d_encsize, d_fullcarry);
  cudaFree(d_fullcarry);
  cudaDeviceSynchronize();
  double runtime = dtimer.stop();

  // get encoded GPU result
  int dencsize = 0;
  cudaMemcpy(&dencsize, d_encsize, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(dencoded, d_encoded, dencsize, cudaMemcpyDeviceToHost);
  printf("encoded size: %d bytes\n", dencsize);
  CheckCuda(__LINE__);
  
  //// COMMENTED OUT -- addressed in "NEW" below
  //const float CR = (100.0 * dencsize) / insize;
  //printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  //if (perf) {
  //  printf("encoding time: %.6f s\n", runtime);
  //  double throughput = insize * 0.000000001 / runtime;
  //  printf("encoding throughput: %8.3f Gbytes/s\n", throughput);
  //  CheckCuda(__LINE__);
  //}
  //// END COMMENTED SECTION
  //// NEW
  const float CR = (100.0 * dencsize) / new_insize;
  printf("ratio: %6.2f%% %7.3fx\n", CR, 100.0 / CR);

  if (perf) {
    printf("encoding time: %.6f s\n", runtime);
    double throughput = new_insize * 0.000000001 / runtime;
    printf("encoding throughput: %8.3f Gbytes/s\n", throughput);
    CheckCuda(__LINE__);
  }
  //// END NEW

  // write to file
  FILE* const fout = fopen(argv[2], "wb");
  fwrite(dencoded, 1, dencsize, fout);
  fclose(fout);

  // clean up GPU memory
  cudaFree(d_input);
  cudaFree(d_encoded);
  cudaFree(d_encsize);
  CheckCuda(__LINE__);

  // clean up
  delete [] input;
  cudaFreeHost(dencoded);
  return 0;
}
