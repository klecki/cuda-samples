#ifndef CMN_H_
#define CMN_H_

#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/common/block_setup.h"

namespace dali {

namespace kernels {

template <typename Out, typename In, int spatial_ndim>
struct SampleDesc {
  Surface2D<Out> out;
  Surface2D<const In> in;
  Roi<spatial_ndim> bounds;

  const float *__restrict__ norm_add;
  const float *__restrict__ norm_mul;
  const void *__restrict__ fill_values;

  // i64vec<spatial_ndim> out_strides;
  // int64_t out_channel_stride;
  // i64vec<spatial_ndim> in_strides;
  // int64_t in_channel_stride;
};



template <int static_channels, typename Out, typename In>
__device__ void SliceNormalizeKernel_2D_NoPad_Ch(const SampleDesc<Out, In, 2> &sample,
                                                 const BlockDesc<2> &tile) {
  auto fill_values = static_cast<const Out *>(sample.fill_values);
  for (int y = threadIdx.y + tile.start.y; y < tile.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + tile.start.x; x < tile.end.x; x += blockDim.x) {
      #pragma unroll static_channels
      for (int c = 0; c < static_channels; c++) {
        float fpin = sample.in(x, y, c);
        float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
        sample.out(x, y, c) = fpout;  //TODO//ConvertSat<Out>(fpout);
      }
    }
  }
}

template <typename Out, typename In>
__global__ void SliceNormalizeKernel_2D_NoPad(const SampleDesc<Out, In, 2> *samples,
                                              const BlockDesc<2> *tiles) {
  const auto tile = tiles[blockIdx.x];
  const auto sample = samples[tile.sample_idx];
  constexpr int static_channels = 3;
  SliceNormalizeKernel_2D_NoPad_Ch<static_channels>(sample, tile);
}

static constexpr int H = 1000, W = 1000, C = 3, NUM_ITERS = 100;

using input_t = uint8_t;
using output_t = float;

void RunSN(int num_samples, uint8_t *input, float *output, float *norm_add, float *norm_mul, cudaStream_t stream) {
  constexpr int spatial_ndim = 2;
  constexpr int channel_dim = 2;
  using Out = float;
  using In = uint8_t;
  using Tile = kernels::BlockDesc<spatial_ndim>;
  using Sample = SampleDesc<Out, In, spatial_ndim>;

  std::vector<Sample> sample_descs;
  sample_descs.resize(num_samples);

  // This goes by x, y - inverted compared to the shape.
  vec<spatial_ndim, int32_t> in_size = {W, H};
  vec<spatial_ndim, int64_t> in_stride = {C, W * C};
  int in_ch_stride = 1;
  int img_size = H * W * C;


  vec<spatial_ndim, int32_t> out_size = {W, H};
  vec<spatial_ndim, int64_t> out_stride = {1, W}; // HMMM, WTH
  int out_ch_stride = H * W;


  // Roi<2> bounds = {ivec{0, 0}, out_size};

  TensorListShape<3> shape = uniform_list_shape(num_samples, TensorShape<3>{H, W, C});


  for (int i = 0; i < num_samples; i++) {
    sample_descs[i].in = {input + i * img_size, in_size, C, in_stride, in_ch_stride};
    sample_descs[i].out = {output + i * img_size, out_size, C, out_stride, out_ch_stride};
    // sample_descs[i].bounds = bounds;
    sample_descs[i].norm_add = norm_add + i * C;
    sample_descs[i].norm_mul = norm_mul + i * C;
    // sample_descs[i].fill_values = fill_values_gpu + i * out_nchannels_; unused here

  }




  BlockSetup<spatial_ndim, channel_dim> block_setup;
  block_setup.SetDefaultBlockSize({64, 64});
  block_setup.SetBlockDim(dim3(32, 32, 1));
  // TODO CHW vs HWC?
  // block_setup.SetupBlocks(out_shape_orig_, true);
  block_setup.SetupBlocks(shape, true);
  auto tiles_cpu = block_setup.Blocks();
  auto grid_dim = block_setup.GridDim();
  auto block_dim = block_setup.BlockDim();




  Sample *samples_gpu = nullptr;
  Tile *tiles_gpu = nullptr;

  cudaMalloc((void **)&samples_gpu, num_samples * sizeof(Sample));
  cudaMalloc((void **)&tiles_gpu, tiles_cpu.size() * sizeof(Tile));
  cudaMemcpy(samples_gpu, sample_descs.data(), num_samples * sizeof(Sample), cudaMemcpyHostToDevice);
  cudaMemcpy(tiles_gpu, tiles_cpu.data(), tiles_cpu.size() * sizeof(Tile), cudaMemcpyHostToDevice);


  SliceNormalizeKernel_2D_NoPad<Out, In>
    <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, tiles_gpu);


  // CUDA events
  cudaEvent_t start, stop;
  // initialize events
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start, stream);

  for (int i = 0; i < NUM_ITERS; i++) {
    SliceNormalizeKernel_2D_NoPad<Out, In>
      <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, tiles_gpu);
  }


  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  float kernelTime;
  checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, stop));

      // report effective bandwidths
  float kernelBandwidth = num_samples * 1000.0f * (H * W * C * (sizeof(uint8_t) + sizeof(float))) / (1024 * 1024 * 1024) /
    (kernelTime / NUM_ITERS);
  printf(
  "transpose %s, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 "
  "elements, NumDevsUsed = %u, Workgroup = %u\n",
  "CMN", kernelBandwidth, kernelTime / NUM_ITERS, (64 * 64),
  1, 32 * 32);


  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  cudaFree(samples_gpu);
  cudaFree(tiles_gpu);

}

void prepare_and_run(int num_samples) {
  input_t *input_gpu;
  output_t *output_gpu;
  float *norm_add_gpu, *norm_mul_gpu;
  cudaMalloc((void **)&input_gpu, num_samples * sizeof(input_t) * H * W * C);
  cudaMalloc((void **)&output_gpu, num_samples * sizeof(output_t) * H * W * C);
  cudaMalloc((void **)&norm_add_gpu, num_samples * sizeof(float) * C);
  cudaMalloc((void **)&norm_mul_gpu, num_samples * sizeof(float) * C);

  std::vector<float> norm_add(num_samples * C, 0.f), norm_mul(num_samples * C, 1.f);
  cudaMemcpy(norm_add_gpu, norm_add.data(), num_samples * sizeof(float) * C, cudaMemcpyHostToDevice);
  cudaMemcpy(norm_mul_gpu, norm_mul.data(), num_samples * sizeof(float) * C, cudaMemcpyHostToDevice);


  // prepare input 0, 1, 2, 3, 4...
  // prepare gold output 0,3,6.... 1,4,7,... 2,5,8,...
  std::vector<input_t> input_cpu;
  input_cpu.resize(H * W * C);
  for (int i = 0; i < H * W * C; i++) {
    input_cpu[i] = i;
  }

  std::vector<input_t> gold_cpu;
  gold_cpu.resize(H * W * C);
  for (int c = 0; c < C; c++) {
    for (int i = 0; i < H * W; i++) {
      gold_cpu[c * H*W + i] = c + i * C;
    }
  }

  cudaMemcpy(input_gpu, input_cpu.data(), sizeof(input_t) * H * W * C, cudaMemcpyHostToDevice);
  for (int i = 1; i < num_samples; i++) {
    cudaMemcpy(input_gpu + i * H * W * C, input_gpu, sizeof(input_t) * H * W * C, cudaMemcpyDeviceToDevice);
  }


  std::vector<output_t> output_cpu;
  output_cpu.resize(num_samples * H * W * C);


  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  RunSN(num_samples, input_gpu, output_gpu, norm_add_gpu, norm_mul_gpu, stream);

  cudaMemcpy(output_cpu.data(), output_gpu, sizeof(output_t) * num_samples *  H * W * C, cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_samples; i++) {
    bool res = compareData(gold_cpu.data(), output_cpu.data() + i * H * W * C, H * W * C, 0.01f, 0.0f);
    if (res == false) {
      printf("*** %s kernel FAILED ***\n", "CMN");
    }
  }

  cudaFree(input_gpu);
  cudaFree(output_gpu);
  cudaFree(norm_add_gpu);
  cudaFree(norm_mul_gpu);
  cudaStreamDestroy(stream);
}


}
}


#endif  // CMN_H
