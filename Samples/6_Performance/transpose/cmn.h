#ifndef CMN_H_
#define CMN_H_

#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/core/fast_div.h"
#include "dali/core/convert.h"

// #include "dbg.h"

namespace dali {

namespace kernels {

static constexpr int kBlockSizeMul = 24;
static constexpr int kBlockWidth = 128;
static constexpr int kStaticChannels = 3;

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

template <typename Out, typename In>
struct SimpleSampleDesc {
  Out *__restrict__ out;
  const In *__restrict__ in;
  int H, W, C;
  // didn't work
  //  TensorShape<3> shape;

  const float *__restrict__ norm_add;
  const float *__restrict__ norm_mul;
  const void *__restrict__ fill_values;

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
        sample.out(x, y, c) = ConvertSat<Out>(fpout);
      }
    }
  }
}

template <int static_channels, typename Out, typename In>
__device__ void sort_channels_hwc_to_chw(const SimpleSampleDesc<Out, In> &sample, const BlockDesc<1> &tile) {

}


template <typename Out, typename In>
__global__ void SortChannels(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  // SliceNormalizeKernel_2D_NoPad_Ch<static_channels>(sample, tile);
  int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  for (int64_t idx = threadIdx.x + block.start.x; idx < block.end.x; idx += blockDim.x) {
    // todo fast_div
    int y = idx / y_stride;
    int y_rem = idx - y * y_stride;
    int x = y_rem / sample.C;
    int c = y_rem - x * sample.C;
    if (y < sample.H && x < sample.W) {
      float fpin = sample.in[idx];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + y * sample.W + x] = ConvertSat<Out>(fpout);
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsFastDiv(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  // SliceNormalizeKernel_2D_NoPad_Ch<static_channels>(sample, tile);
  int y_stride_ = sample.shape[1] * sample.shape[2];  // TODO: make channels always 3?
  fast_div<uint32_t> y_stride(y_stride_);
  fast_div<uint32_t> x_stride(sample.shape[2]);
  for (uint32_t idx = threadIdx.x + block.start.x; idx < block.end.x; idx += blockDim.x) {
    // todo fast_div
    uint32_t y, y_rem, x, c;
    y = div_mod(y_rem, idx, y_stride);
    x = div_mod(c, y_rem, x_stride);
    if (y < sample.shape[0] && x < sample.shape[1]) {
      float fpin = sample.in[idx];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.shape[0] * sample.shape[1] + y * sample.shape[1] + x] = ConvertSat<Out>(fpout);
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsSharedIn(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ In tile[kStaticChannels][(kBlockSizeMul / kStaticChannels) * kBlockWidth];
  int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  for (int64_t idx = threadIdx.x + block.start.x, base_x = threadIdx.x; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
    // todo fast_div
    int c = idx % sample.C;
    tile[c][base_x / sample.C] = sample.in[idx];
  }
  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[c][base_x];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreload(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ In tile[kBlockSizeMul * kBlockWidth];
  int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  for (int64_t idx = threadIdx.x + block.start.x, base_x = threadIdx.x; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
    // todo fast_div
    // int c = idx % sample.C;
    tile[base_x] = sample.in[idx];
  }
  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}

// Works only on aligned data!!!
template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloat(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ In tile[kBlockSizeMul * kBlockWidth];
  int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  uint32_t *tmp = reinterpret_cast<uint32_t*>(tile);
  const uint32_t *in = reinterpret_cast<const uint32_t*>(sample.in);
  for (int64_t idx = threadIdx.x + block.start.x / 4, base_x = threadIdx.x; idx < block.end.x / 4; idx += blockDim.x, base_x += blockDim.x) {
    // todo fast_div
    // int c = idx % sample.C;
    // tile[base_x] = sample.in[idx];
    tmp[base_x] = in[idx];
  }
  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatCond(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ In tile[kBlockSizeMul * kBlockWidth];
  int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  if ((reinterpret_cast<std::uintptr_t>(sample.in + block.start.x)) % 4 == 0) {
    uint32_t *tmp = reinterpret_cast<uint32_t*>(tile);
    const uint32_t *in = reinterpret_cast<const uint32_t*>(sample.in + block.start.x);
    for (int64_t idx = threadIdx.x + block.start.x / 4, base_x = threadIdx.x; idx < block.end.x / 4; idx += blockDim.x, base_x += blockDim.x) {
      // todo fast_div
      // int c = idx % sample.C;
      // tile[base_x] = sample.in[idx];
      tmp[base_x] = in[base_x];
    }

    int iters = (block.end.x / 4 - block.start.x / 4) / blockDim.x;
    // In case we are not divisible by 4, we need to process up to last 3 elements
    // loop iters - we can count them, or have a counter, check which is faster
    // TODO(klecki): Introduce alignment and padding, at least on the per-sample basis.

    int64_t last_read = (block.end.x / 4) * 4 - 1;
    int64_t last_written = last_read - block.start.x;

    for (int64_t idx = threadIdx.x + last_read, base_x = threadIdx.x + last_written; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
      // todo fast_div
      // int c = idx % sample.C;
      tile[base_x] = sample.in[idx];
    }

  } else {
    for (int64_t idx = threadIdx.x + block.start.x, base_x = threadIdx.x; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
      // todo fast_div
      // int c = idx % sample.C;
      tile[base_x] = sample.in[idx];
    }
  }
  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}



//  THIS VERSION SKIPS THE TAIL DATA LOAD DUE TO PADDING ISSUE TO MEASURE THE PERF IMPACT
template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatCondWrong(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ In tile[kBlockSizeMul * kBlockWidth];
  int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  if ((reinterpret_cast<std::uintptr_t>(sample.in + block.start.x)) % 4 == 0) {
    uint32_t *tmp = reinterpret_cast<uint32_t*>(tile);
    const uint32_t *in = reinterpret_cast<const uint32_t*>(sample.in + block.start.x);
    for (int64_t idx = threadIdx.x + block.start.x / 4, base_x = threadIdx.x; idx < block.end.x / 4; idx += blockDim.x, base_x += blockDim.x) {
      // todo fast_div
      // int c = idx % sample.C;
      // tile[base_x] = sample.in[idx];
      tmp[base_x] = in[base_x];
    }

    //  THIS PART IS COMMENTED OUT TO MEASURE THEORETICAL PERF IMPACT
    // int64_t last_read = (block.end.x / 4) * 4 - 1;
    // int64_t last_written = last_read - block.start.x;

    // for (int64_t idx = threadIdx.x + last_read, base_x = threadIdx.x + last_written; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
    //   // todo fast_div
    //   // int c = idx % sample.C;
    //   tile[base_x] = sample.in[idx];
    // }

  } else {
    for (int64_t idx = threadIdx.x + block.start.x, base_x = threadIdx.x; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
      // todo fast_div
      // int c = idx % sample.C;
      tile[base_x] = sample.in[idx];
    }
  }
  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ In tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + block.start.x);
  auto aligned_in_start = align_up(in_start, 32 * 4);
  auto bytes_skipped = aligned_in_start - in_start;

  In *aligned_tile = tile + 32 * 4;
  In *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + block.start.x;


  uint32_t *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uint32_t *aligned_in_u32 = reinterpret_cast<const uint32_t*>(sample.in + block.start.x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = block.end.x - block.start.x - bytes_skipped;

  // aligned load
  for (int64_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    aligned_tile_u32[idx] = aligned_in_u32[idx];
  }

  // epilogue
  In *epilogue_tile = reinterpret_cast<In*>(aligned_tile_u32 + left_after_prologue / 4);
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_u32 + left_after_prologue / 4);

  int64_t left_after_main = left_after_prologue - (left_after_prologue /  4) * 4;
  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = prologue_tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      // printf("%f %f\n", fpout, fpout);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueF32(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ float tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + block.start.x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  float *aligned_tile = tile + 32 * 4;
  float *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + block.start.x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + block.start.x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = block.end.x - block.start.x - bytes_skipped;

  // aligned load
  for (int64_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    uchar4 in = aligned_in_char4[idx];
    aligned_tile[idx * 4 + 0] = in.x;
    aligned_tile[idx * 4 + 1] = in.y;
    aligned_tile[idx * 4 + 2] = in.z;
    aligned_tile[idx * 4 + 3] = in.w;
  }

  int64_t processed_in_main = (left_after_prologue /  4) * 4;
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  float *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = prologue_tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueF32JustRead(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ float tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + block.start.x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  float *aligned_tile = tile + 32 * 4;
  float *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + block.start.x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + block.start.x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = block.end.x - block.start.x - bytes_skipped;

  // aligned load
  for (int64_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    uchar4 in = aligned_in_char4[idx];
    aligned_tile[idx * 4 + 0] = in.x;
    aligned_tile[idx * 4 + 1] = in.y;
    aligned_tile[idx * 4 + 2] = in.z;
    aligned_tile[idx * 4 + 3] = in.w;
  }

  int64_t processed_in_main = (left_after_prologue /  4) * 4;
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  float *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  int64_t out_offset = block.start.x / kStaticChannels;
  for (int64_t idx = threadIdx.x; idx < block.end.x - block.start.x; idx += blockDim.x) {
    int64_t plane_offset = idx / kStaticChannels;
    int c = idx - plane_offset * kStaticChannels;
    float fpin = prologue_tile[idx];
    float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
    sample.out[c * sample.H * sample.W + plane_offset + out_offset] = ConvertSat<Out>(fpout);
  }

  // // idx is not divided by the static channels (mostly the block.start.x)
  // for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
  //   idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
  //     sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
  //   }
  // }
}


// Worse
template <typename Out, typename In>
__global__ void SortChannelsSharedOut(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ Out tile[kStaticChannels][(kBlockSizeMul / kStaticChannels) * kBlockWidth];
  int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  for (int64_t idx = threadIdx.x + block.start.x, base_x = threadIdx.x; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
    // todo fast_div
    int c = idx % sample.C;
    tile[c][base_x / sample.C] = sample.in[idx];
  }
  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
       idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {

    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[c][base_x];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}

// Similar perf to the original
template <typename Out, typename In>
__global__ void SortChannelsInPlace0(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  // __shared__ Out tile[kStaticChannels][(kBlockSizeMul / kStaticChannels) * kBlockWidth];
  // int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  // for (int64_t idx = threadIdx.x + block.start.x, base_x = threadIdx.x; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
  //   // todo fast_div
  //   tile[c][base_x / sample.C] = sample.in[idx];
  // }
  // __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
       idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = sample.in[idx * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsInPlace1(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  // __shared__ Out tile[kStaticChannels][(kBlockSizeMul / kStaticChannels) * kBlockWidth];
  // int y_stride = sample.W * sample.C;  // TODO: make channels always 3?
  // for (int64_t idx = threadIdx.x + block.start.x, base_x = threadIdx.x; idx < block.end.x; idx += blockDim.x, base_x += blockDim.x) {
  //   // todo fast_div
  //   tile[c][base_x / sample.C] = sample.in[idx];
  // }
  // __syncthreads();

  float fpin[kStaticChannels];
  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
       idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {



    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      fpin[c] = sample.in[idx * sample.C + c];
    }
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      // float fpin = sample.in[idx * sample.C + c];
      float fpout = fmaf(fpin[c], sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
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

static constexpr int /*H = 999, W = 999, C = 3, */ NUM_ITERS = 100;

using input_t = uint8_t;
using output_t = float;

void RunSN(int num_samples, input_t *input, output_t *output, float *norm_add, float *norm_mul, cudaStream_t stream, int id, int H, int W, int C) {
  constexpr int spatial_ndim = 2;
  constexpr int channel_dim = 2;
  using Out = output_t;
  using In = input_t;
  using Tile = kernels::BlockDesc<spatial_ndim>;
  using Sample = SampleDesc<Out, In, spatial_ndim>;
  using CollapsedBlock = kernels::BlockDesc<1>;
  using SimpleSample = SimpleSampleDesc<Out, In>;

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
  TensorListShape<1> collapsed_shape = collapse_dims<1>(shape, {{0, 3}});


  for (int i = 0; i < num_samples; i++) {
    sample_descs[i].in = {input + i * img_size, in_size, C, in_stride, in_ch_stride};
    sample_descs[i].out = {output + i * img_size, out_size, C, out_stride, out_ch_stride};
    // sample_descs[i].bounds = bounds;
    sample_descs[i].norm_add = norm_add + i * C;
    sample_descs[i].norm_mul = norm_mul + i * C;
    // sample_descs[i].fill_values = fill_values_gpu + i * out_nchannels_; unused here

  }

  std::vector<SimpleSample> simple_sample_desc(num_samples);
  for (int i = 0; i < num_samples; i++) {
    simple_sample_desc[i].in = input + i * img_size;
    simple_sample_desc[i].out = output + i * img_size;
    simple_sample_desc[i].H = H;
    simple_sample_desc[i].W = W;
    simple_sample_desc[i].C = C;
    // doesn't work, I thought it would, but it does not
    // simple_sample_desc[i].shape = {H, W, C};
    simple_sample_desc[i].norm_add = norm_add + i * C;
    simple_sample_desc[i].norm_mul = norm_mul + i * C;

  }



  BlockSetup<1, -1> collapsed_block_setup(1); // why do I need to set this multiplier here? xDDDD
  collapsed_block_setup.SetDefaultBlockSize({kBlockSizeMul * kBlockWidth});
  collapsed_block_setup.SetBlockDim(dim3{128, 1, 1});
  collapsed_block_setup.SetupBlocks(collapsed_shape, true);
  auto collapsed_blocks_cpu = collapsed_block_setup.Blocks();
  auto collapsed_grid_dim = collapsed_block_setup.GridDim();
  auto collapsed_block_dim = collapsed_block_setup.BlockDim();


  std::vector<BlockDesc<1>> collapsed_blocks_aligned_manual;
  constexpr int kTileSize = kBlockSizeMul * kBlockWidth;
  for (int sample_idx = 0; sample_idx < collapsed_shape.num_samples(); sample_idx++) {
    int64_t len = collapsed_shape.tensor_shape(sample_idx)[0];
    auto sample_base = reinterpret_cast<uintptr_t>(simple_sample_desc[sample_idx].in);
    int start_alignment = sample_base % 4;  // TODO(sizeof(uint32_t))

    // We can start at byte 0, 1, 2, 3, we want to move by the multiple of 3 channels to the
    // 4-byte boundary, so we can move by, 0, 3, 6 or 9 respectively. and we
    // add some kBlockWidth * 6 elements to do some processing in the first tile.
    // All other tiles are divisible by 3 and 4, so they are both aligned with 4-byte boundary
    // and with channels.
    static_assert(kBlockWidth % 4 == 0, "Block is already a multiple of alignment");
    int first_tile_from_alignment[4] = {0, 3 + kBlockWidth * 6, 6 + kBlockWidth * 6, 9 + kBlockWidth * 6};

    // auto aligned_start = align_up(sample_base, kTileSize);
    auto first_tile_size = first_tile_from_alignment[start_alignment];
    BlockDesc<1> blk;
    blk.sample_idx = sample_idx;
    blk.start[0] = 0;
    if (first_tile_size) {
      blk.end[0] = first_tile_size;
      collapsed_blocks_aligned_manual.push_back(blk);
    }
    for (int64_t start = first_tile_size; start < len; start += kTileSize) {
      blk.start[0] = start;
      blk.end[0] = std::min(start + kTileSize, len);
      collapsed_blocks_aligned_manual.push_back(blk);
    }
  }




  // std::cout << kBlockSizeMul * kBlockWidth << std::endl;

  // for (auto &block : collapsed_blocks_cpu) {
  //   std::cout << block.start << " " << block.end << std::endl;
  // }

  // std::cout << "(" << collapsed_grid_dim.x << ", " << collapsed_grid_dim.y << ", " << collapsed_grid_dim.z << ") "
  //           << "(" << collapsed_block_dim.x << ", " << collapsed_block_dim.y << ", " << collapsed_block_dim.z << ") " << std::endl;

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


  SimpleSample *simple_samples_gpu = nullptr;
  CollapsedBlock *collapsed_blocks_gpu = nullptr;



  cudaMalloc((void **)&simple_samples_gpu, num_samples * sizeof(SimpleSample));
  cudaMemcpy(simple_samples_gpu, simple_sample_desc.data(), num_samples * sizeof(SimpleSample), cudaMemcpyHostToDevice);

  constexpr bool USE_ALIGNMENT = true;
  if (USE_ALIGNMENT) {
    cudaMalloc((void **)&collapsed_blocks_gpu, collapsed_blocks_aligned_manual.size() * sizeof(CollapsedBlock));
    cudaMemcpy(collapsed_blocks_gpu, collapsed_blocks_aligned_manual.data(), collapsed_blocks_aligned_manual.size() * sizeof(CollapsedBlock), cudaMemcpyHostToDevice);

    // TODO DODODODODO: VERY IMPORTANT, ALIGN THE GRID:
    collapsed_grid_dim = dim3(collapsed_blocks_aligned_manual.size(), 1, 1);

  } else {

    cudaMalloc((void **)&collapsed_blocks_gpu, collapsed_blocks_cpu.size() * sizeof(CollapsedBlock));
    cudaMemcpy(collapsed_blocks_gpu, collapsed_blocks_cpu.data(), collapsed_blocks_cpu.size() * sizeof(CollapsedBlock), cudaMemcpyHostToDevice);
  }

  // printf("Running kernel: (%d %d %d) x (%d %d %d)\n",
  //       collapsed_grid_dim.x, collapsed_grid_dim.y, collapsed_grid_dim.z,
  //       collapsed_block_dim.x, collapsed_block_dim.y, collapsed_block_dim.z);

  if (id == 0) {
    SliceNormalizeKernel_2D_NoPad<Out, In>
          <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, tiles_gpu);
  }
  // if (id == 1)
  //   SortChannels<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  // if (id == 2)
  //   // SortChannelsFastDiv<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  if (id == 1) {
    SortChannelsSharedIn<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 2) {
    SortChannelsSharedOut<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 3) {
    SortChannelsInPlace0<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 4) {
    SortChannelsInPlace1<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 5) {
    SortChannelsSharedPreload<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 6) {
    SortChannelsSharedPreloadFloat<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 7) {
    SortChannelsSharedPreloadFloatCond<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 8) {
    SortChannelsSharedPreloadFloatCondWrong<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 9) {
    SortChannelsSharedPreloadFloatPrologueEpilogue<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 10) {
    SortChannelsSharedPreloadFloatPrologueEpilogueF32<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 11) {
    SortChannelsSharedPreloadFloatPrologueEpilogueF32JustRead<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }


  // CUDA events
  cudaEvent_t start, stop;
  // initialize events
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start, stream);

  for (int i = 0; i < NUM_ITERS; i++) {
    if (id == 0) {
      SliceNormalizeKernel_2D_NoPad<Out, In>
            <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, tiles_gpu);
    }
    // if (id == 1)
    //   SortChannels<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    // if (id == 2)
    //   // SortChannelsFastDiv<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    if (id == 1) {
      SortChannelsSharedIn<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 2) {
      SortChannelsSharedOut<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 3) {
      SortChannelsInPlace0<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 4) {
      SortChannelsInPlace1<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 5) {
      SortChannelsSharedPreload<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 6) {
      SortChannelsSharedPreloadFloat<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 7) {
      SortChannelsSharedPreloadFloatCond<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 8) {
      SortChannelsSharedPreloadFloatCondWrong<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 9) {
      SortChannelsSharedPreloadFloatPrologueEpilogue<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 10) {
      SortChannelsSharedPreloadFloatPrologueEpilogueF32<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 11) {
      SortChannelsSharedPreloadFloatPrologueEpilogueF32JustRead<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
  }


  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  float kernelTime;
  checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, stop));

      // report effective bandwidths
  float kernelBandwidth = num_samples * 1000.0f * (H * W * C * (sizeof(uint8_t) + sizeof(float))) / (1024 * 1024 * 1024) /
    (kernelTime / NUM_ITERS);
  printf(
  "CMN %d, Throughput = %.4f GB/s, Time = %.5f ms, Size = %d x %d x %d\n",
  id, kernelBandwidth, kernelTime / NUM_ITERS, H, W, C);


  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  cudaFree(samples_gpu);
  cudaFree(tiles_gpu);

}

template <typename T>
void print_planes(T *data, int H, int W, int C) {
  for (int c = 0; c < C; c++) {
    printf("\n\nPlane: %d: =============\n", c);
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        printf("%.0f, ", static_cast<float>(data[c * H * W + y * W + x]));
      }
      printf("|||\n");
    }
  }
}

void prepare_and_run(int num_samples, int H, int W, int C) {
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

  std::vector<output_t> gold_cpu;
  gold_cpu.resize(H * W * C);
  for (int c = 0; c < C; c++) {
    for (int i = 0; i < H * W; i++) {
      gold_cpu[c * H*W + i] = ConvertSat<float>((c + i * C) % 256);
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
  for (int id = 0; id < 12; id++) {
    if (id == 6 && H * W * C % 4) {
      printf("Unaligned version, skipping 6\n");
      continue;
    }
    cudaMemset(output_gpu, 0, num_samples * sizeof(output_t) * H * W * C);
    RunSN(num_samples, input_gpu, output_gpu, norm_add_gpu, norm_mul_gpu, stream, id, H, W, C);

    cudaMemcpy(output_cpu.data(), output_gpu, sizeof(output_t) * num_samples *  H * W * C, cudaMemcpyDeviceToHost);
    if (id == 8) {
      printf("Skipping the check for 8, trailing data is wrong\n");
      continue;
    }
    for (int i = 0; i < num_samples; i++) {
      bool res = compareData(gold_cpu.data(), output_cpu.data() + i * H * W * C, H * W * C, 0.01f, 0.0f);
      // printf("Expected: %d, sample %d\n", id, i);
      // print_planes(gold_cpu.data(), H, W, C);
      // printf("\n\n=============================================================\nComputed: %d, sample %d\n", id, i);
      // print_planes(output_cpu.data() + i * H * W * C, H, W, C);
      if (res == false) {
        printf("*** %s kernel FAILED ***\n", "CMN");
      }
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
