#ifndef CMN_H_
#define CMN_H_

#include "dali/kernels/imgproc/surface.h"
#include "dali/kernels/imgproc/roi.h"
#include "dali/core/geom/vec.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/core/fast_div.h"
#include "dali/core/convert.h"

#include <cuda/pipeline>
#include <cooperative_groups.h>

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
  const float *__restrict__ norm_add;
  const float *__restrict__ norm_mul;
  const void *__restrict__ fill_values;

  int64_t sample_size;
  uint32_t first_block;

  int H, W, C;
  int input_W, input_C;


  // TODO(klecki): This is specifically less generic to save on compute and memsize:
  // We support only cropping in X, as cropping in Y can be done on tile level.
  // cropping channels is even weirder, but we can easily support it by adjusting the
  // second loop range (the one writing output).


  bool mirror;
};


template <typename Out, typename In>
inline __device__ uint32_t FindSampleIdx(const SimpleSampleDesc<Out, In> *samples,
                                         uint32_t num_samples) {
  uint32_t i = 0;
  for (uint32_t jump = (1 << (32 - __clz(num_samples) - 1)); jump; jump >>= 1) {
    if (i + jump < num_samples && samples[i + jump].first_block <= blockIdx.x)
      i += jump;
  }
  return i;
}

inline __device__ uint32_t FindSampleIdx(uint32_t *first_blocks,
                                         uint32_t num_samples) {
  uint32_t i = 0;
  for (uint32_t jump = (1 << (32 - __clz(num_samples) - 1)); jump; jump >>= 1) {
    if (i + jump < num_samples && first_blocks[i + jump] <= blockIdx.x)
      i += jump;
  }
  return i;
}


template <typename Out, typename In>
__global__ void SliceNormalizeKernel_2D(const SampleDesc<Out, In, 2> *samples,
                                        const ::dali::kernels::BlockDesc<2> *tiles) {
  const auto tile = tiles[blockIdx.x];
  const auto sample = samples[tile.sample_idx];
  auto fill_values = static_cast<const Out *>(sample.fill_values);
  for (int y = threadIdx.y + tile.start.y; y < tile.end.y; y += blockDim.y) {
    for (int x = threadIdx.x + tile.start.x; x < tile.end.x; x += blockDim.x) {
      int c = 0;
      if (!sample.bounds.contains(ivec2{x, y})) {
        for (; c < sample.out.channels; c++) {
          sample.out(x, y, c) = fill_values[c];
        }
      } else {
        for (; c < sample.in.channels; c++) {
          float fpin = sample.in(x, y, c);
          float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
          sample.out(x, y, c) = ConvertSat<Out>(fpout);
        }
        for (; c < sample.out.channels; c++) {
          sample.out(x, y, c) = fill_values[c];
        }
      }
    }
  }
}

template <int static_channels, typename Out, typename In>
__device__ void SliceNormalizeKernel_2D_NoPad_Ch(const SampleDesc<Out, In, 2> &sample,
                                                 const BlockDesc<2> &tile) {
  auto fill_values = static_cast<const Out *>(sample.fill_values);

  // THIS OPTIMIZATION DOESN'T WORK HERE
  // float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  // #pragma unroll kStaticChannels
  // for (int c = 0; c < kStaticChannels; c++) {
  //   norm_mul[c] = sample.norm_mul[c];
  //   norm_add[c] = sample.norm_add[c];
  // }

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
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];
  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }
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
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + y * sample.W + x] = ConvertSat<Out>(fpout);
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsFastDiv(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];
  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }
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
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.shape[0] * sample.shape[1] + y * sample.shape[1] + x] = ConvertSat<Out>(fpout);
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsSharedIn(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];
  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }
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
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueF32HWC2CHW(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(samples, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);


  __shared__ float tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  float *aligned_tile = tile + 32 * 4;
  float *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
    idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = prologue_tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }

  // // idx is not divided by the static channels (mostly the start_x)
  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
}



template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueF32(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);


  __shared__ float tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  float *aligned_tile = tile + 32 * 4;
  float *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)
  for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
    idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

    fast_div<uint32_t> channel(kStaticChannels);
    int c = idx % channel;
    float fpin = prologue_tile[base_x];
    float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
    sample.out[idx] = ConvertSat<Out>(fpout);
  }
}
template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueU8(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ uint8_t tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  uint8_t *aligned_tile = tile + 32 * 4;
  uint8_t *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  uint8_t *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
    idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

    fast_div<uint32_t> channel(kStaticChannels);
    int c = idx % channel;
    float fpin = prologue_tile[base_x];
    float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
    sample.out[idx] = ConvertSat<Out>(fpout);
  }

}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueU8_Pad(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ uint8_t tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  uint8_t *aligned_tile = tile + 32 * 4;
  uint8_t *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  uint8_t *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < end_x_padded; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      float fpin = prologue_tile[base_offset * sample.input_C + c];
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_half(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ __half tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  __half *aligned_tile = tile + 32 * 4;
  __half *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  __half *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < end_x_padded; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      __half fpin = prologue_tile[base_offset * sample.input_C + c];
      __half fpout = __hfma(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_half_unroll(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ __half tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  __half *aligned_tile = tile + 32 * 4;
  __half *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  __half *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded / 4, base_x = threadIdx.x;
    idx < end_x_padded / 4; idx += blockDim.x, base_x += blockDim.x) {


    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      __half fpin = prologue_tile[base_x * sample.input_C + c];
      __half fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
    }

    for (int c = kStaticChannels; c < 4; c++) {
      sample.out[idx * sample.C + c] = 42.f + c;
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_half2_Pad_half(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ __half tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  __half2 *aligned_tile = reinterpret_cast<__half2*>(tile + 32 * 4);
  __half *prologue_tile = tile + 32 * 4 - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

  // aligned load
  for (int64_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    uchar4 in = aligned_in_char4[idx];
    aligned_tile[idx * 2 + 0] = make_half2(in.x, in.y);
    aligned_tile[idx * 2 + 1] = make_half2(in.z, in.w);
  }

  int64_t processed_in_main = (left_after_prologue /  4) * 4;
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  __half *epilogue_tile = tile + 32 * 4 + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < end_x_padded; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      __half fpin = prologue_tile[base_offset * sample.input_C + c];
      __half fpout = __hfma(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_half(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ float tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  float *aligned_tile = tile + 32 * 4;
  float *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < end_x_padded; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      __half fpin = prologue_tile[base_offset * sample.input_C + c];
      __half fpout = __hfma(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueU8_Pad_half(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ uint8_t tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  uint8_t *aligned_tile = tile + 32 * 4;
  uint8_t *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  uint8_t *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < end_x_padded; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      __half fpin = prologue_tile[base_offset * sample.input_C + c];
      __half fpout = __hfma(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_half2(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels + 1], norm_add[kStaticChannels + 1];

  __shared__ __half tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }
  norm_mul[3] = 0;
  norm_add[3] = 42.f + 3;

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  __half *prologue_tile = tile;
  __half *aligned_tile = tile + bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

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
  __half *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (int64_t idx = threadIdx.x; idx < left_after_main; idx++) {
    epilogue_tile[idx] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;
  // todo align up
  __half2 *prologue_tile_h2 = reinterpret_cast<__half2*>(prologue_tile);

  __half2 mul_lo = make_half2(norm_mul[0], norm_mul[1]);
  __half2 mul_hi = make_half2(norm_mul[2], norm_mul[3]);
  __half2 add_lo = make_half2(norm_add[0], norm_add[1]);
  __half2 add_hi = make_half2(norm_add[2], norm_add[3]);

  auto out_start = reinterpret_cast<std::uintptr_t>(sample.out + start_x_padded);
  auto aligned_out_start = align_up(out_start, 4);
  auto out_bytes_skipped = aligned_out_start - out_start;

  __half2 *out = reinterpret_cast<__half2*>(sample.out +  out_bytes_skipped / sizeof(__half));
  return;

  for (int64_t base_x = threadIdx.x * 3; base_x < kBlockSize; base_x += blockDim.x) {
    __half2 elems[3];
    #pragma unroll 3
    for (int k = 0; k < 3; k++) {
      elems[k] = prologue_tile_h2[base_x + k];
    }
    // unpack
    __half2 a = elems[0];
    __half2 b = make_half2(elems[1].x, 0);
    __half2 c = make_half2(elems[1].y, elems[2].x);
    __half2 d = make_half2(elems[2].y, 0);

    a = __hfma2(a, mul_lo, add_lo);
    b = __hfma2(b, mul_hi, add_hi);
    c = __hfma2(c, mul_lo, add_lo);
    d = __hfma2(d, mul_hi, add_hi);


    int64_t base_2 = (base_x / 3) * 4;
    out[base_2 + 0] = a;
    out[base_2 + 1] = b;
    out[base_2 + 2] = c;
    out[base_2 + 3] = d;
  }
}

template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueF32Crop(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];
  __shared__ float tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8


  // // calculate the "jump" to previous aligned
  int in_stride = sample.input_W * sample.C;
  int out_stride = sample.W * sample.C;

  // // calculate output coordinate based on the flat indexing
  // int64_t actual_index = idx * 4 + bytes_skipped;
  // int y = actual_index / out_stride;
  // int xc = actual_index - y * out_stride;
  // int x = xc / sample.C;
  // int c = xc - x * sample.C;

  // if y == 0 -> we are in first row, calculate till last viable x or xc?
  // int skip_writes = xc + 3 - out_stride ?
  // if y == 1 -> add just the offset from first row, to back-alginment, skip preceding writes :C
  // if y > 1 -> do the same, but probably with uniform skipping?

  // for now, it looks super complicated. maybe let's do a width-loop?


  // recalculate source offset


  // left in row:
  // int out_stride
  // maybe it is actually easier to have a block loop
  int y_start = block.start.x / out_stride;
  int y_end = block.end.x / out_stride + 1;

  int xc_start = block.start.x - y_start * out_stride;

  float *tile_row = tile;

  for (int y = y_start; y < y_end; y++) {
    int xc_start, xc_end;

    if (y == y_start) {
      xc_start = block.start.x - y_start * out_stride;

    } else {
      xc_start = 0;
    }

    if (y == y_end - 1) {
      xc_end = block.end.x - (y_end - 1) * out_stride;  // + 1? - nope
    } else {
      xc_end = out_stride;
    }

    const In *prologue_in = sample.in + y * in_stride + xc_start;

    auto in_start = reinterpret_cast<std::uintptr_t>(prologue_in);
    // align to 4
    auto aligned_in_start = align_up(in_start, 4);
    auto bytes_skipped = aligned_in_start - in_start;

    float *prologue_tile = tile_row;
    float *aligned_tile = tile_row + bytes_skipped;


    // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
    const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(prologue_in + bytes_skipped);

    // prologue
    for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
      prologue_tile[idx] = prologue_in[idx];
    }

    int64_t left_after_prologue = xc_end - xc_start - bytes_skipped;

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
    tile_row += (xc_end - xc_start);
  }

  // int left_in_row = std::min(block.end.x - block.start.x, );
  // loop over row and reinitialize




  __syncthreads();

  // idx is not divided by the static channels (mostly the block.start.x)
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels, base_x = threadIdx.x;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}



template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogueF32CropCooperative(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];
  __shared__ In tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto group = cooperative_groups::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  constexpr auto stages_count = 2;
  __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);


  // // calculate the "jump" to previous aligned
  int in_stride = sample.input_W * sample.C;
  int out_stride = sample.W * sample.C;


  // left in row:
  // int out_stride
  // maybe it is actually easier to have a block loop
  int y_start = block.start.x / out_stride;
  int y_end = block.end.x / out_stride + 1;

  int xc_start = block.start.x - y_start * out_stride;

  In *tile_row = tile;

  int elements_stage_0, elements_stage_1, total_consumed = 0;

  pipeline.producer_acquire();

  // lazy
  for (int y = y_start; y < y_start + 1; y++) {
    int xc_start, xc_end;

    if (y == y_start) {
      xc_start = block.start.x - y_start * out_stride;

    } else {
      xc_start = 0;
    }

    if (y == y_end - 1) {
      xc_end = block.end.x - (y_end - 1) * out_stride;  // + 1? - nope
    } else {
      xc_end = out_stride;
    }

    const In *prologue_in = sample.in + y * in_stride + xc_start;

    cuda::memcpy_async(group, tile_row, prologue_in, sizeof(In) * (xc_end - xc_start), pipeline);

    tile_row += (xc_end - xc_start);
    elements_stage_0 = (xc_end - xc_start);
  }

  pipeline.producer_commit();

  for (int y = y_start + 1; y < y_end; y++) {

    pipeline.producer_acquire();
    int xc_start, xc_end;

    if (y == y_start) {
      xc_start = block.start.x - y_start * out_stride;

    } else {
      xc_start = 0;
    }

    if (y == y_end - 1) {
      xc_end = block.end.x - (y_end - 1) * out_stride;  // + 1? - nope
    } else {
      xc_end = out_stride;
    }

    const In *prologue_in = sample.in + y * in_stride + xc_start;

    cuda::memcpy_async(group, tile_row, prologue_in, sizeof(In) * (xc_end - xc_start), pipeline);

    tile_row += (xc_end - xc_start);
    elements_stage_1 = (xc_end - xc_start);

    pipeline.producer_commit();

    pipeline.consumer_wait();
    for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels + total_consumed, base_x = threadIdx.x + total_consumed;
      base_x < total_consumed + elements_stage_0 / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
      #pragma unroll kStaticChannels
      for (int c = 0; c < kStaticChannels; c++) {
        float fpin = tile[base_x * sample.C + c];
        float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
        sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
      }
    }
    total_consumed += (elements_stage_0 / kStaticChannels);
    elements_stage_0 = elements_stage_1;
    pipeline.consumer_release();
  }


  pipeline.consumer_wait();
  for (int64_t idx = threadIdx.x + block.start.x / kStaticChannels + total_consumed, base_x = threadIdx.x + total_consumed;
    idx < block.end.x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }

  pipeline.consumer_release();

  // idx is not divided by the static channels (mostly the block.start.x)

}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_halfPlanes(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ __half tile[3][kBlockSizeMul * kBlockWidth / 3 + 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 3*4);  // hmm, can we ensure the pixel?
  auto bytes_skipped = aligned_in_start - in_start;

  // __half *aligned_tile = tile + 32 * 4;
  // __half *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  fast_div<uint32_t> channel(kStaticChannels);
  // prologue
  for (uint32_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    // uint32_t flat_offset = idx;  // start_x is multiple of channels, so w do not need to include it
    uint32_t xy, c;
    xy = div_mod(c, idx, channel);
    tile[c][xy] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

  // aligned load
  for (uint32_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    uint32_t flat_idx = idx * 4 + bytes_skipped;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    uchar4 in = aligned_in_char4[idx];
    tile[c][xy] = in.x;

    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.y;

    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.z;


    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.w;
  }

  int64_t processed_in_main = (left_after_prologue /  4) * 4;
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  // __half *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (uint32_t idx = threadIdx.x; idx < left_after_main; idx++) {
    uint32_t flat_idx = processed_in_main + idx;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    tile[c][xy] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < end_x_padded; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      __half fpin = tile[c][base_offset];
      __half fpout = __hfma(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }
}


template <typename Out, typename In, typename Intermediate>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanes(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels], norm_add[kStaticChannels];

  __shared__ Intermediate tile[3][kBlockSizeMul * kBlockWidth / 3 + 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 3*4);  // hmm, can we ensure the pixel?
  auto bytes_skipped = aligned_in_start - in_start;

  // __half *aligned_tile = tile + 32 * 4;
  // __half *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  fast_div<uint32_t> channel(kStaticChannels);
  // prologue
  for (uint32_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    // uint32_t flat_offset = idx;  // start_x is multiple of channels, so w do not need to include it
    uint32_t xy, c;
    xy = div_mod(c, idx, channel);
    tile[c][xy] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

  // aligned load
  for (uint32_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    uint32_t flat_idx = idx * 4 + bytes_skipped;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    uchar4 in = aligned_in_char4[idx];
    tile[c][xy] = in.x;

    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.y;

    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.z;


    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.w;
  }

  int64_t processed_in_main = (left_after_prologue /  4) * 4;
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  // __half *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (uint32_t idx = threadIdx.x; idx < left_after_main; idx++) {
    uint32_t flat_idx = processed_in_main + idx;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    tile[c][xy] = epilogue_in[idx];
  }

  __syncthreads();
  // for (int64_t idx = threadIdx.x + start_x / kStaticChannels, base_x = threadIdx.x;
  //   idx < end_x / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
  //   #pragma unroll kStaticChannels
  //   for (int c = 0; c < kStaticChannels; c++) {
  //     float fpin = prologue_tile[base_x * sample.C + c];
  //     float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //     sample.out[idx * sample.C + c] = ConvertSat<Out>(fpout);
  //   }
  // }

  // idx is not divided by the static channels (mostly the start_x)

  // for (int64_t idx = threadIdx.x + start_x, base_x = threadIdx.x;
  //   idx < end_x; idx += blockDim.x, base_x += blockDim.x) {

  //   fast_div<uint32_t> channel(kStaticChannels);
  //   int c = idx % channel;
  //   float fpin = prologue_tile[base_x];
  //   float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
  //   sample.out[idx] = ConvertSat<Out>(fpout);
  // }
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < end_x_padded; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      __half fpin = tile[c][base_offset];
      __half fpout = __hfma(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }
}



template <typename Out, typename In, typename Intermediate>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanesPacked(const SimpleSampleDesc<Out, In> *samples, uint32_t *first_blocks,
                             uint32_t num_samples) {

  static constexpr int kBlockSize = kBlockWidth * kBlockSizeMul;
  int sample_idx = FindSampleIdx(first_blocks, num_samples);
  const auto sample = samples[sample_idx];

  int64_t start_x = (blockIdx.x - sample.first_block) * kBlockSize;
  int64_t end_x = ::min(start_x + kBlockSize, sample.sample_size);

  __half norm_mul[kStaticChannels + 1], norm_add[kStaticChannels + 1];

  __shared__ Intermediate tile[3][kBlockSizeMul * kBlockWidth / 3 + 4];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }
  norm_mul[3] = 0;
  norm_add[3] = 42.f + 3;

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + start_x);
  auto aligned_in_start = align_up(in_start, 3*4);  // hmm, can we ensure the pixel?
  auto bytes_skipped = aligned_in_start - in_start;

  // __half *aligned_tile = tile + 32 * 4;
  // __half *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + start_x;


  // float *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uchar4 *aligned_in_char4 = reinterpret_cast<const uchar4*>(sample.in + start_x + bytes_skipped);

  fast_div<uint32_t> channel(kStaticChannels);
  // prologue
  for (uint32_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    // uint32_t flat_offset = idx;  // start_x is multiple of channels, so w do not need to include it
    uint32_t xy, c;
    xy = div_mod(c, idx, channel);
    tile[c][xy] = prologue_in[idx];
  }

  int64_t left_after_prologue = end_x - start_x - bytes_skipped;

  // aligned load
  for (uint32_t idx = threadIdx.x; idx < left_after_prologue / 4; idx += blockDim.x) {
    uint32_t flat_idx = idx * 4 + bytes_skipped;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    uchar4 in = aligned_in_char4[idx];
    tile[c][xy] = in.x;

    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.y;

    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.z;


    c++;
    if (c == 3) {
      c = 0;
      xy++;
    }
    tile[c][xy] = in.w;
  }

  int64_t processed_in_main = (left_after_prologue /  4) * 4;
  int64_t left_after_main = left_after_prologue - processed_in_main;

  // epilogue
  // __half *epilogue_tile = aligned_tile + processed_in_main;
  const In *epilogue_in = reinterpret_cast<const In*>(aligned_in_char4 + processed_in_main / 4);

  for (uint32_t idx = threadIdx.x; idx < left_after_main; idx++) {
    uint32_t flat_idx = processed_in_main + idx;
    uint32_t xy, c;
    xy = div_mod(c, flat_idx, channel);
    tile[c][xy] = epilogue_in[idx];
  }

  __syncthreads();
  int64_t block_4 = (kBlockSize / sample.input_C) * sample.C;
  int64_t sample_size_4 = (sample.sample_size / sample.input_C) * sample.C;
  int64_t start_x_padded = static_cast<int64_t>(blockIdx.x - sample.first_block) * block_4;
  int64_t end_x_padded = ::min(start_x_padded + block_4, sample_size_4);

  int64_t processed = start_x / sample.input_C;


  auto out_start = reinterpret_cast<std::uintptr_t>(sample.out + start_x_padded);
  uint32_t values_skipped_out = 0;
  auto aligned_out_start = out_start;
  while (aligned_out_start % 4) {
    values_skipped_out += 3;
    aligned_out_start += 3 * sizeof(Out);
  }

  auto *out_aligned = sample.out + values_skipped_out + start_x_padded;

  for (int64_t idx = threadIdx.x + start_x_padded, base_x = threadIdx.x;
    idx < start_x_padded + values_skipped_out; idx += blockDim.x, base_x += blockDim.x) {

    int offset = idx >> 2;
    int base_offset = base_x >> 2;
    int c = idx & 3;
    if (c < kStaticChannels) {
      __half fpin = tile[c][base_offset];
      __half fpout = __hfma(fpin, norm_mul[c], norm_add[c]);
      sample.out[idx] = ConvertSat<Out>(fpout);
    } else {
      sample.out[idx] = 42.f + c;
    }
  }

  auto to_write = end_x_padded - start_x_padded - values_skipped_out;
  auto *out = reinterpret_cast<__half2*>(out_aligned);


  __half2 mul_lo = make_half2(norm_mul[0], norm_mul[1]);
  __half2 mul_hi = make_half2(norm_mul[2], norm_mul[3]);
  __half2 add_lo = make_half2(norm_add[0], norm_add[1]);
  __half2 add_hi = make_half2(norm_add[2], norm_add[3]);

  for (int64_t base_x = threadIdx.x; base_x < (to_write) / 2; base_x += blockDim.x) {
    int64_t idx = base_x * 2 + start_x;
    int base_offset = base_x / 2;
    int c = idx & 3;
    if (c == 0) {
      __half fpin0 = tile[0][base_offset];
      __half fpin1 = tile[1][base_offset];
      __half2 fpin = make_half2(fpin0, fpin1);

      __half2 fpout = __hfma2(fpin, mul_lo, add_lo);
      out[base_x] = fpout;
    } else {

      __half fpin0 = tile[2][base_offset];
      __half fpin1 = 42.f + 3;
      __half2 fpin = make_half2(fpin0, fpin1);

      __half2 fpout = __hfma2(fpin, mul_hi, add_hi);
      out[base_x] = fpout;
    }
  }
}



// Worse
template <typename Out, typename In>
__global__ void SortChannelsSharedOut(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

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
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}

// Similar perf to the original
template <typename Out, typename In>
__global__ void SortChannelsInPlace0(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

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
      float fpout = fmaf(fpin, norm_mul[c], norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = ConvertSat<Out>(fpout);
    }
  }
}

template <typename Out, typename In>
__global__ void SortChannelsInPlace1(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto block = blocks[blockIdx.x];
  const auto sample = samples[block.sample_idx];

  float norm_mul[kStaticChannels], norm_add[kStaticChannels];

  #pragma unroll kStaticChannels
  for (int c = 0; c < kStaticChannels; c++) {
    norm_mul[c] = sample.norm_mul[c];
    norm_add[c] = sample.norm_add[c];
  }

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
      float fpout = fmaf(fpin[c], norm_mul[c], norm_add[c]);
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
using output_t = __half;

void RunSN(int num_samples, input_t *input, output_t *output, float *norm_add, float *norm_mul, cudaStream_t stream, int id, int H, int W, int C, int oC) {
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
  int out_img_size = H * W * oC;


  vec<spatial_ndim, int32_t> out_size = in_size; //{W, H};
  vec<spatial_ndim, int64_t> out_stride = {oC, W * oC}; // {1, W}; // HMMM, WTH
  int out_ch_stride = H * W;

  // Roi<2> bounds = {ivec{0, 0}, out_size};

  TensorListShape<3> shape = uniform_list_shape(num_samples, TensorShape<3>{H, W, C});
  TensorListShape<1> collapsed_shape = collapse_dims<1>(shape, {{0, 3}});


  for (int i = 0; i < num_samples; i++) {
    sample_descs[i].in = {input + i * img_size, in_size, C, in_stride, in_ch_stride};
    sample_descs[i].out = {output + i * img_size, out_size, C, out_stride, in_ch_stride};
    // sample_descs[i].bounds = bounds;
    sample_descs[i].norm_add = norm_add + i * C;
    sample_descs[i].norm_mul = norm_mul + i * C;
    // sample_descs[i].fill_values = fill_values_gpu + i * out_nchannels_; unused here

  }


  uint32_t offset_blk = 0;
  int nonempty_samples = 0;

  std::vector<SimpleSample> simple_sample_desc(num_samples);
  std::vector<uint32_t> first_blocks(num_samples);
  for (int i = 0; i < num_samples; i++) {
    int64_t sample_size = collapsed_shape[i][0];

    if (sample_size == 0) {
      continue;
    }

    auto &sample_desc = simple_sample_desc[nonempty_samples++];

    sample_desc.first_block = offset_blk;
    first_blocks[nonempty_samples - 1] = offset_blk;
    sample_desc.sample_size = sample_size;
    offset_blk += div_ceil(sample_size, kBlockSizeMul * kBlockWidth);


    sample_desc.in = input + i * img_size;
    sample_desc.out = output + i * out_img_size;
    sample_desc.H = H;
    sample_desc.W = W;
    sample_desc.C = oC;
    sample_desc.input_W = W;
    sample_desc.input_C = C;
    // doesn't work, I thought it would, but it does not
    // simple_sample_desc[i].shape = {H, W, C};
    sample_desc.norm_add = norm_add + i * C;
    sample_desc.norm_mul = norm_mul + i * C;
    sample_desc.mirror = false;

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
  uint32_t *first_blocks_gpu = nullptr;

  cudaMalloc((void **)&samples_gpu, num_samples * sizeof(Sample));
  cudaMalloc((void **)&tiles_gpu, tiles_cpu.size() * sizeof(Tile));
  cudaMalloc((void **)&first_blocks_gpu, num_samples * sizeof(uint32_t));
  cudaMemcpy(samples_gpu, sample_descs.data(), num_samples * sizeof(Sample), cudaMemcpyHostToDevice);
  cudaMemcpy(tiles_gpu, tiles_cpu.data(), tiles_cpu.size() * sizeof(Tile), cudaMemcpyHostToDevice);
  cudaMemcpy(first_blocks_gpu, first_blocks.data(), num_samples * sizeof(uint32_t), cudaMemcpyHostToDevice);


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


  if (id == 8) {
    std::cout << "Hwc2Cwh" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogueF32HWC2CHW<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }

  if (id == 9) {

    std::cout << "Hwc2Hwc orginal" <<std::endl;
    SliceNormalizeKernel_2D_NoPad<Out, In>
          <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, tiles_gpu);
  }
  if (id == 10) {
    std::cout << "Hwc2Hwc via float" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogueF32<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }
  if (id == 11) {
    std::cout << "Hwc2Hwc via u8" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogueU8<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }
  if (id == 12) {
    std::cout << "Hwc2Hwc via u8 Pad" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogueU8_Pad<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }
  if (id == 13) {
    std::cout << "Hwc2Hwc via half to half computation" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }
  if (id == 14) {
    std::cout << "Hwc2Hwc via half to half Planes" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_halfPlanes<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }

  if (id == 15) {
    std::cout << "Hwc2Hwc via float to half Planes" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanes<Out, In, float><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }

  if (id == 16) {
    std::cout << "Hwc2Hwc via u8 to half Planes" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanes<Out, In, uint8_t><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }

  if (id == 17) {
    std::cout << "Hwc2Hwc via half to half Planes Packed" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanesPacked<Out, In, __half><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }
  if (id == 18) {
    std::cout << "Hwc2Hwc via u8 to half Planes Packed" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanesPacked<Out, In, uint8_t><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }
  if (id == 19) {
    std::cout << "Hwc2Hwc via float to half Planes Packed" <<std::endl;
    SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanesPacked<Out, In, float><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  }
  // if (id == 15) {
  //   std::cout << "Hwc2Hwc via u8 to half computation" <<std::endl;
  //   SortChannelsSharedPreloadFloatPrologueEpilogueU8_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  // }
  // if (id == 16) {
  //   std::cout << "Hwc2Hwc via half2 to half computation" <<std::endl;
  //   SortChannelsSharedPreloadFloatPrologueEpilogue_half2_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  // }
  // if (id == 17) {
  //   std::cout << "Hwc2Hwc via half to half unroll" <<std::endl;
  //   SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_half_unroll<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  // }
  // if (id == 18) {
  //   std::cout << "Hwc2Hwc via float to half computation" <<std::endl;
  //   SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
  // }





  // CUDA events
  cudaEvent_t start, stop;
  // initialize events
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start, stream);

  for (int i = 0; i < NUM_ITERS; i++) {

    if (id == 8) {
      SortChannelsSharedPreloadFloatPrologueEpilogueF32HWC2CHW<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }

    if (id == 9) {
      SliceNormalizeKernel_2D_NoPad<Out, In>
            <<<grid_dim, block_dim, 0, stream>>>(samples_gpu, tiles_gpu);
    }
    if (id == 10) {
      SortChannelsSharedPreloadFloatPrologueEpilogueF32<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }
    if (id == 11) {
      SortChannelsSharedPreloadFloatPrologueEpilogueU8<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }

    if (id == 12) {
      SortChannelsSharedPreloadFloatPrologueEpilogueU8_Pad<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }
    if (id == 13) {
      SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }
    if (id == 14) {
      SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_halfPlanes<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }
    if (id == 15) {
      SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanes<Out, In, float><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }

    if (id == 16) {
      SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanes<Out, In, uint8_t><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }

    if (id == 17) {
      SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanesPacked<Out, In, __half><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }
    if (id == 18) {
      SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanesPacked<Out, In, uint8_t><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }
    if (id == 19) {
      SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_halfPlanesPacked<Out, In, float><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    }
    // if (id == 15) {
    //   SortChannelsSharedPreloadFloatPrologueEpilogueU8_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    // }

    // if (id == 16) {
    //   SortChannelsSharedPreloadFloatPrologueEpilogue_half2_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    // }
    // if (id == 17) {
    //   SortChannelsSharedPreloadFloatPrologueEpilogue_half_Pad_half_unroll<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    // }
    // if (id == 18) {
    //   SortChannelsSharedPreloadFloatPrologueEpilogue_float_Pad_half<Out, In><<<offset_blk, 128, 0, stream>>>(simple_samples_gpu, first_blocks_gpu, nonempty_samples);
    // }

  }


  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  float kernelTime;
  checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, stop));

      // report effective bandwidths
  float kernelBandwidth = num_samples * 1000.0f * (H * W * C * sizeof(uint8_t) + H * W * oC * sizeof(output_t)) / (1024 * 1024 * 1024) /
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

void prep_gold(output_t *out, const uint8_t *in, int oH, int oW, int oC, int iH, int iW, int iC, int transpose = false,
               int Y = 0, int X = 0) {
  for (int y = 0; y < oH; y++) {
    for (int x = 0; x < oW; x++) {

      for (int c = 0; c < iC; c++) {
        if (transpose) {
          out[c * oH * oW + y * oW + x] = in[(y + Y) * iW * iC + (x + X) * iC + c];
        } else {
          out[y * oW * oC + x * oC + c] = in[(y + Y) * iW * iC + (x + X) * iC + c];
        }
      }

      for (int c = iC; c < oC; c++) {;
        if (transpose) {
          out[c * oH * oW + y * oW + x] = 42.f + c;
        } else {
          out[y * oW * oC + x * oC + c] = 42.f + c;
        }
      }
    }
  }
}

void prepare_and_run(int num_samples, int H, int W, int C, int oC, int start = 8, int stop = 12) {
  input_t *input_gpu;
  output_t *output_gpu;
  float *norm_add_gpu, *norm_mul_gpu;
  cudaMalloc((void **)&input_gpu, num_samples * sizeof(input_t) * H * W * C);
  cudaMalloc((void **)&output_gpu, num_samples * sizeof(output_t) * H * W * oC);
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
  gold_cpu.resize(H * W * oC);

  cudaMemcpy(input_gpu, input_cpu.data(), sizeof(input_t) * H * W * C, cudaMemcpyHostToDevice);
  for (int i = 1; i < num_samples; i++) {
    cudaMemcpy(input_gpu + i * H * W * C, input_gpu, sizeof(input_t) * H * W * C, cudaMemcpyDeviceToDevice);
  }


  std::vector<output_t> output_cpu;
  output_cpu.resize(num_samples * H * W * oC);


  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  for (int id = start; id < stop; id++) {

    prep_gold(gold_cpu.data(), input_cpu.data(), H, W, oC, H, W, C, id < 9);
    if ((id == 6) && H * W * C % 4) {
      printf("Unaligned version, skipping 6\n");
      continue;
    }
    cudaMemset(output_gpu, 0, num_samples * sizeof(output_t) * H * W * oC);
    RunSN(num_samples, input_gpu, output_gpu, norm_add_gpu, norm_mul_gpu, stream, id, H, W, C, oC);

    cudaMemcpy(output_cpu.data(), output_gpu, sizeof(output_t) * num_samples *  H * W * oC, cudaMemcpyDeviceToHost);
    // if (id == 8 || id == 12) {
    //   printf("Skipping the check for 8, trailing data is wrong, 12 - mirror draft\n");
    //   continue;
    // }
    for (int i = 0; i < num_samples; i++) {
      auto res = compareData(gold_cpu.data(), output_cpu.data() + i * H * W * oC, H * W * oC, 0.01f, 0.0f);
      // printf("Expected: %d, sample %d\n", id, i);
      // print_planes(gold_cpu.data(), H, W, C);
      // printf("\n\n=============================================================\nComputed: %d, sample %d\n", id, i);
      // print_planes(output_cpu.data() + i * H * W * C, H, W, C);
      if (res) {
        printf("*** %s kernel FAILED: %u sample: %d***\n", "CMN", res, i);
      }
    }
  }

  cudaFree(input_gpu);
  cudaFree(output_gpu);
  cudaFree(norm_add_gpu);
  cudaFree(norm_mul_gpu);
  cudaStreamDestroy(stream);
}




void RunSN_Crop(int num_samples, input_t *input, output_t *output, float *norm_add, float *norm_mul, cudaStream_t stream,
                int id, int H, int W, int C, int iH, int iW, int Y, int X) {
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
  int64_t in_img_size = iH * iW * C;
  int64_t out_img_size = H * W * C;


  TensorListShape<3> shape = uniform_list_shape(num_samples, TensorShape<3>{H, W, C});
  TensorListShape<1> collapsed_shape = collapse_dims<1>(shape, {{0, 3}});


  std::vector<SimpleSample> simple_sample_desc(num_samples);
  for (int i = 0; i < num_samples; i++) {
    simple_sample_desc[i].in = input + i * in_img_size + Y * iW * C + X * C;
    simple_sample_desc[i].out = output + i * out_img_size;
    simple_sample_desc[i].H = H;
    simple_sample_desc[i].W = W;
    simple_sample_desc[i].C = C;
    // doesn't work, I thought it would, but it does not
    // simple_sample_desc[i].shape = {H, W, C};
    simple_sample_desc[i].norm_add = norm_add + i * C;
    simple_sample_desc[i].norm_mul = norm_mul + i * C;
    simple_sample_desc[i].mirror = false;
    simple_sample_desc[i].input_W = iW;

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

  SimpleSample *simple_samples_gpu = nullptr;
  CollapsedBlock *collapsed_blocks_gpu = nullptr;


  cudaMalloc((void **)&simple_samples_gpu, num_samples * sizeof(SimpleSample));
  cudaMemcpy(simple_samples_gpu, simple_sample_desc.data(), num_samples * sizeof(SimpleSample), cudaMemcpyHostToDevice);

  constexpr bool USE_ALIGNMENT = false;
  if (USE_ALIGNMENT) {
    cudaMalloc((void **)&collapsed_blocks_gpu, collapsed_blocks_aligned_manual.size() * sizeof(CollapsedBlock));
    cudaMemcpy(collapsed_blocks_gpu, collapsed_blocks_aligned_manual.data(), collapsed_blocks_aligned_manual.size() * sizeof(CollapsedBlock), cudaMemcpyHostToDevice);

    // TODO DODODODODO: VERY IMPORTANT, ALIGN THE GRID:
    collapsed_grid_dim = dim3(collapsed_blocks_aligned_manual.size(), 1, 1);

  } else {

    cudaMalloc((void **)&collapsed_blocks_gpu, collapsed_blocks_cpu.size() * sizeof(CollapsedBlock));
    cudaMemcpy(collapsed_blocks_gpu, collapsed_blocks_cpu.data(), collapsed_blocks_cpu.size() * sizeof(CollapsedBlock), cudaMemcpyHostToDevice);
  }
  if (id == 0) {
    SortChannelsSharedPreloadFloatPrologueEpilogueF32Crop<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }
  if (id == 1) {
    SortChannelsSharedPreloadFloatPrologueEpilogueF32CropCooperative<<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
  }


  // CUDA events
  cudaEvent_t start, stop;
  // initialize events
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaEventRecord(start, stream);

  for (int i = 0; i < NUM_ITERS; i++) {
    if (id == 0) {
      SortChannelsSharedPreloadFloatPrologueEpilogueF32Crop<Out, In><<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }
    if (id == 1) {
      SortChannelsSharedPreloadFloatPrologueEpilogueF32CropCooperative<<<collapsed_grid_dim, collapsed_block_dim, 0, stream>>>(simple_samples_gpu, collapsed_blocks_gpu);
    }


  }


  checkCudaErrors(cudaEventRecord(stop, stream));
  checkCudaErrors(cudaEventSynchronize(stop));
  float kernelTime;
  checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, stop));

      // report effective bandwidths
  float kernelBandwidth = num_samples * 1000.0f * (H * W * C * (sizeof(uint8_t) + sizeof(output_t))) / (1024 * 1024 * 1024) /
    (kernelTime / NUM_ITERS);
  printf(
  "CMN %d, Throughput = %.4f GB/s, Time = %.5f ms, Size = %d x %d x %d\n",
  id, kernelBandwidth, kernelTime / NUM_ITERS, H, W, C);


  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}


void prepare_and_run_crop(int num_samples, int H, int W, int C, int iH, int iW, int Y, int X) {
  input_t *input_gpu;
  output_t *output_gpu;
  float *norm_add_gpu, *norm_mul_gpu;
  cudaMalloc((void **)&input_gpu, num_samples * sizeof(input_t) * iH * iW * C);
  cudaMalloc((void **)&output_gpu, num_samples * sizeof(output_t) * H * W * C);
  cudaMalloc((void **)&norm_add_gpu, num_samples * sizeof(float) * C);
  cudaMalloc((void **)&norm_mul_gpu, num_samples * sizeof(float) * C);

  std::vector<float> norm_add(num_samples * C, 0.f), norm_mul(num_samples * C, 1.f);
  cudaMemcpy(norm_add_gpu, norm_add.data(), num_samples * sizeof(float) * C, cudaMemcpyHostToDevice);
  cudaMemcpy(norm_mul_gpu, norm_mul.data(), num_samples * sizeof(float) * C, cudaMemcpyHostToDevice);


  // prepare input 0, 1, 2, 3, 4...
  // prepare gold output 0,3,6.... 1,4,7,... 2,5,8,...
  std::vector<input_t> input_cpu;
  input_cpu.resize(iH * iW * C);
  for (int i = 0; i < iH * iW * C; i++) {
    input_cpu[i] = i;
  }

  std::vector<output_t> gold_cpu;
  gold_cpu.resize(H * W * C);
  prep_gold(gold_cpu.data(), input_cpu.data(), H, W, C, iH, iW, C, true, Y, X);

  cudaMemcpy(input_gpu, input_cpu.data(), sizeof(input_t) * iH * iW * C, cudaMemcpyHostToDevice);
  for (int i = 1; i < num_samples; i++) {
    cudaMemcpy(input_gpu + i * iH * iW * C, input_gpu, sizeof(input_t) * iH * iW * C, cudaMemcpyDeviceToDevice);
  }


  std::vector<output_t> output_cpu;
  output_cpu.resize(num_samples * H * W * C);


  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  for (int id = 0; id < 2; id++) {
    cudaMemset(output_gpu, 0, num_samples * sizeof(output_t) * H * W * C);
    RunSN_Crop(num_samples, input_gpu, output_gpu, norm_add_gpu, norm_mul_gpu, stream, id, H, W, C, iH, iW, X, Y);

    cudaMemcpy(output_cpu.data(), output_gpu, sizeof(output_t) * num_samples *  H * W * C, cudaMemcpyDeviceToHost);
    for (int i = 0; i < num_samples; i++) {
      auto res = compareData(gold_cpu.data(), output_cpu.data() + i * H * W * C, H * W * C, 0.01f, 0.0f);
      // printf("Expected: %d, sample %d\n", id, i);
      // print_planes(gold_cpu.data(), H, W, C);
      // printf("\n\n=============================================================\nComputed: %d, sample %d\n", id, i);
      // print_planes(output_cpu.data() + i * H * W * C, H, W, C);
      if (res) {
        printf("*** %s kernel FAILED: %u ***\n", "CMN", res);
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
