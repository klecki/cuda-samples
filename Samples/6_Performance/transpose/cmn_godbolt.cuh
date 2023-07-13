// Type your code here, or load an example.

static constexpr int kBlockSizeMul = 24;
static constexpr int kBlockWidth = 128;
static constexpr int kStaticChannels = 3;

#include <cstdint>
template <int ndim>
struct BlockDesc {
  int sample_idx;
  int start, end;
};

/**
 * @brief Block descriptor specifying range in given sample.
 *
 * Specialization for 1 dim to support 64bit addressing range.
 */
template <>
struct BlockDesc<1> {
  int sample_idx;
  int64_t start, end;
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

template <typename Value, typename Alignment>
__host__ __device__
constexpr Value align_up(Value v, Alignment a) {
  return v + ((a - 1) & -v);
}


template <typename Out, typename In>
__global__ void SortChannelsSharedPreloadFloatPrologueEpilogue(const SimpleSampleDesc<Out, In> *samples,
                             const BlockDesc<1> *blocks) {
  const auto &block = blocks[blockIdx.x];
  const auto &sample = samples[block.sample_idx];
  __shared__ In tile[kBlockSizeMul * kBlockWidth + 33 * 4];

  // TODO: assumes u8

  auto in_start = reinterpret_cast<std::uintptr_t>(sample.in + block.start);
  auto aligned_in_start = align_up(in_start, 32*4);
  auto bytes_skipped = aligned_in_start - in_start;

  In *aligned_tile = tile + 32 * 4;
  In *prologue_tile = aligned_tile - bytes_skipped;
  const In *prologue_in = sample.in + block.start;


  uint32_t *aligned_tile_u32 = reinterpret_cast<uint32_t*>(aligned_tile);
  const uint32_t *aligned_in_u32 = reinterpret_cast<const uint32_t*>(sample.in + block.start + bytes_skipped);

  // prologue
  for (int64_t idx = threadIdx.x; idx < bytes_skipped; idx += blockDim.x) {
    prologue_tile[idx] = prologue_in[idx];
  }

  int64_t left_after_prologue = block.end - block.start - bytes_skipped;

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
  for (int64_t idx = threadIdx.x + block.start / kStaticChannels, base_x = threadIdx.x;
    idx < block.end / kStaticChannels; idx += blockDim.x, base_x += blockDim.x) {
    #pragma unroll kStaticChannels
    for (int c = 0; c < kStaticChannels; c++) {
      float fpin = prologue_tile[base_x * sample.C + c];
      float fpout = fmaf(fpin, sample.norm_mul[c], sample.norm_add[c]);
      sample.out[c * sample.H * sample.W + idx] = fpout;
    }
  }
}

template __global__ void
SortChannelsSharedPreloadFloatPrologueEpilogue<float, uint8_t>(const SimpleSampleDesc<float, uint8_t> *samples,
                             const BlockDesc<1> *blocks);
