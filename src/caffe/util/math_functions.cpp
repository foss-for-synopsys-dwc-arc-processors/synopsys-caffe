#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#define SIGNED_SATURATE_MAX 2047
#define SIGNED_SATURATE_MIN -2048
#define UNSIGNED_SATURATE_MAX 4095
#define SIGNED_8BIT_SATURATE_MAX 127
#define SIGNED_8BIT_SATURATE_MIN -128
#define UNSIGNED_8BIT_SATURATE_MAX 255

namespace caffe {

template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_div_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] /= alpha;
  }
}

template <>
void caffe_div_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] /= alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<bool>(const int N, const bool* X, bool* Y);
template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
  vsSqrt(n, a, y);
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
  vdSqrt(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
    vdAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<Dtype> random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <typename Dtype>
void caffe_cpu_saturate(const int n, Dtype* x, Dtype saturate_method) {
  if (saturate_method == ConvolutionParameter_SaturateMethod_Signed)
    caffe_cpu_signed_saturate(n, x);
  if (saturate_method == ConvolutionParameter_SaturateMethod_Unsigned)
    caffe_cpu_unsigned_saturate(n, x);
  if (saturate_method == ConvolutionParameter_SaturateMethod_Signed_8bit)
    caffe_cpu_signed_8bit_saturate(n, x);
  if (saturate_method == ConvolutionParameter_SaturateMethod_Unsigned_8bit)
    caffe_cpu_unsigned_8bit_saturate(n, x);
}

template void caffe_cpu_saturate<float>(const int n, float* x, float saturate_method);

template void caffe_cpu_saturate<double>(const int n, double* x, double saturate_method);

template <typename Dtype>
void caffe_cpu_universal_saturate(const int n, Dtype* x, Dtype SATURATE_MAX, Dtype SATURATE_MIN) {
  for (int i = 0; i < n; ++i) {
    if (x[i] > SATURATE_MAX) {
      x[i] = SATURATE_MAX;
    }
    if (x[i] < SATURATE_MIN) {
      x[i] = SATURATE_MIN;
    }
  }
}

template <>
void caffe_cpu_signed_saturate<float>(const int n, float* x) {
  caffe_cpu_universal_saturate<float>(n, x, SIGNED_SATURATE_MAX, SIGNED_SATURATE_MIN);
}

template <>
void caffe_cpu_signed_saturate<double>(const int n, double* x) {
  caffe_cpu_universal_saturate<double>(n, x, SIGNED_SATURATE_MAX, SIGNED_SATURATE_MIN);
}

template <>
void caffe_cpu_unsigned_saturate<float>(const int n, float* x) {
  caffe_cpu_universal_saturate<float>(n, x, UNSIGNED_SATURATE_MAX, 0);
}

template <>
void caffe_cpu_unsigned_saturate<double>(const int n, double* x) {
  caffe_cpu_universal_saturate<double>(n, x, UNSIGNED_SATURATE_MAX, 0);
}

template <>
void caffe_cpu_signed_8bit_saturate<float>(const int n, float* x) {
  caffe_cpu_universal_saturate<float>(n, x, SIGNED_8BIT_SATURATE_MAX, SIGNED_8BIT_SATURATE_MIN);
}

template <>
void caffe_cpu_signed_8bit_saturate<double>(const int n, double* x) {
  caffe_cpu_universal_saturate<double>(n, x, SIGNED_8BIT_SATURATE_MAX, SIGNED_8BIT_SATURATE_MIN);
}

template <>
void caffe_cpu_unsigned_8bit_saturate<float>(const int n, float* x) {
  caffe_cpu_universal_saturate<float>(n, x, UNSIGNED_8BIT_SATURATE_MAX, 0);
}
template <>
void caffe_cpu_unsigned_8bit_saturate<double>(const int n, double* x) {
  caffe_cpu_universal_saturate<double>(n, x, UNSIGNED_8BIT_SATURATE_MAX, 0);
}

template <typename Dtype>
void caffe_cpu_round(const int n, Dtype *x) {
  for (int i = 0; i < n; ++i) {
    x[i] = std::rint(x[i]);
  }
}

template void caffe_cpu_round<float>(const int n, float* x);

template void caffe_cpu_round<double>(const int n, double* x);

template <typename Dtype>
void caffe_cpu_quantize(const int n, Dtype* x, const double scale, const int zero_point){
  if (scale != Dtype(1.0)) {
    caffe_div_scalar<Dtype>(n, scale, x);
    for (int i = 0; i < n; ++i) {
      // TfLiteRound == std::round
      // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/cppmath.h#L36
      x[i] = std::round(x[i]);
    }
  }
  if (zero_point != 0) {
    caffe_add_scalar<Dtype>(n, Dtype(zero_point), x);
  }
}

template void caffe_cpu_quantize<float>(const int n, float* x, const double scale, const int zero_point);

template void caffe_cpu_quantize<double>(const int n, double* x, const double scale, const int zero_point);

template <typename Dtype>
void caffe_cpu_dequantize(const int n, Dtype* x, const double scale, const int zero_point){
  if (zero_point != 0) {
    caffe_add_scalar<Dtype>(n, Dtype(-zero_point), x);
  }
  if (scale != Dtype(1.0)) {
    caffe_scal<Dtype>(n, scale, x);
  }
}

template void caffe_cpu_dequantize<float>(const int n, float* x, const double scale, const int zero_point);

template void caffe_cpu_dequantize<double>(const int n, double* x, const double scale, const int zero_point);

template <typename Dtype, typename Stype>
void caffe_cpu_scale_double_round(const int n, const Stype scale, Dtype* x){
  // refer to https://github.com/google/gemmlowp/blob/master/doc/quantization.md#implementation-of-quantized-matrix-multiplication
  int shift;
  Stype mul = std::frexp(scale, &shift); // multiplier in normalized interval [0.5, 1.0)
  shift = -shift;
  shift = (1<<shift);
  for (int i = 0; i < n; ++i) {
    x[i] = std::round(x[i] * mul);
    x[i] = std::round(x[i] / shift);
  }
}

template void caffe_cpu_scale_double_round<float, float>(const int n, const float scale, float* x);

template void caffe_cpu_scale_double_round<double, double>(const int n, const double scale, double* x);

template void caffe_cpu_scale_double_round<double, float>(const int n, const float scale, double* x);

template void caffe_cpu_scale_double_round<float, double>(const int n, const double scale, float* x);

template <typename Dtype>
void MultiplyByQuantizedMultiplierVR(const int n, Dtype* x, const int mul, const int shift, const int round_mode){
  // MultiplyByQuantizedMultiplier ; V for vector, R for round_mode
  // simulate x[i] * mul * 2^31* 2^shift
  CHECK_EQ(round_mode >= 1 && round_mode <= 2, true);
  int shf = -shift;

  if (round_mode == 1) {
    shf += 31;
    long long round = (1ll << (shf-1));
    // round half to positive inf
    // https://github.com/tensorflow/tensorflow/blob/cfa91be9863a91d5105a3b4941096044ab32036b/tensorflow/core/kernels/quantized_conv_ops.cc#L73
    // also found ruy::MultiplyByQuantizedMultiplier using single-rounding
    // https://github.com/google/ruy/blob/master/ruy/apply_multiplier.cc#L48
    for (int i = 0; i < n; ++i) {
      long long v = (long long) x[i];
      v *= mul;
      v += round;
      v = v>>shf;
      x[i] = v;
    }
  } else if (round_mode == 2) {
    for(int i = 0; i < n; ++i) {
      x[i] = tfl_RoundingDivideByPOT(tfl_SaturatingRoundingDoublingHighMul((int)x[i], mul), shf);
    }
  }
}

template void MultiplyByQuantizedMultiplierVR<float>(const int n, float* x, const int mul, const int shift, const int round_mode);

template void MultiplyByQuantizedMultiplierVR<double>(const int n, double* x, const int mul, const int shift, const int round_mode);

int tfl_SaturatingRoundingDoublingHighMul(int a, int b) {
  // https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h#L340
  bool overflow = a == b && a== std::numeric_limits<std::int32_t>::min();
  CHECK_NE(overflow, true);
  long long a_64 = (long long) a;
  long long b_64 = (long long) b;
  long long ab_64 = a_64 * b_64;
  int nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
  //int nudge = (1 << 30);// - (ab_64 < 0);
  int ab_x2_high32 = (int) ((ab_64 + nudge) / (1ll << 31));
  return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}
int tfl_RoundingDivideByPOT(int x, int exp) {
  // https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h#L368
  CHECK_GE(exp, 0);
  CHECK_LE(exp, 31);
  const int mask = (1ll << exp) - 1;
  const int remainder = x & mask;
  const int threshold = (mask >> 1) + (x < 0);
  return (x >> exp) + (remainder > threshold);
}

int tfl_QuantizeMultiplier(double scale, int *shift) {
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/kernels/internal/quantization_util.cc#L53
  CHECK_NE(scale, 0.0);
  double q_mul = std::frexp(scale, shift);
  long long quantized_multiplier = (long long) std::round(q_mul * (1ll<<31));
  CHECK_LT(quantized_multiplier, 1ll<<31);
  CHECK_GE(*shift, -31);
  return (int) quantized_multiplier;
}
int tfl_MultiplyByQuantizedMultiplier(int x, int q_mul, int shift) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;
  // divide by power of two
  return tfl_RoundingDivideByPOT(tfl_SaturatingRoundingDoublingHighMul(x * (1 << left_shift), q_mul), right_shift);
}

}  // namespace caffe
