/*
 The code below is inserted in caffe's conv_layer.cpp to change rounding behavior
 for a convolution.

 In EV hardware an accumulated value in a convolution or innerproduct is reduced
 to fit into the destination blob size (e.g. 8 bits).

 To implement the reduction, the accumulator is effectively multiplied by a
 floating-point number with a 15-bit mantissa.  Integer multiplication and shift
 implement this multiplication, as the hardware does not have floating point:

 - A s32 x s16 multiplication is done, producing an s48-bit result
 - The s48 result is shifted right and rounded symmetrically (also called round-to-even).

 Next:
 - If there is a zero point, it is added,
 and the result saturated in the range [0,2*N-1] for an N-bit blob
 - otherwise, relu bounds, if any, are applied before storing into the destination blob.

 The computation of the 15-bit mantissa is derived from a floating-point number F.
 F is the scale of the output blob divided by the scale of the accumulator.
 Note: the EV notion of scale is the inverse of Tensorflow; i.e. the floating
 point number represented by a pixel is F = (pixel - zero_point) / scale .
 Thus in Synopsys caffe,

 F = input_scale * (double)weight_scale / output_scale;
 ^^^ compute accumulator scale ^^^

 The function normalize_fractional below takes the floating point number and computes
 the 15-bit mantiss and its accompanying shift, taking care to produce a number
 <= 32767 and a shift >= 1, in case hardware doesn't support a shift of 0.
 In addition if the computed integer is even it's shifted right to remove 0s
 solely for representational efficiency.

 So, in summary:
 - compute a floating-point number that reduces the accumulator to the desired
 output scale
 - Convert the floating-point number to a 15-bit integer and 6-bit shift
 - Multiply the accumulator by the integer and shift with convergent rounding

 The macro
 define LL_ROUND(X,shift) / (unbiased) round-to-even / \
       ((X + ((X >> (shift)) & 1) + (LLSHL1(shift-1)-1)) >> (shift))
 implements round-to-even -- i.e., unbiased (convergent) rounding.

 The environment variable
 set CAFFE_QUANTIZED_ROUND=EV
 engages this alternative rounding.

 */

typedef double Scale_type;
#include <stdlib.h>
#include <cmath>

static void normalize_fractional(Scale_type F, unsigned &mpy, unsigned &shift) {
  // Adapted from python code in evgencnn.
  int frac_bits = 16;
  Scale_type hi = 0.5;
  int nudge_power = frac_bits;
  // Due to symmetric rounding, >= 32767.5 rounds to 32768, which is invalid
  // as a 16-bit signed number.
  // So the high value should be shifted by 32767.49/32768 so rounding
  // will produce at most 32767.  Nudge avoids that.
  unsigned two_to_nudge = 1 << nudge_power;
  hi *= (Scale_type(two_to_nudge) - 0.51) / two_to_nudge;
  Scale_type lo = hi / 2;
  int frac_adjust_shift = 0;
  F = fabs(F);
  Scale_type oldF = F;
  while (F >= hi) {
    frac_adjust_shift -= 1;
    F /= 2;
  }
  while (F < lo) {
    frac_adjust_shift += 1;
    F *= 2;
  }

  int max_shift = 63;
  while (frac_bits + frac_adjust_shift > max_shift) {
    frac_adjust_shift--;
  }
  int total_shift = frac_bits + frac_adjust_shift;
  // printf("F=%f fas=%d\n", F, frac_adjust_shift);
  // printf("newF=%f\n", oldF * (1 << total_shift));
  mpy = std::round(oldF * (1 << frac_bits) * (1 << frac_adjust_shift));
  // Now if mpy is even, divide by 2 and reduce the shift.
  shift = frac_bits + frac_adjust_shift;
  const int MINSHIFT = 1; // Not knowing whether HW likes shift of 0, we make min 1.
  while ((mpy & 1) == 0 && shift > MINSHIFT) {
    // The end result is an odd fractional.
    mpy >>= 1;
    shift -= 1;
  }
}

template<typename Dtype>
void caffe_cpu_scale_better_round(const std::string &name, const int n,
    const Scale_type scale, Dtype* x) {
  // refer to https://github.com/google/gemmlowp/blob/master/doc/quantization.md#implementation-of-quantized-matrix-multiplication
  Scale_type mul = scale; // multiplier in normalized interval [0.5, 1.0)
  enum Rmode {
    R_double_round, R_single_round, R_ev_round
  };
  auto tell = []() {
    const char* QR = getenv("CAFFE_QUANTIZED_ROUND");
    if (QR == 0) return R_double_round;
    return
    strcmp(QR,"SR")==0?R_single_round:
    strcmp(QR,"EV")==0?R_ev_round:
    (printf("Unrecognized rounding mode %s\n",QR), R_double_round);
  };
  static const Rmode QR = tell();
  switch (QR) {
  case R_double_round:
  case R_single_round: {
    if (QR != R_double_round)
      printf("    Layer %s: round mode %d by %18.15f\n", name.c_str(), QR,
          scale);
    bool SR = QR == R_single_round;
    int shift = 0;
    while (mul < 0.5) {
      mul *= 2.0;
      ++shift;
    }
    shift = (1 << shift);
    for (int i = 0; i < n; ++i) {
      x[i] = SR ? x[i] * mul : std::round(x[i] * mul);
      x[i] = std::round(x[i] / shift);
    }
  }
    break;
  case R_ev_round: {
#define LLSHL1(x) (1LL<<(x))
#define LL_ROUND(X,shift) /* (unbiased) round-to-even */ \
        ((X + ((X >> (shift)) & 1) + (LLSHL1(shift-1)-1)) >> (shift))
    unsigned mpy, shift;
    // Produces 15-bit mantissa and an exponent.  The mantissa is
    // thus less precise than that of a 32-bit floating-point number.
    normalize_fractional(scale, mpy, shift);
    printf("    Layer %s: round mode %d by %18.15f = mpy %d shift %d\n",
        name.c_str(), QR, scale, mpy, shift);
    typedef signed long long SLL;
    for (int i = 0; i < n; ++i) {
      SLL acc = SLL(x[i]);    // Assumed to be an integer already.
      acc *= mpy;
      x[i] = LL_ROUND(acc, shift);
    }
  }
    break;
  }
}

//#define caffe_cpu_scale_double_round(A,B,C) \
//    caffe_cpu_scale_better_round(this->layer_param_.name(),A,B,C)

template<typename Dtype>
void Multiply_better(const int n, Dtype* x, const int mul, const int shift,
    const int round_mode, const std::string &name, const Scale_type scale) {
  enum Rmode {
    R_double_round, R_ev_round
  };
  auto tell = []() {
    const char* QR = getenv("CAFFE_QUANTIZED_ROUND");
    if (QR == 0) return R_double_round;
    return
    strcmp(QR,"EV")==0?R_ev_round:
    (printf("Unrecognized rounding mode %s\n",QR), R_double_round);
  };
  static const Rmode QR = tell();
  static bool show_data_bool = getenv("CAFFE_SHOW_DATA") != 0;
  auto show_data = [&](const char *when) {
    printf("Data %s\n",when);
    for (int i = 0; i < n; i++) {
      printf("%4d = %f\n",i,x[i]);
    }
    return 0;
  };
  switch (QR) {
  case R_double_round: {
    MultiplyByQuantizedMultiplierVR(n, x, mul, shift, round_mode);
  }
    break;
  case R_ev_round: {
#define LLSHL1(x) (1LL<<(x))
#define LL_ROUND(X,shift) /* (unbiased) round-to-even */ \
        ((X + ((X >> (shift)) & 1) + (LLSHL1(shift-1)-1)) >> (shift))
    unsigned mpy, shift;
    // Produces 15-bit mantissa and an exponent.  The mantissa is
    // thus less precise than that of a 32-bit floating-point number.
    normalize_fractional(scale, mpy, shift);
    printf("    Layer %s: round mode %d by %18.15f = mpy %d shift %d\n",
        name.c_str(), QR, scale, mpy, shift);
    typedef signed long long SLL;
    if (show_data_bool)
      show_data("before scaling {");
    for (int i = 0; i < n; ++i) {
      SLL acc = SLL(x[i]);    // Assumed to be an integer already.
      acc *= mpy;
      x[i] = double(LL_ROUND(acc, shift));
    }
    if (show_data_bool)
      show_data("after scaling }");
  }
    break;
  }
}

#define MultiplyByQuantizedMultiplierVR(A,B,C,D,E) \
    Multiply_better(A,B,C,D,E,this->layer_param_.name(), out_scal)
