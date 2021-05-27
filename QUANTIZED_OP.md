# Quantized models in different frameworks
As we keep floating-point `scale` and integer `zero-point` for quantized models, and meanwhile some has `Quantize/Dequantize` operators. This section briefs the data types they use.   
|       | TFLite | ONNX  | Caffe2 |
| ----- | ------ | ----- | ------ |
| scale | double | float | float  |
| fp    | [template](https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/lite/kernels/internal/reference/dequantize.h#L41) |  [float](https://github.com/onnx/onnx/blob/master/docs/Operators.md#outputs-29)   | float |
| round half | away zero  | toward even | toward even
| std:: | [round](https://github.com/tensorflow/tensorflow/blob/b58b895a5f64663b88177b1935d39c09fb6278ae/tensorflow/lite/kernels/internal/cppmath.h#L36) | rint | [nearbyint](https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/caffe2/operators/quantized/int8_utils.h#L51)

fp generally denotes the data_type of  
  * Input tensor for `Quantize` and
  * Output tensor for `Dequantize`,

Some operators (in some frameworks) would invoke intermediate floating-point representation, it's usually `float`; not seen double used so far.

ONNXruntime generally handles output_scale by [MlasRequantizeOutput](https://github.com/microsoft/onnxruntime/blob/8d737f977056444a307f1b7f0bcd402fba62d790/onnxruntime/core/mlas/lib/quantize.cpp#L357)(int Input, int Output, float scale); which uses intermediate floating-point representation -- `float`.

## Quantized Convolutions
`output_multiplier` = `input_scale` * `weight_scale` / `output_scale`
Reminded that TFLite uses <double>, while ONNXruntime and Caffe2 use <float> for scales.
### TFLite
The quantized multiplier is calculated as (the `shift` is a power-of-two normalizer to normalize output_multiplier in [0.5,1) )
```cpp=
output_multiplier = <double>input_scale * <double>weight_scale / <double> output_scale;
quantized_multiplier = std::round(std::frexp(output_multiplier, &shift) * (1<<31));
```

For convolutions, TFLite transfrom to DepthwiseConv if `group` = `in_ch` = `out_ch`.  
Then, different roundings are derived in SNPS-Caffe to match TFLite:

| Scales \ group | 1 | Depthwise | Pointwise* |
| --------- | ------ | --------- | ---------  |
| PerTensor | A1     | A2        |     A1*    |
| PerChannel| B1     | B2        |     B2*

Two kinds of rounding are used to multiply quantized numbers.
* [SaturatingRoundingDoublingHighMul](https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h#L340)
* [RoundingDivideByPOT](https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h#L368)

#### **A1** (Double Precision + Double Roundings)
```cpp=
scaled_acc = SaturatingRoundingDoublingHighMul(<int>acc,<int>quantized_multiplier)
out_acc = RoundingDivideByPOT(scaled_acc, shift)
// The approximate result := out_acc = scaled_acc / (1<<31) / (1<<shift),
// where roundings are used
```

#### **A2** (Single Precision + Double Roundings)
Use **`<float>`** to calculate output_multiplier, then apply **A1**.

#### **B1**  (Double Precision + Single Rounding)
Calculate the `output_multiplier` as per channel
```cpp=
output_multiplier[ch] = <double>input_scale * <double>weight_scale[ch] / <double> output_scale;
```
But it uses simpler rounding to calculate the approximate result
```cpp=
scaled_acc = <int>acc * <int>quantized_multiplier
out_acc = (scaled_acc + (1<<(31+shift-1)) >> (31+shift-1)
// which is, it rounds (only once) half toward positive inf
```

#### **B2** (Double Precision + Double Roundings)
The per-channel `output_multiplier` is calculated as **B1**.  
But it applies the `Roundings` in **A1**.

#### **Pointwise Convolution***
When I try to match bit-exactness result, the combination of `PerTensor-A1` and `PerChannel-B2` is found by brute-force.

### ONNX runtime
It casts `<int>acc` to `<float>`, multiply by <float>output_multiplier, and requantize the result.

### Caffe2
It uses single-precision scales, the computation is the same as mentioned **A2**.

