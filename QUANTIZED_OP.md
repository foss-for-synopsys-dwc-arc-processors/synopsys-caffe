# Quantized models in different frameworks (Editing)
As we keep floating-point `scale` and integer `zero-point` for quantized models, and meanwhile some has `Quantize/Dequantize` operators that having floating-point input or output. This section briefs the data types they use, and in SNPS-Caffe implementation we use the target floating-precision.   
|       | TFLite | ONNX  | Caffe2 | SNPS Caffe |
| ----- | ------ | ----- | ------ | ---------- |
| scale | double | float | float  | 
| fp    | [template](https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/lite/kernels/internal/reference/dequantize.h#L41) |  [float](https://github.com/onnx/onnx/blob/master/docs/Operators.md#outputs-29)   | float | float |
| round half | away zero  | toward even | toward even
| std:: | [round](https://github.com/tensorflow/tensorflow/blob/b58b895a5f64663b88177b1935d39c09fb6278ae/tensorflow/lite/kernels/internal/cppmath.h#L36) | rint | [nearbyint](https://github.com/pytorch/pytorch/blob/c371542efc31b1abfe6f388042aa3ab0cef935f2/caffe2/operators/quantized/int8_utils.h#L51)

fp generally denotes the data_type of  
  * Input tensor for `Quantize` and
  * Output tensor for `Dequantize`,
  * Intermediate tensor type if some specific operators use floating-point registers for computation or handling output_scale.
      * e.g. ONNXruntime generally handles the `input_scale-to-output_scale` transformation by [MlasRequantizeOutput](https://github.com/microsoft/onnxruntime/blob/8d737f977056444a307f1b7f0bcd402fba62d790/onnxruntime/core/mlas/lib/quantize.cpp#L357)(int Input, int Output, float scale); which uses intermediate floating-point representation -- `float`.

## Quick Look-Up for Implementations in SNPS Caffe
We support the implementations from different frameworks, which leverages the layer parameter `quantize_method` when their results fail bit-exactness. You can also refer to [FEATURES.md](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-caffe/blob/development/FEATURES.md#custom-quantization-related) for other quantization-related parameters.
We denote TFLite/ONNXruntime/Caffe2 implementations by **t**/**o**/**c**. Since some quantized operators may have bit-exactness results between the frameworks, we don't elaborate the specific implementation.

| `operator` \ `quantize_method` | TFLite |  ONNX | Caffe2|
| ----------- | ------ | ----- | ----- |
| AveragePooling  | **t**  | **o** | **c** |
| Bias        |        |       | **c** |
| Convolution | **t**  | **o** | **c** |
| EltwiseSum  | **t**  | **c** | **c** |
| InnerProduct| **t**  | **o** |       |
| Power*       | **t**  | **o** | **c** |
| Concat*       |  |  |  |
| ResizeBilinear*|   |  |  |


#### Notes
1. Our model zoo doesn't cover all quantized operators over the frameworks. The entry is left empty if the `(framework,operator)` combination is not seen yet.
    * Quantized bias_layer only occurs in ONNX (does not support FC+Bias fusion yet).    
2. Only Quantize and Dequantize operators are mapped to Power_layer.
3. For ResizeBilinear/Concat layers, we use Dequantize+Quantize to implment the affine transformation.
 


## Quantized Convolutions
`output_multiplier` = `input_scale` * `weight_scale` / `output_scale`.  
Reminded that TFLite uses `double`, while ONNXruntime and Caffe2 use `float` for scales.
### TFLite
The quantized multiplier is calculated as (the `shift` is a power-of-two normalizer to normalize output_multiplier in [0.5,1) )
```cpp=
output_multiplier = <double>input_scale * <double>weight_scale / <double> output_scale;
quantized_multiplier = std::round(std::frexp(output_multiplier, &shift) * (1<<31));
// or for channel-wise quantization
// output_multiplier[ch] = <double>input_scale * <double>weight_scale[ch] / <double> output_scale;
// quantized_multiplier[ch] = std::round(std::frexp(output_multiplier[ch], &shift[ch]) * (1<<31));
```

For convolutions, TFLite transfrom to DepthwiseConv if `group` = `in_ch` = `out_ch`.  
Then, different implementations are derived in SNPS-Caffe to match TFLite:

| Scales \ group | 1 | Depthwise | Pointwise* |
| --------- | ------ | --------- | ---------  |
| PerTensor | D2     | F2        |     F2*    |
| PerChannel| D1     | D2        |     D1*

Two kinds of rounding are used to approximate the affine transformation (from `input_scale` to `output_scale`, using the quantized multiplier).
1. The first splits it into two steps, denoted by **2-steps-rounding**
    * [SaturatingRoundingDoublingHighMul](https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h#L340), and
    * [RoundingDivideByPOT](https://github.com/google/gemmlowp/blob/master/fixedpoint/fixedpoint.h#L368)
2. The second implments `rounding half toward positive infinity`, denoted by **1-step-rounding**

#### **D2** (Double Precision + 2-Steps-Rounding)
```cpp=
scaled_acc = SaturatingRoundingDoublingHighMul(<int>acc,<int>quantized_multiplier)
out_acc = RoundingDivideByPOT(scaled_acc, shift)
// The approximate result := out_acc = scaled_acc / (1<<31) / (1<<shift),
// where roundings are used
```

#### **F2** (Single Precision + 2-Steps-Rounding)
Use **`<float>`** to calculate output_multiplier, then apply 2-steps-rounding in **D2**.

#### **D1**  (Double Precision + 1-Step-Rounding)
Calculate the `output_multiplier` as per channel

Also it uses simpler rounding to calculate the approximate result
```cpp=
scaled_acc = <int>acc * <int>quantized_multiplier
out_acc = (scaled_acc + (1<<(31+shift-1)) >> (31+shift-1)
// which is, it rounds (only once) half toward positive inf
```

#### **Pointwise Convolution***
When I try to match bit-exactness result, the combination of `PerTensor-A1` and `PerChannel-B2` is found by brute-force.

### ONNX runtime
It casts `<int>acc` to `<float>`, multiply by `<float>output_multiplier`, then requantize the result.

### Caffe2
It uses single-precision scales, the computation is the same as mentioned **F2**.
