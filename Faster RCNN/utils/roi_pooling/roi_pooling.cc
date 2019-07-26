#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cstdio>
#include <iostream>
#include <typeinfo>

using namespace tensorflow;

REGISTER_OP("RoiPooling")
.Input("input: float32")
.Input("rois: int32")
.Attr("pooling_height: int")
.Attr("pooling_width: int")
.Output("output: float32")
.Output("argmax_output: int32")

#define Dtype float
void RoiPoolingKernelGPU(const float* input, const int* rois, 
                        int num_rois, int channels, int height, int width,
                        int pooling_height, int pooling_width, 
                        Dtype* output, int* argmax_output);

class RoiPoolingOp : public OpKernel {
    private: 
        int pooling_height_, pooling_width_;
    public:
        explicit RoiPoolingOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("pooling_height", &pooling_height_));
            OP_REQUIRES_OK(context, context->GetAttr("pooling_width", &pooling_width_));
        }

    void Compute(OpKernelContext* context) override {
        const Tensor& input_tensor = context->input(0);
        const Tensor& rois_tensor = context->input(1);

        auto input = input_tensor.flat<float>();
        auto rois = rois_tensor.flat<int>();

        Tensor* output_tensor = nullptr;
        Tensor* argmax_output_tensor = nullptr;

        auto input_shape = input_tensor.shape();
        auto rois_shape = rois_tensor.shape();

        int num_rois = rois_shape.dim_size(0);
        int height = input_shape.dim_size(1);
        int width = input_shape.dim_size(2);
        int channels = input_shape.dim_size(3);


        TensorShape output_shape = TensorShape({static_cast<int>(num_rois),
                                                static_cast<int>(channels),
                                                static_cast<int>(pooling_height_),
                                                static_cast<int>(pooling_width_)});

        OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, oupput_shape, &argmax_output_tensor));

        auto output = output_tnesor->template flat<float>();
        auto argmax_output = argmax_output_tensor->template flat<int>();

        RoiPoolingKernelGPU(input.data(), rois.data(), 
                        num_rois, channels, height, width, pooling_height_, pooling_width_,
                        output.data(), argmax_output.data());
    }
};

REGISTER_KERNEL_BUILDER(Name('RoiPooling').Device(DEVICE_GPU), RoiPoolingOp)


