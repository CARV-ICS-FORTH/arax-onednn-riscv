#include <assert.h>
#include <chrono>
#include <vector>
#include <unordered_map>
#include <functional>
#include <iostream>
#include <cstring>
#include "dnnl.hpp"

using namespace dnnl;

const char *dnnl_status2str(dnnl_status_t v) {
    if (v == dnnl_success) return "success";
    if (v == dnnl_out_of_memory) return "out_of_memory";
    if (v == dnnl_invalid_arguments) return "invalid_arguments";
    if (v == dnnl_unimplemented) return "unimplemented";
    if (v == dnnl_iterator_ends) return "iterator_ends";
    if (v == dnnl_runtime_error) return "runtime_error";
    if (v == dnnl_not_required) return "not_required";
    assert(!"unknown status");
    return "unknown status";
}

inline int handle_example_errors(
        std::function<void()> example) {
    int exit_code = 0;

    try {
        example();
    } catch (dnnl::error &e) {
        std::cout << "oneDNN error caught: " << std::endl
                  << "\tStatus: " << dnnl_status2str(e.status) << std::endl
                  << "\tMessage: " << e.what() << std::endl;
        exit_code = 1;
    } catch (std::exception &e) {
        std::cout << "Error in the example: " << e.what() << "." << std::endl;
	throw e;
        exit_code = 2;
    }

    std::cout << "Example " << (exit_code ? "failed" : "passed") << " on "
              << "cpu." << std::endl;
    return exit_code;
}

// Read from handle, write to memory
static inline void write_to_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *dst = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            dst[i] = ((uint8_t *)handle)[i];
    } else {
        assert(!"Engine kind not CPU!");
    }
}

static inline void read_from_dnnl_memory(void *handle, dnnl::memory &mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t *src = static_cast<uint8_t *>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i)
            ((uint8_t *)handle)[i] = src[i];
    } else {
        assert(!"Engine kind not CPU!");
    }
}

void fill_input(std::vector<float> & input)
{
  float val= 1;
  for(auto & elem : input)
  {
    elem = val;
    val += 1;
  }
}

void run_convolution() {
    using tag = memory::format_tag;
    using dt = memory::data_type;

    // Initialize engine and stream
    engine eng(dnnl::engine::kind::cpu, 0);
    stream s(eng);

    // Declare the network as a sequence of oneDNN primitives
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;

    // Declare the convolution primitive parameters
    const memory::dim batch = 1;
    const memory::dim IC = 256, OC = 256;
    const memory::dim IH = 14, KH = 3, OH = 14;
    const memory::dim IW = 14, KW = 3, OW = 14;

    memory::dims conv_src_tz = {batch, IC, IH, IW};
    memory::dims conv_weights_tz = {OC, IC, KH, KW};
    memory::dims conv_dst_tz = {batch, OC, OH, OW};
    memory::dims conv_strides = {1, 1};
    memory::dims conv_padding = {1, 1};

    /// Allocate buffers for input and output data, weights, and bias.
    std::vector<float> user_src(batch * IC * IH * IW);
    std::vector<float> user_dst(batch * OC * OH * OW);
    std::vector<float> conv_weights(OC * IC * KH * KW);

fill_input(user_src);
fill_input(conv_weights);

    /// Create memory that describes data layout in the buffers.
    /// This example uses tag::nchw (batch-channels-height-width) for
    /// input data and tag::oihw for weights.
    auto user_src_mem = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
    auto user_dst_mem = memory({{conv_dst_tz}, dt::f32, tag::nchw}, eng);
    auto user_wei_mem = memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);

    write_to_dnnl_memory(user_src.data(), user_src_mem);
    write_to_dnnl_memory(conv_weights.data(), user_wei_mem);

    /// Create memory descriptors with layout tag::any. The `any` format enables
    /// the convolution primitive to choose the data format that will result in
    /// best performance based on its input parameters (convolution kernel
    /// sizes, strides, padding, and so on). If the resulting format is different
    /// from `nchw`, the user data must be transformed to the format required for
    /// the convolution (as explained below).
    auto conv_src_md     = memory::desc({conv_src_tz}, dt::f32, tag::any);
    auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
    auto conv_dst_md     = memory::desc({conv_dst_tz}, dt::f32, tag::any);

    /// Create a convolution descriptor by specifying propagation kind,
    /// convolution algorithm, shapes of input, weights, bias, output,
    /// convolution strides, padding, and kind of padding. Propagation kind is
    /// set to prop_kind::forward_inference to optimize for inference execution
    /// and omit computations that are necessary only for backward propagation.
    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_dst_md, conv_strides, conv_padding,
            conv_padding);

    /// Create a convolution primitive descriptor. Once created, this
    /// descriptor has specific formats instead of the `any` format specified
    /// in the convolution descriptor.
    auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

    /// Check whether data and weights formats required by convolution is different
    /// from the user format. In case it is different change the layout using
    /// reorder primitive.
    auto conv_src_memory = user_src_mem;
    if (conv_prim_desc.src_desc() != user_src_mem.get_desc()) {
        conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
        net.push_back(reorder(user_src_mem, conv_src_memory));
        net_args.push_back({{DNNL_ARG_FROM, user_src_mem},
                {DNNL_ARG_TO, conv_src_memory}});
    }

    auto conv_weights_memory = user_wei_mem;
    if (conv_prim_desc.weights_desc() != user_wei_mem.get_desc()) {
        conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
        reorder(user_wei_mem, conv_weights_memory)
                .execute(s, user_wei_mem, conv_weights_memory);
    }

    /// Create a memory primitive for output.
    auto conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);

    /// Create a convolution primitive and add it to the net.
    net.push_back(convolution_forward(conv_prim_desc));
    net_args.push_back({{DNNL_ARG_SRC, conv_src_memory},
            {DNNL_ARG_WEIGHTS, conv_weights_memory},
            {DNNL_ARG_DST, conv_dst_memory}});

    /// Finally, execute the primitives. For this example, the net is executed
    /// multiple times and each execution is timed individually.
    assert(net.size() == net_args.size() && "something is missing");
    for (size_t i = 0; i < net.size(); ++i)
        net.at(i).execute(s, net_args.at(i));
    s.wait();

    std::vector<float> user_results(batch * OC * OH * OW);
    read_from_dnnl_memory(user_results.data(),user_dst_mem);

    std::string sep = "";
    for(auto elem : user_results)
    {
      std::cerr << sep << elem;
      sep = ", ";
    }
}

int main() {
    run_convolution();
    return 0;
}
