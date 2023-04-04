#include "PrimitiveMap.h"
#include "common/c_types_map.hpp"

#define PS_ITEM(TYPE) {TYPE,sizeof(TYPE##_desc_t)}

std::unordered_map<dnnl_primitive_kind_t,std::size_t> Primitive::Size = {
	{dnnl_reorder,sizeof(dnnl::impl::dnnl_reorder_desc_t)},
	PS_ITEM(dnnl_shuffle),
	{dnnl_concat,sizeof(dnnl::impl::dnnl_concat_desc_t)},
	{dnnl_sum,sizeof(dnnl::impl::dnnl_sum_desc_t)},
	PS_ITEM(dnnl_convolution),
	PS_ITEM(dnnl_deconvolution),
	PS_ITEM(dnnl_eltwise),
	PS_ITEM(dnnl_lrn),
	PS_ITEM(dnnl_batch_normalization),
	PS_ITEM(dnnl_inner_product),
	PS_ITEM(dnnl_rnn),
	{dnnl_gemm,sizeof(dnnl::impl::dnnl_gemm_desc_t)},
	PS_ITEM(dnnl_binary),
	PS_ITEM(dnnl_matmul),
	PS_ITEM(dnnl_resampling),
	PS_ITEM(dnnl_pooling),
	PS_ITEM(dnnl_reduction),
	PS_ITEM(dnnl_prelu),
	PS_ITEM(dnnl_softmax),
	PS_ITEM(dnnl_layer_normalization)
};
