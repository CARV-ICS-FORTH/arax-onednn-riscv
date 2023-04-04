#pragma once
#include <unordered_map>
#include <dnnl_types.h>


namespace Primitive {

	extern std::unordered_map<dnnl_primitive_kind_t,std::size_t> Size;

}
