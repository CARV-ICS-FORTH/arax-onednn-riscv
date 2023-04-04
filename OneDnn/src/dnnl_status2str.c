#include "dnnl.h"
#include "arax.h"

#define STRS 7

static const char * strs[STRS] = {
	"dnnl_success",
	"dnnl_out_of_memory",
	"dnnl_invalid_arguments",
	"dnnl_unimplemented",
	"dnnl_iterator_ends",
	"dnnl_runtime_error",
	"dnnl_not_required"
};

const char * dnnl_status2str(dnnl_status_t s)
{
	return strs[s];
}
