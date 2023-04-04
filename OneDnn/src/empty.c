#include <stdio.h>


#define EMPTY_FUNC(func)					\
int func()									\
{											\
	fprintf(stderr,"%c[7mCall " #func "%c[0m\n",27,27);		\
	return 3;								\
}

EMPTY_FUNC(dnnl_engine_get_count)

EMPTY_FUNC(dnnl_primitive_desc_destroy)
EMPTY_FUNC(dnnl_primitive_destroy)
EMPTY_FUNC(dnnl_primitive_attr_destroy);

// EMPTY_FUNC(dnnl_status2str)
