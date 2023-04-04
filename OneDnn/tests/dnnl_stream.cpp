#include <catch2/catch_test_macros.hpp>

#include "dnnl.h"

TEST_CASE( "Init") {

	SECTION("dnnl_engine")
	{
		dnnl_engine_t engine;

		REQUIRE(dnnl_engine_create(&engine,dnnl_cpu,0 ) == dnnl_success);

		SECTION("get_kind")
		{
			dnnl_engine_kind_t kind;
			REQUIRE(dnnl_engine_get_kind(engine, &kind) == dnnl_success);
			REQUIRE(kind == dnnl_cpu);
		}

		SECTION("dnnl_memory_desc")
		{
			dnnl_memory_desc_t memory_desc;
			dnnl_dim_t dims[] = {2,2};
			float data[] = {1,2,3,4};

			REQUIRE(dnnl_memory_desc_init_by_tag(&memory_desc,2,dims,dnnl_f32,dnnl_ab) == dnnl_success);

			SECTION("dnnl_memory")
			{
				dnnl_memory_t memory;

				REQUIRE(dnnl_memory_create(&memory,&memory_desc,engine,data) == dnnl_success);

				SECTION("get_engine")
				{
					dnnl_engine_t mem_engine;
					REQUIRE(dnnl_memory_get_engine(memory,&mem_engine) == dnnl_success);
					dnnl_engine_kind_t kind = dnnl_any_engine;
					dnnl_engine_get_kind(mem_engine, &kind);
					REQUIRE(kind == dnnl_cpu);
				}

				SECTION("get_data_handle")
				{
					void *ptr = (void*)0xBAADF00D;
					REQUIRE(dnnl_memory_get_data_handle(memory,&ptr) == dnnl_success);
					REQUIRE(ptr != (void*)(0xBAADF00D) );
				}

				REQUIRE(dnnl_memory_destroy(memory) == dnnl_success);
			}
		}
		SECTION("dnnl_stream")
		{
			dnnl_stream_t stream;
			REQUIRE(dnnl_stream_create(&stream,engine,0) == dnnl_success);
			REQUIRE(dnnl_stream_destroy(stream) == dnnl_success);
		}
	}
}
