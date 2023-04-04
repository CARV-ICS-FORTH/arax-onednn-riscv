#include "DataPack.h"
#include "Utils.h"
#include "arax.h"
#include "dnnl.h"
#include <iostream>
#include "PrimitiveMap.h"
// Politically incorect inclusion of private headers ... i feel dirty
#include "common/c_types_map.hpp"

union op_desc_union {
  dnnl_convolution_desc_t conv_desc;
};

#ifdef ARAX_HANDLERS
ARAX_HANDLER(dnnl_primitive_desc_iterator_create, CPU) {
  TRACE_CALL();
  Array<dnnl_status_t> status(1);
  Array<dnnl_primitive_desc_iterator_t> iterator(1);
  size_t op_desc_size;

  const_dnnl_primitive_attr_t attr;
  dnnl_engine_t engine;
  const_dnnl_primitive_desc_t hint_forward_primitive_desc;

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> iterator >> op_desc_size;

  Array<char> op_desc(op_desc_size);

  upack >> op_desc >> attr >> engine >>
      hint_forward_primitive_desc;

  status[0] = dnnl_primitive_desc_iterator_create(
      &iterator[0], &op_desc[0], attr, engine, hint_forward_primitive_desc);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_primitive_desc_iterator_fetch, CPU) {
  TRACE_CALL();
  DataUnpack upack((arax_task_msg_s *)task);

  Array<dnnl_primitive_desc_t> desc(1);
  Array<const_dnnl_primitive_desc_iterator_t> iterator(1);

  upack >> desc;
  upack >> iterator;

  desc[0] = dnnl_primitive_desc_iterator_fetch(iterator[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_primitive_desc_query_md, CPU) {
  TRACE_CALL();
  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_memory_desc_t> ret(1);

  const_dnnl_primitive_desc_t primitive_desc;
  dnnl_query_t what;
  int index;

  upack >> ret;
  upack >> primitive_desc;
  upack >> what;
  upack >> index;

  ret[0] = *dnnl_primitive_desc_query_md(primitive_desc, what, index);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_primitive_attr_create, CPU) {
  TRACE_CALL();
  DataUnpack upack((arax_task_msg_s *)task);

  Array<dnnl_status_t> ret(1);
  Array<dnnl_primitive_attr_t> attr(1);

  upack >> ret;
  upack >> attr;

  ret[0] = dnnl_primitive_attr_create(&attr[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_reorder_primitive_desc_create, CPU) {
  TRACE_CALL();
  DataUnpack upack((arax_task_msg_s *)task);

  Array<dnnl_status_t> ret(1);
  Array<dnnl_primitive_desc_t> reorder_primitive_desc(1);
  Array<const dnnl_memory_desc_t> src_desc(1);
  dnnl_engine_t src_engine;
  Array<const dnnl_memory_desc_t> dst_desc(1);
  dnnl_engine_t dst_engine;
  const_dnnl_primitive_attr_t attr;

  upack >> ret;
  upack >> reorder_primitive_desc;
  upack >> src_desc;
  upack >> src_engine;
  upack >> dst_desc;
  upack >> dst_engine;
  upack >> attr;

  ret[0] = dnnl_reorder_primitive_desc_create(&reorder_primitive_desc[0],
                                              &src_desc[0], src_engine,
                                              &dst_desc[0], dst_engine, attr);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_primitive_create, CPU) {
  TRACE_CALL();
  Array<dnnl_status_t> ret(1);
  Array<dnnl_primitive_t> primitive(1);
  Array<const_dnnl_primitive_desc_t> primitive_desc(1);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> ret;
  upack >> primitive;
  upack >> primitive_desc;

  ret[0] = dnnl_primitive_create(&primitive[0], primitive_desc[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_primitive_execute, CPU) {
  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_status_t> ret(1);
  Array<const_dnnl_primitive_t> primitive(1);
  dnnl_stream_t stream;
  int nargs;

  upack >> ret;
  upack >> primitive;
  upack >> stream;
  upack >> nargs;

  Array<const dnnl_exec_arg_t> args(nargs);
  upack >> args;

  ret[0] = dnnl_primitive_execute(primitive[0], stream, nargs, &args[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

#endif

#ifdef ARAX_WRAPPERS
extern "C" {

dnnl_status_t DNNL_API dnnl_primitive_desc_iterator_create(
    dnnl_primitive_desc_iterator_t *iterator, const_dnnl_op_desc_t op_desc,
    const_dnnl_primitive_attr_t attr, dnnl_engine_t engine,
    const_dnnl_primitive_desc_t hint_forward_primitive_desc) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;

  dnnl_primitive_kind_t * kind = (dnnl_primitive_kind_t *)op_desc;

  pack << status;
  pack << Array<dnnl_primitive_desc_iterator_t>(iterator,1);
  pack << Primitive::Size.at(*kind);
  pack << Array<char>((char *)op_desc, Primitive::Size.at(*kind));
  pack << attr;
  pack << engine;
  pack << hint_forward_primitive_desc;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status;
  upack >> Array<dnnl_primitive_desc_iterator_t>(iterator,1);
  size_t temp;
  upack >> temp;
  upack >>
      Array<char>((char *)op_desc, Primitive::Size.at(*kind));
  upack >> attr;
  upack >> engine;
  upack >> hint_forward_primitive_desc;

  arax_task_free(task);
  return status;
}

dnnl_primitive_desc_t DNNL_API dnnl_primitive_desc_iterator_fetch(
    const_dnnl_primitive_desc_iterator_t iterator) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_primitive_desc_t desc;

  pack << desc;
  pack << iterator;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> desc;
  upack >> iterator;

  arax_task_free(task);

  return desc;
}

const dnnl_memory_desc_t *DNNL_API dnnl_primitive_desc_query_md(
    const_dnnl_primitive_desc_t primitive_desc, dnnl_query_t what, int index) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_memory_desc_t *ret = new dnnl_memory_desc_t;

  std::cerr << __func__ << " allocated md: " << ret << std::endl;

  pack << Array<const dnnl_memory_desc_t>(ret, 1);
  pack << primitive_desc;
  pack << what;
  pack << index;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> Array<dnnl_memory_desc_t>(ret, 1);
  upack >> primitive_desc;
  upack >> what;
  upack >> index;

  arax_task_free(task);

  return ret;
}

dnnl_status_t DNNL_API dnnl_primitive_attr_create(dnnl_primitive_attr_t *attr) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;

  dnnl_status_t ret;
  Array<dnnl_status_t> _ret(&ret, 1);

  pack << _ret;
  pack << Array<dnnl_primitive_attr_t>(attr, 1);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> _ret;
  upack >> Array<dnnl_primitive_attr_t>(attr, 1);

  arax_task_free(task);

  return ret;
}

dnnl_status_t DNNL_API dnnl_reorder_primitive_desc_create(
    dnnl_primitive_desc_t *reorder_primitive_desc,
    const dnnl_memory_desc_t *src_desc, dnnl_engine_t src_engine,
    const dnnl_memory_desc_t *dst_desc, dnnl_engine_t dst_engine,
    const_dnnl_primitive_attr_t attr) {
  dnnl_status_t ret = dnnl_runtime_error;
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;

  pack << ret;
  pack << Array<dnnl_primitive_desc_t>(reorder_primitive_desc, 1);
  pack << Array<const dnnl_memory_desc_t>(src_desc, 1);
  pack << src_engine;
  pack << Array<const dnnl_memory_desc_t>(dst_desc, 1);
  pack << dst_engine;
  pack << attr;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> ret;
  upack >> Array<dnnl_primitive_desc_t>(reorder_primitive_desc, 1);
  upack >> Array<const dnnl_memory_desc_t>(src_desc, 1);
  upack >> src_engine;
  upack >> Array<const dnnl_memory_desc_t>(dst_desc, 1);
  upack >> dst_engine;
  upack >> attr;

  arax_task_free(task);

  return ret;
}

dnnl_status_t DNNL_API dnnl_primitive_create(
    dnnl_primitive_t *primitive, const_dnnl_primitive_desc_t primitive_desc) {
  dnnl_status_t ret = dnnl_runtime_error;
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;

  pack << ret;
  pack << Array<dnnl_primitive_t>(primitive, 1);
  pack << primitive_desc;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> ret;
  upack >> Array<dnnl_primitive_t>(primitive, 1);
  upack >> primitive_desc;

  arax_task_free(task);

  return ret;
}

dnnl_status_t DNNL_API dnnl_primitive_execute(const_dnnl_primitive_t primitive,
                                              dnnl_stream_t stream, int nargs,
                                              const dnnl_exec_arg_t *args) {
  dnnl_status_t ret = dnnl_runtime_error;
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;

  pack << ret;
  pack << Array<const_dnnl_primitive_t>(&primitive, 1);
  pack << stream;
  pack << nargs;
  pack << Array<const dnnl_exec_arg_t>(args, nargs);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> ret;
  upack >> Array<const_dnnl_primitive_t>(&primitive, 1);
  upack >> stream;
  upack >> nargs;
  upack >> Array<const dnnl_exec_arg_t>(args, nargs);

  arax_task_free(task);

  return ret;
}

dnnl_status_t DNNL_API dnnl_primitive_desc_iterator_destroy(dnnl_primitive_desc_iterator_t) {
  dnnl_status_t ret = dnnl_success;
  TRACE_CALL();
  return ret;
}
}
#endif
