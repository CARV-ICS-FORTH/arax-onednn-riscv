#include "DataPack.h"
#include "Utils.h"
#include "arax.h"
#include "dnnl.h"

#ifdef ARAX_HANDLERS

ARAX_HANDLER(dnnl_memory_desc_init_by_tag, CPU) {
  TRACE_CALL();
  Array<dnnl_memory_desc_t> memory_desc(1);
  int ndims;
  Array<long> dims(ndims);
  dnnl_data_type_t data_type;
  dnnl_format_tag_t tag;
  Array<dnnl_status_t> status(1);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> memory_desc >> ndims >> dims >> data_type >> tag;

  status[0] = dnnl_memory_desc_init_by_tag(&memory_desc[0], ndims, &dims[0],
                                           data_type, tag);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_memory_desc_init_by_strides, CPU) {
  TRACE_CALL();
  Array<dnnl_status_t> status(1);
  Array<dnnl_memory_desc_t> _memory_desc(1);
  int ndims;
  Array<const long> dims(12);
  dnnl_data_type_t data_type;
  Array<const long> strides(12);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status;
  upack >> _memory_desc;
  upack >> ndims;
  upack >> dims;
  upack >> data_type;
  upack >> strides;

  status[0] = dnnl_memory_desc_init_by_strides(
    &_memory_desc[0], ndims, &dims[0], data_type, &strides[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_memory_desc_get_size, CPU) {
  TRACE_CALL();
  DataUnpack upack((arax_task_msg_s *)task);
  Array<size_t> size(1);
  dnnl_memory_desc_t desc;

  upack >> size >> desc;

  size[0] = dnnl_memory_desc_get_size(&desc);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}
#endif

#ifdef ARAX_WRAPPERS
extern "C" {
dnnl_status_t DNNL_API dnnl_memory_desc_init_by_tag(
    dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
    dnnl_data_type_t data_type, dnnl_format_tag_t tag) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  Array<long> _dims((long *)dims, ndims);
  pack << status << Array<dnnl_memory_desc_t>(memory_desc, 1) << ndims << _dims
       << data_type << tag;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  dnnl_dims_t trash;

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> Array<dnnl_memory_desc_t>(memory_desc, 1) >> ndims >>
      Array<long>(trash, ndims);

  upack >> data_type >> tag;

  arax_task_free(task);
  return status;
}

dnnl_status_t DNNL_API dnnl_memory_desc_init_by_strides(
  dnnl_memory_desc_t *memory_desc, int ndims, const dnnl_dims_t dims,
  dnnl_data_type_t data_type, const dnnl_dims_t strides) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  Array<dnnl_memory_desc_t> _memory_desc(memory_desc, 1);

  pack << status;
  pack << _memory_desc;
  pack << ndims;
  pack << Array<const long>(dims, 12);
  pack << data_type;
  pack << Array<const long>(strides, 12);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status;
  upack >> _memory_desc;
  upack >> ndims;
  upack >> Array<const long>(dims, 12);
  upack >> data_type;
  upack >> Array<const long>(strides, 12);

  arax_task_free(task);
  return status;
  }

size_t DNNL_API
dnnl_memory_desc_get_size(const dnnl_memory_desc_t *memory_desc) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  size_t size;
  pack << size << *memory_desc;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  dnnl_memory_desc_t temp;
  upack >> size >> temp;

  return size;
}
}
#endif
