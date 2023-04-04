#include "DataPack.h"
#include "Utils.h"
#include "arax.h"
#include "dnnl.h"
#include <iostream>
#include "core/arax_ptr.h"
#include "arax_pipe.h"

#ifdef ARAX_HANDLERS

ARAX_HANDLER(dnnl_memory_create, CPU) {
  TRACE_CALL();

  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_status_t> status(1);
  Array<dnnl_memory_desc_t> _memory_desc(1);
  Array<dnnl_memory_t> _memory(1);
  dnnl_engine_t engine;
  void *handle;
  upack >> status >> _memory >> _memory_desc >> engine >> handle;

  status[0] = dnnl_memory_create(&_memory[0], &_memory_desc[0], engine, handle);

  void *org_handle;
  dnnl_memory_get_data_handle(_memory[0],&org_handle);

  if(arax_ptr_valid(org_handle) == false)
  {
    const dnnl_memory_desc_t* md = 0;
    dnnl_memory_get_memory_desc(_memory[0], &md);
    size_t size = dnnl_memory_desc_get_size(md);

    arax_pipe_s *arax = arax_init();

    void * arax_ptr = arch_alloc_allocate(&(arax->allocator), size);
    // Not sure if this leaks org_handle?
    dnnl_memory_set_data_handle(_memory[0], arax_ptr);
  }

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_memory_get_engine, CPU) {
  TRACE_CALL();
  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_status_t> status(1);
  dnnl_memory_t memory;
  Array<dnnl_engine_t> engine(1);

  upack >> status >> memory >> engine;

  status[0] = dnnl_memory_get_engine(memory, &engine[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_memory_get_memory_desc, CPU) {
  TRACE_CALL();
  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_status_t> status(1);
  dnnl_memory_t memory;
  Array<dnnl_memory_desc_t> memory_desc(1);

  upack >> status;
  upack >> memory;
  upack >> memory_desc;

  const dnnl_memory_desc_t *temp = 0;

  status[0] = dnnl_memory_get_memory_desc(memory, &temp);

  memory_desc[0] = *temp;

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_memory_destroy, CPU) {
  TRACE_CALL();

  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_status_t> status(1);
  Array<dnnl_memory_t> _memory(1);
  upack >> status >> _memory;

  status[0] = dnnl_memory_destroy(_memory[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_memory_get_data_handle, CPU) {
  TRACE_CALL();

  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_status_t> status(1);
  Array<dnnl_memory_t> _memory(1);
  Array<void *> _handle(1);

  upack >> status >> _memory >> _handle;

  status[0] = dnnl_memory_get_data_handle(_memory[0], &_handle[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_memory_desc_equal, CPU) {
  TRACE_CALL();

  DataUnpack upack((arax_task_msg_s *)task);
  Array<int> ret(1);
  Array<const dnnl_memory_desc_t> lhs(1);
  Array<const dnnl_memory_desc_t> rhs(1);

  upack >> ret;
  upack >> lhs;
  upack >> rhs;

  ret[0] = dnnl_memory_desc_equal(&lhs[0], &rhs[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

#endif

#ifdef ARAX_WRAPPERS
extern "C" {
dnnl_status_t DNNL_API dnnl_memory_create(dnnl_memory_t *memory,
                                          const dnnl_memory_desc_t *memory_desc,
                                          dnnl_engine_t engine, void *handle) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << Array<dnnl_memory_t>(memory, 1)
       << Array<dnnl_memory_desc_t>((dnnl_memory_desc_t *)memory_desc, 1)
       << engine << handle;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_memory_desc_t> _memory_desc(1);
  upack >> status;
  upack >> Array<dnnl_memory_t>(memory, 1);
  upack >> _memory_desc;
  upack >> engine;
  upack >> handle;

  return status;
}

dnnl_status_t DNNL_API dnnl_memory_get_engine(const_dnnl_memory_t memory,
                                              dnnl_engine_t *engine) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << memory << Array<dnnl_engine_t>(engine, 1);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> memory >> Array<dnnl_engine_t>(engine, 1);
  return status;
}

dnnl_status_t DNNL_API dnnl_memory_get_memory_desc(
    const_dnnl_memory_t memory, const dnnl_memory_desc_t **memory_desc) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  pack << status;
  pack << memory;
  pack << Array<const dnnl_memory_desc_t>(1);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  dnnl_memory_desc_t *dmd = new dnnl_memory_desc_t;
  auto temp = Array<const dnnl_memory_desc_t>(dmd, 1);
  upack >> status;
  upack >> memory;
  upack >> temp;

  // Maybe should be in shm?
  std::cerr << __func__ << " alloc: " << dmd << std::endl;

  memcpy(dmd, &temp[0], sizeof(*dmd));

  *memory_desc = dmd;
  return status;
}

dnnl_status_t DNNL_API dnnl_memory_get_data_handle(const_dnnl_memory_t memory,
                                                   void **handle) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;

  pack << status << memory << handle;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> status >> memory >> Array(handle, 1);

  return status;
}

dnnl_status_t DNNL_API dnnl_memory_destroy(dnnl_memory_t memory) {
  TRACE_CALL();
  arax_init();

  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << Array<dnnl_memory_t>(&memory, 1);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  Array<dnnl_memory_t> _memory(1);
  upack >> status >> _memory;

  return status;
}

int DNNL_API dnnl_memory_desc_equal(const dnnl_memory_desc_t *lhs,
                                    const dnnl_memory_desc_t *rhs) {
  TRACE_CALL();
  arax_init();

  if (lhs == rhs) // Obvioulsy the same
    return 1;

  static arax_proc *proc = arax_proc_get(__func__);
  int ret;
  DataPack pack;

  pack << Array<int>(&ret, 1);
  pack << Array<const dnnl_memory_desc_t>(lhs, 1);
  pack << Array<const dnnl_memory_desc_t>(rhs, 1);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> ret;
  upack >> Array<const dnnl_memory_desc_t>(lhs, 1);
  upack >> Array<const dnnl_memory_desc_t>(rhs, 1);

  return ret;
}
}
#endif
