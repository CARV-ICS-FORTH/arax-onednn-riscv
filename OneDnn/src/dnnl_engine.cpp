#include "DataPack.h"
#include "Utils.h"
#include "arax.h"
#include "dnnl.h"

#ifdef ARAX_HANDLERS

ARAX_HANDLER(dnnl_engine_create, CPU) {
  TRACE_CALL();
  Array<dnnl_engine_t> engine(1);
  dnnl_engine_kind_t kind;
  size_t index;
  Array<dnnl_status_t> status(1);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> status >> engine >> kind >> index;

  status[0] = dnnl_engine_create(&engine[0], kind, index);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_engine_destroy, CPU) {
  TRACE_CALL();
  Array<dnnl_engine_t> engine(1);
  dnnl_engine_kind_t kind;
  size_t index;
  Array<dnnl_status_t> status(1);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> status >> engine;

  status[0] = dnnl_engine_destroy(engine[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_engine_get_kind, CPU) {
  TRACE_CALL();
  Array<dnnl_status_t> status(1);
  Array<dnnl_engine_kind_t> kind(1);
  Array<dnnl_engine_t> engine(1);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> status >> engine >> kind;

  status[0] = dnnl_engine_get_kind(engine[0], &kind[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}
#endif

#ifdef ARAX_WRAPPERS
extern "C" dnnl_status_t dnnl_engine_create(dnnl_engine_t *engine,
                                            dnnl_engine_kind_t kind,
                                            size_t index) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << engine << kind << index;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> Array<dnnl_engine_t>(engine, 1) >> kind >> index;

  arax_task_free(task);
  return status;
}

extern "C" dnnl_status_t dnnl_engine_destroy(dnnl_engine_t engine) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << engine;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> Array<dnnl_engine_t>(&engine, 1);

  arax_task_free(task);
  return status;
}

dnnl_status_t dnnl_engine_get_kind(dnnl_engine_t engine,
                                   dnnl_engine_kind_t *kind) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << engine << dnnl_any_engine;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> Array<dnnl_engine_t>(&engine, 1) >> *kind;

  arax_task_free(task);
  return status;
}
#endif
