#include "DataPack.h"
#include "Utils.h"
#include "arax.h"
#include "dnnl.h"

#ifdef ARAX_HANDLERS

ARAX_HANDLER(dnnl_stream_create, CPU) {
  TRACE_CALL();
  Array<dnnl_stream_t> stream(1);
  dnnl_engine_t engine;
  unsigned flags;
  Array<dnnl_status_t> status(1);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> status >> stream >> engine >> flags;

  status[0] = dnnl_stream_create(&stream[0], engine, flags);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

ARAX_HANDLER(dnnl_stream_wait, CPU) {
  TRACE_CALL();

  DataUnpack upack((arax_task_msg_s *)task);
  Array<dnnl_status_t> status(1);
  Array<dnnl_stream_t> stream(1);
  upack >> status;
  upack >> stream;

  status[0] = dnnl_stream_wait(stream[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}

#endif

#ifdef ARAX_WRAPPERS
extern "C" {

dnnl_status_t DNNL_API dnnl_stream_create(dnnl_stream_t *stream,
                                          dnnl_engine_t engine,
                                          unsigned flags) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  dnnl_status_t status = dnnl_runtime_error;
  Array<dnnl_stream_t> _stream(1);
  pack << status << _stream << engine << flags;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status >> Array<dnnl_stream_t>(stream, 1) >> engine >> flags;

  arax_task_free(task);
  return status;
}

dnnl_status_t DNNL_API dnnl_stream_wait(dnnl_stream_t stream) {
  dnnl_status_t status = dnnl_runtime_error;
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;

  pack << status;
  pack << Array<dnnl_stream_t>(&stream, 1);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);
  upack >> status;
  upack >> Array<dnnl_stream_t>(&stream, 1);

  arax_task_free(task);
  return status;
}

dnnl_status_t DNNL_API dnnl_stream_destroy(dnnl_stream_t stream) {
  TRACE_CALL();
  return dnnl_success;
}
}
#endif
