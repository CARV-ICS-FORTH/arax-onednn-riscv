#include "DataPack.h"
#include "Utils.h"
#include "arax.h"
#include "dnnl.h"

#ifdef ARAX_HANDLERS
ARAX_HANDLER(dnnl_eltwise_forward_desc_init, CPU) {
  TRACE_CALL();
  Array<dnnl_status_t> status(1);
  Array<dnnl_eltwise_desc_t> eltwise_desc(1);
  dnnl_prop_kind_t prop_kind;
  dnnl_alg_kind_t alg_kind;
  Array<const dnnl_memory_desc_t> data_desc(1);
  float alpha;
  float beta;

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> status >> eltwise_desc >> prop_kind >> alg_kind >> data_desc >>
      alpha >> beta;

  status[0] = dnnl_eltwise_forward_desc_init(
      &eltwise_desc[0], prop_kind, alg_kind, &data_desc[0], alpha, beta);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}
#endif

#ifdef ARAX_WRAPPERS
extern "C" {
dnnl_status_t DNNL_API dnnl_eltwise_forward_desc_init(
    dnnl_eltwise_desc_t *eltwise_desc, dnnl_prop_kind_t prop_kind,
    dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *data_desc, float alpha,
    float beta) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  Array<dnnl_eltwise_desc_t> _eltwise_desc(eltwise_desc, 1);
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << _eltwise_desc << prop_kind << alg_kind
       << Array<const dnnl_memory_desc_t>(data_desc, 1) << alpha << beta;

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  dnnl_memory_desc_t temp;
  upack >> status >> _eltwise_desc >> prop_kind >> alg_kind >>
      Array<dnnl_memory_desc_t>(&temp, 1) >> alpha >> beta;

  arax_task_free(task);
  return status;
}
}

#endif
