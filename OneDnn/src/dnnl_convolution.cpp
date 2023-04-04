#include "DataPack.h"
#include "Utils.h"
#include "arax.h"
#include "dnnl.h"

#ifdef ARAX_HANDLERS
ARAX_HANDLER(dnnl_convolution_forward_desc_init, CPU) {
  TRACE_CALL();
  Array<dnnl_status_t> status(1);
  Array<dnnl_convolution_desc_t> _conv_desc(1);
  dnnl_prop_kind_t prop_kind;
  dnnl_alg_kind_t alg_kind;
  Array<dnnl_memory_desc_t> src_desc(1);
  Array<dnnl_memory_desc_t> weights_desc(1);
  Array<dnnl_memory_desc_t> bias_desc(1);
  Array<dnnl_memory_desc_t> dst_desc(1);
  Array<long> strides(12);
  Array<long> padding_l(12);
  Array<long> padding_r(12);

  DataUnpack upack((arax_task_msg_s *)task);

  upack >> status >> _conv_desc >> prop_kind >> alg_kind >> src_desc >>
      weights_desc >> bias_desc >> dst_desc >> strides >> padding_l >>
      padding_r;

  status[0] = dnnl_convolution_forward_desc_init(
      &_conv_desc[0], prop_kind, alg_kind, &src_desc[0], &weights_desc[0],
      /*&bias_desc[0]*/ 0, &dst_desc[0], &strides[0], &padding_l[0],
      &padding_r[0]);

  arax_task_mark_done(task, task_completed);
  return task_completed;
}
#endif

#ifdef ARAX_WRAPPERS
extern "C" {
dnnl_status_t dnnl_convolution_forward_desc_init(
    dnnl_convolution_desc_t *conv_desc, dnnl_prop_kind_t prop_kind,
    dnnl_alg_kind_t alg_kind, const dnnl_memory_desc_t *src_desc,
    const dnnl_memory_desc_t *weights_desc, const dnnl_memory_desc_t *bias_desc,
    const dnnl_memory_desc_t *dst_desc, const dnnl_dims_t strides,
    const dnnl_dims_t padding_l, const dnnl_dims_t padding_r) {
  TRACE_CALL();
  arax_init();
  static arax_proc *proc = arax_proc_get(__func__);
  DataPack pack;
  Array<dnnl_convolution_desc_t> _conv_desc(conv_desc, 1);
  dnnl_status_t status = dnnl_runtime_error;
  pack << status << _conv_desc << prop_kind << alg_kind
       << Array<const dnnl_memory_desc_t>(src_desc, 1)
       << Array<const dnnl_memory_desc_t>(weights_desc, 1)
       << Array<const dnnl_memory_desc_t>(bias_desc, 1)
       << Array<const dnnl_memory_desc_t>(dst_desc, 1)
       << Array<const long>(strides, 12) << Array<const long>(padding_l, 12)
       << Array<const long>(padding_r, 12);

  auto task = arax_task_issue(getControllVAC(), proc, pack.ptr(), pack.size(),
                              0, 0, 0, 0);
  arax_task_wait(task);

  DataUnpack upack((arax_task_msg_s *)task);

  dnnl_memory_desc_t temp;
  upack >> status >> _conv_desc >> prop_kind >> alg_kind >>
      Array<dnnl_memory_desc_t>(&temp, 1) >>
      Array<dnnl_memory_desc_t>(&temp, 1) >>
      Array<dnnl_memory_desc_t>(&temp, 1) >>
      Array<dnnl_memory_desc_t>(&temp, 1) >> Array<const long>(strides, 12) >>
      Array<const long>(padding_l, 12) >> Array<const long>(padding_r, 12);

  arax_task_free(task);
  return status;
}
}

#endif
