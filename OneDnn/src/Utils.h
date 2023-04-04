#ifndef ONEDNN_ARAX_UTILS
#define ONEDNN_ARAX_UTILS
#include "core/arax_vaccel.h"
#include <string>

arax_vaccel_s * getControllVAC();

class LogTracer
{
    public:
        LogTracer(std::string msg);
        ~LogTracer();
    private:
        std::string msg;
};

#ifdef TRACE_CALLS
#define TRACE_CALL() LogTracer _lt(__func__)
#else
#define TRACE_CALL() ;
#endif

#endif
