#ifndef SCALAR_PACK_HEADER
#define SCALAR_PACK_HEADER
#include <vector>
#include <cstring>
#include <cstdint>
#include <core/arax_task.h>

template<class T>
struct Array
{
  /**
   * Reference initialization.
   * base pointer will be loaded from DataUnpack>> operation
   */
  Array(size_t len)
  : base(0), len(len) {}
  Array(T *base,size_t len)
  : base(base), len(len) {}
  T *base;
  size_t len;

  T & operator[](int index)
  {
    return base[index];
  }

  size_t Size() const
  {
    return sizeof(T)*len;
  }
};


class DataPack
{
  public:
    void grow(size_t size,const void *data = 0);
    size_t size();
    void * ptr();
    void Hash();
  private:
    std::vector<char> buff;
};

template<class T>
DataPack & operator<<(DataPack &sp, T val)
{
  sp.grow(sizeof(val), (void*)&val);
  return sp;
}

template<class T,size_t N>
DataPack & operator<<(DataPack &sp, T (&array)[N])
{
  sp.grow(sizeof(T)*N,array);
  return sp;
}

template<>
DataPack & operator<<(DataPack &sp, char val[]);

template<>
DataPack & operator<<(DataPack &sp, const char *val);

template<class T>
DataPack & operator<<(DataPack &sp, Array<T> val)
{
  sp.grow(val.Size(), (void*)val.base);
  return sp;
}

class DataUnpack
{
  public:
    DataUnpack(arax_task_msg_s *task)
    : size(task->host_size), ptr(arax_task_host_data(task,size)) {/*Hash();*/}
    DataUnpack(void *ptr,size_t size)
    : size(size), ptr(ptr) {}
    void * shrink(size_t size);
    size_t nextSize() const;
    size_t nextSizeChecked(size_t expected) const;
    void Hash();
    ~DataUnpack() noexcept(false);
    template<class T>
    operator T&()
    {
        return *(T*)shrink(nextSize());
    }
  private:
    size_t size;
    void *ptr;
};

template<class T>
DataUnpack & operator>>(DataUnpack &su, T & val)
{
  val = *(T*)su.shrink(sizeof(T));
  return su;
}

template<>
DataUnpack & operator>>(DataUnpack &sp, char const *&val);

template<class T>
DataUnpack & operator>>(DataUnpack &sp,const Array<T> &arr)
{
  size_t size = sp.nextSizeChecked(arr.Size());
  char *src = (char*)sp.shrink(size);
  std::copy(src,src+size,(char*)arr.base);
  return sp;
}

template<class T>
DataUnpack & operator>>(DataUnpack &sp,Array<T> &arr)
{
  size_t size = sp.nextSize();
  char *src = (char*)sp.shrink(size);
  if(arr.base)
    std::copy(src,src+size,(char*)arr.base);
  else
    arr.base = (T*)src;
  return sp;
}

#endif
