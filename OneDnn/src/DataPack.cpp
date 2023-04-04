#include "DataPack.h"
#include <iostream>
#include <stddef.h>
#include <stdexcept>

void *hash_it(char *start, size_t size) {
  size_t hash = 0;
  while (size--) {
    hash += (*start);
    start++;
  }
  return (void *)hash;
}

#define MASK8 (SIZE_MAX ^ 7)

#define ALLIGN8(SIZE) (((SIZE) + 7) & MASK8)

void DataPack ::grow(size_t size, const void *data) {
  size_t size8 = ALLIGN8(size) + 8;
  buff.resize(buff.size() + size8);
  auto end = buff.end();

  std::copy((uint8_t *)&size, ((uint8_t *)&size) + sizeof(size_t),
            end - size8);

  if (data)
    std::copy((uint8_t *)data, ((uint8_t *)data) + size,
              end - (size8 - sizeof(size_t)));
}

size_t DataPack ::size() { return buff.size(); }

void *DataPack ::ptr() {
  // Hash();
  return (void *)&buff.front();
}

void DataPack ::Hash() {
  std::cerr << (void *)this << " Phash: " << hash_it(&buff[0], buff.size())
            << std::endl;
}

template <> DataPack &operator<<(DataPack &sp, char val[]) {
  sp.grow(strlen(val) + 1, (void *)val);
  return sp;
}

template <> DataPack &operator<<(DataPack &sp, const char *val) {
  size_t len = strlen(val) + 1;
  sp.grow(len, (void *)val);
  return sp;
}

template DataPack &operator<<(DataPack &sp, Array<int> val);

void *DataUnpack ::shrink(size_t size) {
  size_t size8 = ALLIGN8(size); // Size padded to be multiple of 8
  if (size + sizeof(size_t) > this->size)
    throw std::runtime_error("Buffer Overrun");
  if (size != nextSize())
    throw std::runtime_error("Size Missmatch");
  ptr = ((char *)ptr) + sizeof(size_t);
  this->size -= sizeof(size_t);
  void *ret = ptr;
  ptr = ((char *)ptr) + size8;
  this->size -= size8;
  return ret;
}

size_t DataUnpack ::nextSize() const { return *(size_t *)ptr; }

size_t DataUnpack ::nextSizeChecked(size_t expected) const {
  size_t next = nextSize();
  if (next != expected)
    throw std::runtime_error(
        "DataUnpack size missmatch (Next:" + std::to_string(next) +
        " Expected: " + std::to_string(expected) + ")");
  return next;
}

void DataUnpack ::Hash() {
  std::cerr << (void *)this << " Uhash: " << hash_it((char *)ptr, size)
            << std::endl;
}

DataUnpack ::~DataUnpack() noexcept(false) {
  if (size)
    throw std::runtime_error("DataUnpack Partially Unused (" +
                             std::to_string(size) + ")");
}

template <> DataUnpack &operator>>(DataUnpack &sp, char const *&val) {
  size_t size = sp.nextSize();
  char *src = (char *)sp.shrink(size);
  val = src;
  return sp;
}
