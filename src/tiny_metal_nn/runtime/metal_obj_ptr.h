#pragma once

// Move-only RAII wrapper for Objective-C objects under MRC.
// Only meaningful when included from an Objective-C++ translation unit
// (-fno-objc-arc on the .mm). Compiled out in pure C++ to avoid leaking
// id<...> types into runtime .cpp files.

#ifdef __OBJC__

#include <objc/objc.h>  // for nil

namespace tmnn::detail {

template <typename T>
class MetalObjPtr {
public:
  MetalObjPtr() noexcept = default;

  // adopt(): caller transfers an existing +1 retain (e.g. from
  // -newBufferWithLength: or +alloc/-init).
  static MetalObjPtr adopt(T obj) noexcept { return MetalObjPtr(obj); }

  // retain(): caller has a non-owning reference; we add a +1 retain.
  static MetalObjPtr retain(T obj) noexcept {
    if (obj) [obj retain];
    return MetalObjPtr(obj);
  }

  ~MetalObjPtr() noexcept {
    if (obj_) [obj_ release];
  }

  MetalObjPtr(MetalObjPtr&& other) noexcept : obj_(other.obj_) {
    other.obj_ = nil;
  }

  MetalObjPtr& operator=(MetalObjPtr&& other) noexcept {
    if (this != &other) {
      reset(other.release());
    }
    return *this;
  }

  MetalObjPtr(const MetalObjPtr&) = delete;
  MetalObjPtr& operator=(const MetalObjPtr&) = delete;

  [[nodiscard]] T get() const noexcept { return obj_; }
  [[nodiscard]] explicit operator bool() const noexcept { return obj_ != nil; }

  // Hands +1 retain back to caller; *this is left null.
  [[nodiscard]] T release() noexcept {
    T tmp = obj_;
    obj_ = nil;
    return tmp;
  }

  void reset(T new_obj = nil) noexcept {
    if (obj_) [obj_ release];
    obj_ = new_obj;
  }

private:
  explicit MetalObjPtr(T obj) noexcept : obj_(obj) {}
  T obj_ = nil;
};

}  // namespace tmnn::detail

#endif  // __OBJC__
