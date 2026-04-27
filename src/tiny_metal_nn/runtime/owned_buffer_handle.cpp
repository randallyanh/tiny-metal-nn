#include "tiny_metal_nn/runtime/owned_buffer_handle.h"

#include "tiny_metal_nn/runtime/buffer_arena.h"

namespace tmnn::detail {

OwnedBufferHandle::~OwnedBufferHandle() noexcept {
  if (arena_ && arena_->is_valid(handle_)) {
    arena_->release(handle_);
  }
}

OwnedBufferHandle::OwnedBufferHandle(OwnedBufferHandle&& other) noexcept
    : arena_(other.arena_), handle_(other.handle_) {
  other.arena_ = nullptr;
  other.handle_ = {};
}

OwnedBufferHandle& OwnedBufferHandle::operator=(
    OwnedBufferHandle&& other) noexcept {
  if (this != &other) {
    if (arena_ && arena_->is_valid(handle_)) {
      arena_->release(handle_);
    }
    arena_ = other.arena_;
    handle_ = other.handle_;
    other.arena_ = nullptr;
    other.handle_ = {};
  }
  return *this;
}

bool OwnedBufferHandle::valid() const noexcept {
  return arena_ != nullptr && arena_->is_valid(handle_);
}

BorrowedBufferView OwnedBufferHandle::view() const noexcept {
  if (!valid()) {
    return {};
  }
  return BorrowedBufferView(arena_, arena_->view(handle_));
}

BorrowedBufferView OwnedBufferHandle::sub_view(std::size_t offset,
                                               std::size_t bytes) const noexcept {
  if (!valid()) {
    return {};
  }
  return BorrowedBufferView(arena_,
                            BufferArena::sub_view(arena_->view(handle_),
                                                  offset, bytes));
}

OwnedBufferHandle::Raw OwnedBufferHandle::release() noexcept {
  Raw r{arena_, handle_};
  arena_ = nullptr;
  handle_ = {};
  return r;
}

bool BorrowedBufferView::valid() const noexcept {
  return arena_ != nullptr && arena_->is_valid(view_.handle);
}

BorrowedBufferView BorrowedBufferView::sub_view(std::size_t offset,
                                                std::size_t bytes) const noexcept {
  return BorrowedBufferView(arena_,
                            BufferArena::sub_view(view_, offset, bytes));
}

BufferBinding BorrowedBufferView::bind(std::uint32_t binding_index) const noexcept {
  return BufferArena::bind(view_, binding_index);
}

}  // namespace tmnn::detail
