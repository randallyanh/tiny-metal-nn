#pragma once

// RAII layer over BufferArena's slot pool: OwnedBufferHandle holds a slot
// and releases it on destruction; BorrowedBufferView is a non-owning view
// whose validity is generation-checked against the owning arena.
//
// These types live in tmnn::detail and never appear in include/tiny-metal-nn/.
// They wrap the existing BufferArena/BufferHandle/BufferView vocabulary
// without changing it; Phase 3 will retire the raw `BufferHandle` API once
// all internal call sites have migrated.

#include "tiny_metal_nn/runtime/buffer_handle.h"

#include <cstddef>
#include <cstdint>

namespace tmnn {

class BufferArena;

namespace detail {

class BorrowedBufferView;

// Move-only owning handle. dtor calls arena->release(handle); copy is deleted
// to enforce single ownership.
class OwnedBufferHandle {
public:
  OwnedBufferHandle() noexcept = default;
  ~OwnedBufferHandle() noexcept;

  OwnedBufferHandle(OwnedBufferHandle&& other) noexcept;
  OwnedBufferHandle& operator=(OwnedBufferHandle&& other) noexcept;
  OwnedBufferHandle(const OwnedBufferHandle&) = delete;
  OwnedBufferHandle& operator=(const OwnedBufferHandle&) = delete;

  // Generation-checked. False for null, false for moved-from, false if the
  // arena has recycled this slot under us (defensive; should not happen
  // during normal lifetime).
  [[nodiscard]] bool valid() const noexcept;

  [[nodiscard]] BufferHandle raw() const noexcept { return handle_; }
  [[nodiscard]] BufferArena* arena() const noexcept { return arena_; }

  // Borrowed view of the whole buffer. The view is non-owning; do not
  // outlive *this.
  [[nodiscard]] BorrowedBufferView view() const noexcept;
  [[nodiscard]] BorrowedBufferView sub_view(std::size_t offset,
                                            std::size_t bytes) const noexcept;

  // Carrier for the non-RAII state used to hand ownership across an external
  // boundary (e.g. adopt_external in §3.4). Caller becomes responsible for
  // calling arena->release(handle) when done.
  struct Raw {
    BufferArena* arena = nullptr;
    BufferHandle handle{};
  };
  [[nodiscard]] Raw release() noexcept;

private:
  friend class tmnn::BufferArena;
  OwnedBufferHandle(BufferArena* a, BufferHandle h) noexcept
      : arena_(a), handle_(h) {}

  BufferArena* arena_ = nullptr;
  BufferHandle handle_{};  // {0, 0} = null state
};

// Non-owning view. valid() is the runtime guard against use-after-release
// of the owning OwnedBufferHandle.
class BorrowedBufferView {
public:
  BorrowedBufferView() noexcept = default;

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] BufferHandle handle() const noexcept { return view_.handle; }
  [[nodiscard]] std::size_t bytes() const noexcept { return view_.bytes; }
  [[nodiscard]] std::size_t offset() const noexcept { return view_.offset; }
  [[nodiscard]] void* cpu_data() const noexcept { return view_.data; }
  [[nodiscard]] void* gpu_buffer() const noexcept { return view_.gpu_buffer; }
  [[nodiscard]] const BufferView& raw() const noexcept { return view_; }

  [[nodiscard]] BorrowedBufferView sub_view(std::size_t offset,
                                            std::size_t bytes) const noexcept;
  [[nodiscard]] BufferBinding bind(std::uint32_t binding_index) const noexcept;

private:
  friend class OwnedBufferHandle;
  BorrowedBufferView(BufferArena* a, BufferView v) noexcept
      : arena_(a), view_(v) {}

  BufferArena* arena_ = nullptr;
  BufferView view_{};
};

}  // namespace detail
}  // namespace tmnn
