#pragma once

/**
 * @file network_with_input_encoding.h
 * @brief Lightweight composition of Encoding + Network descriptors.
 */

#include "tiny-metal-nn/encoding.h"
#include "tiny-metal-nn/network.h"
#include "tiny-metal-nn/detail/network_planning.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace tmnn {

class NetworkWithInputEncoding final : public Module {
public:
  NetworkWithInputEncoding(std::shared_ptr<Encoding> encoding,
                           std::shared_ptr<Network> network)
      : encoding_(std::move(encoding)), network_(std::move(network)) {
    if (!encoding_)
      throw std::invalid_argument(
          "NetworkWithInputEncoding: encoding must not be null");
    if (!network_)
      throw std::invalid_argument(
          "NetworkWithInputEncoding: network must not be null");
    if (encoding_->n_output_dims() != network_->n_input_dims()) {
      throw std::invalid_argument(
          "NetworkWithInputEncoding: encoding output dims must match network "
          "input dims");
    }
  }

  [[nodiscard]] int n_input_dims() const override {
    return encoding_->n_input_dims();
  }

  [[nodiscard]] int n_output_dims() const override {
    return network_->n_output_dims();
  }

  [[nodiscard]] int n_params() const override {
    return encoding_->n_params() + network_->n_params();
  }

  [[nodiscard]] std::string name() const override {
    return "NetworkWithInputEncoding";
  }

  [[nodiscard]] const std::shared_ptr<Encoding> &encoding() const {
    return encoding_;
  }

  [[nodiscard]] const std::shared_ptr<Network> &network() const {
    return network_;
  }

  [[nodiscard]] NetworkPlan
  plan(const NetworkFactoryOptions &options = {}) const;

private:
  std::shared_ptr<Encoding> encoding_;
  std::shared_ptr<Network> network_;
};

} // namespace tmnn
