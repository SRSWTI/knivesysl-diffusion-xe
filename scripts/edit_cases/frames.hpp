#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace frame_demo {

using Bytes = std::vector<std::uint8_t>;
using Frames = std::vector<Bytes>;

struct Statistics {
    std::size_t bytes_received = 0;
    std::size_t frames_delivered = 0;
    std::size_t protocol_errors = 0;
};

// Wire format: a four-byte unsigned big-endian payload length, then payload.
// Empty payloads are valid. There is no delimiter, padding, or terminator.
inline Bytes encode_frame(const Bytes& payload) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error("payload cannot fit in the wire header");
    }
    const auto length = static_cast<std::uint32_t>(payload.size());
    Bytes wire;
    wire.reserve(payload.size() + 4);
    wire.push_back(static_cast<std::uint8_t>(length >> 24));
    wire.push_back(static_cast<std::uint8_t>(length >> 16));
    wire.push_back(static_cast<std::uint8_t>(length >> 8));
    wire.push_back(static_cast<std::uint8_t>(length));
    wire.insert(wire.end(), payload.begin(), payload.end());
    return wire;
}

inline Bytes concatenate(const Frames& frames) {
    Bytes wire;
    for (const auto& payload : frames) {
        auto encoded = encode_frame(payload);
        wire.insert(wire.end(), encoded.begin(), encoded.end());
    }
    return wire;
}

class Decoder {
public:
    explicit Decoder(std::size_t maximum_payload = 1024 * 1024)
        : maximum_payload_(maximum_payload) {
        if (maximum_payload_ > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("maximum payload exceeds wire limit");
        }
    }

    // Chunk boundaries are arbitrary: a header or payload may span many calls.
    // Oversized headers cause length_error and discard pending input. The same
    // decoder can be reused for a fresh connection after that exception.
    Frames feed(const Bytes& chunk) {
        stats_.bytes_received += chunk.size();
        pending_.insert(pending_.end(), chunk.begin(), chunk.end());
        Frames output;
        while (pending_.size() >= 4) {
            const std::uint32_t length =
                std::uint32_t(pending_[0]) |
                (std::uint32_t(pending_[1]) << 8) |
                (std::uint32_t(pending_[2]) << 16) |
                (std::uint32_t(pending_[3]) << 24);
            if (length > maximum_payload_) {
                ++stats_.protocol_errors;
                pending_.clear();
                throw std::length_error("frame exceeds configured maximum");
            }
            if (pending_.size() < 4 + std::size_t(length)) {
                pending_.clear();
                break;
            }
            if (length != 0) {
                output.emplace_back(pending_.begin() + 4,
                                    pending_.begin() + 4 + length);
                ++stats_.frames_delivered;
            }
            pending_.erase(pending_.begin(), pending_.begin() + 4 + length);
        }
        return output;
    }

    const Statistics& statistics() const noexcept {
        return stats_;
    }

    std::size_t buffered_bytes() const noexcept {
        return pending_.size();
    }

    std::size_t maximum_payload() const noexcept {
        return maximum_payload_;
    }

    bool at_frame_boundary() const noexcept {
        return pending_.empty();
    }

    void reset() noexcept {
        pending_.clear();
        stats_ = {};
    }

private:
    std::size_t maximum_payload_;
    Bytes pending_;
    Statistics stats_;
};

class Writer {
public:
    void append(Bytes payload) {
        auto wire = encode_frame(payload);
        pending_.insert(pending_.end(), wire.begin(), wire.end());
    }

    Bytes take(std::size_t maximum_bytes) {
        const auto count = std::min(maximum_bytes, pending_.size() - offset_);
        Bytes result(pending_.begin() + offset_, pending_.begin() + offset_ + count);
        offset_ += count;
        if (offset_ == pending_.size()) {
            pending_.clear();
            offset_ = 0;
        }
        return result;
    }

    std::size_t remaining() const noexcept {
        return pending_.size() - offset_;
    }

private:
    Bytes pending_;
    std::size_t offset_ = 0;
};

inline Bytes as_bytes(std::string_view text) {
    return Bytes(text.begin(), text.end());
}

inline std::string as_text(const Bytes& bytes) {
    return std::string(bytes.begin(), bytes.end());
}

inline Frames decode_chunks(const std::vector<Bytes>& chunks, std::size_t maximum_payload) {
    Decoder decoder(maximum_payload);
    Frames frames;
    for (const auto& chunk : chunks) {
        auto next = decoder.feed(chunk);
        frames.insert(frames.end(), next.begin(), next.end());
    }
    if (!decoder.at_frame_boundary()) {
        throw std::invalid_argument("stream ended in the middle of a frame");
    }
    return frames;
}

} // namespace frame_demo
