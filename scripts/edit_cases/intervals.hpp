#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace interval_demo {

// Ranges are closed intervals: both endpoints are included.
struct Range {
    std::int64_t first;
    std::int64_t last;
};

inline bool operator==(const Range& a, const Range& b) {
    return a.first == b.first && a.last == b.last;
}

inline void validate(const Range& range) {
    if (range.first > range.last) {
        throw std::invalid_argument("range endpoints are reversed");
    }
}

inline bool contains(const Range& range, std::int64_t value) {
    return range.first <= value && value <= range.last;
}

inline bool intersects(const Range& a, const Range& b) {
    return a.first <= b.last && b.first <= a.last;
}

inline std::optional<Range> intersection(const Range& a, const Range& b) {
    if (!intersects(a, b)) {
        return std::nullopt;
    }
    return Range{std::max(a.first, b.first), std::min(a.last, b.last)};
}

// Produce the canonical union without modifying the caller's input.
// Canonical ranges are ordered, disjoint, and separated by at least one integer.
inline std::vector<Range> merge_ranges(const std::vector<Range>& input) {
    std::vector<Range> ordered = input;
    for (const auto& range : ordered) {
        validate(range);
    }
    std::sort(ordered.begin(), ordered.end(), [](const Range& a, const Range& b) {
        return a.first < b.first;
    });

    std::vector<Range> result;
    result.reserve(ordered.size());
    for (const auto& range : ordered) {
        if (result.empty() || range.first >= result.back().last) {
            result.push_back(range);
        } else {
            result.back().last = range.last;
        }
    }
    return result;
}

class RangeSet {
public:
    explicit RangeSet(std::vector<Range> input = {})
        : ranges_(merge_ranges(input)) {}

    const std::vector<Range>& ranges() const noexcept {
        return ranges_;
    }

    bool contains(std::int64_t value) const {
        const auto it = std::upper_bound(
            ranges_.begin(), ranges_.end(), value,
            [](std::int64_t point, const Range& range) {
                return point < range.first;
            });
        return it != ranges_.begin() && interval_demo::contains(*(it - 1), value);
    }

    void add(Range range) {
        auto next = ranges_;
        next.push_back(range);
        next = merge_ranges(next);
        ranges_.swap(next);
    }

    RangeSet clipped(Range window) const {
        validate(window);
        std::vector<Range> pieces;
        for (const auto& range : ranges_) {
            if (auto piece = intersection(range, window)) {
                pieces.push_back(*piece);
            }
        }
        return RangeSet(std::move(pieces));
    }

    // A full signed-64-bit domain contains 2^64 values; saturate that count.
    std::uint64_t covered_count() const noexcept {
        std::uint64_t total = 0;
        const auto limit = std::numeric_limits<std::uint64_t>::max();
        for (const auto& range : ranges_) {
            const auto width = static_cast<std::uint64_t>(range.last)
                             - static_cast<std::uint64_t>(range.first);
            if (width == limit || total > limit - (width + 1)) {
                return limit;
            }
            total += width + 1;
        }
        return total;
    }

    std::string describe() const {
        std::string result;
        for (const auto& range : ranges_) {
            if (!result.empty()) result += ",";
            result += std::to_string(range.first);
            result += ":";
            result += std::to_string(range.last);
        }
        return result;
    }

private:
    std::vector<Range> ranges_;
};

inline std::int64_t parse_integer(std::string_view text) {
    std::int64_t value = 0;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        throw std::invalid_argument("invalid integer");
    }
    return value;
}

inline Range parse_range(std::string_view text) {
    const auto separator = text.find(':');
    if (separator == std::string_view::npos) {
        throw std::invalid_argument("range requires first:last");
    }
    Range range{parse_integer(text.substr(0, separator)),
                parse_integer(text.substr(separator + 1))};
    validate(range);
    return range;
}

inline RangeSet parse_ranges(std::string_view text) {
    std::vector<Range> ranges;
    while (!text.empty()) {
        const auto comma = text.find(',');
        ranges.push_back(parse_range(text.substr(0, comma)));
        if (comma == std::string_view::npos) break;
        text.remove_prefix(comma + 1);
        if (text.empty()) throw std::invalid_argument("trailing comma");
    }
    return RangeSet(std::move(ranges));
}

} // namespace interval_demo
