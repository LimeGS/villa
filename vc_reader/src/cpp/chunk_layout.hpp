#pragma once
// Numeric Zarr filters and stored-order conversion, shared by snapshot and array decoding.
#include "element_value.hpp"
#include "reader_types.hpp"

#include <memory>

namespace vcr {

struct ChunkLayout {
    struct Filter {
        enum Kind { Shuffle, Delta } kind;
        std::size_t before, after, elementsize = 1;
        ElementType decoded, encoded;
    };
    Index shape;
    std::size_t itemsize, bytes, encodedBytes;
    bool fortran;
    std::vector<Filter> filters;

    static std::size_t multiply(std::size_t a, std::size_t b)
    {
        if (b && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::invalid_argument("chunk layout size overflow");
        return a * b;
    }

    explicit ChunkLayout(const nlohmann::json& spec)
        : shape(spec.at("chunks").get<Index>()), itemsize(parseDtype(spec.at("dtype").get<std::string>()).itemsize),
          bytes(itemsize), fortran(spec.at("order") == "F")
    {
        if (shape.empty() || (spec.at("order") != "C" && !fortran))
            throw std::invalid_argument("invalid chunk layout");
        for (auto n : shape) {
            if (n <= 0) throw std::invalid_argument("invalid chunk extent");
            bytes = multiply(bytes, std::size_t(n));
        }
        encodedBytes = bytes;
        const auto& configs = spec.at("filters");
        if (!configs.is_null() && !configs.is_array())
            throw std::invalid_argument("Zarr filters must be an array");
        for (const auto& config : configs) {
            Filter filter{};
            filter.before = encodedBytes;
            const auto id = config.at("id").get<std::string>();
            if (id == "shuffle") {
                filter.kind = Filter::Shuffle;
                const auto size = config.value("elementsize", std::int64_t(4));
                if (size <= 0 || encodedBytes % std::size_t(size))
                    throw std::invalid_argument("invalid shuffle element size");
                filter.elementsize = std::size_t(size);
            } else if (id == "delta") {
                filter.kind = Filter::Delta;
                const auto dtype = config.at("dtype").get<std::string>();
                filter.decoded = elementOf(dtype);
                filter.encoded = elementOf(config.value("astype", dtype));
                if (encodedBytes % filter.decoded.itemsize)
                    throw std::invalid_argument("delta dtype does not divide chunk bytes");
                encodedBytes = multiply(encodedBytes / filter.decoded.itemsize, filter.encoded.itemsize);
            } else {
                throw std::invalid_argument("unsupported numeric Zarr filter: " + id);
            }
            filter.after = encodedBytes;
            filters.push_back(filter);
        }
    }

    template <class T>
    static void store(std::uint8_t* out, ElementType type, T value)
    {
        const auto put = [&](auto converted) { std::memcpy(out, &converted, sizeof(converted)); };
        if (type.kind == 'f') {
            if (type.itemsize == 2) put(float16Bits(double(value)));
            else if (type.itemsize == 4) put(float(value));
            else put(double(value));
        } else {
            // Unsigned arithmetic preserves modular integer deltas, including int64 overflow.
            std::uint64_t bits;
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(value) || value < -0x1p63 || value >= 0x1p64)
                    throw std::runtime_error("delta value is outside integer conversion range");
                bits = value < 0 ? std::uint64_t(std::int64_t(value)) : std::uint64_t(value);
            } else {
                bits = std::uint64_t(value);
            }
            switch (type.itemsize) {
            case 1: put(std::uint8_t(bits)); break;
            case 2: put(std::uint16_t(bits)); break;
            case 4: put(std::uint32_t(bits)); break;
            case 8: put(bits); break;
            }
        }
        if (type.swap) std::reverse(out, out + type.itemsize);
    }

    void decode(std::vector<std::uint8_t> work, std::uint8_t* out, std::size_t size) const
    {
        if (size != bytes || work.size() != encodedBytes)
            throw std::runtime_error("invalid chunk layout byte size");
        for (auto it = filters.rbegin(); it != filters.rend(); ++it) {
            const auto& f = *it;
            std::vector<std::uint8_t> next(f.before);
            if (f.kind == Filter::Shuffle) {
                const auto count = f.before / f.elementsize;
                for (std::size_t b = 0; b < f.elementsize; ++b)
                    for (std::size_t i = 0; i < count; ++i)
                        next[i * f.elementsize + b] = work[b * count + i];
            } else {
                // NumPy's accumulate selects a common input/output dtype before summing.
                auto precision = [](ElementType t) -> std::size_t {
                    return t.kind == 'f' ? t.itemsize : t.itemsize == 1 ? 2 : t.itemsize == 2 ? 4 : 8;
                };
                const bool mixed64 = f.encoded.kind != f.decoded.kind &&
                    ((f.encoded.kind == 'u' && f.encoded.itemsize == 8) ||
                     (f.decoded.kind == 'u' && f.decoded.itemsize == 8));
                const bool integral = f.encoded.kind != 'f' && f.decoded.kind != 'f' && !mixed64;
                const auto width = std::max(precision(f.encoded), precision(f.decoded));
                f.encoded.visit_load([&](auto load) {
                    const auto count = f.before / f.decoded.itemsize;
                    const auto accumulate = [&]<class T>() {
                        T total = 0;
                        for (std::size_t i = 0; i < count; ++i) {
                            total += T(load(work.data() + i * f.encoded.itemsize));
                            if (!integral && width == 2) {
                                const auto half = float16Bits(double(total));
                                total = T(ElementType{'f', 2, false}.load(reinterpret_cast<const std::uint8_t*>(&half)));
                            }
                            store(next.data() + i * f.decoded.itemsize, f.decoded, total);
                        }
                    };
                    if (integral)
                        accumulate.template operator()<std::uint64_t>();
                    else if (width == 8)
                        accumulate.template operator()<double>();
                    else
                        accumulate.template operator()<float>();
                });
            }
            work = std::move(next);
        }
        if (!fortran) {
            std::memcpy(out, work.data(), size);
            return;
        }
        std::vector<std::size_t> strides(shape.size(), 1);
        for (std::size_t axis = 1; axis < shape.size(); ++axis)
            strides[axis] = strides[axis - 1] * std::size_t(shape[axis - 1]);
        for (std::size_t i = 0; i < bytes / itemsize; ++i) {
            std::size_t remaining = i, source = 0;
            for (std::size_t axis = shape.size(); axis-- > 0;) {
                source += (remaining % std::size_t(shape[axis])) * strides[axis];
                remaining /= std::size_t(shape[axis]);
            }
            std::memcpy(out + i * itemsize, work.data() + source * itemsize, itemsize);
        }
    }
};
}  // namespace vcr
