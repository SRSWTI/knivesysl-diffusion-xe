#include "subject.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
using namespace frame_demo;
static int checks = 0;
static void require(bool condition, const char* label) {
    ++checks;
    if (!condition) { std::cerr << "FAIL " << label << " at check " << checks << '\n'; std::exit(1); }
}
int main() {
    Frames messages{{},{1,2,3},Bytes(257,0xab),{0,255,0},Bytes(1024,7),{}};
    const auto wire = concatenate(messages);
    for (std::size_t split = 0; split <= wire.size(); ++split) {
        Decoder decoder(1024);
        auto first = decoder.feed(Bytes(wire.begin(),wire.begin()+split));
        require(decoder.feed({}).empty(), "empty chunk cannot manufacture frames");
        auto second = decoder.feed(Bytes(wire.begin()+split,wire.end()));
        first.insert(first.end(),second.begin(),second.end());
        require(first == messages, "every header/payload split roundtrip");
        require(decoder.at_frame_boundary(), "buffer fully consumed");
        require(decoder.statistics().bytes_received == wire.size(), "received-byte accounting");
        require(decoder.statistics().frames_delivered == messages.size(), "empty frames count as delivered");
    }
    std::mt19937 random(90210);
    for (int trial = 0; trial < 100; ++trial) {
        Decoder decoder(1024);
        Frames observed;
        std::size_t position = 0;
        while (position < wire.size()) {
            const auto count = std::min<std::size_t>(1+random()%17,wire.size()-position);
            auto next = decoder.feed(Bytes(wire.begin()+position,wire.begin()+position+count));
            observed.insert(observed.end(),next.begin(),next.end());
            position += count;
        }
        require(observed == messages, "random chunking roundtrip");
    }
    Decoder tiny(3);
    bool rejected = false;
    try { tiny.feed({0,0,0,4}); } catch (const std::length_error&) { rejected = true; }
    require(rejected, "oversized frame rejected immediately from header");
    require(tiny.buffered_bytes() == 0 && tiny.statistics().protocol_errors == 1, "protocol error clears pending input");
    require(tiny.feed(encode_frame({2,3})) == Frames{{2,3}}, "decoder reusable after protocol error");
    Decoder boundary(0);
    require(boundary.feed({0,0,0,0}) == Frames{{}}, "zero maximum permits empty payload");
    Decoder partial(8);
    require(partial.feed({0,0}).empty() && partial.buffered_bytes()==2, "retain incomplete header");
    partial.reset();
    require(partial.at_frame_boundary() && partial.statistics().bytes_received==0, "reset clears counters");
    Writer writer;
    for (const auto& frame : messages) writer.append(frame);
    std::vector<Bytes> chunks;
    while (writer.remaining()) chunks.push_back(writer.take(3));
    require(decode_chunks(chunks,1024)==messages,"writer/decoder integration");
    std::cout << "PASS " << checks << " frame checks\n";
}
