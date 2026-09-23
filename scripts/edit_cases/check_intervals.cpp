#include "subject.hpp"
#include <iostream>
#include <random>
#include <set>
#include <cstdlib>
using namespace interval_demo;
static int checks = 0;
static void require(bool condition, const char* label) {
    ++checks;
    if (!condition) { std::cerr << "FAIL " << label << " at check " << checks << '\n'; std::exit(1); }
}
int main() {
    using I = std::int64_t;
    const I lo = std::numeric_limits<I>::min(), hi = std::numeric_limits<I>::max();
    require(merge_ranges({}).empty(), "empty union");
    require(merge_ranges({{1,10},{2,3}}) == std::vector<Range>{{1,10}}, "nested interval must not shrink");
    require(merge_ranges({{4,7},{1,4},{8,10}}) == std::vector<Range>{{1,10}}, "overlap and inclusive adjacency");
    require(merge_ranges({{hi-1,hi},{hi,hi}}) == std::vector<Range>{{hi-1,hi}}, "INT64_MAX overlap");
    require(merge_ranges({{lo,lo},{lo+1,lo+2}}) == std::vector<Range>{{lo,lo+2}}, "INT64_MIN adjacency");
    require(merge_ranges({{lo,lo},{hi,hi}}).size() == 2, "distant extreme endpoints");
    bool rejected = false;
    try { (void)merge_ranges({{8,7}}); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "reversed range rejection");
    std::mt19937 random(814);
    for (int trial = 0; trial < 500; ++trial) {
        std::vector<Range> input;
        std::set<I> points;
        for (int j = 0; j < trial % 23; ++j) {
            I a = I(random() % 81) - 40, b = I(random() % 81) - 40;
            if (a > b) std::swap(a,b);
            input.push_back({a,b});
            for (I p = a; p <= b; ++p) points.insert(p);
        }
        const auto saved = input;
        const auto merged = merge_ranges(input);
        require(input == saved, "input mutation");
        std::vector<Range> oracle;
        for (I point : points) {
            if (oracle.empty() || point != oracle.back().last + 1) oracle.push_back({point,point});
            else oracle.back().last = point;
        }
        require(merged == oracle, "randomized canonical union");
        RangeSet set(input);
        require(set.covered_count() == points.size(), "covered count");
        for (I point = -42; point <= 42; ++point) require(set.contains(point) == points.contains(point), "contains lookup");
    }
    require(parse_ranges("8:10,1:3,4:7").describe() == "1:10", "parser and union integration");
    require(RangeSet({{lo,hi}}).covered_count() == std::numeric_limits<std::uint64_t>::max(), "full domain saturation");
    require(RangeSet({{0,10},{20,30}}).clipped({5,25}).ranges() == std::vector<Range>{{5,10},{20,25}}, "clip integration");
    std::cout << "PASS " << checks << " interval checks\n";
}
