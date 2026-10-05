#include "ngram-cache.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

static void check(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void compare(const common_ngram_cache & expected, const common_ngram_cache_static & actual) {
    for (const auto & item : expected) {
        if (item.first.tokens[2] != LLAMA_TOKEN_NULL || item.first.tokens[1] == LLAMA_TOKEN_NULL) {
            continue;
        }
        const auto part = common_ngram_cache_static_find(actual, item.first);
        check(part.size == item.second.size(), "static part size differs");
        check(std::equal(item.second.begin(), item.second.end(), part.entries), "static counts differ");
    }
    const llama_token absent[] = {90001, 90002};
    check(common_ngram_cache_static_find(actual, common_ngram(absent, 2)).size == 0, "absent n-gram matched");
}

static std::vector<char> read(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

static void write(const std::string & path, const std::vector<char> & bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), bytes.size());
}

template<class F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::ios_base::failure &) { rejected = true; }
    check(rejected, "invalid cache or path was accepted");
}

static void check_thresholds(const std::string & path) {
    using counts = std::vector<std::pair<llama_token, int32_t>>;
    auto draft = [&](const counts & primary, const counts & weights, bool dynamic, int ngram_size) {
        std::vector<llama_token> input = {1, 2, 3, 4};
        common_ngram_cache context_cache, dynamic_cache, static_cache;
        common_ngram_cache_part part;
        for (const auto & item : primary) { part.emplace(item.first, item.second); }
        auto & cache = dynamic ? dynamic_cache : context_cache;
        cache.emplace(common_ngram(input.data() + input.size() - ngram_size, ngram_size), part);
        std::shared_ptr<const common_ngram_cache_static> static_part;
        if (!weights.empty()) {
            common_ngram_cache_part weighted;
            for (const auto & item : weights) { weighted.emplace(item.first, item.second); }
            static_cache.emplace(common_ngram(input.data() + 2, 2), weighted);
            common_ngram_cache_static_save(static_cache, path);
            static_part = common_ngram_cache_static_load(path);
        }
        std::vector<llama_token> result = {input.back()};
        common_ngram_cache_draft(input, result, 1, 1, 4, context_cache, dynamic_cache, static_part.get());
        return result.size() == 1 ? LLAMA_TOKEN_NULL : result.back();
    };
    check(draft({{10, 1}}, {}, false, 1) == LLAMA_TOKEN_NULL, "context sample minimum bypassed");
    check(draft({{10, 2}}, {}, false, 1) == 10, "context sample boundary rejected");
    check(draft({{10, 1}, {20, 1}, {30, 1}}, {}, false, 1) == LLAMA_TOKEN_NULL, "diffuse context accepted");
    check(draft({{10, 66}, {20, 34}}, {}, false, 1) == 10, "context percentage boundary rejected");
    check(draft({{10, 65}, {20, 35}}, {}, false, 1) == LLAMA_TOKEN_NULL, "below context percentage accepted");
    check(draft({{10, 3}}, {}, true, 1) == LLAMA_TOKEN_NULL, "dynamic sample minimum bypassed");
    check(draft({{10, 3}, {20, 1}}, {}, true, 1) == 10, "dynamic percentage boundary rejected");
    check(draft({{10, 2}, {20, 2}}, {}, true, 1) == LLAMA_TOKEN_NULL, "below dynamic percentage accepted");
    check(draft({{10, 2}, {20, 2}}, {}, false, 2) == 10, "unweighted tie order changed");
    check(draft({{10, 2}, {20, 2}}, {{10, 1}, {20, 2}}, false, 2) == 20, "static weighting ignored");
    check(draft({{10, 3}, {20, 1}}, {{10, 1}, {20, 4}, {30, 4}}, false, 1) == LLAMA_TOKEN_NULL,
          "weighted winner bypassed final primary percentage check");
    check(draft({{10, 3}, {20, 1}}, {{10, 1}, {20, 1}, {30, 1}}, false, 1) == 10,
          "valid weighted winner rejected");
    check(draft({{10, 1}}, {{20, 4}}, false, 1) == 20, "early rejection skipped static fallback");
    std::filesystem::remove(path);
    std::puts("PASS: context/dynamic thresholds, exact boundaries, weighted winner, tie order, static fallback");
}

int main(int argc, char ** argv) {
    try {
        ggml_time_init();
        check(argc == 2, "expected a scratch directory");
        const auto dir = std::filesystem::path(argv[1]);
        std::filesystem::create_directories(dir);
        check_thresholds((dir / "thresholds.bin").string());
        const auto legacy = (dir / "legacy.bin").string();
        const auto modern = (dir / "static.bin").string();
        const auto merged = (dir / "merged.bin").string();
        const auto old_pr = (dir / "upstream-v1.bin").string();
        const auto invalid = (dir / "invalid.bin").string();
        const auto empty = (dir / "empty.bin").string();

        for (size_t n = 0; n < 513; ++n) {
            common_ngram_cache_part part;
            for (size_t i = n; i > 0; --i) { part.emplace(llama_token(i * 3), 1); }
            for (llama_token token = -1; token <= llama_token(n * 3 + 1); ++token) {
                const auto expected = std::lower_bound(part.begin(), part.end(), token,
                    [](const auto & value, llama_token key) { return value.first < key; });
                const size_t actual = common_ngram_cache_part::lower_bound(part.entries.data(), n, token);
                check(actual == size_t(expected - part.begin()), "binary search differs from std::lower_bound");
            }
        }

        std::vector<llama_token> corpus;
        for (int i = 0; i < 100; ++i) {
            for (llama_token token : {1, 2, 3, 4, 1, 2, 3, 5}) { corpus.push_back(token); }
        }
        common_ngram_cache source;
        common_ngram_cache_update(source, 1, 4, corpus, int(corpus.size()), false);
        common_ngram_cache_save(source, legacy);
        common_ngram_cache_static_save(source, modern);
        const auto converted = common_ngram_cache_static_load(legacy);
        const auto loaded = common_ngram_cache_static_load(modern);
        compare(source, *converted);
        compare(source, *loaded);
        const auto shared = loaded;
        check(shared.get() == loaded.get(), "static cache was copied");
        check(loaded->buffer.size() * sizeof(uint64_t) < std::filesystem::file_size(modern), "key table was loaded for inference");

        for (size_t start = 4; start < 40; ++start) {
            std::vector<llama_token> input(corpus.begin(), corpus.begin() + start);
            common_ngram_cache context, dynamic;
            common_ngram_cache_update(context, 1, 4, input, int(input.size()), false);
            std::vector<llama_token> a = {input.back()}, b = a;
            common_ngram_cache_draft(input, a, 16, 1, 4, context, dynamic, converted.get());
            common_ngram_cache_draft(input, b, 16, 1, 4, context, dynamic, loaded.get());
            check(a == b && a.size() > 1, "legacy and modern drafts differ or are empty");
        }

        auto expanded = common_ngram_cache_load(modern);
        auto mixed = common_ngram_cache_load(legacy);
        common_ngram_cache_merge(mixed, expanded);
        for (const auto & item : source) {
            const auto actual = mixed.find(item.first);
            check(actual != mixed.end(), "merged key missing");
            for (const auto & token : item.second) {
                const auto count = actual->second.find(token.first);
                const int factor = item.first.tokens[1] != LLAMA_TOKEN_NULL && item.first.tokens[2] == LLAMA_TOKEN_NULL ? 2 : 1;
                check(count != actual->second.end() && count->second == factor * token.second, "merged count differs");
            }
        }
        common_ngram_cache_save(mixed, merged);
        compare(mixed, *common_ngram_cache_static_load(merged));

        auto bytes = read(modern);
        uint64_t n_entries, map_size;
        memcpy(&n_entries, bytes.data() + 8, 8);
        memcpy(&map_size, bytes.data() + 16, 8);
        std::vector<char> v1(bytes.begin(), bytes.begin() + 24);
        const uint64_t magic_v1 = 0x70616d63676e6767ull;
        memcpy(v1.data(), &magic_v1, 8);
        v1.insert(v1.end(), bytes.begin() + 32, bytes.begin() + 32 + n_entries * 8 + map_size);
        write(old_pr, v1);
        compare(source, *common_ngram_cache_static_load(old_pr));
        rejects([&] { common_ngram_cache_load(old_pr); });

        write(invalid, {'x', 'y', 'z'});
        rejects([&] { common_ngram_cache_static_load(invalid); });
        auto truncated = bytes;
        truncated.resize(16);
        write(invalid, truncated);
        rejects([&] { common_ngram_cache_static_load(invalid); });
        auto corrupt = bytes;
        const uint64_t enormous = UINT64_MAX;
        memcpy(corrupt.data() + 8, &enormous, 8);
        write(invalid, corrupt);
        rejects([&] { common_ngram_cache_static_load(invalid); });
        corrupt = bytes;
        memcpy(corrupt.data() + 32 + n_entries * 8 + map_size + 8, &enormous, 8);
        write(invalid, corrupt);
        rejects([&] { common_ngram_cache_load(invalid); });
        rejects([&] { common_ngram_cache_static_save(source, (dir / "missing" / "file.bin").string()); });

        common_ngram_cache no_entries;
        common_ngram_cache_save(no_entries, empty);
        check(common_ngram_cache_static_load(empty)->n_entries == 0, "empty legacy cache failed");
        common_ngram_cache_static_save(no_entries, empty);
        check(common_ngram_cache_static_load(empty)->n_entries == 0, "empty static cache failed");
        check(common_ngram_cache_load(empty).empty(), "empty static merge failed");

        for (const auto & path : {legacy, modern, merged, old_pr, invalid, empty}) { std::filesystem::remove(path); }
        std::puts("PASS: ngram search, legacy/new round trips, shared cache, drafts, merge, v1 reads, malformed files, empty caches");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
