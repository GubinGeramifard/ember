#pragma once
//
// A fixed-size thread pool offering a single primitive: parallel_for, which
// splits [0, total) into one contiguous range per worker (the calling thread
// included) and returns once every range is done.
//
// Workers are created once and parked on a condition variable between jobs, so
// there is no per-call thread creation. Dispatch is allocation-free: instead of
// wrapping the callable in a std::function (which would heap-allocate on every
// matmul), the job is type-erased by hand into a void* context plus a trampoline
// function pointer. The callable lives on the caller's stack for the duration.
//
#include <condition_variable>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace ember {

class ThreadPool {
public:
    explicit ThreadPool(unsigned n_threads) : n_(n_threads < 1 ? 1 : n_threads) {
        // Worker 0 is always the calling thread, so only spawn n_ - 1 workers.
        for (unsigned w = 1; w < n_; ++w)
            workers_.emplace_back([this, w] { loop(w); });
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
            ++epoch_;
        }
        work_cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] unsigned size() const noexcept { return n_; }

    // Run fn(begin, end) over contiguous sub-ranges covering [0, total),
    // blocking until all workers finish. fn must be safe to call concurrently.
    template <class F>
    void parallel_for(int total, F&& fn) {
        if (total <= 0) return;
        if (n_ == 1) {  // no workers: just run it inline
            fn(0, total);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            ctx_ = const_cast<void*>(static_cast<const void*>(std::addressof(fn)));
            trampoline_ = [](void* p, int b, int e) {
                (*static_cast<std::remove_reference_t<F>*>(p))(b, e);
            };
            total_ = total;
            busy_ = n_ - 1;
            ++epoch_;
        }
        work_cv_.notify_all();

        auto [b, e] = chunk(0, total, static_cast<int>(n_));
        if (e > b) fn(b, e);  // the caller does chunk 0

        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return busy_ == 0; });
        ctx_ = nullptr;
    }

private:
    // Split `total` items into T as-even-as-possible contiguous chunks; return
    // the [begin, end) range owned by chunk i (the first `rem` chunks get one extra).
    static std::pair<int, int> chunk(int i, int total, int T) noexcept {
        const int base = total / T;
        const int rem = total % T;
        const int begin = i * base + (i < rem ? i : rem);
        const int count = base + (i < rem ? 1 : 0);
        return {begin, begin + count};
    }

    void loop(unsigned w) {
        unsigned seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            work_cv_.wait(lk, [this, &seen] { return stop_ || epoch_ != seen; });
            if (stop_) return;
            seen = epoch_;
            void* ctx = ctx_;
            auto tr = trampoline_;
            const int total = total_;
            lk.unlock();

            auto [b, e] = chunk(static_cast<int>(w), total, static_cast<int>(n_));
            if (e > b) tr(ctx, b, e);

            lk.lock();
            if (--busy_ == 0) done_cv_.notify_one();
        }
    }

    unsigned n_;
    std::vector<std::thread> workers_;

    std::mutex m_;
    std::condition_variable work_cv_;  // workers wait here for the next job
    std::condition_variable done_cv_;  // caller waits here for completion
    unsigned epoch_ = 0;               // bumped to release workers for a new job
    bool stop_ = false;
    void* ctx_ = nullptr;
    void (*trampoline_)(void*, int, int) = nullptr;
    int total_ = 0;
    int busy_ = 0;                     // workers still running the current job
};

}  // namespace ember
