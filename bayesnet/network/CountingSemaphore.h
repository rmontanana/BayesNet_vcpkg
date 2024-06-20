#ifndef COUNTING_SEMAPHORE_H
#define COUNTING_SEMAPHORE_H
#include <mutex>
#include <condition_variable>
class CountingSemaphore {
public:
    explicit CountingSemaphore(size_t max_count) : max_count_(max_count), count_(max_count) {}

    // Acquires a permit, blocking if necessary until one becomes available
    void acquire()
    {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [this]() { return count_ > 0; });
        --count_;
    }

    // Releases a permit, potentially waking up a blocked acquirer
    void release()
    {
        std::lock_guard<std::mutex> lock(mtx_);
        ++count_;
        if (count_ <= max_count_) {
            cv_.notify_one();
        }
    }

private:
    std::mutex mtx_;
    std::condition_variable cv_;
    size_t max_count_;
    size_t count_;
};
#endif