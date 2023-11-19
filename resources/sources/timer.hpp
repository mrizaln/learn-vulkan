#ifndef TIMER_HPP_EUYQAR14
#define TIMER_HPP_EUYQAR14

#include <chrono>
#include <iostream>
#include <ratio>
#include <string>

// #define TIMER_FORCE_PRINT

class Timer
{
public:
    // type aliases
    using clock_type    = std::chrono::steady_clock;
    using duration_type = std::chrono::duration<double, std::ratio<1, 1'000>>;    // milliseconds

    inline static bool s_doPrint{ true };

public:
    Timer(const std::string& name, bool doAutoPrint = true)
        : m_name{ name }
        , m_doAutoPrint{ doAutoPrint }
    {
    }

    // #if (!defined(NDEBUG) or defined(TIMER_FORCE_PRINT)) and !defined(TIMER_SUPPRESS_PRINT)
    ~Timer()
    {
        if (s_doPrint && m_doAutoPrint) {
            print();
        }
    }
    // #endif

    void reset() { m_beginning = clock_type::now(); }

    duration_type elapsed() const { return std::chrono::duration_cast<duration_type>(clock_type::now() - m_beginning); }

    void print() { std::cout << m_name << ": " << elapsed() << " ms\n"; }

private:
    const std::string                   m_name;
    const bool                          m_doAutoPrint;
    std::chrono::time_point<clock_type> m_beginning{ clock_type::now() };
};

#endif /* end of include guard: TIMER_HPP_EUYQAR14 */
