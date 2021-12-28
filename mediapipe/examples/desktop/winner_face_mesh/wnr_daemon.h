#ifndef WNR_DAEMON_HPP_
#define WNR_DAEMON_HPP_

#include <csignal>
#include <functional>

namespace
{
    class WnrDaemon
    {
    public:
        static WnrDaemon &instance()
        {
            static WnrDaemon instance;
            return instance;
        }

        void setReloadFunction(std::function<void()> func);

        bool IsRunning();

    private:
        std::function<void()> m_reloadFunc;
        bool m_isRunning;
        bool m_reload;

        WnrDaemon();
        WnrDaemon(WnrDaemon const &) = delete;
        void operator=(WnrDaemon const &) = delete;

        void Reload();

        static void signalHandler(int signal);
    };
}

#endif // WNR_DAEMON_HPP_
