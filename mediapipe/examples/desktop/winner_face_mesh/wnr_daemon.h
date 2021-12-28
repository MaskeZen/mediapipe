#include <csignal>
#include <functional>

namespace winnerPy
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
