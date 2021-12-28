
#include "wnr_daemon.h"

#include "mediapipe/framework/port/logging.h"

namespace {

  WnrDaemon::WnrDaemon() {
      m_isRunning = true;
      m_reload = false;
      signal(SIGINT, WnrDaemon::signalHandler);
      signal(SIGTERM, WnrDaemon::signalHandler);
      signal(SIGHUP, WnrDaemon::signalHandler);
  }

  void WnrDaemon::setReloadFunction(std::function<void()> func) {
      m_reloadFunc = func;
  }

  bool WnrDaemon::IsRunning() {
      if (m_reload) {
          m_reload = false;
          m_reloadFunc();
      }
      return m_isRunning;
  }

  void WnrDaemon::signalHandler(int signal) {
      LOG(INFO) << "Interrup signal number [", signal, "] recived.";
      switch (signal) {
          case SIGINT:
          case SIGTERM: {
              WnrDaemon::instance().m_isRunning = false;
              break;
          }
          case SIGHUP: {
              WnrDaemon::instance().m_reload = true;
              break;
          }
      }
  }

}