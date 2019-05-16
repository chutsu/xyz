#ifndef PROTOTYPE_COMM_TCP_HPP
#define PROTOTYPE_COMM_TCP_HPP

#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>

#include <vector>
#include <string>
#include <map>

#include "prototype/core/log.hpp"

namespace proto {

/**
 * TCP server
 */
struct tcp_server_t {
  int port = 8080;
  int sockfd = -1;
  std::vector<int> conns;
  void *(*conn_thread)(void *) = nullptr;

  tcp_server_t(int port_=8080);
};

/**
 * TCP client
 */
struct tcp_client_t{
  std::string server_ip;
  int server_port = 8080;
  int sockfd = -1;
  int (*loop_cb)(tcp_client_t &) = nullptr;

  tcp_client_t(const std::string &server_ip_="127.0.0.1",
               int server_port_=8080);
};

/**
 * Configure TCP server
 */
int tcp_server_config(tcp_server_t &server);

/**
 * Loop TCP server
 */
int tcp_server_loop(tcp_server_t &server);

/**
 * Configure TCP client
 */
int tcp_client_config(tcp_client_t &client);

/**
 * Loop TCP client
 */
int tcp_client_loop(tcp_client_t &client);

} //  namespace proto
#endif // PROTOTYPE_COMM_TCP_HPP
