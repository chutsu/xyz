#ifndef UBX_H
#define UBX_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/poll.h>

// UBX Class IDs
#define UBX_NAV 0x01
#define UBX_RXM 0x02
#define UBX_INF 0x04
#define UBX_ACK 0x05
#define UBX_CFG 0x06
#define UBX_UPD 0x09
#define UBX_MON 0x0A
#define UBX_TIM 0x0D
#define UBX_MGA 0x13
#define UBX_LOG 0x21
#define UBX_SEC 0x27

// UBX Class CFG
#define UBX_ACK_ACK 0x01
#define UBX_ACK_NAK 0x00

// UBX Class CFG
#define UBX_CFG_VALDEL 0x8C
#define UBX_CFG_VALGET 0x8B
#define UBX_CFG_VALSET 0x8A

// UBX Class MON Monitoring Messages
#define UBX_MON_COMMS 0x36
#define UBX_MON_GNSS 0x28
#define UBX_MON_HW2 0x0B
#define UBX_MON_HW3 0x37
#define UBX_MON_HW 0x09
#define UBX_MON_IO 0x02
#define UBX_MON_MSGPP 0x06
#define UBX_MON_PATCH 0x27
#define UBX_MON_RF 0x38
#define UBX_MON_RXBUF 0x07
#define UBX_MON_RXR 0x21
#define UBX_MON_TXBUF 0x08
#define UBX_MON_VER 0x04

// UBX Class NAV Navigation Results Messages
#define UBX_NAV_CLOCK 0x22
#define UBX_NAV_DOP 0x04
#define UBX_NAV_EOE 0x61
#define UBX_NAV_GEOFENCE 0x39
#define UBX_NAV_HPPOSECEF 0x13
#define UBX_NAV_HPPOSLLH 0x14
#define UBX_NAV_ODO 0x09
#define UBX_NAV_ORB 0x34
#define UBX_NAV_POSECEF 0x01
#define UBX_NAV_POSLLH 0x02
#define UBX_NAV_PVT 0x07
#define UBX_NAV_RELPOSNED 0x3C
#define UBX_NAV_RESETODO 0x10
#define UBX_NAV_SAT 0x35
#define UBX_NAV_SIG 0x43
#define UBX_NAV_STATUS 0x03
#define UBX_NAV_SVIN 0x3B
#define UBX_NAV_TIMEBDS 0x24
#define UBX_NAV_TIMEGAL 0x25
#define UBX_NAV_TIMEGLO 0x23
#define UBX_NAV_TIMEGPS 0x20
#define UBX_NAV_TIMELS 0x26
#define UBX_NAV_TIMEUTC 0x21
#define UBX_NAV_VELECEF 0x11
#define UBX_NAV_VELNED 0x12

// UBX Class RXM Receiver Manager Messages
#define UBX_RXM_MEASX 0x14
#define UBX_RXM_PMREQ 0x41
#define UBX_RXM_RAWX 0x15
#define UBX_RXM_RLM 0x59
#define UBX_RXM_RTCM 0x32
#define UBX_RXM_SFRBX 0x13

/**
 * Configuration Aliases (Incomplete - Refer to manual for more info)
 * These configuration are just the ones I found useful to setup GPS mode and
 * RTK-GPS mode on the UBlox ZED-F9P. It may not work with all GPS sensors.
 */
#define CFG_SIGNAL_GPS_ENA 0x1031001f
#define CFG_SIGNAL_GPS_L1CA_ENA 0x10310001
#define CFG_SIGNAL_QZSS_ENA 0x10310024
#define CFG_SIGNAL_BDS_B2_ENA 0x1031000e

#define CFG_RATE_MEAS 0x30210001
#define CFG_UART1_BAUDRATE 0x40520001
#define CFG_USBOUTPROT_NMEA 0x10780002

#define CFG_MSGOUT_RTCM_3X_TYPE1005_USB 0x209102c0
#define CFG_MSGOUT_RTCM_3X_TYPE1077_USB 0x209102cf
#define CFG_MSGOUT_RTCM_3X_TYPE1087_USB 0x209102d4
#define CFG_MSGOUT_RTCM_3X_TYPE1097_USB 0x2091031b
#define CFG_MSGOUT_RTCM_3X_TYPE1127_USB 0x209102d9
#define CFG_MSGOUT_RTCM_3X_TYPE1230_USB 0x20910306

#define CFG_MSGOUT_UBX_NAV_CLOCK_USB 0x20910068
#define CFG_MSGOUT_UBX_NAV_DOP_USB 0x2091003b
#define CFG_MSGOUT_UBX_NAV_EOE_USB 0x20910162
#define CFG_MSGOUT_UBX_NAV_HPPOSEECF_USB 0x20910031
#define CFG_MSGOUT_UBX_NAV_HPPOSLLH_USB 0x20910036
#define CFG_MSGOUT_UBX_NAV_RELPOSNED_USB 0x20910090
#define CFG_MSGOUT_UBX_NAV_STATUS_USB 0x2091001d
#define CFG_MSGOUT_UBX_NAV_SVIN_USB 0x2091008b
#define CFG_MSGOUT_UBX_NAV_PVT_USB 0x20910009
#define CFG_MSGOUT_UBX_NAV_VELNED_USB 0x20910045
#define CFG_MSGOUT_UBX_MON_RF_USB 0x2091035c
#define CFG_MSGOUT_UBX_RXM_RTCM_USB 0x2091026b

#define CFG_TMODE_MODE 0x20030001
#define CFG_TMODE_SVIN_MIN_DUR 0x40030010
#define CFG_TMODE_SVIN_ACC_LIMIT 0x40030011

#define CFG_NAVSPG_DYNMODEL 0x20110021

/*****************************************************************************
 * UBX UTILS
 ****************************************************************************/

// DEBUG
#ifdef NDEBUG
#define UBX_DEBUG(...)
#else
#define UBX_DEBUG(...)                                                         \
  fprintf(stderr, "[UBX DEBUG] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__); \
  fprintf(stderr, __VA_ARGS__);
#endif

// LOG
#ifndef UBX_ERROR
#define UBX_ERROR(...)                                                         \
  fprintf(stderr, "[UBX ERROR] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__); \
  fprintf(stderr, __VA_ARGS__);
#endif

#ifndef UBX_WARN
#define UBX_WARN(...)                                                          \
  fprintf(stderr, "[UBX WARN] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__);  \
  fprintf(stderr, __VA_ARGS__);
#endif

#ifndef UBX_INFO
#define UBX_INFO(...)                                                          \
  fprintf(stderr, "[UBX INFO] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__);  \
  fprintf(stderr, __VA_ARGS__);
#endif

// FATAL
#ifndef UBX_FATAL
#define UBX_FATAL(...)                                                         \
  fprintf(stderr, "[UBX FATAL] [%s:%d:%s()]: ", __FILE__, __LINE__, __func__); \
  fprintf(stderr, __VA_ARGS__);                                                \
  exit(-1);
#endif

int8_t ubx_int8(const uint8_t *data, const size_t offset);
uint8_t ubx_uint8(const uint8_t *data, const size_t offset);
int16_t ubx_int16(const uint8_t *data, const size_t offset);
uint16_t ubx_uint16(const uint8_t *data, const size_t offset);
int32_t ubx_int32(const uint8_t *data, const size_t offset);
uint32_t ubx_uint32(const uint8_t *data, const size_t offset);
int ubx_ip_port_info(const int sockfd, char *ip, int *port);

/*****************************************************************************
 * UBX UART
 ****************************************************************************/

typedef struct ubx_uart_t {
  int connected;
  int connfd;

  char port[100];
  int speed;
  int parity;
} ubx_uart_t;

int ubx_uart_connect(ubx_uart_t *uart, char *port);
int ubx_uart_disconnect(ubx_uart_t *uart);
int ubx_uart_write(const ubx_uart_t *uart,
                   const uint8_t *payload,
                   const size_t length);
int ubx_uart_read(const ubx_uart_t *uart,
                  uint8_t *payload,
                  const size_t length);

/*****************************************************************************
 * UBX Message
 ****************************************************************************/

#define UBX_MON_RF_MAX_BLOCKS 100

typedef struct ubx_msg_t {
  uint8_t ok;

  uint8_t msg_class;
  uint8_t msg_id;
  uint16_t payload_length;
  uint8_t payload[1024];
  uint8_t ck_a;
  uint8_t ck_b;
} ubx_msg_t;

typedef struct ubx_nav_dop_t {
  uint32_t itow;
  uint16_t gdop;
  uint16_t pdop;
  uint16_t tdop;
  uint16_t vdop;
  uint16_t hdop;
  uint16_t ndop;
  uint16_t edop;
} ubx_nav_dop_t;

typedef struct ubx_nav_eoe_t {
  uint32_t itow;
} ubx_nav_eoe_t;

typedef struct ubx_nav_hpposllh_t {
  uint8_t version;
  uint32_t itow;
  int32_t lon;
  int32_t lat;
  int32_t height;
  int32_t hmsl;
  int8_t lon_hp;
  int8_t lat_hp;
  int8_t height_hp;
  int8_t hmsl_hp;
  uint32_t hacc;
  uint32_t vacc;
} ubx_nav_hpposllh_t;

typedef struct ubx_nav_pvt_t {
  uint32_t itow;
  uint16_t year;
  uint8_t month;
  uint8_t day;
  uint8_t hour;
  uint8_t min;
  uint8_t sec;
  uint8_t valid;
  uint32_t tacc;
  int32_t nano;
  uint8_t fix_type;
  uint8_t flags;
  uint8_t flags2;
  uint8_t num_sv;
  uint32_t lon;
  uint32_t lat;
  uint32_t height;
  uint32_t hmsl;
  int32_t hacc;
  int32_t vacc;
  int32_t veln;
  int32_t vele;
  int32_t veld;
  int32_t gspeed;
  int32_t headmot;
  uint32_t sacc;
  uint32_t headacc;
  uint16_t pdop;
  int32_t headveh;
  int16_t magdec;
  uint16_t magacc;
} ubx_nav_pvt_t;

typedef struct ubx_nav_status_t {
  uint32_t itow;
  uint8_t fix;
  uint8_t flags;
  uint8_t fix_status;
  uint8_t flags2;
  uint32_t ttff;
  uint32_t msss;
} ubx_nav_status_t;

typedef struct ubx_nav_svin_t {
  uint32_t itow;
  uint32_t dur;
  int32_t mean_x;
  int32_t mean_y;
  int32_t mean_z;
  int8_t mean_xhp;
  int8_t mean_yhp;
  int8_t mean_zhp;
  uint32_t mean_acc;
  uint32_t obs;
  uint8_t valid;
  uint8_t active;
} ubx_nav_svin_t;

typedef struct ubx_nav_velned_t {
  uint32_t itow;
  int32_t veln;
  int32_t vele;
  int32_t veld;
  uint32_t speed;
  uint32_t gspeed;
  int32_t heading;
  uint32_t sacc;
  uint32_t cacc;
} ubx_nav_velned_t;

typedef struct ubx_rxm_rtcm_t {
  uint8_t flags;
  uint16_t sub_type;
  uint16_t ref_station;
  uint16_t msg_type;
} ubx_rxm_rtcm_t;

typedef struct ubx_mon_rf_t {
  uint8_t version;
  uint32_t nblocks;

  uint8_t block_id[UBX_MON_RF_MAX_BLOCKS];
  uint8_t flags[UBX_MON_RF_MAX_BLOCKS];
  uint8_t ant_status[UBX_MON_RF_MAX_BLOCKS];
  uint8_t ant_power[UBX_MON_RF_MAX_BLOCKS];
  uint32_t post_status[UBX_MON_RF_MAX_BLOCKS];
  uint16_t noise_per_ms[UBX_MON_RF_MAX_BLOCKS];
  uint16_t agc_cnt[UBX_MON_RF_MAX_BLOCKS];
  uint8_t jam_ind[UBX_MON_RF_MAX_BLOCKS];
  int8_t ofs_i[UBX_MON_RF_MAX_BLOCKS];
  uint8_t mag_i[UBX_MON_RF_MAX_BLOCKS];
  int8_t ofs_q[UBX_MON_RF_MAX_BLOCKS];
  uint8_t mag_q[UBX_MON_RF_MAX_BLOCKS];
} ubx_mon_rf_t;

void ubx_msg_init(ubx_msg_t *msg);
void ubx_msg_checksum(const uint8_t msg_class,
                      const uint8_t msg_id,
                      const uint16_t payload_length,
                      const uint8_t *payload,
                      uint8_t *ck_a,
                      uint8_t *ck_b);
uint8_t ubx_msg_is_valid(const ubx_msg_t *msg);
void ubx_msg_build(ubx_msg_t *msg,
                   const uint8_t msg_class,
                   const uint8_t msg_id,
                   const uint16_t length,
                   const uint8_t *payload);
void ubx_msg_parse(ubx_msg_t *msg, const uint8_t *data);
void ubx_msg_serialize(const ubx_msg_t *msg,
                       uint8_t *frame,
                       size_t *frame_size);
void ubx_msg_print(const ubx_msg_t *msg);
ubx_nav_dop_t ubx_nav_dop(const ubx_msg_t *msg);
ubx_nav_eoe_t ubx_nav_eoe(const ubx_msg_t *msg);
ubx_nav_hpposllh_t ubx_nav_hpposllh(const ubx_msg_t *msg);
ubx_nav_pvt_t ubx_nav_pvt(const ubx_msg_t *msg);
ubx_nav_status_t ubx_nav_status(const ubx_msg_t *msg);
ubx_nav_svin_t ubx_nav_svin(const ubx_msg_t *msg);
ubx_nav_velned_t ubx_nav_velned(const ubx_msg_t *msg);
ubx_rxm_rtcm_t ubx_rxm_rtcm(const ubx_msg_t *msg);
ubx_mon_rf_t ubx_mon_rf(const ubx_msg_t *msg);
void print_ubx_nav_hpposllh(const ubx_nav_hpposllh_t *msg);
void print_ubx_nav_pvt(const ubx_nav_pvt_t *msg);
void print_ubx_nav_status(const ubx_nav_status_t *msg);
void print_ubx_nav_svin(const ubx_nav_svin_t *msg);
void print_ubx_rxm_rtcm(const ubx_rxm_rtcm_t *msg);

/*****************************************************************************
 * UBX Stream Parser
 ****************************************************************************/

// UBX Stream Parser States
#define SYNC_1 0
#define SYNC_2 1
#define MSG_CLASS 2
#define MSG_ID 3
#define PAYLOAD_LENGTH_LOW 4
#define PAYLOAD_LENGTH_HI 5
#define PAYLOAD_DATA 6
#define CK_A 7
#define CK_B 8

// UBX Stream Parser
typedef struct ubx_parser_t {
  uint8_t state;
  uint8_t buf_data[9046];
  size_t buf_pos;
  ubx_msg_t msg;
} ubx_parser_t;

void ubx_parser_init(ubx_parser_t *parser);
void ubx_parser_reset(ubx_parser_t *parser);
int ubx_parser_update(ubx_parser_t *parser, uint8_t data);

/*****************************************************************************
 * RTCM3 Stream Parser
 ****************************************************************************/

// RTCM3 Stream Parser
typedef struct rtcm3_parser_t {
  uint8_t buf_data[9046];
  size_t buf_pos;
  size_t msg_len;
  size_t msg_type;
} rtcm3_parser_t;

void rtcm3_parser_init(rtcm3_parser_t *parser);
void rtcm3_parser_reset(rtcm3_parser_t *parser);
int rtcm3_parser_update(rtcm3_parser_t *parser, uint8_t data);

/*****************************************************************************
 * UBlox
 ****************************************************************************/

#define UBLOX_MAX_CONNS 10
#define UBLOX_READY 0
#define UBLOX_PARSING_UBX 1
#define UBLOX_PARSING_RTCM3 2

typedef struct ublox_t ublox_t;
typedef void (*ubx_msg_callback)(ublox_t *ublox);
typedef void (*rtcm3_msg_callback)(ublox_t *ublox);

// UBlox
typedef struct ublox_t {
  int state;
  uint8_t ok;
  ubx_uart_t *uart;

  int sockfd;
  int conns[UBLOX_MAX_CONNS];
  size_t nb_conns;

  ubx_parser_t ubx_parser;
  rtcm3_parser_t rtcm3_parser;

  ubx_msg_callback ubx_cb;
  rtcm3_msg_callback rtcm3_cb;

} ublox_t;

int ublox_init(ublox_t *ublox, ubx_uart_t *uart);
void ublox_disconnect(ublox_t *ublox);
int ublox_reset(ublox_t *ublox);
int ubx_write(const ublox_t *ublox,
              uint8_t msg_class,
              uint8_t msg_id,
              uint16_t length,
              uint8_t *payload);
int ubx_poll(const ublox_t *ublox,
             const uint8_t msg_class,
             const uint8_t msg_id,
             uint16_t *payload_length,
             uint8_t *payload,
             const uint8_t expect_ack,
             const int retry);
int ubx_read_ack(const ublox_t *ublox,
                 const uint8_t msg_class,
                 const uint8_t msg_id);
int ubx_get(const ublox_t *ublox,
            const uint8_t layer,
            const uint32_t key,
            uint32_t *val);
int ubx_set(const ublox_t *ublox,
            const uint8_t layer,
            const uint32_t key,
            const uint32_t val,
            const uint8_t val_size);

//***************************** Ublox GPS Mode *****************************

void ublox_version(const ublox_t *ublox);
int ublox_parse_ubx(ublox_t *ublox, uint8_t data);
int ublox_gps_config(ublox_t *ublox);
int ublox_run(ublox_t *ublox, int *loop);

//*************************** Ublox Base Station ***************************

void ublox_broadcast_rtcm3(ublox_t *ublox);
int ublox_parse_rtcm3(ublox_t *ublox, uint8_t data);
int ublox_base_station_config(ublox_t *base);
int ublox_base_run(ublox_t *base, const int port, int *loop);

//****************************** Ublox Rover *******************************

int ublox_rover_config(ublox_t *rover);
int ublox_rover_run(ublox_t *rover,
                    const char *base_ip,
                    const int base_port,
                    int *loop);

#endif // UBX_H

//////////////////////////////////////////////////////////////////////////////
//                             IMPLEMENTATION                               //
//////////////////////////////////////////////////////////////////////////////

#ifdef UBX_IMPLEMENTATION

/*****************************************************************************
 * UBX UTILS
 ****************************************************************************/

int8_t ubx_int8(const uint8_t *data, const size_t offset) {
  return (int8_t)(data[offset]);
}

uint8_t ubx_uint8(const uint8_t *data, const size_t offset) {
  return (uint8_t)(data[offset]);
}

int16_t ubx_int16(const uint8_t *data, const size_t offset) {
  return (int16_t)((data[offset + 1] << 8) | (data[offset]));
}

uint16_t ubx_uint16(const uint8_t *data, const size_t offset) {
  return (uint16_t)((data[offset + 1] << 8) | (data[offset]));
}

int32_t ubx_int32(const uint8_t *data, const size_t offset) {
  return (int32_t)((data[offset + 3] << 24) | (data[offset + 2] << 16) |
                   (data[offset + 1] << 8) | (data[offset]));
}

uint32_t ubx_uint32(const uint8_t *data, const size_t offset) {
  return (uint32_t)((data[offset + 3] << 24) | (data[offset + 2] << 16) |
                    (data[offset + 1] << 8) | (data[offset]));
}

//* Obtain the IP and Port number from the network socket file descriptor *
int ubx_ip_port_info(const int sockfd, char *ip, int *port) {
  struct sockaddr_storage addr;
  socklen_t len = sizeof addr;
  if (getpeername(sockfd, (struct sockaddr *) &addr, &len) != 0) {
    return -1;
  }

  // Deal with both IPv4 and IPv6:
  char ipstr[INET6_ADDRSTRLEN];

  if (addr.ss_family == AF_INET) {
    // IPV4
    struct sockaddr_in *s = (struct sockaddr_in *) &addr;
    *port = ntohs(s->sin_port);
    inet_ntop(AF_INET, &s->sin_addr, ipstr, sizeof(ipstr));
  } else {
    // IPV6
    struct sockaddr_in6 *s = (struct sockaddr_in6 *) &addr;
    *port = ntohs(s->sin6_port);
    inet_ntop(AF_INET6, &s->sin6_addr, ipstr, sizeof(ipstr));
  }
  strcpy(ip, ipstr);
  // ip = std::string{ipstr};

  return 0;
}

/*****************************************************************************
 * UBX UART
 ****************************************************************************/

// Source: https://stackoverflow.com/a/38318768/154688
static int ubx_set_interface_attributes(int fd, int speed) {
  struct termios tty;
  if (tcgetattr(fd, &tty) < 0) {
    printf("Error from tcgetattr: %s\n", strerror(errno));
    return -1;
  }

  cfsetospeed(&tty, (speed_t) speed);
  cfsetispeed(&tty, (speed_t) speed);

  tty.c_cflag |= (CLOCAL | CREAD); // ignore modem controls
  tty.c_cflag &= ~CSIZE;
  tty.c_cflag |= CS8;      // 8-bit characters
  tty.c_cflag &= ~PARENB;  // no parity bit
  tty.c_cflag &= ~CSTOPB;  // only need 1 stop bit
  tty.c_cflag &= ~CRTSCTS; // no hardware flowcontrol

  // Setup for non-canonical mode
  tty.c_iflag &=
      ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
  tty.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
  tty.c_oflag &= ~OPOST;

  // Fetch bytes as they become available
  tty.c_cc[VMIN] = 1;
  tty.c_cc[VTIME] = 1;

  if (tcsetattr(fd, TCSANOW, &tty) != 0) {
    printf("Error from tcsetattr: %s\n", strerror(errno));
    return -1;
  }
  return 0;
}

int ubx_uart_connect(ubx_uart_t *uart, char *port) {
  // Connect
  strcpy(uart->port, port);
  uart->connfd = open(uart->port, O_RDWR | O_NOCTTY | O_SYNC);
  if (uart->connfd < 0) {
    return -1;
  }
  uart->connected = 1;

  // Set serial attributes
  if (ubx_set_interface_attributes(uart->connfd, uart->speed) != 0) {
    return -1;
  }

  return 0;
}

int ubx_uart_disconnect(ubx_uart_t *uart) {
  if (close(uart->connfd) != 0) {
    return -1;
  }
  uart->connected = -1;
  uart->connfd = -1;

  return 0;
}

int ubx_uart_write(const ubx_uart_t *uart,
                   const uint8_t *payload,
                   const size_t length) {
  ssize_t retval = write(uart->connfd, payload, length);
  if (retval != (ssize_t) length) {
    return -1;
  }

  return 0;
}

int ubx_uart_read(const ubx_uart_t *uart,
                  uint8_t *payload,
                  const size_t length) {
  ssize_t retval = read(uart->connfd, payload, length);
  if (retval < 0) {
    return -1;
  }

  return 0;
}

/*****************************************************************************
 * UBX Message
 ****************************************************************************/

void ubx_msg_init(ubx_msg_t *msg) {
  msg->ok = 0;

  msg->msg_class = 0;
  msg->msg_id = 0;
  msg->payload_length = 0;
  memset(msg->payload, '\0', sizeof(uint8_t) * 1024);
  msg->ck_a = 0;
  msg->ck_b = 0;
}

void ubx_msg_checksum(const uint8_t msg_class,
                      const uint8_t msg_id,
                      const uint16_t payload_length,
                      const uint8_t *payload,
                      uint8_t *ck_a,
                      uint8_t *ck_b) {
  *ck_a = 0;
  *ck_b = 0;

  *ck_a = *ck_a + msg_class;
  *ck_b = *ck_b + *ck_a;

  *ck_a = *ck_a + msg_id;
  *ck_b = *ck_b + *ck_a;

  *ck_a = *ck_a + (payload_length & 0x00FF);
  *ck_b = *ck_b + *ck_a;

  *ck_a = *ck_a + ((payload_length & 0xFF00) >> 8);
  *ck_b = *ck_b + *ck_a;

  for (uint16_t i = 0; i < payload_length; i++) {
    *ck_a = *ck_a + payload[i];
    *ck_b = *ck_b + *ck_a;
  }
}

uint8_t ubx_msg_is_valid(const ubx_msg_t *msg) {
  uint8_t expected_ck_a = 0;
  uint8_t expected_ck_b = 0;
  ubx_msg_checksum(msg->msg_class,
                   msg->msg_id,
                   msg->payload_length,
                   msg->payload,
                   &expected_ck_a,
                   &expected_ck_b);

  if (expected_ck_a == msg->ck_a && expected_ck_b == msg->ck_b) {
    return 1;
  }

  return 0;
}

void ubx_msg_build(ubx_msg_t *msg,
                   const uint8_t msg_class,
                   const uint8_t msg_id,
                   const uint16_t length,
                   const uint8_t *payload) {
  // OK
  msg->ok = 1;

  // Header
  msg->msg_class = msg_class;
  msg->msg_id = msg_id;
  msg->payload_length = length;

  // Payload
  if (payload) {
    for (size_t i = 0; i < length; i++) {
      msg->payload[i] = payload[i];
    }
  }

  // Checksum
  ubx_msg_checksum(msg->msg_class,
                   msg->msg_id,
                   msg->payload_length,
                   msg->payload,
                   &msg->ck_a,
                   &msg->ck_b);
}

void ubx_msg_parse(ubx_msg_t *msg, const uint8_t *data) {
  // Check SYNC_1 and SYNC_2
  if (data[0] != 0xB5 || data[1] != 0x62) {
    msg->ok = 0;
    return;
  }

  // Message class and id
  msg->msg_class = data[2];
  msg->msg_id = data[3];

  // Payload
  msg->payload_length = (data[5] << 8) | (data[4]);
  for (uint16_t i = 0; i < msg->payload_length; i++) {
    msg->payload[i] = data[6 + i];
  }

  // Checksum
  msg->ck_a = data[msg->payload_length + 6];
  msg->ck_b = data[msg->payload_length + 6 + 1];
  msg->ok = ubx_msg_is_valid(msg);
}

void ubx_msg_serialize(const ubx_msg_t *msg,
                       uint8_t *frame,
                       size_t *frame_size) {
  // Form packet frame (header[6] + payload length + checksum[2])
  *frame_size = 6 + msg->payload_length + 2;

  // -- Form header
  frame[0] = 0xB5;                              // Sync Char 1
  frame[1] = 0x62;                              // Sync Char 2
  frame[2] = msg->msg_class;                    // Message class
  frame[3] = msg->msg_id;                       // Message id
  frame[4] = msg->payload_length & 0xFF;        // Length
  frame[5] = (msg->payload_length >> 8) & 0xFF; // Length

  // -- Form payload
  for (size_t i = 0; i < msg->payload_length; i++) {
    frame[6 + i] = msg->payload[i];
  }

  // -- Form checksum
  frame[*frame_size - 2] = msg->ck_a;
  frame[*frame_size - 1] = msg->ck_b;
}

void ubx_msg_print(const ubx_msg_t *msg) {
  printf("msg_class: 0x%02x\n", msg->msg_class);
  printf("msg_id: 0x%02x\n", msg->msg_id);
  printf("payload_length: 0x%02x\n", msg->payload_length);
  for (size_t i = 0; i < msg->payload_length; i++) {
    printf("payload[%zu]: 0x%02x\n", i, msg->payload[i]);
  }
  printf("ck_a: 0x%02x\n", msg->ck_a);
  printf("ck_b: 0x%02x\n", msg->ck_b);
}

ubx_nav_dop_t ubx_nav_dop(const ubx_msg_t *msg) {
  ubx_nav_dop_t nav_dop;

  nav_dop.itow = ubx_uint32(msg->payload, 0);
  nav_dop.gdop = ubx_int16(msg->payload, 4);
  nav_dop.pdop = ubx_int16(msg->payload, 6);
  nav_dop.tdop = ubx_int16(msg->payload, 8);
  nav_dop.vdop = ubx_int16(msg->payload, 10);
  nav_dop.hdop = ubx_int16(msg->payload, 12);
  nav_dop.ndop = ubx_int16(msg->payload, 14);
  nav_dop.edop = ubx_int16(msg->payload, 16);

  return nav_dop;
}

ubx_nav_eoe_t ubx_nav_eoe(const ubx_msg_t *msg) {
  ubx_nav_eoe_t ubx_nav_eoe;
  ubx_nav_eoe.itow = ubx_uint32(msg->payload, 0);
  return ubx_nav_eoe;
}

ubx_nav_hpposllh_t ubx_nav_hpposllh(const ubx_msg_t *msg) {
  ubx_nav_hpposllh_t ubx_nav_hpposllh;

  ubx_nav_hpposllh.version = ubx_uint8(msg->payload, 0);
  ubx_nav_hpposllh.itow = ubx_uint32(msg->payload, 4);
  ubx_nav_hpposllh.lon = ubx_int32(msg->payload, 8);
  ubx_nav_hpposllh.lat = ubx_int32(msg->payload, 12);
  ubx_nav_hpposllh.height = ubx_int32(msg->payload, 16);
  ubx_nav_hpposllh.hmsl = ubx_int32(msg->payload, 20);
  ubx_nav_hpposllh.lon_hp = ubx_int8(msg->payload, 24);
  ubx_nav_hpposllh.lat_hp = ubx_int8(msg->payload, 25);
  ubx_nav_hpposllh.height_hp = ubx_int8(msg->payload, 26);
  ubx_nav_hpposllh.hmsl_hp = ubx_int8(msg->payload, 27);
  ubx_nav_hpposllh.hacc = ubx_uint32(msg->payload, 28);
  ubx_nav_hpposllh.vacc = ubx_uint32(msg->payload, 32);

  return ubx_nav_hpposllh;
}

ubx_nav_pvt_t ubx_nav_pvt(const ubx_msg_t *msg) {
  ubx_nav_pvt_t ubx_nav_pvt;

  ubx_nav_pvt.itow = ubx_uint32(msg->payload, 0);
  ubx_nav_pvt.year = ubx_uint16(msg->payload, 4);
  ubx_nav_pvt.month = ubx_uint8(msg->payload, 6);
  ubx_nav_pvt.day = ubx_uint8(msg->payload, 7);
  ubx_nav_pvt.hour = ubx_uint8(msg->payload, 8);
  ubx_nav_pvt.min = ubx_uint8(msg->payload, 9);
  ubx_nav_pvt.sec = ubx_uint8(msg->payload, 10);
  ubx_nav_pvt.valid = ubx_uint8(msg->payload, 11);
  ubx_nav_pvt.tacc = ubx_uint32(msg->payload, 12);
  ubx_nav_pvt.nano = ubx_int32(msg->payload, 16);
  ubx_nav_pvt.fix_type = ubx_uint8(msg->payload, 20);
  ubx_nav_pvt.flags = ubx_uint8(msg->payload, 21);
  ubx_nav_pvt.flags2 = ubx_uint8(msg->payload, 22);
  ubx_nav_pvt.num_sv = ubx_uint8(msg->payload, 23);
  ubx_nav_pvt.lon = ubx_int32(msg->payload, 24);
  ubx_nav_pvt.lat = ubx_int32(msg->payload, 28);
  ubx_nav_pvt.height = ubx_int32(msg->payload, 32);
  ubx_nav_pvt.hmsl = ubx_int32(msg->payload, 36);
  ubx_nav_pvt.hacc = ubx_uint32(msg->payload, 40);
  ubx_nav_pvt.vacc = ubx_uint32(msg->payload, 44);
  ubx_nav_pvt.veln = ubx_int32(msg->payload, 48);
  ubx_nav_pvt.vele = ubx_int32(msg->payload, 52);
  ubx_nav_pvt.veld = ubx_int32(msg->payload, 56);
  ubx_nav_pvt.gspeed = ubx_int32(msg->payload, 60);
  ubx_nav_pvt.headmot = ubx_int32(msg->payload, 64);
  ubx_nav_pvt.sacc = ubx_uint32(msg->payload, 68);
  ubx_nav_pvt.headacc = ubx_uint32(msg->payload, 72);
  ubx_nav_pvt.pdop = ubx_uint16(msg->payload, 76);
  ubx_nav_pvt.headveh = ubx_int32(msg->payload, 84);
  ubx_nav_pvt.magdec = ubx_int16(msg->payload, 88);
  ubx_nav_pvt.magacc = ubx_uint16(msg->payload, 90);

  return ubx_nav_pvt;
}

ubx_nav_status_t ubx_nav_status(const ubx_msg_t *msg) {
  ubx_nav_status_t ubx_nav_status;

  ubx_nav_status.itow = ubx_uint32(msg->payload, 0);
  ubx_nav_status.fix = ubx_uint8(msg->payload, 4);
  ubx_nav_status.flags = ubx_uint8(msg->payload, 5);
  ubx_nav_status.fix_status = ubx_uint8(msg->payload, 6);
  ubx_nav_status.flags2 = ubx_uint8(msg->payload, 7);
  ubx_nav_status.ttff = ubx_uint32(msg->payload, 8);
  ubx_nav_status.msss = ubx_uint32(msg->payload, 12);

  return ubx_nav_status;
}

ubx_nav_svin_t ubx_nav_svin(const ubx_msg_t *msg) {
  ubx_nav_svin_t ubx_nav_svin;

  ubx_nav_svin.itow = ubx_uint32(msg->payload, 4);
  ubx_nav_svin.dur = ubx_uint32(msg->payload, 8);
  ubx_nav_svin.mean_x = ubx_int32(msg->payload, 12);
  ubx_nav_svin.mean_y = ubx_int32(msg->payload, 16);
  ubx_nav_svin.mean_z = ubx_int32(msg->payload, 20);
  ubx_nav_svin.mean_xhp = ubx_int8(msg->payload, 24);
  ubx_nav_svin.mean_yhp = ubx_int8(msg->payload, 25);
  ubx_nav_svin.mean_zhp = ubx_int8(msg->payload, 26);
  ubx_nav_svin.mean_acc = ubx_uint32(msg->payload, 28);
  ubx_nav_svin.obs = ubx_uint32(msg->payload, 32);
  ubx_nav_svin.valid = ubx_uint8(msg->payload, 36);
  ubx_nav_svin.active = ubx_uint8(msg->payload, 37);

  return ubx_nav_svin;
}

ubx_nav_velned_t ubx_nav_velned(const ubx_msg_t *msg) {
  ubx_nav_velned_t ubx_nav_velned;

  ubx_nav_velned.itow = ubx_uint32(msg->payload, 0);
  ubx_nav_velned.veln = ubx_int32(msg->payload, 4);
  ubx_nav_velned.vele = ubx_int32(msg->payload, 8);
  ubx_nav_velned.veld = ubx_int32(msg->payload, 12);
  ubx_nav_velned.speed = ubx_uint32(msg->payload, 16);
  ubx_nav_velned.gspeed = ubx_uint32(msg->payload, 20);
  ubx_nav_velned.heading = ubx_int32(msg->payload, 24);
  ubx_nav_velned.sacc = ubx_uint32(msg->payload, 28);
  ubx_nav_velned.cacc = ubx_uint32(msg->payload, 32);

  return ubx_nav_velned;
}

ubx_rxm_rtcm_t ubx_rxm_rtcm(const ubx_msg_t *msg) {
  ubx_rxm_rtcm_t ubx_rxm_rtcm;

  ubx_rxm_rtcm.flags = msg->payload[1];
  ubx_rxm_rtcm.sub_type = ubx_uint16(msg->payload, 2);
  ubx_rxm_rtcm.ref_station = ubx_uint16(msg->payload, 4);
  ubx_rxm_rtcm.msg_type = ubx_uint16(msg->payload, 6);

  return ubx_rxm_rtcm;
}

ubx_mon_rf_t ubx_mon_rf(const ubx_msg_t *msg) {
  ubx_mon_rf_t ubx_mon_rf;

  ubx_mon_rf.version = ubx_uint8(msg->payload, 0);
  ubx_mon_rf.nblocks = ubx_uint8(msg->payload, 1);
  for (uint32_t i = 0; i < ubx_mon_rf.nblocks; i++) {
    const uint32_t offset = 24 * i;
    ubx_mon_rf.block_id[i] = ubx_uint8(msg->payload, 4 + offset);
    ubx_mon_rf.flags[i] = ubx_uint8(msg->payload, 5 + offset);
    ubx_mon_rf.ant_status[i] = ubx_uint8(msg->payload, 6 + offset);
    ubx_mon_rf.ant_power[i] = ubx_uint8(msg->payload, 7 + offset);
    ubx_mon_rf.post_status[i] = ubx_uint8(msg->payload, 8 + offset);
    ubx_mon_rf.noise_per_ms[i] = ubx_uint8(msg->payload, 16 + offset);
    ubx_mon_rf.agc_cnt[i] = ubx_uint8(msg->payload, 18 + offset);
    ubx_mon_rf.jam_ind[i] = ubx_uint8(msg->payload, 20 + offset);
    ubx_mon_rf.ofs_i[i] = ubx_uint8(msg->payload, 21 + offset);
    ubx_mon_rf.mag_i[i] = ubx_uint8(msg->payload, 22 + offset);
    ubx_mon_rf.ofs_q[i] = ubx_uint8(msg->payload, 23 + offset);
    ubx_mon_rf.mag_q[i] = ubx_uint8(msg->payload, 24 + offset);
  }

  return ubx_mon_rf;
}

void print_ubx_nav_hpposllh(const ubx_nav_hpposllh_t *msg) {
  printf("[nav-hpposllh] ");
  printf("itow: %d", msg->itow);
  printf("\t");
  printf("lon: %d", (int32_t) msg->lon);
  printf("\t");
  printf("lat: %d", (int32_t) msg->lat);
  printf("\t");
  printf("height: %d", msg->height);
  printf("\t");
  printf("hmsl: %d", msg->hmsl);
  printf("\t");
  printf("lon_hp: %d", msg->lon_hp);
  printf("\t");
  printf("lat_hp: %d", msg->lat_hp);
  printf("\t");
  printf("height_hp: %d", msg->height_hp);
  printf("\t");
  printf("hmsl_hp: %d", msg->hmsl_hp);
  printf("\t");
  printf("hacc: %d", msg->hacc);
  printf("\t");
  printf("vacc: %d", msg->vacc);
  printf("\n");
}

void print_ubx_nav_pvt(const ubx_nav_pvt_t *msg) {
  printf("[nav-hpposllh] ");
  printf("itow: %d", msg->itow);
  printf("\t");
  printf("lon: %d", (int32_t) msg->lon);
  printf("\t");
  printf("lat: %d", (int32_t) msg->lat);
  printf("\t");
  printf("height: %d", msg->height);
  printf("\t");
  switch (msg->fix_type) {
    case 0:
      printf("Fix type: no fix");
      break;
    case 1:
      printf("Fix type: dead reckoning only");
      break;
    case 2:
      printf("Fix type: 2D-fix");
      break;
    case 3:
      printf("Fix type: 3D-fix");
      break;
    case 4:
      printf("Fix type: GNSS + dead reckoning combined");
      break;
    case 5:
      printf("Fix type: time only fix");
      break;
  }
  printf("\n");
}

void print_ubx_nav_status(const ubx_nav_status_t *msg) {
  printf("[nav-status] ");
  printf("itow: %d", msg->itow);

  printf("\t");
  switch (msg->fix) {
    case 0x00:
      printf("fix: no fix");
      break;
    case 0x01:
      printf("fix: dead reckoning only");
      break;
    case 0x02:
      printf("fix: 2D-fix");
      break;
    case 0x03:
      printf("fix: 3D-fix");
      break;
    case 0x04:
      printf("fix: GNSS + dead reckoning combined");
      break;
    case 0x05:
      printf("fix: time only fix");
      break;
  }
  printf("\t");

  printf("flags: %d", msg->flags);
  printf("\t");

  const uint8_t diff_corr = (msg->fix_status & 0x1); // 0b00000001
  if (diff_corr) {
    printf("diff corr avail?: true");
  } else {
    printf("diff corr avail?: false");
  }
  printf("\t");
  printf("\t");

  const uint8_t map_matching = (msg->fix_status & 0xC0); // 0b11000000
  if (map_matching == 0x0) {
    printf("map matching: none");
  } else if (map_matching == 0x1) {
    printf("map matching: valid but not used");
  } else if (map_matching == 0x2) {
    printf("map matching: valid and used");
  } else if (map_matching == 0x3) {
    printf("map matching: valid and used");
  }
  printf("\t");

  printf("flags2: %d", msg->flags2);
  printf("\t");

  printf("ttff: %d", msg->ttff);
  printf("\t");

  printf("msss: %d", msg->msss);
  printf("\n");
}

void print_ubx_nav_svin(const ubx_nav_svin_t *msg) {
  printf("[nav-svin] ");
  printf("itow: %d", msg->itow);
  printf("\t");
  printf("dur: %d", msg->dur);
  printf("\t");
  printf("mean_x: %d", msg->mean_x);
  printf("\t");
  printf("mean_y: %d", msg->mean_y);
  printf("\t");
  printf("mean_z: %d", msg->mean_z);
  printf("\t");
  printf("active: 0x%02x", msg->active);
  printf("\t");
  printf("valid: 0x%02x", msg->valid);
  printf("\n");
}

void print_ubx_rxm_rtcm(const ubx_rxm_rtcm_t *msg) {
  printf("GOT RTCM3 msg type: [%d]", msg->msg_type);
  printf("\t");
  if (msg->flags == 0) {
    printf("RTCM OK!");
  } else {
    printf("RTCM NOT OK!");
  }
  printf("\n");
}

/*****************************************************************************
 * UBX Stream Parser
 ****************************************************************************/

void ubx_parser_init(ubx_parser_t *parser) {
  parser->state = SYNC_1;
  memset(parser->buf_data, '\0', 9046);
  parser->buf_pos = 0;
}

void ubx_parser_reset(ubx_parser_t *parser) {
  parser->state = SYNC_1;
  for (size_t i = 0; i < 1024; i++) {
    parser->buf_data[i] = 0;
  }
  parser->buf_pos = 0;
}

int ubx_parser_update(ubx_parser_t *parser, uint8_t data) {
  // Add byte to buffer
  parser->buf_data[parser->buf_pos++] = data;

  // Parse byte
  switch (parser->state) {
    case SYNC_1:
      if (data == 0xB5) {
        parser->state = SYNC_2;
      } else {
        ubx_parser_reset(parser);
      }
      break;
    case SYNC_2:
      if (data == 0x62) {
        parser->state = MSG_CLASS;
      } else {
        ubx_parser_reset(parser);
      }
      break;
    case MSG_CLASS:
      parser->state = MSG_ID;
      break;
    case MSG_ID:
      parser->state = PAYLOAD_LENGTH_LOW;
      break;
    case PAYLOAD_LENGTH_LOW:
      parser->state = PAYLOAD_LENGTH_HI;
      break;
    case PAYLOAD_LENGTH_HI:
      parser->state = PAYLOAD_DATA;
      break;
    case PAYLOAD_DATA: {
      uint8_t length_low = parser->buf_data[4];
      uint8_t length_hi = parser->buf_data[5];
      uint16_t payload_length = (length_hi << 8) | (length_low);
      if (parser->buf_pos == 6 + payload_length) {
        parser->state = CK_A;
      }
      if (parser->buf_pos >= 1022) {
        ubx_parser_reset(parser);
        return -2;
      }
      break;
    }
    case CK_A:
      parser->state = CK_B;
      break;
    case CK_B:
      ubx_msg_parse(&parser->msg, parser->buf_data);
      ubx_parser_reset(parser);
      return 1;
    // default: UBX_FATAL("Invalid Parser State!"); break;
    default:
      ubx_parser_reset(parser);
  }

  return 0;
}

/*****************************************************************************
 * RTCM3 Stream Parser
 ****************************************************************************/

void rtcm3_parser_init(rtcm3_parser_t *parser) {
  for (size_t i = 0; i < 9046; i++) {
    parser->buf_data[i] = 0;
  }
  parser->buf_pos = 0;
  parser->msg_len = 0;
}

void rtcm3_parser_reset(rtcm3_parser_t *parser) {
  rtcm3_parser_init(parser);
}

/**
 * RTCM 3.2 Frame
 * --------------
 * Byte 0: Always 0xD3
 * Byte 1: 6-bits of zero
 * Byte 2: 10-bits of length of this packet including the first two-ish header
 *         bytes, + 6.
 * byte 3 + 4: Msg type 12 bits
 *
 * Example [Msg type 1087]:
 *
 *   D3 00 7C 43 F0 ...
 *
 * Where 0x7C is the payload size = 124
 * = 124 + 6 [header]
 * = 130 total bytes in this packet
 */
int rtcm3_parser_update(rtcm3_parser_t *parser, uint8_t data) {
  // Add byte to buffer
  parser->buf_data[parser->buf_pos] = data;

  // Parse message
  if (parser->buf_data[0] != 0xD3) {
    rtcm3_parser_init(parser);

  } else if (parser->buf_pos == 1) {
    // Get the last two bits of this byte. Bits 8 and 9 of 10-bit length
    parser->msg_len = (data & 0x03) << 8;

  } else if (parser->buf_pos == 2) {
    parser->msg_len |= data; // Bits 0-7 of packet length
    parser->msg_len += 6;
    // There are 6 additional bytes of what we presume is
    // header, msgType, CRC, and stuff

  } else if (parser->buf_pos == 3) {
    parser->msg_type = data << 4; // Message Type, most significant 4 bits

  } else if (parser->buf_pos == 4) {
    parser->msg_type |= (data >> 4); // Message Type, bits 0-7
  }
  parser->buf_pos++;

  // Check if end of message
  if (parser->buf_pos == parser->msg_len) {
    return 1;
  }

  return 0;
}

/*****************************************************************************
 * UBlox
 ****************************************************************************/

int ublox_init(ublox_t *ublox, ubx_uart_t *uart) {
  ublox->state = UBLOX_READY;

  // UART
  if (uart->connected != 1) {
    if (ubx_uart_connect(uart, ublox->uart->port) != 0) {
      ublox->ok = -1;
      return -1;
    }
  }
  ublox->ok = 1;
  ublox->uart = uart;

  // Socket
  ublox->sockfd = -1;

  // Connections
  for (size_t i = 0; i < UBLOX_MAX_CONNS; i++) {
    ublox->conns[i] = -1;
  }
  ublox->nb_conns = 0;

  // Parsers
  ubx_parser_init(&ublox->ubx_parser);
  rtcm3_parser_init(&ublox->rtcm3_parser);

  // Callbacks
  ublox->ubx_cb = NULL;
  ublox->rtcm3_cb = NULL;

  return 0;
}

void ublox_disconnect(ublox_t *ublox) {
  // Serial
  if (ublox->uart->connected) {
    ubx_uart_disconnect(ublox->uart);
  }
  ublox->ok = -1;

  // Connections
  for (size_t i = 0; i < ublox->nb_conns; i++) {
    close(ublox->conns[i]);
    ublox->conns[i] = -1;
  }
  ublox->nb_conns = 0;

  // Socket
  if (ublox->sockfd != -1) {
    close(ublox->sockfd);
  }
  ublox->sockfd = -1;
}

int ublox_reset(ublox_t *ublox) {
  ublox_disconnect(ublox);
  return ublox_init(ublox, ublox->uart);
}

int ubx_write(const ublox_t *ublox,
              uint8_t msg_class,
              uint8_t msg_id,
              uint16_t length,
              uint8_t *payload) {
  // Build UBX message
  ubx_msg_t msg;
  ubx_msg_build(&msg, msg_class, msg_id, length, payload);

  // Serialize the message
  uint8_t frame[1024] = {0};
  size_t frame_size = 0;
  ubx_msg_serialize(&msg, frame, &frame_size);

  // Transmit msg
  size_t retval = ubx_uart_write(ublox->uart, frame, frame_size);
  if (retval != 0) {
    UBX_ERROR("Failed to send data to UART!");
    return -1;
  }

  return 0;
}

int ubx_poll(const ublox_t *ublox,
             const uint8_t msg_class,
             const uint8_t msg_id,
             uint16_t *payload_length,
             uint8_t *payload,
             const uint8_t expect_ack,
             const int retry) {
  int attempts = 0;
  ubx_parser_t parser;
  ubx_parser_init(&parser);

request:
  // Request
  attempts++;
  if (attempts > retry) {
    payload_length = 0;
    return -1;
  }
  ubx_write(ublox, msg_class, msg_id, *payload_length, payload);

  // Arbitrary counter for response timeout
  int counter = 0;
response:
  // Response
  while (counter < 1024) {
    uint8_t data = 0;
    if (ubx_uart_read(ublox->uart, &data, 1) == 0) {
      if (ubx_parser_update(&parser, data) == 1) {
        break;
      }
    }

    counter++;
  }

  // Check parsed message
  if (parser.msg.ok == 0) {
    UBX_WARN("Checksum failed, retrying ...");
    goto request;
  }

  // Try sending the request again?
  if (counter == 1024) {
    goto request;
  }

  // Check message
  const uint8_t msg_is_ack = (parser.msg.msg_class == UBX_ACK);
  const uint8_t match_class = (parser.msg.msg_class == msg_class);
  const uint8_t match_id = (parser.msg.msg_id == msg_id);
  if (!msg_is_ack && match_class && match_id) {
    // Copy payload length and data
    for (uint16_t i = 0; i < parser.msg.payload_length; i++) {
      payload[i] = parser.msg.payload[i];
    }
    *payload_length = parser.msg.payload_length;

    // Get another message (hopefully an ACK)
    if (expect_ack) {
      counter = 0;
      goto response;
    } else {
      return 0;
    }
  }
  if (!msg_is_ack && !match_class && !match_id) {
    // Get another message
    goto response;

  } else if (expect_ack && msg_is_ack) {
    // Check the ACK message
    const uint8_t match_class = (msg_class == parser.msg.payload[0]);
    const uint8_t match_id = (msg_id == parser.msg.payload[1]);
    const uint8_t is_ack = (parser.msg.msg_id == UBX_ACK_ACK);
    if (match_class && match_id && is_ack) {
      return 0;
    } else {
      return -2;
    }
  }

  return 0;
}

int ubx_read_ack(const ublox_t *ublox,
                 const uint8_t msg_class,
                 const uint8_t msg_id) {
  ubx_parser_t parser;
  ubx_parser_init(&parser);

  // Get Ack
  int counter = 0; // Arbitrary counter for timeout
  while (counter != 1024) {
    uint8_t data = 0;
    if (ubx_uart_read(ublox->uart, &data, 1) != 0) {
      continue;
    }

    if (ubx_parser_update(&parser, data) == 1) {
      const uint8_t is_ack_msg = (parser.msg.msg_class == UBX_ACK);
      const uint8_t ack_msg_class = parser.msg.payload[0];
      const uint8_t ack_msg_id = parser.msg.payload[1];
      const uint8_t ack_msg_class_match = (ack_msg_class == msg_class);
      const uint8_t ack_msg_id_match = (ack_msg_id == msg_id);

      if (is_ack_msg && ack_msg_class_match && ack_msg_id_match) {
        break;
      }
    }

    counter++;
  }

  // Try again?
  if (counter == 1024) {
    return 1;
  }

  return (parser.msg.msg_id == UBX_ACK_ACK) ? 0 : -1;
}

int ubx_get(const ublox_t *ublox,
            const uint8_t layer,
            const uint32_t key,
            uint32_t *val) {
  // Build message
  uint16_t payload_len = 4 + 4;
  uint8_t payload[1024] = {0};
  payload[0] = 0; // Version
  payload[1] = layer;
  payload[4 + 0] = key >> 0;
  payload[4 + 1] = key >> 8;
  payload[4 + 2] = key >> 16;
  payload[4 + 3] = key >> 24;

  // Poll
  if (ubx_poll(ublox, UBX_CFG, UBX_CFG_VALGET, &payload_len, payload, 1, 1) !=
      0) {
    return -1;
  }

  *val = ubx_uint32(payload, 8);
  return 0;
}

int ubx_set(const ublox_t *ublox,
            const uint8_t layer,
            const uint32_t key,
            const uint32_t val,
            const uint8_t val_size) {
  uint32_t bit_masks[4] = {0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000};

  // Build message
  uint16_t payload_length = 4 + 4 + val_size;
  uint8_t payload[1024] = {0};
  payload[0] = 0; // Version
  payload[1] = layer;
  payload[2] = 0;

  payload[4 + 0] = (key & bit_masks[0]);
  payload[4 + 1] = (key & bit_masks[1]) >> 8;
  payload[4 + 2] = (key & bit_masks[2]) >> 16;
  payload[4 + 3] = (key & bit_masks[3]) >> 24;

  for (uint8_t i = 0; i < val_size; i++) {
    payload[4 + 4 + i] = ((val & bit_masks[i]) >> (8 * i));
  }

  // Set value
  int attempts = 0;
retry:
  attempts++;
  if (attempts >= 10) {
    UBX_ERROR("Failed to set configuration!");
    return -1;
  }

  ubx_write(ublox, UBX_CFG, UBX_CFG_VALSET, payload_length, payload);
  switch (ubx_read_ack(ublox, UBX_CFG, UBX_CFG_VALSET)) {
    case 0:
      return 0;
    case 1:
      goto retry;
    case -1:
    default:
      UBX_ERROR("Failed to set configuration!");
      return -1;
  }
}

//***************************** Ublox GPS Mode *****************************

void ublox_version(const ublox_t *ublox) {
  uint16_t length = 0;
  uint8_t payload[1024] = {0};
  if (ubx_poll(ublox, UBX_MON, 0x04, &length, payload, 0, 5) == 0) {
    printf("SW VERSION: %s\n", payload);
    printf("HW VERSION: %s\n", payload + 30);
    printf("%s\n", payload + 40);
  } else {
    UBX_ERROR("Failed to obtain UBlox version!");
  }
}

int ublox_parse_ubx(ublox_t *ublox, uint8_t data) {
  if (ubx_parser_update(&ublox->ubx_parser, data) == 1) {
    UBX_DEBUG("[UBX]\tmsg_class: %d\tmsg_id: %d",
              ublox->ubx_parser.msg.msg_class,
              ublox->ubx_parser.msg.msg_id);

    // UBX message callback
    if (ublox->ubx_cb) {
      ublox->ubx_cb(ublox);
    }
    ublox->state = UBLOX_READY;
    return 1;
  }

  return 0;
}

int ublox_gps_config(ublox_t *ublox) {
  // Configure ublox
  const uint8_t layer = 1; // RAM
  int retval = 0;
  retval += ubx_set(ublox, layer, CFG_RATE_MEAS, 100, 2); // 100ms = 10Hz
  retval += ubx_set(ublox, layer, CFG_USBOUTPROT_NMEA, 0, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_CLOCK_USB, 0, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_DOP_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_EOE_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_HPPOSEECF_USB, 0, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_HPPOSLLH_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_STATUS_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_SVIN_USB, 0, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_PVT_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_NAV_VELNED_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_MON_RF_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_MSGOUT_UBX_RXM_RTCM_USB, 1, 1);
  retval += ubx_set(ublox, layer, CFG_TMODE_MODE, 0, 1);
  retval += ubx_set(ublox, layer, CFG_NAVSPG_DYNMODEL, 6, 1);
  if (retval != 0) {
    return -1;
  }

  return 0;
}

int ublox_run(ublox_t *ublox, int *loop) {
  // Configure GPS mode
  if (ublox_gps_config(ublox) != 0) {
    UBX_ERROR("Failed to configure Ublox into GPS mode!");
    return -1;
  }

  // Setup poll file descriptors
  const int timeout = 1; // 1ms
  struct pollfd fds[1];
  fds[0].fd = ublox->uart->connfd;
  fds[0].events = POLLIN;

  // Poll
  while (poll(fds, 1, timeout) >= 0 && *loop == 1) {
    // Read byte from UART and parse
    if (fds[0].revents & POLLIN) {
      uint8_t data = 0;
      if (ubx_uart_read(ublox->uart, &data, 1) != 0) {
        continue;
      }
      ublox_parse_ubx(ublox, data);
    }
  }

  return 0;
}

//*************************** Ublox Base Station ***************************

void ublox_broadcast_rtcm3(ublox_t *ublox) {
  const uint8_t *msg_data = ublox->rtcm3_parser.buf_data;
  const size_t msg_len = ublox->rtcm3_parser.msg_len;
  const int msg_flags = MSG_DONTWAIT | MSG_NOSIGNAL;

  // Broad cast RTCM3 to clients and check client connection
  int good_conns[UBLOX_MAX_CONNS] = {0};
  size_t nb_conns = 0;

  for (size_t i = 0; i < ublox->nb_conns; i++) {
    const int conn_fd = ublox->conns[i];
    if (send(conn_fd, msg_data, msg_len, msg_flags) == -1) {
      UBX_ERROR("Rover diconnected!");
    } else {
      good_conns[nb_conns] = conn_fd;
      nb_conns++;
    }
  }

  // Clear connections
  for (size_t i = 0; i < UBLOX_MAX_CONNS; i++) {
    ublox->conns[i] = 0;
  }
  ublox->nb_conns = 0;

  // Copy good connections back to ublox->conns
  for (size_t i = 0; i < nb_conns; i++) {
    ublox->conns[i] = good_conns[i];
  }
  ublox->nb_conns = nb_conns;
}

int ublox_parse_rtcm3(ublox_t *ublox, uint8_t data) {
  if (rtcm3_parser_update(&ublox->rtcm3_parser, data) == 1) {
    // Debug
    UBX_DEBUG("[RTCM3]\tmsg type: %zu\tmsg length: %zu",
              ublox->rtcm3_parser.msg_type,
              ublox->rtcm3_parser.msg_len);

    // RTCM3 message callback
    if (ublox->rtcm3_cb) {
      ublox->rtcm3_cb(ublox);
    }

    // Reset parser and msg type
    rtcm3_parser_reset(&ublox->rtcm3_parser);
    ublox->state = UBLOX_READY;

    return 1;
  }

  return 0;
}

int ublox_base_station_config(ublox_t *base) {
  const uint8_t layer = 1; // RAM
  // const uint8_t layer = 2; #<{(| BBR |)}>#
  int retval = 0;
  retval += ubx_set(base, layer, CFG_RATE_MEAS, 1000, 2); // 1000ms = 1Hz
  retval += ubx_set(base, layer, CFG_USBOUTPROT_NMEA, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_RTCM_3X_TYPE1005_USB, 1, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_RTCM_3X_TYPE1077_USB, 1, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_RTCM_3X_TYPE1087_USB, 1, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_RTCM_3X_TYPE1097_USB, 1, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_RTCM_3X_TYPE1127_USB, 1, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_RTCM_3X_TYPE1230_USB, 1, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_CLOCK_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_DOP_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_EOE_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_HPPOSEECF_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_HPPOSLLH_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_STATUS_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_SVIN_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_PVT_USB, 1, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_NAV_VELNED_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_MON_RF_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_MSGOUT_UBX_RXM_RTCM_USB, 0, 1);
  retval += ubx_set(base, layer, CFG_TMODE_MODE, 1, 1);
  retval += ubx_set(base, layer, CFG_TMODE_SVIN_MIN_DUR, 60, 4);
  retval += ubx_set(base, layer, CFG_TMODE_SVIN_ACC_LIMIT, 50000, 4);
  if (retval != 0) {
    UBX_ERROR("Failed to configure Ublox into BASE_STATION mode!");
    return -1;
  }

  base->rtcm3_cb = ublox_broadcast_rtcm3;

  return 0;
}

int ublox_base_run(ublox_t *base, const int port, int *loop) {
  // Configure base station
  if (ublox_base_station_config(base) != 0) {
    UBX_ERROR("Failed to configure Ublox into BASE_STATION mode!");
    return -1;
  }

  // Socket create and verification
  base->sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (base->sockfd == -1) {
    UBX_ERROR("TCP Socket creation failed...");
    return -1;
  } else {
    UBX_DEBUG("TCP created!");
  }

  // Change server socket into non-blocking state
  fcntl(base->sockfd, F_SETFL, O_NONBLOCK);

  // Socket options
  const int level = SOL_SOCKET;
  const int val = 1;
  const socklen_t len = sizeof(int);
  if (setsockopt(base->sockfd, level, SO_REUSEADDR, &val, len) < 0) {
    UBX_ERROR("setsockopt(SO_REUSEADDR) failed");
  }
  if (setsockopt(base->sockfd, SOL_SOCKET, SO_REUSEPORT, &val, len) < 0) {
    UBX_ERROR("setsockopt(SO_REUSEPORT) failed");
  }

  // Assign IP, PORT
  struct sockaddr_in server;
  bzero(&server, sizeof(server));
  server.sin_family = AF_INET;
  server.sin_addr.s_addr = htonl(INADDR_ANY);
  server.sin_port = htons(port);

  // Bind newly created socket to given IP
  int retval = bind(base->sockfd, (struct sockaddr *) &server, sizeof(server));
  if (retval != 0) {
    UBX_ERROR("TCP Socket bind failed...");
    UBX_ERROR("%s", strerror(errno));
    return -1;
  } else {
    UBX_DEBUG("TCP Socket binded!");
  }

  // Server is ready to listen
  if ((listen(base->sockfd, 20)) != 0) {
    UBX_ERROR("Listen failed...");
    return -1;
  } else {
    UBX_DEBUG("Server running");
  }

  // Obtain RTCM3 Messages from receiver and transmit them to rover
  while (*loop) {
    // Accept the data packet from client and verification
    struct sockaddr_in client;
    socklen_t len = sizeof(client);
    int connfd = accept(base->sockfd, (struct sockaddr *) &client, &len);
    if (connfd >= 0) {
      char ip[INET6_ADDRSTRLEN] = {0};
      int port = 0;
      ubx_ip_port_info(connfd, ip, &port);
      UBX_INFO("Server connected with UBlox client [%s:%d]", ip, port);

      base->conns[base->nb_conns] = connfd;
      base->nb_conns++;
    }

    // Read byte
    uint8_t data = 0;
    if (ubx_uart_read(base->uart, &data, 1) != 0) {
      continue;
    }

    // Parse data
    switch (base->state) {
      case UBLOX_READY:
        if (data == 0xB5) {
          base->state = UBLOX_PARSING_UBX;
        } else if (data == 0xD3) {
          base->state = UBLOX_PARSING_RTCM3;
        }
        break;
      case UBLOX_PARSING_UBX:
        ublox_parse_ubx(base, data);
        break;
      case UBLOX_PARSING_RTCM3:
        ublox_parse_rtcm3(base, data);
        break;
    }
  }

  // Clean up
  ublox_disconnect(base);

  return 0;
}

//****************************** Ublox Rover *******************************

int ublox_rover_config(ublox_t *rover) {
  // Configure rover
  const uint8_t layer = 1; // RAM
  int retval = 0;
  retval += ubx_set(rover, layer, CFG_RATE_MEAS, 100, 2); // 100ms = 10Hz
  retval += ubx_set(rover, layer, CFG_USBOUTPROT_NMEA, 0, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_CLOCK_USB, 0, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_DOP_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_EOE_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_HPPOSEECF_USB, 0, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_HPPOSLLH_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_STATUS_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_SVIN_USB, 0, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_PVT_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_NAV_VELNED_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_MON_RF_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_MSGOUT_UBX_RXM_RTCM_USB, 1, 1);
  retval += ubx_set(rover, layer, CFG_TMODE_MODE, 0, 1);
  retval += ubx_set(rover, layer, CFG_NAVSPG_DYNMODEL, 6, 1);
  if (retval != 0) {
    return -1;
  }

  return 0;
}

int ublox_rover_run(ublox_t *rover,
                    const char *base_ip,
                    const int base_port,
                    int *loop) {
  // Configure rover
  if (ublox_rover_config(rover) != 0) {
    UBX_ERROR("Failed to configure Ublox into ROVER mode!");
    return -1;
  }

  // Create socket
  rover->sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (rover->sockfd == -1) {
    UBX_ERROR("TCP Socket creation failed!");
    return -1;
  } else {
    UBX_DEBUG("Created TCP socket!");
  }

  // Assign IP, PORT
  struct sockaddr_in server;
  bzero(&server, sizeof(server));
  server.sin_family = AF_INET;
  server.sin_addr.s_addr = inet_addr(base_ip);
  server.sin_port = htons(base_port);

  // Connect to server
  const socklen_t len = sizeof(server);
  int retval = connect(rover->sockfd, (struct sockaddr *) &server, len);
  if (retval != 0) {
    UBX_ERROR("Connection with the server failed!");
    return -1;
  } else {
    UBX_DEBUG("Connected to the server!");
  }

  // Poll for UBX from UART and RTCM3 Messages from TCP connection
  // -- Setup poll file descriptors
  struct pollfd fds[2];
  // -- UART
  fds[0].fd = rover->uart->connfd;
  fds[0].events = POLLIN;
  // -- Server
  fds[1].fd = rover->sockfd;
  fds[1].events = POLLIN;

  // Poll
  const int timeout = 1; // 1ms
  while (poll(fds, 2, timeout) >= 0 && *loop == 1) {
    // Read byte from UART and parse
    if (fds[0].revents & POLLIN) {
      uint8_t data = 0;
      if (ubx_uart_read(rover->uart, &data, 1) != 0) {
        continue;
      }
      ublox_parse_ubx(rover, data);
    }

    // Read byte from TCP socket (assuming its the RTCM3)
    if (fds[1].fd != -1 && (fds[1].revents & POLLIN)) {
      // Read byte
      uint8_t data = 0;
      if (read(rover->sockfd, &data, 1) != 1) {
        UBX_ERROR("Failed to read RTCM3 byte from server!");
        UBX_ERROR("Ignoring server for now!");
        fds[1].fd = -1;
      }

      // Transmit RTCM3 packet if its ready
      if (rtcm3_parser_update(&rover->rtcm3_parser, data)) {
        const uint8_t *msg_data = rover->rtcm3_parser.buf_data;
        const size_t msg_len = rover->rtcm3_parser.msg_len;
        ubx_uart_write(rover->uart, msg_data, msg_len);
        rtcm3_parser_reset(&rover->rtcm3_parser);
      }
    }
  }

  // Clean up
  ublox_disconnect(rover);

  return 0;
}

#endif // UBX_IMPLEMENTATION

//////////////////////////////////////////////////////////////////////////////
//                                UNITTESTS                                 //
//////////////////////////////////////////////////////////////////////////////

#ifdef UBX_UNITTEST

#include <signal.h>

// UNITESTS GLOBAL VARIABLES
static int nb_tests = 0;
static int nb_passed = 0;
static int nb_failed = 0;

#define ENABLE_TERM_COLORS 0
#if ENABLE_TERM_COLORS == 1
#define TERM_RED "\x1B[1;31m"
#define TERM_GRN "\x1B[1;32m"
#define TERM_WHT "\x1B[1;37m"
#define TERM_NRM "\x1B[1;0m"
#else
#define TERM_RED
#define TERM_GRN
#define TERM_WHT
#define TERM_NRM
#endif

/**
 * Run unittests
 * @param[in] test_name Test name
 * @param[in] test_ptr Pointer to unittest
 */
void run_test(const char *test_name, int (*test_ptr)(void)) {
  printf("-> [%s] ", test_name);

  if ((*test_ptr)() == 0) {
    printf(TERM_GRN "OK!\n" TERM_NRM);
    fflush(stdout);
    nb_passed++;
  } else {
    printf(TERM_RED "FAILED!\n" TERM_NRM);
    fflush(stdout);
    nb_failed++;
  }
  nb_tests++;
}

/**
 * Add unittest
 * @param[in] TEST Test function
 */
#define TEST(TEST_FN) run_test(#TEST_FN, TEST_FN);

/**
 * Unit-test assert
 * @param[in] TEST Test condition
 */
#define TEST_ASSERT(TEST)                                                      \
  do {                                                                         \
    if ((TEST) == 0) {                                                         \
      printf(TERM_RED "ERROR!" TERM_NRM " [%s:%d] %s FAILED!\n",               \
             __func__,                                                         \
             __LINE__,                                                         \
             #TEST);                                                           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

int loop = 1;
static void signal_handler(int sig) {
  loop = 0;
}

/*****************************************************************************
 * UBX Message
 ****************************************************************************/

int test_ubx_msg_init(void) {
  ubx_msg_t msg;
  ubx_msg_init(&msg);

  TEST_ASSERT(msg.ok == 0);
  TEST_ASSERT(msg.msg_class == 0);
  TEST_ASSERT(msg.msg_id == 0);
  TEST_ASSERT(msg.payload_length == 0);
  TEST_ASSERT(msg.payload[0] == 0);
  TEST_ASSERT(msg.ck_a == 0);
  TEST_ASSERT(msg.ck_b == 0);

  return 0;
}

int test_ubx_msg_checksum(void) {
  // Create frame
  uint8_t data[10] = {0};
  data[0] = 0xB5; // SYNC 1
  data[1] = 0x62; // SYNC 2
  data[2] = 0x01; // Message class
  data[3] = 0x22; // Message id
  data[4] = 0x0;  // Payload length low
  data[5] = 0x0;  // Payload length hi

  // Calculate checksum
  ubx_msg_checksum(data[2],
                   data[3],
                   (data[4] << 8) | (data[5]),
                   NULL,
                   &data[6],
                   &data[7]);

  TEST_ASSERT(data[6] == 0x23);
  TEST_ASSERT(data[7] == 0x6a);

  return 0;
}

int test_ubx_msg_is_valid(void) {
  return 0;
}

int test_ubx_msg_build(void) {
  ubx_msg_t msg;
  uint8_t msg_class = 0x01;
  uint8_t msg_id = 0x22;
  uint16_t payload_length = 0x00;
  ubx_msg_build(&msg, msg_class, msg_id, payload_length, NULL);

  TEST_ASSERT(msg.msg_class == msg_class);
  TEST_ASSERT(msg.msg_id == msg_id);
  TEST_ASSERT(msg.payload_length == payload_length);

  return 0;
}

int test_ubx_msg_parse_and_serialize(void) {
  // Create frame
  uint8_t data[10] = {0};
  data[0] = 0xB5; // SYNC 1
  data[1] = 0x62; // SYNC 2
  data[2] = 0x01; // Message class
  data[3] = 0x22; // Message id
  data[4] = 0x0;  // Payload length low
  data[5] = 0x0;  // Payload length hi
  // -- Calculate checksum
  ubx_msg_checksum(data[2],
                   data[3],
                   (data[4] << 8) | (data[5]),
                   NULL,
                   &data[6],
                   &data[7]);

  // Parse
  ubx_msg_t msg;
  ubx_msg_parse(&msg, data);
  TEST_ASSERT(msg.ok);
  TEST_ASSERT(msg.msg_class == 0x01);
  TEST_ASSERT(msg.msg_id == 0x22);
  TEST_ASSERT(msg.payload_length == 0x0);
  TEST_ASSERT(msg.ck_a == 0x23);
  TEST_ASSERT(msg.ck_b == 0x6a);

  // Serialize
  uint8_t frame[1024] = {0};
  size_t frame_size = 0;
  ubx_msg_serialize(&msg, frame, &frame_size);

  TEST_ASSERT(frame_size == 8);
  for (size_t i = 0; i < frame_size; i++) {
    TEST_ASSERT(frame[i] == data[i]);
  }

  return 0;
}

int test_ubx_msg_print(void) {
  ubx_msg_t msg;

  uint8_t msg_class = 0x01;
  uint8_t msg_id = 0x22;
  uint16_t payload_length = 0x00;
  ubx_msg_build(&msg, msg_class, msg_id, payload_length, NULL);
  ubx_msg_print(&msg);

  return 0;
}

/*****************************************************************************
 * UBX Stream Parser
 ****************************************************************************/

int test_ubx_parser_init(void) {
  ubx_parser_t parser;

  ubx_parser_init(&parser);
  TEST_ASSERT(parser.state == SYNC_1);
  TEST_ASSERT(parser.buf_data[0] == '\0');
  TEST_ASSERT(parser.buf_pos == 0);

  return 0;
}

int test_ubx_parser_reset(void) {
  ubx_parser_t parser;

  ubx_parser_reset(&parser);
  TEST_ASSERT(parser.state == SYNC_1);
  TEST_ASSERT(parser.buf_data[0] == '\0');
  TEST_ASSERT(parser.buf_pos == 0);

  return 0;
}

int test_ubx_parser_update(void) {
  ubx_parser_t parser;

  ubx_parser_update(&parser, 0x0);

  return 0;
}

/*****************************************************************************
 * UBLOX
 ****************************************************************************/

int test_ublox_init(void) {
  // Setup UART connection to UBlox
  ubx_uart_t uart;
  if (ubx_uart_connect(&uart, "/dev/ttyACM0") != 0) {
    UBX_ERROR("Failed to connect to ublox!");
    return -1;
  }

  // Setup UBlox
  ublox_t ublox;
  if (ublox_init(&ublox, &uart) != 0) {
    UBX_ERROR("Failed to setup ublox!");
    return -1;
  }

  // Clean up
  ublox_disconnect(&ublox);
  sleep(2);

  return 0;
}

int test_ublox_version(void) {
  // Setup UART connection to UBlox
  ubx_uart_t uart;
  if (ubx_uart_connect(&uart, "/dev/ttyACM0") != 0) {
    UBX_ERROR("Failed to connect to ublox!");
    return -1;
  }

  // Setup UBlox
  ublox_t ublox;
  if (ublox_init(&ublox, &uart) != 0) {
    UBX_ERROR("Failed to setup ublox!");
    return -1;
  }

  // Print UBlox version
  ublox_version(&ublox);

  // Clean up
  ublox_disconnect(&ublox);
  sleep(1);

  return 0;
}

int test_ubx_set_and_get(void) {
  // Setup UART connection to UBlox
  ubx_uart_t uart;
  if (ubx_uart_connect(&uart, "/dev/ttyACM0") != 0) {
    UBX_ERROR("Failed to connect to ublox!");
    return -1;
  }

  // Setup UBlox
  ublox_t ublox;
  if (ublox_init(&ublox, &uart) != 0) {
    UBX_ERROR("Failed to setup ublox!");
    return -1;
  }

  uint32_t key = CFG_MSGOUT_RTCM_3X_TYPE1005_USB;
  uint32_t val = 1;
  uint8_t val_size = 1;
  ubx_set(&ublox, 1, key, val, val_size);

  ublox_reset(&ublox);

  uint32_t value = 0;
  ubx_get(&ublox, 0, key, &value);
  TEST_ASSERT(value == val);

  // Clean up
  ublox_disconnect(&ublox);
  sleep(1);

  return 0;
}

int test_ublox_parse_rtcm3(void) {
  // Setup UART connection to UBlox
  ubx_uart_t uart;
  if (ubx_uart_connect(&uart, "/dev/ttyACM0") != 0) {
    UBX_ERROR("Failed to connect to ublox!");
    return -1;
  }

  // Setup UBlox
  ublox_t ublox;
  if (ublox_init(&ublox, &uart) != 0) {
    UBX_ERROR("Failed to setup ublox!");
    return -1;
  }

  // clang-format off
  uint8_t data[25 + 129 + 201 + 176 + 14] = {
    // RTCM3 1005
    0xD3, 0x00, 0x13, 0x3E, 0xD0, 0x00, 0x03, 0x89, 0x43, 0x50, 0xA5, 0x6B,
    0xBF, 0xF8, 0xEC, 0xBB, 0xD8, 0x0B, 0x91, 0x87, 0xEA, 0xA2, 0x09, 0xAA,
    0xF3,
    // RTCM3 1074
    0xD3, 0x00, 0x7B, 0x43, 0x20, 0x00, 0x4B, 0x63, 0xBE, 0x62, 0x00, 0x00,
    0x00, 0x00, 0x8C, 0x35, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x80, 0x00,
    0x55, 0xFE, 0x88, 0x9C, 0x8C, 0x86, 0x94, 0xA0, 0xA5, 0x6B, 0x9C, 0x6E,
    0xCE, 0x73, 0x19, 0x2B, 0xE5, 0x08, 0x8A, 0x1E, 0xE7, 0x79, 0x5D, 0xF1,
    0x9D, 0x62, 0x57, 0xDD, 0x9D, 0xB7, 0x60, 0x77, 0xD0, 0xDE, 0x3E, 0x71,
    0x1C, 0x7D, 0x01, 0x15, 0x14, 0xFB, 0xB6, 0x13, 0xCB, 0x11, 0x5F, 0x19,
    0xF9, 0xBC, 0x4E, 0x26, 0xF7, 0x5B, 0x67, 0xDB, 0xF0, 0xF0, 0x77, 0x77,
    0x01, 0xB8, 0xB4, 0xF9, 0xBB, 0x9B, 0xE5, 0x08, 0xBD, 0xDD, 0xDD, 0xDD,
    0xDD, 0xDD, 0x00, 0x18, 0x59, 0x7E, 0x76, 0x5D, 0x75, 0xD3, 0x46, 0xF8,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4B, 0x63, 0xF7,
    // RTCM3 1077
    0xD3, 0x00, 0xC3, 0x43, 0x50, 0x00, 0x4B, 0x63, 0xDD, 0xA2, 0x00, 0x00,
    0x00, 0x00, 0x8C, 0x35, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x80, 0x00,
    0x55, 0xFE, 0x88, 0x9C, 0x8C, 0x86, 0x94, 0xA0, 0xA4, 0x00, 0x00, 0x00,
    0x16, 0xA9, 0xBE, 0xEC, 0xE7, 0x51, 0x72, 0xC6, 0x53, 0x01, 0xF2, 0x13,
    0x07, 0xEF, 0xFF, 0xB9, 0x04, 0x25, 0xEC, 0x9F, 0xA9, 0xC6, 0xE8, 0x33,
    0xB5, 0x67, 0x87, 0x80, 0xF7, 0x85, 0x6C, 0x98, 0x1E, 0xC6, 0x1B, 0x78,
    0xA1, 0x7A, 0x46, 0x7A, 0xA6, 0xC7, 0x61, 0xE9, 0x86, 0x35, 0x37, 0xFE,
    0x4E, 0x1B, 0xA7, 0xD5, 0xED, 0x8A, 0x14, 0x1E, 0x10, 0xF9, 0xE1, 0x61,
    0x9D, 0xE0, 0x93, 0xA0, 0x06, 0xC9, 0xC8, 0x06, 0x0B, 0x76, 0x1E, 0x96,
    0x14, 0x1D, 0x6C, 0xDD, 0xE1, 0x73, 0xD3, 0xE0, 0x80, 0xC8, 0xDA, 0xB6,
    0xAD, 0xAB, 0x6A, 0xDB, 0xB6, 0xAD, 0xBB, 0x6A, 0xDB, 0xB6, 0xAD, 0xB8,
    0x00, 0xC0, 0x2C, 0x0B, 0xC3, 0x30, 0xB0, 0x2E, 0x0B, 0x82, 0xE0, 0x98,
    0x23, 0x07, 0xC0, 0x40, 0xD8, 0x2D, 0x8E, 0x04, 0x3D, 0xB1, 0xBB, 0x5E,
    0x0F, 0x67, 0x1E, 0x4F, 0xC5, 0x6F, 0x8C, 0x1E, 0x22, 0xFC, 0x66, 0xA0,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1C, 0x39, 0xE7,
    // RTCM3 1097
    0xD3, 0x00, 0xAA, 0x44, 0x90, 0x00, 0x4B, 0x63, 0xCE, 0x02, 0x00, 0x00,
    0x20, 0x18, 0x00, 0xC0, 0x08, 0x00, 0x00, 0x00, 0x20, 0x01, 0x00, 0x00,
    0x7F, 0xFA, 0xBA, 0x42, 0x52, 0x6A, 0x62, 0xC8, 0x00, 0x00, 0x07, 0x77,
    0xEC, 0xC7, 0xD4, 0x97, 0x89, 0xAF, 0x03, 0xC8, 0x06, 0xB7, 0xD8, 0x1F,
    0x09, 0x80, 0x04, 0x10, 0xCE, 0xCE, 0x59, 0x6C, 0x15, 0x30, 0x95, 0x09,
    0x87, 0xDC, 0x48, 0xB9, 0x48, 0x8A, 0x2F, 0x81, 0xD9, 0x66, 0x1F, 0x99,
    0x3F, 0x1A, 0x8F, 0xF1, 0x21, 0x6F, 0xBB, 0xB5, 0x7B, 0xDC, 0x2F, 0xB3,
    0x2B, 0x57, 0xB0, 0xEB, 0x60, 0x25, 0x52, 0x98, 0x1F, 0xC6, 0xD0, 0x2D,
    0x5D, 0x50, 0x28, 0x8A, 0x70, 0x77, 0x09, 0x68, 0x7E, 0x2F, 0xD7, 0xC6,
    0x96, 0x3F, 0xC4, 0xBC, 0xBF, 0xEF, 0xCB, 0x2F, 0xF0, 0x26, 0x13, 0x68,
    0xDB, 0xB6, 0x8D, 0xBB, 0x68, 0xDA, 0x36, 0x8D, 0xBB, 0x68, 0xDB, 0xB6,
    0x8D, 0xB8, 0x00, 0x5A, 0x16, 0x05, 0xA1, 0x58, 0x56, 0x15, 0x85, 0xA1,
    0x70, 0x60, 0x17, 0x85, 0x21, 0x50, 0x1A, 0x1E, 0x34, 0x16, 0x0D, 0x10,
    0x1B, 0x47, 0x08, 0x5E, 0x0F, 0x39, 0x7F, 0xB3, 0x0F, 0x8E, 0x31, 0x1C,
    0x4A, 0x4D, 0x6C, 0x99, 0xB8, 0x05, 0xDC, 0x86,
    // RTCM3 1230
    0xD3, 0x00, 0x08, 0x4C, 0xE0, 0x00, 0x8A, 0x00, 0x00, 0x00, 0x00, 0xA8,
    0xF7, 0x2A,
  };
  // clang-format on

  for (int i = 0; i < 25 + 129 + 201 + 176 + 14; i++) {
    if (ublox_parse_rtcm3(&ublox, data[i])) {
      TEST_ASSERT(ublox.rtcm3_parser.buf_pos == 0);
      TEST_ASSERT(ublox.rtcm3_parser.msg_len == 0);
    }
  }

  // Clean up
  ublox_disconnect(&ublox);
  sleep(1);

  return 0;
}

int test_ublox_run(void) {
  // Setup UART connection to UBlox
  ubx_uart_t uart;
  if (ubx_uart_connect(&uart, "/dev/ttyACM0") != 0) {
    UBX_ERROR("Failed to connect to ublox!");
    return -1;
  }

  // Setup ublox
  ublox_t ublox;
  if (ublox_init(&ublox, &uart) != 0) {
    UBX_ERROR("Failed to initialize ublox!");
    return -1;
  }

  // Configure and run Ublox in GPS mode
  loop = 1;
  ublox_run(&ublox, &loop);

  return 0;
}

int test_ublox_base(void) {
  // Setup UART connection to UBlox
  ubx_uart_t uart;
  if (ubx_uart_connect(&uart, "/dev/ttyACM0") != 0) {
    UBX_ERROR("Failed to connect to ublox!");
    return -1;
  }

  // Setup UBlox
  ublox_t base;
  if (ublox_init(&base, &uart) != 0) {
    UBX_ERROR("Failed to initialize ublox!");
    return -1;
  }

  // Run base station
  loop = 1;
  ublox_base_run(&base, 1234, &loop);

  return 0;
}

int test_ublox_rover(void) {
  // Setup UART connection to UBlox
  ubx_uart_t uart;
  if (ubx_uart_connect(&uart, "/dev/ttyACM0") != 0) {
    UBX_ERROR("Failed to connect to ublox!");
    return -1;
  }

  // Setup rover
  ublox_t rover;
  if (ublox_init(&rover, &uart) != 0) {
    UBX_ERROR("Failed to initialize ublox!");
    return -1;
  }

  // Run rover
  char *ip = "127.0.0.1";
  int port = 1234;
  loop = 1;
  ublox_rover_run(&rover, ip, port, &loop);

  return 0;
}

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);

  TEST(test_ubx_msg_init);
  TEST(test_ubx_msg_checksum);
  TEST(test_ubx_msg_is_valid);
  TEST(test_ubx_msg_build);
  TEST(test_ubx_msg_parse_and_serialize);
  TEST(test_ubx_msg_print);

  TEST(test_ubx_parser_init);
  TEST(test_ubx_parser_reset);
  TEST(test_ubx_parser_update);

  TEST(test_ublox_init);
  TEST(test_ublox_version);
  TEST(test_ubx_set_and_get);
  TEST(test_ublox_parse_rtcm3);
  TEST(test_ublox_run);
  // TEST(test_ublox_base);
  // TEST(test_ublox_rover);

  return (nb_failed) ? -1 : 0;
}

#endif // UBX_UNITTEST
