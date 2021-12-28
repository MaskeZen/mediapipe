namespace winnerPy
{

  const int IMG_WIDTH = 512;
  const int IMG_HEIGHT = 512;
  const int IMG_CHANNELS = 3;
  const int IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS;
  const int IMG_SHM_KEY = 411367;
  int last_msg_id = 0;

  const int IMG_OUT_WIDTH = 112;
  const int IMG_OUT_HEIGHT = 112;
  const int IMG_OUT_CHANNELS = 3;
  const int IMG_OUT_SIZE = IMG_OUT_WIDTH * IMG_OUT_HEIGHT * IMG_OUT_CHANNELS;

  struct datos_imagen
  {
    int msg_id;
    int msg_reply;
    int status;
    float yaw;
    float pitch;
    float roll;
    float detection_certainty;
    unsigned char imagen[IMG_SIZE];
  };

}