import glob

import cv2
import numpy as np
from PIL import Image


def YUV420p(byteArray, size):
    npArray = np.frombuffer(byteArray, dtype=np.uint8)

    e = size[0] * size[1]

    Y = npArray[0:e]
    Y = np.reshape(Y, size)

    V = npArray[e:int(e * 1.25)]
    V = np.reshape(V, (int(size[0] / 2), int(size[1] / 2)))
    V = V.repeat(2, axis=0).repeat(2, axis=1)

    U = npArray[int(e * 1.25):]
    U = np.reshape(U, (int(size[0] / 2), int(size[1] / 2)))
    U = U.repeat(2, axis=0).repeat(2, axis=1)

    return (np.dstack([Y, U, V])).astype(np.uint8)


class VideoCaptureYUV:
    def __init__(self, filename, size):
        self.height, self.width = size
        self.frame_len = int(self.width * self.height * 1.5)
        self.f = open(filename, 'rb')
        self.shape = (int(self.height * 1.5), self.width)

    def read_raw(self):
        try:

            raw = self.f.read()
            l_video = np.empty((0, self.height, self.width), int)
            for i in range(0, len(raw), self.frame_len):
                yuv = YUV420p(raw[i:i+self.frame_len], (self.height, self.width))
                rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)

                l_video = np.append(l_video, [l], axis = 0)
                print(i/self.frame_len, len(raw)/self.frame_len)

            #raw = self.f.read(self.frame_len)
            #yuv = YUV420p(raw, (self.height, self.width))
        except Exception as e:
            print(str(e))
            return False, None
        return True, l_video

    def read(self):
        ret, l_video = self.read_raw()
        if not ret:
            return ret, l_video
        return ret, l_video


if __name__ == "__main__":
    # filename = "data/20171214180916RGB.yuv"

    videos = sorted(glob.glob('./yuv/dis/*.yuv'))
    for video_path in videos:
        print(video_path)
        size = (1080, 1920)
        cap = VideoCaptureYUV(video_path, size)

        ret, l_video = cap.read()
        if ret:
            np_path = video_path[:video_path.find('.yuv')] + '_l.npy'
            with open(np_path, 'wb') as f:
                np.save(f, l_video)
                '''
    # cv2.imshow("frame", frame)
    # cv2.waitKey(30)'''