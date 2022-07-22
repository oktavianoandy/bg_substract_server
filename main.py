import numpy as np
import cv2
from collections import Counter
from werkzeug.utils import secure_filename
import statistics
from PIL import Image
import io
import base64


def preprocessing(file):
    print("proses prepocessing berjalan")
    video = cv2.VideoCapture(file)

    kernel = np.ones((5, 5), np.uint8)

    background_model = getBackgroundModel(file)
    background_model = cv2.resize(background_model, (600, 600))
    background_model = cv2.dilate(background_model, kernel)

    # cv2.imshow("background model", background_model)
    # cv2.waitKey(0)

    akumulasi = []
    frames = []

    while video.isOpened:
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.resize(frame, (600, 600))

        foreground_dilate = cv2.dilate(frame, kernel)
        foreground_diff = cv2.absdiff(foreground_dilate, background_model)

        # cv2.imshow("foregorund diff", foreground_diff)
        # cv2.waitKey(0)

        foreground_normalisasi = cv2.normalize(foreground_diff, None, alpha=0, beta=255
                                               , norm_type=cv2.NORM_MINMAX
                                               , dtype=cv2.CV_8UC1)

        # cv2.imshow("foregorund norm", foreground_normalisasi)
        # cv2.waitKey(0)

        foreground_final = cv2.cvtColor(foreground_normalisasi, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("foregorund gray", foreground_final)
        # cv2.waitKey(0)

        ret, foreground_tresh = cv2.threshold(foreground_final, 100, 255, cv2.THRESH_BINARY)

        # cv2.imshow("foregorund final", foreground_tresh)
        # cv2.waitKey(0)

        output = cv2.connectedComponentsWithStats(foreground_tresh, 4, cv2.CV_32S)
        last, frame = ccl(frame, output, foreground_tresh)

        frames.append(frame)
        akumulasi.append(last)

    jumlah_benih, final_frame = hasil(akumulasi, frames)

    print("Jumlah lele yang terdeteksi : " + str(jumlah_benih))

    pil_img = Image.fromarray(final_frame)
    buff = io.BytesIO()
    pil_img.save(buff, format("PNG"))
    img_str = base64.b64encode(buff.getvalue())

    print(img_str)

    cv2.imshow("Final Frame", final_frame)
    cv2.waitKey(0)


def getBackgroundModel(file):
    print("proses pencarian bg model berjalan")
    cap = cv2.VideoCapture(file)
    list_img = []
    gray = []

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            list_img.append(frame)
        else:
            break

    list_img = np.array(list_img, dtype=int)
    gray = np.array(gray, dtype=int)
    new_img = np.zeros((list_img.shape[1], list_img.shape[2], list_img.shape[3])).astype(np.uint8)
    for index, i in enumerate(new_img):
        for index_2, j in enumerate(i):
            new_img[index, index_2, 0] = np.bincount(list_img[:, index, index_2, 0]).argmax()
            new_img[index, index_2, 1] = np.bincount(list_img[:, index, index_2, 1]).argmax()
            new_img[index, index_2, 2] = np.bincount(list_img[:, index, index_2, 2]).argmax()

    background = new_img

    return background


def ccl(frame, output, thresh):
    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(thresh.shape, dtype="uint8")
    last = 0

    for i in range(1, numLabels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area >= 350 and w >= 5:
            last += 1
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, str(last), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    # cv2.imshow("frame result", list_img)
    # cv2.waitKey(0)

    return last, frame


def hasil(akumulasi, frames):
    print("proses perhitungan benih berjalan")

    modus = Counter(akumulasi)
    modus = modus.most_common()[0][0]

    for i in range(len(akumulasi)):
        if akumulasi[i] == modus:
            frame_result = frames[i]

    return modus, frame_result


if __name__ == "__main__":
    preprocessing("Videos/5.mp4")
