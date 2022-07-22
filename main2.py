import numpy as np
import cv2
from collections import Counter
from werkzeug.utils import secure_filename
import statistics


def preprocessing(file):
    video = cv2.VideoCapture(file)
    # background_model = getBackgroundModel(file)
    # background_model = cv2.resize(background_model, (500, 500))

    akumulasi = []

    while video.isOpened:
        ret, frame = video.read()

        if not ret:
            break

        frame = cv2.resize(frame, (500, 500))
        rgb_image = cv2.split(frame)
        result_image = []
        result_norm_image = []
        for image in rgb_image:
            kernel_erosi = np.ones((5, 5), np.uint8)
            frame_dilasi = cv2.dilate(image, kernel_erosi)
            frame_median = cv2.medianBlur(frame_dilasi, 21)

            cv2.imshow("frame median", frame_median)
            cv2.waitKey(0)

            frame_diff = 255 - cv2.absdiff(image, frame_median)

            cv2.imshow("frame diff", frame_diff)
            cv2.waitKey(0)

            frame_normalisasi = cv2.normalize(frame_diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
                                              , dtype=cv2.CV_8UC1)

            cv2.imshow("frame norm", frame_normalisasi)
            cv2.waitKey(0)

            result_image.append(frame_diff)
            result_norm_image.append(frame_normalisasi)

        result = cv2.merge(result_image)
        result_norm = cv2.merge(result_norm_image)
        frame_final = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(frame_final, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

        cv2.imshow("frame thresh", thresh)
        cv2.waitKey(0)

        output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)

        akumulasi.append(ccl(frame, output, thresh))

    hasil(akumulasi, 5)


def getBackgroundModel(file):
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

        if area >= 450 and w >= 15:
            last += 1
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, str(last), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("frame result", frame)
    cv2.waitKey(0)

    return last


def hasil(akumulasi, jumlah_asli):
    modus = Counter(akumulasi)
    modus = modus.most_common(1)[0][0]

    em = abs(jumlah_asli - modus)
    er = (em / jumlah_asli) * 100
    akurasi = 100 - er

    print(f'jumlah lele yang terdeteksi: {modus}')
    print(f'jumlah lele asli: {jumlah_asli}')
    print(f'akurasi: {akurasi}')


if __name__ == "__main__":
    preprocessing("Videos/1.mp4")