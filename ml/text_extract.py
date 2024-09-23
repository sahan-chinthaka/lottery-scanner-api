import cv2
import numpy as np
import easyocr


imgQ_mahajana = cv2.imread("./ml/references/mahajana-sampatha.jpg")
imgQ_mahajana = cv2.cvtColor(imgQ_mahajana, cv2.COLOR_BGR2GRAY)
h_mahajana, w_mahajana = imgQ_mahajana.shape

orb_mahajana = cv2.ORB_create(10000)
kp1_mahajana, des1_mahajana = orb_mahajana.detectAndCompute(imgQ_mahajana, None)

imgQ_govisetha = cv2.imread("./ml/references/govisetha.jpg")
imgQ_govisetha = cv2.cvtColor(imgQ_govisetha, cv2.COLOR_BGR2GRAY)
h_govisetha, w_govisetha = imgQ_govisetha.shape

orb_govisetha = cv2.ORB_create(10000)
kp1_govisetha, des1_govisetha = orb_govisetha.detectAndCompute(imgQ_govisetha, None)

imgQ_koti_kapruka = cv2.imread("./ml/references/koti-kapruka.jpg")
imgQ_koti_kapruka = cv2.cvtColor(imgQ_koti_kapruka, cv2.COLOR_BGR2GRAY)
h_koti_kapruka, w_koti_kapruka = imgQ_koti_kapruka.shape

orb_koti_kapruka = cv2.ORB_create(10000)
kp1_koti_kapruka, des1_koti_kapruka = orb_koti_kapruka.detectAndCompute(
    imgQ_koti_kapruka, None
)

reader = easyocr.Reader(["en"], gpu=True)


def extract_mahajana_sampatha(image):
    area_draw = [320, 630], [470, 740]
    area_date = [50, 630], [300, 730]
    area_numb = [30, 505], [430, 630]

    draw_data = []
    date_data = []
    numb_data = []

    imgC = np.array(image)
    imgC = imgC[:, :, ::-1].copy()

    img = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb_mahajana.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1_mahajana)

    good = sorted(matches, key=lambda x: x.distance)[:100]

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1_mahajana[m.trainIdx].pt for m in good]).reshape(
        -1, 1, 2
    )

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(imgC, M, (w_mahajana, h_mahajana))
    result = reader.readtext(imgScan)

    for bbox, text, _ in result:
        top_left = bbox[0]
        bottom_right = bbox[2]

        try:
            imgScan = cv2.rectangle(imgScan, top_left, bottom_right, (255, 255, 255), 5)
            imgScan = cv2.putText(
                imgScan, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

            if (
                area_draw[0][0] <= top_left[0]
                and area_draw[0][1] <= top_left[1]
                and area_draw[1][0] >= bottom_right[0]
                and area_draw[1][1] >= bottom_right[1]
            ):
                draw_data.append([top_left, text])

            if (
                area_date[0][0] <= top_left[0]
                and area_date[0][1] <= top_left[1]
                and area_date[1][0] >= bottom_right[0]
                and area_date[1][1] >= bottom_right[1]
            ):
                date_data.append([top_left, text])

            if (
                area_numb[0][0] <= top_left[0]
                and area_numb[0][1] <= top_left[1]
                and area_numb[1][0] >= bottom_right[0]
                and area_numb[1][1] >= bottom_right[1]
            ):
                numb_data.append([top_left, text])
        except Exception:
            pass

    numb_data = sorted(numb_data, key=lambda p: p[0][0])
    date_data = sorted(date_data, key=lambda p: p[0][0])
    draw_data = sorted(draw_data, key=lambda p: p[0][0])

    return (
        list(map(lambda x: x[1], numb_data)),
        list(map(lambda x: x[1], draw_data)),
        list(map(lambda x: x[1], date_data)),
    )


def extract_govisetha(image):
    area_draw = [105, 590], [165, 635]
    area_date = [105, 555], [245, 600]
    area_numb = [20, 400], [400, 455]
    area_spec = [320, 550], [400, 630]
    area_doub = [200, 500], [350, 555]

    draw_data = []
    date_data = []
    numb_data = []
    spec_data = []
    doub_data = []

    imgC = np.array(image)
    imgC = imgC[:, :, ::-1].copy()

    img = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb_govisetha.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1_govisetha)

    good = sorted(matches, key=lambda x: x.distance)[:100]

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1_govisetha[m.trainIdx].pt for m in good]).reshape(
        -1, 1, 2
    )

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    imgScan = cv2.warpPerspective(imgC, M, (w_govisetha, h_govisetha))
    result = reader.readtext(imgScan)

    for bbox, text, _ in result:
        top_left = bbox[0]
        bottom_right = bbox[2]

        try:
            imgScan = cv2.rectangle(imgScan, top_left, bottom_right, (255, 255, 255), 5)
            imgScan = cv2.putText(
                imgScan, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

            if (
                area_draw[0][0] <= top_left[0]
                and area_draw[0][1] <= top_left[1]
                and area_draw[1][0] >= bottom_right[0]
                and area_draw[1][1] >= bottom_right[1]
            ):
                draw_data.append([top_left, text])

            if (
                area_date[0][0] <= top_left[0]
                and area_date[0][1] <= top_left[1]
                and area_date[1][0] >= bottom_right[0]
                and area_date[1][1] >= bottom_right[1]
            ):
                date_data.append([top_left, text])

            if (
                area_numb[0][0] <= top_left[0]
                and area_numb[0][1] <= top_left[1]
                and area_numb[1][0] >= bottom_right[0]
                and area_numb[1][1] >= bottom_right[1]
            ):
                numb_data.append([top_left, text])

            if (
                area_spec[0][0] <= top_left[0]
                and area_spec[0][1] <= top_left[1]
                and area_spec[1][0] >= bottom_right[0]
                and area_spec[1][1] >= bottom_right[1]
            ):
                spec_data.append([top_left, text])

            if (
                area_doub[0][0] <= top_left[0]
                and area_doub[0][1] <= top_left[1]
                and area_doub[1][0] >= bottom_right[0]
                and area_doub[1][1] >= bottom_right[1]
            ):
                doub_data.append([top_left, text])
        except Exception:
            pass

    numb_data = list(sorted(numb_data, key=lambda p: p[0][0]))
    date_data = list(sorted(date_data, key=lambda p: p[0][0]))
    draw_data = list(sorted(draw_data, key=lambda p: p[0][0]))
    spec_data = list(sorted(spec_data, key=lambda p: p[0][0]))
    doub_data = list(sorted(doub_data, key=lambda p: p[0][0]))

    return (
        list(map(lambda x: x[1], numb_data)),
        list(map(lambda x: x[1], draw_data)),
        list(map(lambda x: x[1], date_data)),
        list(map(lambda x: x[1], spec_data)),
        list(map(lambda x: x[1], doub_data)),
    )


def extract_koti_kapruka(image):
    area_draw = [320, 560], [400, 600]
    area_date = [95, 550], [250, 605]
    area_numb = [5, 470], [410, 535]
    area_doub = [235, 395], [400, 445]

    draw_data = []
    date_data = []
    numb_data = []
    doub_data = []

    imgC = np.array(image)
    imgC = imgC[:, :, ::-1].copy()

    imgC = cv2.resize(imgC, (w_koti_kapruka, h_koti_kapruka))
    img = cv2.cvtColor(imgC, cv2.COLOR_BGR2GRAY)

    kp2, des2 = orb_koti_kapruka.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1_koti_kapruka)

    good = sorted(matches, key=lambda x: x.distance)[:100]

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1_koti_kapruka[m.trainIdx].pt for m in good]).reshape(
        -1, 1, 2
    )

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

    imgScan = cv2.warpPerspective(imgC, M, (w_koti_kapruka, h_koti_kapruka))
    result = reader.readtext(imgScan)

    for bbox, text, _ in result:
        top_left = bbox[0]
        bottom_right = bbox[2]

        try:
            imgScan = cv2.rectangle(imgScan, top_left, bottom_right, (255, 255, 255), 5)
            imgScan = cv2.putText(
                imgScan, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

            if (
                area_draw[0][0] <= top_left[0]
                and area_draw[0][1] <= top_left[1]
                and area_draw[1][0] >= bottom_right[0]
                and area_draw[1][1] >= bottom_right[1]
            ):
                draw_data.append([top_left, text])

            if (
                area_date[0][0] <= top_left[0]
                and area_date[0][1] <= top_left[1]
                and area_date[1][0] >= bottom_right[0]
                and area_date[1][1] >= bottom_right[1]
            ):
                date_data.append([top_left, text])

            if (
                area_numb[0][0] <= top_left[0]
                and area_numb[0][1] <= top_left[1]
                and area_numb[1][0] >= bottom_right[0]
                and area_numb[1][1] >= bottom_right[1]
            ):
                numb_data.append([top_left, text])

            if (
                area_doub[0][0] <= top_left[0]
                and area_doub[0][1] <= top_left[1]
                and area_doub[1][0] >= bottom_right[0]
                and area_doub[1][1] >= bottom_right[1]
            ):
                doub_data.append([top_left, text])
        except Exception:
            pass

    numb_data = sorted(numb_data, key=lambda p: p[0][0])
    date_data = sorted(date_data, key=lambda p: p[0][0])
    draw_data = sorted(draw_data, key=lambda p: p[0][0])
    doub_data = sorted(doub_data, key=lambda p: p[0][0])

    return (
        list(map(lambda x: x[1], numb_data)),
        list(map(lambda x: x[1], draw_data)),
        list(map(lambda x: x[1], date_data)),
        list(map(lambda x: x[1], doub_data)),
    )
