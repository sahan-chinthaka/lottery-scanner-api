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


def extract_mahajana_sampatha():
    area_draw = [320, 630], [470, 740]
    area_date = [50, 630], [300, 730]
    area_numb = [30, 505], [430, 630]

    draw_data = []
    date_data = []
    numb_data = []

    imgC = cv2.imread("./../model/data/model-3/mahajana-sampatha/527.jpg")
    imgC = cv2.resize(imgC, (w_mahajana, h_mahajana))
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

    # imgScan = cv2.rectangle(imgScan, *area_date, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_draw, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_numb, (255, 0, 0), 5)

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

    numbers = " ".join(list(map(lambda x: x[1], numb_data)))
    draw_number = "".join(list(map(lambda x: x[1], draw_data)))
    date = "".join(list(map(lambda x: x[1], date_data)))

    print(numbers.split(" "))
    print(draw_number)
    print(date)

    cv2.imshow("k3", imgScan)

    cv2.waitKey(0)


def extract_govisetha():
    area_draw = [105, 590], [165, 635]
    area_date = [105, 555], [245, 600]
    area_numb = [50, 400], [400, 455]
    area_spec = [330, 560], [395, 625]
    area_doub = [200, 500], [350, 555]

    draw_data = []
    date_data = []
    numb_data = []
    spec_data = []
    doub_data = []

    imgC = cv2.imread("./../model/data/model-3/govisetha/1000.jpg")
    imgC = cv2.resize(imgC, (w_govisetha, h_govisetha))
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

    # imgScan = cv2.rectangle(imgScan, *area_date, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_draw, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_numb, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_spec, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_doub, (255, 0, 0), 5)

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

    numb_data = sorted(numb_data, key=lambda p: p[0][0])
    date_data = sorted(date_data, key=lambda p: p[0][0])
    draw_data = sorted(draw_data, key=lambda p: p[0][0])
    spec_data = sorted(spec_data, key=lambda p: p[0][0])
    doub_data = sorted(doub_data, key=lambda p: p[0][0])

    numbers = " ".join(list(map(lambda x: x[1], numb_data)))
    draw_number = "".join(list(map(lambda x: x[1], draw_data)))
    date = "".join(list(map(lambda x: x[1], date_data)))
    special = "".join(list(map(lambda x: x[1], spec_data)))
    double = "".join(list(map(lambda x: x[1], doub_data)))

    print(numbers.split(" "))
    print(draw_number)
    print(date)
    print(special)
    print(list(double))

    cv2.imshow("k3", imgScan)

    cv2.waitKey(0)


def extract_koti_kapruka():
    area_draw = [340, 560], [400, 600]
    area_date = [95, 565], [240, 605]
    area_numb = [30, 470], [410, 525]
    area_doub = [235, 395], [400, 445]

    draw_data = []
    date_data = []
    numb_data = []
    doub_data = []

    imgC = cv2.imread("./../model/data/model-3/koti-kapruka/22.jpg")
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

    # imgScan = cv2.rectangle(imgScan, *area_date, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_draw, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_numb, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_spec, (255, 0, 0), 5)
    # imgScan = cv2.rectangle(imgScan, *area_doub, (255, 0, 0), 5)

    result = reader.readtext(imgScan)

    for bbox, text, _ in result:
        top_left = bbox[0]
        bottom_right = bbox[2]

        print(top_left, bottom_right, text)

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

    numbers = " ".join(list(map(lambda x: x[1], numb_data)))
    draw_number = "".join(list(map(lambda x: x[1], draw_data)))
    date = "".join(list(map(lambda x: x[1], date_data)))
    double = "".join(list(map(lambda x: x[1], doub_data)))

    print(numbers.split(" "))
    print(draw_number)
    print(date)
    print(list(double))

    cv2.imshow("k3", imgScan)

    cv2.waitKey(0)
