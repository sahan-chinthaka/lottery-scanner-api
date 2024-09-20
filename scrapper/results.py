import requests
from bs4 import BeautifulSoup
import re

cookie = "1010"


def update_cookie():
    global cookie
    res = requests.get("https://www.nlb.lk/English/results")

    mt = re.search(r"\'([a-f0-9]+)\'", res.text)

    cookie = mt.group(1)


def get_nlb_results():
    global cookie
    results = []

    try:
        res = requests.get(
            "https://www.nlb.lk/English/results",
            headers={
                "Cookie": f"human={cookie}",
            },
        )

        if len(res.text) < 1000 and "setCookie" in res.text:
            update_cookie()
            return get_nlb_results()

        soup = BeautifulSoup(res.text, "html.parser")
        lboxes = soup.find_all("li", attrs={"class": "lbox"})

        for lbox in lboxes:
            w0 = lbox.find("div", attrs={"class": "w0"})
            spans = w0.find_all("span")

            name = spans[0].text
            draw = spans[1].text
            date = spans[2].text

            cur = {
                "name": name,
                "draw": draw,
                "date": date,
                "results": [],
                "type": "nlb",
            }

            Bs = w0.find_all("ol", attrs={"class": "B"})

            for B in Bs:
                lis = B.find_all("li")
                for li in lis:
                    cur["results"].append(li.text)

            results.append(cur)
    except Exception:
        pass

    return results


def get_dlb_results():
    results = []
    try:
        res = requests.get("https://www.dlb.lk/home/en")

        soup = BeautifulSoup(res.text, "html.parser")
        result_box = soup.find("div", attrs={"class": "col-md-12 latest_resultbox"})
        boxes = result_box.findChildren("div", recursive=False)

        for box in boxes:
            lottery_n_d = box.find("p", attrs={"class": "lottery_n_d"}).text

            pattern = re.compile(r"\s+")
            lottery_n_d = re.sub(pattern, " ", lottery_n_d)

            name_d, date = lottery_n_d.split(" | ")
            name, draw = name_d.split(" - ")

            cur = {
                "name": name,
                "draw": draw,
                "date": date,
                "results": [],
                "type": "dlb",
            }

            lis = box.find_all("li")

            for li in lis:
                txt = li.text.strip()
                if txt == "":
                    img = li.find("img")
                    txt = img["src"][:-4].split("/")[-1]
                    p = re.compile(r"\d+")
                    txt = re.sub(p, "", txt)

                cur["results"].append(txt)

            results.append(cur)

    except Exception:
        pass

    return results


def get_govisetha(draw: str):
    res = requests.get(
        f"https://www.nlb.lk/English/results/govisetha/{draw}",
        headers={
            "Cookie": "human=1010",
        },
    )

    soup = BeautifulSoup(res.text, "html5lib")
    lresult = soup.find("div", attrs={"class": "lresult"})

    if lresult is None:
        return None

    ps = lresult.find_all("p")
    draw_no: str = ps[0].text

    if "Draw No.: " in draw_no:
        draw_no = draw_no.replace("Draw No.: ", "")

    date: str = ps[1].text

    if "Date: " in date:
        date = date.replace("Date: ", "")

    Bs = lresult.find_all("ol", attrs={"class": "B"})

    numbers = []

    for li in Bs[0].find_all("li")[:-1]:
        numbers.append(li.text)

    for li in Bs[1].find_all("li")[:-1]:
        numbers.append(li.text)

    return draw_no, date, numbers
