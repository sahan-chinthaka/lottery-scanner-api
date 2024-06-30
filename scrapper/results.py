import requests
from bs4 import BeautifulSoup


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
