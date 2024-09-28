import io
from datetime import datetime, timedelta, timezone

from flask import Flask, jsonify, request
from PIL import Image

from ml import (
    extract_govisetha,
    extract_koti_kapruka,
    extract_mahajana_sampatha,
    get_lottery_type,
)
from scrapper import get_dlb_results, get_nlb_results

from .firebase import db
from .util import upload_results

app = Flask(__name__)


@app.route("/")
def index():
    return "running"


@app.route("/latest-results")
def getLatestResults():
    today = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    day_before_yesterday = today - timedelta(days=2)

    data_ref = db.collection("data")
    query = data_ref.where("date", ">=", day_before_yesterday)

    result = query.stream()

    final = []
    for doc in result:
        final.append(doc.to_dict())

    print(f"Result {len(final)}")

    if len(final) == 0:
        print("Scrapping results")
        dlb = get_dlb_results()
        nlb = get_nlb_results()
        all_res = dlb + nlb
        upload_results(all_res)
        return jsonify(all_res)

    return jsonify(final)


@app.post("/scan-image")
def scanImage():
    image = request.files["file"]

    img = Image.open(io.BytesIO(image.read()))
    lottery_type = get_lottery_type(img)

    res = None

    if lottery_type == "Koti Kapruka":
        res = extract_koti_kapruka(img)
    elif lottery_type == "Govisetha":
        res = extract_govisetha(img)
    elif lottery_type == "Mahajana Sampatha":
        res = extract_mahajana_sampatha(img)

    return jsonify(
        {
            "file": image.filename,
            "type": lottery_type,
            "data": res,
        }
    )


def main(d=True):
    app.run(debug=d)


if __name__ == "__main__":
    main()
