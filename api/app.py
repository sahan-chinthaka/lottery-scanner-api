from flask import Flask, jsonify, request
from scrapper import get_dlb_results, get_nlb_results
from ml import get_lottery_type
from ml import extract_koti_kapruka, extract_govisetha, extract_mahajana_sampatha
from PIL import Image
import io

app = Flask(__name__)


@app.route("/")
def index():
    return "running"


@app.route("/latest-results")
def getLatestResults():
    dlb = get_dlb_results()
    nlb = get_nlb_results()

    return jsonify(nlb + dlb)


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
