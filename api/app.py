from flask import Flask, jsonify
from scrapper import get_dlb_results, get_nlb_results

app = Flask(__name__)


@app.route("/latest-results")
def getLatestResults():
    dlb = get_dlb_results()
    nlb = get_nlb_results()

    return jsonify(nlb + dlb)


if __name__ == "__main__":
    app.run(debug=True)
