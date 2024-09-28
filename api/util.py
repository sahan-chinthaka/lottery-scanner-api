from .firebase import db


def upload_results(results):
    batch = db.batch()

    for item in results:
        doc_ref = db.collection("data").document()

        batch.set(doc_ref, item)
    batch.commit()
