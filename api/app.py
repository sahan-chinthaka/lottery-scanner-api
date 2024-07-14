import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("./api/smart-lottery-adminsdk.json")
firebase_admin.initialize_app(cred,)

store = firestore.client()
col_ref = store.collection("users")


def main():
    doc_ref = col_ref.document()
    doc_ref.set(
        {
            "name": "Test",
        },
    )


if __name__ == "__main__":
    main()
