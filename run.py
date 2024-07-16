from api.app import main
import os

d = True if os.environ.get("debug") == "1" else False

main(d)
