import os
from flask import Flask, render_template, request

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    image_filename = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)
            image_filename = file.filename

    return render_template("index.html", image_filename=image_filename)


if __name__ == "__main__":
    app.run(debug=True)