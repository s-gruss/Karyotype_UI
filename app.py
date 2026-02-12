import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "change-me"

UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Detectron2 imports are optional and lazy-loaded ---
_predictor = None
_cfg = None
_MODEL_WEIGHTS = os.environ.get("DETECTRON2_WEIGHTS")  # set this to the model path or leave None


def load_predictor(weights_path=None, device="cpu"):
    global _predictor, _cfg
    if _predictor is not None:
        return _predictor
    try:
        import cv2
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
    except Exception as e:
        raise RuntimeError(
            "Detectron2 (and opencv) must be installed to run segmentation. Install manually per your CUDA/PyTorch setup."
        ) from e

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

    cfg.INPUT.MIN_SIZE_TEST = 1600
    cfg.INPUT.MAX_SIZE_TEST = 1600
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    if weights_path:
        cfg.MODEL.WEIGHTS = weights_path
    elif _MODEL_WEIGHTS:
        cfg.MODEL.WEIGHTS = _MODEL_WEIGHTS

    cfg.MODEL.DEVICE = device
    _cfg = cfg
    _predictor = DefaultPredictor(cfg)
    return _predictor


@app.route("/", methods=["GET", "POST"])
def index():
    image_filename = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)
            image_filename = filename

    return render_template("index.html", image_filename=image_filename)


@app.route("/segment", methods=["POST"])
def segment():
    image_filename = request.form.get("image_filename")
    if not image_filename:
        flash("No image selected for segmentation.")
        return redirect(url_for("index"))

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(image_filename))
    if not os.path.exists(image_path):
        flash("Uploaded image not found.")
        return redirect(url_for("index"))

    # Load predictor (may raise RuntimeError if detectron2 not installed)
    try:
        predictor = load_predictor()  # uses DETECTRON2_WEIGHTS env var if set
    except RuntimeError as e:
        flash(str(e))
        return redirect(url_for("index"))

    # Run prediction and save visualization
    try:
        import cv2
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        import numpy as np
    except Exception:
        flash("Required visualization libraries (opencv, detectron2) not found.")
        return redirect(url_for("index"))

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        flash("Could not read the image file.")
        return redirect(url_for("index"))

    img_rgb = img_bgr[:, :, ::-1]
    outputs = predictor(img_rgb)

    # Try to get metadata if registered; fallback to empty
    try:
        metadata = MetadataCatalog.get("autokary_test")
    except Exception:
        metadata = None

    v = Visualizer(img_rgb, metadata=metadata, scale=1.0)
    pred_vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    vis_img = pred_vis.get_image()  # RGB
    vis_bgr = vis_img[:, :, ::-1]

    seg_filename = f"seg_{image_filename}"
    seg_path = os.path.join(app.config["UPLOAD_FOLDER"], seg_filename)
    cv2.imwrite(seg_path, vis_bgr)

    return render_template("index.html", image_filename=image_filename, seg_image_filename=seg_filename)


if __name__ == "__main__":
    app.run(debug=True)