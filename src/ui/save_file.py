import os
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, redirect, url_for, render_template, Blueprint
from werkzeug.utils import secure_filename
from utils import label_map_util 

from animal_info import *  # Custom module for animal-related info
from user_info import user_details  # Custom module for user-related info

# Ensure compatibility with TensorFlow v1
tf.compat.v1.disable_eager_execution()

# Model and configuration paths
MODEL_NAME = '../object_detection/model'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('../object_detection/data', 'label.pbtxt')
NUM_CLASSES = 8

# Flask app configuration
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '../object_detection/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

save_blueprint = Blueprint("save_blueprint", __name__)

# Import TensorFlow Object Detection API utilities
try:
    from object_detection.utils import label_map_util
    from object_detection.utils import visualization_utils as vis_util
except ImportError:
    print("Using custom visualization as TensorFlow Object Detection API is not installed.")
    # Custom label map and visualization utilities
     

    def visualize_boxes_on_image(image_np, boxes, classes, scores, category_index, threshold=0.5):
        """
        Custom visualization function for detected objects.
        """
        import cv2
        for i, box in enumerate(boxes):
            if scores[i] > threshold:
                ymin, xmin, ymax, xmax = box
                label = category_index[classes[i]]['name']
                score = scores[i]
                (im_height, im_width, _) = image_np.shape
                (xmin, xmax, ymin, ymax) = (xmin * im_width, xmax * im_width, ymin * im_height, ymax * im_height)
                cv2.rectangle(image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 2)
                cv2.putText(image_np, f"{label}: {score:.2f}", (int(xmin), int(ymin) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image_np

# Load detection graph
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def allowed_file(filename):
    """
    Check if the uploaded file is allowed based on its extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def load_image_into_numpy_array(image):
    """
    Convert an image to a numpy array.
    """
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


@save_blueprint.route('/upload', methods=['POST'])
def upload():
    """
    Handle file uploads and save to the configured directory.
    """
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('save_blueprint.uploaded_file', filename=filename))


@save_blueprint.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Process the uploaded file and run object detection.
    """
    global animal_name
    PATH_TO_TEST_IMAGES_DIR = app.config['UPLOAD_FOLDER']
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, filename)]

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                # Get tensors
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                # Run detection
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualize results
                if 'visualize_boxes_and_labels_on_image_array' in dir(vis_util):
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=8)
                else:
                    visualize_boxes_on_image(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index
                    )

                im = Image.fromarray(image_np)
                filepath = os.path.join('../ui/static/uploads/', filename)
                im.save(filepath)

                mylabel = [category_index.get(value) for index, value in enumerate(classes[0]) if scores[0, index] > 0.5]
                mapped_name = mylabel[0].get('name') if mylabel else "Unknown"
                animal_name = getMappedName(mapped_name)

    animal_info = get_all_details_for(animal_name)
    return render_template("index.html", name=animal_name, mapped_name=mapped_name,
                           users=user_details(), loaded=True, info=animal_info,
                           file_path='../static/uploads/' + filename)


# Register blueprint
app.register_blueprint(save_blueprint)

if __name__ == '__main__':
    app.run(debug=True)
