import collections
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
import io
import tensorflow as tf

# Constants
_TITLE_LEFT_MARGIN = 10
_TITLE_TOP_MARGIN = 10
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige',
    'Bisque', 'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue',
    'AntiqueWhite', 'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk',
    'Crimson', 'Cyan', 'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki',
    'DarkOrange', 'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise',
    'DarkViolet', 'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick',
    'FloralWhite', 'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite',
    'Gold', 'GoldenRod', 'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed',
    'Ivory', 'Khaki', 'Lavender', 'LavenderBlush', 'LawnGreen',
    'LemonChiffon', 'LightBlue', 'LightCoral', 'LightCyan',
    'LightGoldenRodYellow', 'LightGray', 'LightGrey', 'LightGreen',
    'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow',
    'Lime', 'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine',
    'MediumOrchid', 'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue',
    'MediumSpringGreen', 'MediumTurquoise', 'MediumVioletRed', 'MintCream',
    'MistyRose', 'Moccasin', 'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab',
    'Orange', 'OrangeRed', 'Orchid', 'PaleGoldenRod', 'PaleGreen',
    'PaleTurquoise', 'PaleVioletRed', 'PapayaWhip', 'PeachPuff', 'Peru',
    'Pink', 'Plum', 'PowderBlue', 'Purple', 'Red', 'RosyBrown', 'RoyalBlue',
    'SaddleBrown', 'Green', 'SandyBrown', 'SeaGreen', 'SeaShell', 'Sienna',
    'Silver', 'SkyBlue', 'SlateBlue', 'SlateGray', 'SlateGrey', 'Snow',
    'SpringGreen', 'SteelBlue', 'GreenYellow', 'Teal', 'Thistle', 'Tomato',
    'Turquoise', 'Violet', 'Wheat', 'White', 'WhiteSmoke', 'Yellow',
    'YellowGreen'
]

# Functions
def save_image_array_as_png(image, output_path):
    """Saves a numpy image array as a PNG file."""
    image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
    with tf.io.gfile.GFile(output_path, 'wb') as fid:
        image_pil.save(fid, 'PNG')

def encode_image_array_as_png_str(image):
    """Encodes a numpy image array into a PNG string."""
    image_pil = Image.fromarray(np.uint8(image))
    output = io.BytesIO()
    image_pil.save(output, format='PNG')
    return output.getvalue()

def draw_bounding_box_on_image(image, ymin, xmin, ymax, xmax, color='red',
                               thickness=4, display_str_list=(),
                               use_normalized_coordinates=True):
    """Draws a bounding box on an image."""
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (
            xmin * im_width, xmax * im_width,
            ymin * im_height, ymax * im_height
        )
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)

    draw.line(
        [(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
        width=thickness,
        fill=color
    )

    # Attempt to load a custom font, fall back to default.
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    text_bottom = top
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getbbox(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin),
             (left + text_width, text_bottom)],
            fill=color
        )
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font
        )
        text_bottom -= text_height - 2 * margin
