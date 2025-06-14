from langchain_core.tools import tool 
import os
import io
import base64
import uuid
from PIL import Image
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter


# Helper functions for image processing
def encode_image(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def decode_image(base64_string: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def save_image(image: Image.Image, directory: str = "image_outputs") -> str:
    """Save a PIL Image to disk and return the path."""
    os.makedirs(directory, exist_ok=True)
    image_id = str(uuid.uuid4())
    image_path = os.path.join(directory, f"{image_id}.png")
    image.save(image_path)
    return image_path 


@tool
def analyze_image(image_base64: str) -> Dict[str, Any]:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview).
    Args:
        image_base64 (str): Base64 encoded image string
    Returns:
        Dictionary with analysis result
    """
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode

        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            avg_colors = arr.mean(axis=(0, 1))
            dominant = ["Red", "Green", "Blue"][np.argmax(avg_colors[:3])]
            brightness = avg_colors.mean()
            color_analysis = {
                "average_rgb": avg_colors.tolist(),
                "brightness": brightness,
                "dominant_color": dominant,
            }
        else:
            color_analysis = {"note": f"No color analysis for mode {mode}"}

        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)

        return {
            "dimensions": (width, height),
            "mode": mode,
            "color_analysis": color_analysis,
            "thumbnail": thumbnail_base64,
        }
    except Exception as e:
        return {"error": str(e)}



@tool
def transform_image(image_base64: str, 
                    operation: str, 
                    params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Apply transformations: resize, rotate, crop, flip, brightness, contrast, blur, sharpen, grayscale.
    Args:
        image_base64 (str): Base64 encoded input image
        operation (str): Transformation operation
        params (Dict[str, Any], optional): Parameters for the operation
    Returns:
        Dictionary with transformed image (base64)
    """
    try:
        img = decode_image(image_base64)
        params = params or {}

        if operation == "resize":
            img = img.resize(
                (
                    params.get("width", img.width // 2),
                    params.get("height", img.height // 2),
                )
            )
        elif operation == "rotate":
            img = img.rotate(params.get("angle", 90), expand=True)
        elif operation == "crop":
            img = img.crop(
                (
                    params.get("left", 0),
                    params.get("top", 0),
                    params.get("right", img.width),
                    params.get("bottom", img.height),
                )
            )
        elif operation == "flip":
            if params.get("direction", "horizontal") == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif operation == "adjust_brightness":
            img = ImageEnhance.Brightness(img).enhance(params.get("factor", 1.5))
        elif operation == "adjust_contrast":
            img = ImageEnhance.Contrast(img).enhance(params.get("factor", 1.5))
        elif operation == "blur":
            img = img.filter(ImageFilter.GaussianBlur(params.get("radius", 2)))
        elif operation == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)
        elif operation == "grayscale":
            img = img.convert("L")
        else:
            return {"error": f"Unknown operation: {operation}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"transformed_image": result_base64}

    except Exception as e:
        return {"error": str(e)}



@tool
def draw_on_image(
    image_base64: str, drawing_type: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Draw shapes (rectangle, circle, line) or text onto an image.(bounding box/anntation)
    Args:
        image_base64 (str): Base64 encoded input image
        drawing_type (str): Drawing type
        params (Dict[str, Any]): Drawing parameters
    Returns:
        Dictionary with result image (base64)
    """
    try:
        img = decode_image(image_base64)
        draw = ImageDraw.Draw(img)
        color = params.get("color", "red")

        if drawing_type == "rectangle":
            draw.rectangle(
                [params["left"], params["top"], params["right"], params["bottom"]],
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "circle":
            x, y, r = params["x"], params["y"], params["radius"]
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                outline=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "line":
            draw.line(
                (
                    params["start_x"],
                    params["start_y"],
                    params["end_x"],
                    params["end_y"],
                ),
                fill=color,
                width=params.get("width", 2),
            )
        elif drawing_type == "text":
            font_size = params.get("font_size", 20)
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                font = ImageFont.load_default()
            draw.text(
                (params["x"], params["y"]),
                params.get("text", "Text"),
                fill=color,
                font=font,
            )
        else:
            return {"error": f"Unknown drawing type: {drawing_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"result_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def generate_simple_image(image_type: str,
                        width: int = 500,
                        height: int = 500,
                        params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate a simple image (gradient, noise, pattern, chart).
    Args:
        image_type (str): Type of image
        width (int), height (int)
        params (Dict[str, Any], optional): Specific parameters
    Returns:
        Dictionary with generated image (base64)
    """
    try:
        params = params or {}

        if image_type == "gradient":
            direction = params.get("direction", "horizontal")
            start_color = params.get("start_color", (255, 0, 0))
            end_color = params.get("end_color", (0, 0, 255))

            img = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(img)

            if direction == "horizontal":
                for x in range(width):
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * x / width)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * x / width)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * x / width)                   
                    draw.line([(x, 0), (x, height)], fill=(r, g, b))
            else:
                for y in range(height):
                    r = int(start_color[0] + (end_color[0] - start_color[0]) * y / height)
                    g = int(start_color[1] + (end_color[1] - start_color[1]) * y / height)
                    b = int(start_color[2] + (end_color[2] - start_color[2]) * y / height)
                    draw.line([(0, y), (width, y)], fill=(r, g, b))

        elif image_type == "noise":
            noise_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            img = Image.fromarray(noise_array, "RGB")

        else:
            return {"error": f"Unsupported image_type {image_type}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"generated_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


@tool
def combine_images(images_base64: List[str], 
                operation: str, 
                params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Combine multiple images (collage, stack, blend).
    Args:
        images_base64 (List[str]): List of base64 images
        operation (str): Combination type
        params (Dict[str, Any], optional)
    Returns:
        Dictionary with combined image (base64)
    """
    try:
        images = [decode_image(b64) for b64 in images_base64]
        params = params or {}

        if operation == "stack":
            direction = params.get("direction", "horizontal")
            if direction == "horizontal":
                total_width = sum(img.width for img in images)
                max_height = max(img.height for img in images)
                new_img = Image.new("RGB", (total_width, max_height))
                x = 0
                for img in images:
                    new_img.paste(img, (x, 0))
                    x += img.width
            else:
                max_width = max(img.width for img in images)
                total_height = sum(img.height for img in images)
                new_img = Image.new("RGB", (max_width, total_height))
                y = 0
                for img in images:
                    new_img.paste(img, (0, y))
                    y += img.height
        else:
            return {"error": f"Unsupported combination operation {operation}"}

        result_path = save_image(new_img)
        result_base64 = encode_image(result_path)
        return {"combined_image": result_base64}

    except Exception as e:
        return {"error": str(e)}
