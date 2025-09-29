from flask import Flask, request, render_template, send_file, jsonify
import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw
import zipfile
import io
import tempfile
import shutil
from werkzeug.utils import secure_filename
from scipy.ndimage import label

# When packaged with PyInstaller the application assets live in _MEIPASS.
if getattr(sys, 'frozen', False):
    BASE_PATH = sys._MEIPASS  # type: ignore[attr-defined]
    RUNTIME_PATH = os.path.dirname(sys.executable)
else:
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    RUNTIME_PATH = BASE_PATH

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_PATH, 'templates')
)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = os.path.join(RUNTIME_PATH, 'uploads')
app.config['PROCESSED_FOLDER'] = os.path.join(RUNTIME_PATH, 'processed')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_remove_people_icons(image_path, output_path):
    """
    Detect and remove people icons from the image.
    Uses multiple detection methods optimized for small gray people silhouettes.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return False

    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Convert to PIL for easier manipulation
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Focus on the rightmost area where people icons appear
    right_margin = int(width * 0.85)  # Look in the rightmost 15% of the image
    right_region = gray[:, right_margin:]

    icons_removed = 0

    # Method 1: Detect gray circular/oval shapes (people icon silhouettes)
    # Apply morphological operations to enhance small shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    right_processed = cv2.morphologyEx(right_region, cv2.MORPH_CLOSE, kernel)

    # Find contours in the right region
    contours, _ = cv2.findContours(right_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area typical for people icons (adjusted for mobile screenshots)
        if 50 < area < 1000:
            x, y, w, h = cv2.boundingRect(contour)
            actual_x = x + right_margin

            # Check aspect ratio and size constraints
            aspect_ratio = w / h if h > 0 else 0

            # People icons are usually roughly square or slightly taller
            if 0.6 <= aspect_ratio <= 1.4 and 10 <= w <= 40 and 10 <= h <= 40:
                # Check if it's in a likely position for a people icon
                # (not too close to edges, positioned like UI elements)
                if y > 10 and y < height - 10:
                    # Sample background color more carefully
                    bg_color = get_enhanced_background_color(img, actual_x, y, w, h)

                    # Remove the icon with a slightly larger area to ensure complete removal
                    padding = 3
                    draw.rectangle([
                        actual_x - padding,
                        y - padding,
                        actual_x + w + padding,
                        y + h + padding
                    ], fill=bg_color)
                    icons_removed += 1

    # Method 2: Template matching for common people icon patterns
    # Create templates for common people icon shapes
    templates = create_people_icon_templates()

    for template in templates:
        result = cv2.matchTemplate(right_region, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= 0.6)  # Adjust threshold as needed

        for pt in zip(*locations[::-1]):
            x, y = pt
            actual_x = x + right_margin
            th, tw = template.shape

            # Avoid duplicates by checking if we already processed this area
            if not area_already_processed(actual_x, y, tw, th, icons_removed):
                bg_color = get_enhanced_background_color(img, actual_x, y, tw, th)
                draw.rectangle([actual_x-2, y-2, actual_x+tw+2, y+th+2], fill=bg_color)
                icons_removed += 1

    # Method 3: Color-based detection for gray people silhouettes
    # Look for specific gray tones typical of people icons
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define range for gray colors (low saturation, medium value)
    lower_gray = np.array([0, 0, 60])
    upper_gray = np.array([180, 30, 140])

    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
    gray_mask_right = gray_mask[:, right_margin:]

    contours, _ = cv2.findContours(gray_mask_right, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 30 < area < 600:
            x, y, w, h = cv2.boundingRect(contour)
            actual_x = x + right_margin

            aspect_ratio = w / h if h > 0 else 0
            if 0.6 <= aspect_ratio <= 1.4 and 8 <= w <= 35 and 8 <= h <= 35:
                bg_color = get_enhanced_background_color(img, actual_x, y, w, h)
                draw.rectangle([actual_x-2, y-2, actual_x+w+2, y+h+2], fill=bg_color)
                icons_removed += 1

    # Save the processed image
    pil_img.save(output_path)
    return True

def create_people_icon_templates():
    """Create simple templates for people icon shapes"""
    templates = []

    # Template 1: Simple circular head with body
    template1 = np.zeros((20, 16), dtype=np.uint8)
    cv2.circle(template1, (8, 6), 4, 255, -1)  # Head
    cv2.rectangle(template1, (5, 10), (11, 18), 255, -1)  # Body
    templates.append(template1)

    # Template 2: Simple silhouette
    template2 = np.zeros((18, 14), dtype=np.uint8)
    cv2.ellipse(template2, (7, 5), (3, 4), 0, 0, 360, 255, -1)  # Head
    cv2.rectangle(template2, (4, 9), (10, 16), 255, -1)  # Body
    templates.append(template2)

    return templates

def area_already_processed(x, y, w, h, processed_count):
    """Simple check to avoid processing the same area multiple times"""
    # This is a simplified check - in a more sophisticated implementation,
    # you'd maintain a list of processed rectangles
    return False

def get_enhanced_background_color(img, x, y, w, h):
    """
    Enhanced background color sampling that looks at multiple areas around the icon.
    """
    height, width = img.shape[:2]

    # Sample from multiple areas around the icon
    sample_areas = []

    # Left side (most reliable for mobile UI backgrounds)
    if x - 20 >= 0:
        sample_areas.append((x - 15, y + h//2))

    # Above and below
    if y - 15 >= 0:
        sample_areas.append((x + w//2, y - 10))
    if y + h + 15 < height:
        sample_areas.append((x + w//2, y + h + 10))

    # Far left (background area)
    if x - 50 >= 0:
        sample_areas.append((x - 40, y + h//2))

    colors = []
    for sx, sy in sample_areas:
        if 0 <= sx < width and 0 <= sy < height:
            # Sample a small area around the point for better average
            sample_region = img[max(0, sy-2):min(height, sy+3), max(0, sx-2):min(width, sx+3)]
            if sample_region.size > 0:
                avg_color = np.mean(sample_region.reshape(-1, 3), axis=0)
                colors.append(tuple(avg_color[::-1].astype(int)))  # BGR to RGB

    if colors:
        # Return the median color to avoid outliers
        final_color = tuple(int(np.median([c[i] for c in colors])) for i in range(3))
        return final_color

    # Fallback: sample from the left side of the image (likely background)
    if width > 100:
        left_region = img[:, :width//4]
        avg_color = np.mean(left_region.reshape(-1, 3), axis=0)
        return tuple(avg_color[::-1].astype(int))

    # Final fallback to light gray
    return (240, 240, 240)

def get_background_color(img, x, y, radius):
    """
    Sample the background color around a given point, avoiding the icon area.
    """
    height, width = img.shape[:2]

    # Sample from areas around the icon
    sample_areas = [
        (max(0, x - radius - 20), y),  # Left of icon
        (x, max(0, y - radius - 20)),  # Above icon
        (x, min(height-1, y + radius + 20)),  # Below icon
    ]

    colors = []
    for sx, sy in sample_areas:
        if 0 <= sx < width and 0 <= sy < height:
            colors.append(tuple(img[sy, sx][::-1]))  # BGR to RGB

    if colors:
        # Return the most common color or average
        avg_color = tuple(int(np.mean([c[i] for c in colors])) for i in range(3))
        return avg_color

    # Default to white if sampling fails
    return (255, 255, 255)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files = request.files.getlist('files')
    processed_files = []
    debug_info = []

    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{filename}')
            debug_path = os.path.join(app.config['PROCESSED_FOLDER'], f'debug_{filename}')

            file.save(input_path)

            # Remove icons using adaptive detection and inpainting
            result = remove_icons_fixed_position(input_path, output_path, debug_path)
            if result['success']:
                processed_files.append(f'processed_{filename}')
                debug_info.append({
                    'filename': filename,
                    'icons_found': result['icons_found'],
                    'debug_image': f'debug_{filename}'
                })
            else:
                return jsonify({'error': f'Failed to process {filename}'}), 500

    return jsonify({
        'message': f'Successfully processed {len(processed_files)} files',
        'files': processed_files,
        'debug_info': debug_info
    })

def remove_icons_fixed_position(image_path, output_path, debug_path):
    """
    Detect people icons on the right edge and remove them via inpainting.
    The routine focuses on low-saturation grey shapes, validates their geometry,
    and inpaints the merged mask for natural-looking results.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'icons_found': 0}

    height, width = img.shape[:2]

    # Prepare debug canvas the same size as the original image
    debug_img = img.copy()

    # Search only within the right-most band where the icons live
    roi_start_x = int(width * 0.65)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Grey silhouettes have low saturation and mid-level value
    lower_gray = np.array([0, 0, 90], dtype=np.uint8)
    upper_gray = np.array([180, 70, 220], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_gray, upper_gray)
    mask[:, :roi_start_x] = 0  # discard everything outside the region of interest

    # Clean the mask to get solid blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    icon_regions = _collect_candidate_regions(mask, width, height)

    # Fallback: dynamic thresholding on grayscale if the colour heuristic failed
    if not icon_regions:
        icon_regions = _fallback_regions(img, roi_start_x, width, height)

    if not icon_regions:
        # Nothing to do; keep original image and emit debug overlay showing search band
        cv2.rectangle(debug_img, (roi_start_x, 0), (width - 1, height - 1), (0, 255, 255), 2)
        if debug_path:
            cv2.imwrite(debug_path, debug_img)
        cv2.imwrite(output_path, img)
        return {'success': True, 'icons_found': 0}

    # Build inpainting mask from merged regions
    inpaint_mask = np.zeros((height, width), dtype=np.uint8)
    for idx, (x, y, w, h) in enumerate(icon_regions, start=1):
        pad = 6
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)
        inpaint_mask[y1:y2, x1:x2] = 255

        # Draw debug boxes
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(debug_img, f'I{idx}', (x1 - 25, max(15, y1 + 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    dilated_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)

    # Inpaint to naturally fill the removed area
    cleaned = cv2.inpaint(img, dilated_mask, 7, cv2.INPAINT_TELEA)

    if debug_path:
        overlay = debug_img.copy()
        overlay[dilated_mask > 0] = (0, 0, 255)
        debug_overlay = cv2.addWeighted(cleaned, 0.8, overlay, 0.2, 0)
        cv2.putText(debug_overlay, f'Icons removed: {len(icon_regions)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imwrite(debug_path, debug_overlay)

    cv2.imwrite(output_path, cleaned)
    return {'success': True, 'icons_found': len(icon_regions)}


def _collect_candidate_regions(mask, width, height):
    """Collect candidate icon rectangles from a pre-computed mask."""
    icon_regions = []
    if mask is None or not mask.any():
        return icon_regions

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for idx in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[idx]
        aspect_ratio = w / h if h > 0 else 0

        if area < 60 or area > 2000:
            continue
        if w < 12 or h < 12 or w > 70 or h > 70:
            continue
        if not (0.6 <= aspect_ratio <= 1.4):
            continue
        if y < 10 or y + h > height - 10:
            continue

        icon_regions.append((x, y, w, h))

    return icon_regions


def _fallback_regions(img, roi_start_x, width, height):
    """
    Fallback search that uses adaptive thresholding on the right-side grayscale ROI.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[:, roi_start_x:]

    if roi.size == 0:
        return []

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21,
        5
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)

    icon_regions = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(adaptive, connectivity=8)

    for idx in range(1, num_labels):
        x, y, w, h, area = stats[idx]
        aspect_ratio = w / h if h > 0 else 0

        if area < 80 or area > 2200:
            continue
        if w < 12 or h < 12 or w > 80 or h > 80:
            continue
        if not (0.55 <= aspect_ratio <= 1.5):
            continue
        if y < 5 or y + h > height - 5:
            continue

        icon_regions.append((x + roi_start_x, y, w, h))

    return icon_regions

def detect_and_remove_people_icons_with_debug(image_path, output_path, debug_path):
    """
    Simplified approach with extensive debugging to understand what's happening.
    """
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'icons_found': 0}

    original_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Create debug image
    debug_img = img.copy()

    # Add text showing image dimensions
    cv2.putText(debug_img, f'Size: {width}x{height}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Convert to PIL for easier manipulation
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    icons_found = 0
    all_detections = []

    # Method 1: Simple brute force approach - check every possible icon location
    # Focus on the area where people icons actually appear (more to the left)
    search_width = 80
    search_start_x = width - search_width - 20  # Move search area 20px left from edge

    # Draw the search area on debug image
    cv2.rectangle(debug_img, (search_start_x, 0), (search_start_x + search_width, height), (0, 255, 0), 2)
    cv2.putText(debug_img, 'Search Area', (search_start_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Manually define expected icon positions based on your image
    # These are approximate Y positions where icons typically appear
    expected_icon_positions = [
        180, 280, 380, 480, 580, 680, 780, 880, 980, 1080, 1180, 1280, 1380, 1480
    ]

    # Check each expected position
    for i, y_center in enumerate(expected_icon_positions):
        if y_center < height - 40:  # Make sure we don't go out of bounds

            # Define a box around this expected position
            box_size = 40
            y_start = max(0, y_center - box_size//2)
            y_end = min(height, y_center + box_size//2)
            x_start = search_start_x + 20  # Look in the repositioned search area
            x_end = search_start_x + search_width - 10

            # Draw this search box on debug image
            cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
            cv2.putText(debug_img, f'P{i+1}', (x_start-20, y_center), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

            # Extract this region
            search_region = gray[y_start:y_end, x_start:x_end]

            if search_region.size > 0:
                # Calculate region statistics
                mean_val = np.mean(search_region)
                min_val = np.min(search_region)
                max_val = np.max(search_region)

                # Look for darker areas that could be icons
                # Icons should be darker than the background
                dark_pixels = np.sum(search_region < 150)
                total_pixels = search_region.size
                dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0

                # Add debug text showing statistics
                debug_text = f'M:{int(mean_val)} D:{dark_ratio:.2f}'
                cv2.putText(debug_img, debug_text, (x_start, y_start-5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 0), 1)

                # If this region has characteristics of a people icon
                if dark_ratio > 0.15 and mean_val < 200:  # Has some dark content
                    # Consider this a detection
                    detection = {
                        'x': x_start,
                        'y': y_start,
                        'w': x_end - x_start,
                        'h': y_end - y_start,
                        'confidence': dark_ratio,
                        'mean_val': mean_val
                    }
                    all_detections.append(detection)

    # Method 2: Look for ANY dark regions in the repositioned search area
    search_region = gray[:, search_start_x:search_start_x + search_width]

    # Create a threshold to find dark areas
    _, thresh = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.putText(debug_img, f'Found {len(contours)} contours', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Check each contour
    for j, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 10:  # Any reasonable sized area
            x, y, w, h = cv2.boundingRect(contour)

            # Convert back to full image coordinates
            actual_x = x + search_start_x

            # Draw all contours for debugging
            cv2.rectangle(debug_img, (actual_x, y), (actual_x + w, y + h), (0, 255, 255), 1)
            cv2.putText(debug_img, f'C{j}', (actual_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)

            # If it's a reasonable size for an icon
            if 20 < area < 2000 and 5 <= w <= 50 and 5 <= h <= 50:
                detection = {
                    'x': actual_x - 2,
                    'y': y - 2,
                    'w': w + 4,
                    'h': h + 4,
                    'confidence': 0.6,
                    'area': area
                }
                all_detections.append(detection)

    # Apply ALL detections (don't merge for now, to see everything)
    cv2.putText(debug_img, f'Total detections: {len(all_detections)}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for i, detection in enumerate(all_detections):
        x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']

        # Ensure bounds are within image
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)

        if w > 0 and h > 0:
            # Mark on debug image with thick rectangles
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 255), 3)
            cv2.putText(debug_img, f'ICON_{i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 2)

            # Sample background color and remove from main image
            bg_color = get_enhanced_background_color(img, x, y, w, h)

            # Remove with padding
            padding = 5
            draw.rectangle([
                x - padding,
                y - padding,
                x + w + padding,
                y + h + padding
            ], fill=bg_color)
            icons_found += 1

    # Save both processed and debug images
    pil_img.save(output_path)
    cv2.imwrite(debug_path, debug_img)

    return {'success': True, 'icons_found': icons_found}

def is_icon_area(region):
    """
    Determine if a region likely contains a people icon.
    """
    if region.size == 0:
        return False

    # Check various characteristics
    mean_val = np.mean(region)
    std_val = np.std(region)

    # Icons should have moderate brightness and some variation
    if 80 < mean_val < 200 and std_val > 10:
        # Check for dark areas (typical of silhouettes)
        dark_pixels = np.sum(region < 120)
        total_pixels = region.size
        dark_ratio = dark_pixels / total_pixels

        # Should have some dark content but not be completely dark
        if 0.1 < dark_ratio < 0.8:
            return True

    return False

def merge_detections(detections):
    """
    Merge overlapping detections into larger areas.
    """
    if not detections:
        return []

    # Sort by x position
    detections.sort(key=lambda d: d['x'])

    merged = []
    for detection in detections:
        merged_with_existing = False

        for merged_det in merged:
            # Check if this detection overlaps with an existing merged detection
            if (abs(detection['x'] - merged_det['x']) < 50 and
                abs(detection['y'] - merged_det['y']) < 40):

                # Merge by expanding the bounding box
                min_x = min(detection['x'], merged_det['x'])
                min_y = min(detection['y'], merged_det['y'])
                max_x = max(detection['x'] + detection['w'], merged_det['x'] + merged_det['w'])
                max_y = max(detection['y'] + detection['h'], merged_det['y'] + merged_det['h'])

                merged_det['x'] = min_x
                merged_det['y'] = min_y
                merged_det['w'] = max_x - min_x
                merged_det['h'] = max_y - min_y
                merged_det['confidence'] = max(detection['confidence'], merged_det['confidence'])

                merged_with_existing = True
                break

        if not merged_with_existing:
            merged.append(detection.copy())

    # Filter by confidence and size
    final_detections = []
    for det in merged:
        if det['confidence'] > 0.6 and 20 <= det['w'] <= 80 and 20 <= det['h'] <= 80:
            final_detections.append(det)

    return final_detections

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

@app.route('/download_all')
def download_all():
    # Create a zip file with all processed images
    memory_file = io.BytesIO()

    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in os.listdir(app.config['PROCESSED_FOLDER']):
            if allowed_file(filename):
                file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
                zipf.write(file_path, filename)

    memory_file.seek(0)

    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='processed_images.zip'
    )

@app.route('/clear')
def clear_files():
    # Clear upload and processed folders
    for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    return jsonify({'message': 'All files cleared'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
