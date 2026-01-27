import streamlit as st
import numpy as np
import cv2
from inference import get_model
import supervision as sv
import os
import io
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
try:
    # Fallback click-capture component that reliably renders images on Streamlit Cloud
    from streamlit_image_coordinates import streamlit_image_coordinates
except Exception:
    streamlit_image_coordinates = None
from pathlib import Path
import zipfile
import shutil


ROBOFLOW_API_KEY = (
    os.environ.get("ROBOFLOW_API_KEY")
    or st.secrets.get("ROBOFLOW_API_KEY", "")
)
# Compatible rerun helper for old/new Streamlit
def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


@st.cache(allow_output_mutation=True)
def load_model():
    key = ROBOFLOW_API_KEY or os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        st.error("Missing ROBOFLOW_API_KEY.")
        st.stop()
    return get_model(model_id="mouse-optic-nerve-uktj7/6", api_key=key)

model = load_model()


if "app_step" not in st.session_state:
    st.session_state.app_step = "upload"
if "yellow_mask" not in st.session_state:
    st.session_state.yellow_mask = None
if "orig_shape" not in st.session_state:
    st.session_state.orig_shape = None
if "rightmost_point" not in st.session_state:
    st.session_state.rightmost_point = None
if "leftmost_point" not in st.session_state:
    st.session_state.leftmost_point = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_img_idx" not in st.session_state:
    st.session_state.current_img_idx = 0
if "csv_buffers" not in st.session_state:
    st.session_state.csv_buffers = {}
if "microns_per_pixel" not in st.session_state:
    st.session_state["microns_per_pixel"] = 3.07
if "interval_microns" not in st.session_state:
    # default interval = current sampling radius (30 px) * microns_per_pixel
    st.session_state["interval_microns"] = 30 * st.session_state.get("microns_per_pixel", 3.07)
if "sampling_radius" not in st.session_state:
    st.session_state["sampling_radius"] = 30


st.title("🧠 Optic Nerve Mask Segmentatione")


if st.session_state.app_step == "upload":
    st.write("Welcome to the Optic Nerve Mask Segmentation App! This app allows you to upload a folder of optic nerve images, run inference to segment each nerve, and then select chiasm points for further analysis.")

    # microns-per-pixel control shown only on upload step (writes into session_state)
    st.number_input(
        "Image Scale: Microns per pixel (µm/pixel)",
        min_value=0.0001,
        step=0.01,
        format="%.4f",
        key="microns_per_pixel",
        help="Enter the number of microns represented by one pixel for your imaging setup. Default: 3.07"
    )

    # measurement interval in microns (used to compute sampling radius in pixels)
    st.number_input(
        "Measurement interval: microns between sampling lines (µm)",
        min_value=0.01,
        step=0.1,
        format="%.2f",
        key="interval_microns",
        help="Distance between consecutive measurement sections in microns. Default = 30 px × µm/pixel"
    )

    uploaded_files = st.file_uploader(
        "Upload a folder of optic nerve images", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected.")
        # require explicit confirmation so widget value is saved before changing app_step
        # ...existing code...
        if st.button("➡️ Start processing"):
            # copy current widget value into a separate confirmed key (safe to write)
            st.session_state["microns_per_pixel_confirmed"] = float(
                st.session_state.get("microns_per_pixel", 3.07)
            )
            # confirm interval and compute sampling radius (px)
            st.session_state["interval_microns_confirmed"] = float(
                st.session_state.get("interval_microns", 30 * st.session_state.get("microns_per_pixel", 3.07))
            )
            # sampling radius in pixels (rounded int) used in contour stepping
            sampling_px = st.session_state["interval_microns_confirmed"] / st.session_state["microns_per_pixel_confirmed"]
            st.session_state["sampling_radius"] = max(1, int(round(sampling_px)))
            st.session_state.uploaded_files = uploaded_files
            st.session_state.current_img_idx = 0
            st.session_state.csv_buffers = {}
            st.session_state.app_step = "model"
            do_rerun()
# ...existing code...

if st.session_state.app_step == "model":
    uploaded_files = st.session_state.uploaded_files
    current_idx = st.session_state.current_img_idx
    uploaded_file = uploaded_files[current_idx]
    filename_base = os.path.splitext(uploaded_file.name)[0]
    st.session_state.uploaded_filename = filename_base
    st.write(f"**Image {current_idx + 1} of {len(uploaded_files)}:** `{uploaded_file.name}`")

    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Original Image", channels="BGR")

    # Save original dimensions
    st.session_state.orig_shape = image.shape[:2]  # (H, W)

    # Run inference
    results = model.infer(image)[0]
    detections = sv.Detections.from_inference(results)
    masks = detections.mask  # shape: (N, H, W)

    if len(masks) == 0:
        st.warning("⚠️ No masks found.")
        st.stop()
    else:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        if len(masks) >= 2:
            areas = [np.sum(mask) for mask in masks]
            idx_outer = np.argmax(areas)
            idx_inner = np.argmin(areas)

            mask_outer = (masks[idx_outer].astype(np.uint8)) * 255
            mask_inner = (masks[idx_inner].astype(np.uint8)) * 255
            yellow_mask = cv2.bitwise_and(mask_outer, cv2.bitwise_not(mask_inner))

            axs[1].imshow(mask_outer, cmap="gray")
            axs[1].set_title("Outer Mask")
            axs[1].axis("off")

            axs[2].imshow(yellow_mask, cmap="gray")
            axs[2].set_title("Refined (Outer - Inner)")
            axs[2].axis("off")
        else:
            yellow_mask = (masks[0].astype(np.uint8)) * 255
            axs[1].imshow(yellow_mask, cmap="gray")
            axs[1].set_title("Refined Mask (Single)")
            axs[1].axis("off")
            axs[2].axis("off")

        st.pyplot(fig)

        st.session_state.yellow_mask = yellow_mask

    if st.button("➡️ Next: Select Points"):
        st.session_state.app_step = "select"
        do_rerun()

# --- STEP 2: POINT SELECTION ---
if st.session_state.app_step == "select":

    uploaded_files = st.session_state.uploaded_files
    current_idx = st.session_state.current_img_idx
    uploaded_file = uploaded_files[current_idx]
    filename_base = os.path.splitext(uploaded_file.name)[0]
    st.session_state.uploaded_filename = filename_base
    st.write(f"**Image {current_idx + 1} of {len(uploaded_files)}:** `{uploaded_file.name}`")


    st.subheader("📍 Select Chiasm Points")
    st.markdown("👉 Click two points on the nerve (order doesn't matter).")
    st.markdown("**Important:** Ensure that the leftmost point does not go beyond the edges of the optic nerve, and leave about 50 pixels of space on the edge of BOTH sides of the nerve in order to avoid errors.")

    yellow_mask = st.session_state.yellow_mask
    orig_h, orig_w = st.session_state.orig_shape

    display_width = 600
    scale_factor = display_width / orig_w
    display_height = int(orig_h * scale_factor)

    resized = cv2.resize(yellow_mask, (display_width, display_height))
    display_img = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    pil_bg = Image.fromarray(display_img).convert("RGB")

    # Store clicked points in display-space (image coords)
    if "clicked_points_display" not in st.session_state:
        st.session_state.clicked_points_display = []

    # Reset points automatically when switching to a new image
    current_image_key = f"{current_idx}_{filename_base}"
    if st.session_state.get("_last_select_image_key") != current_image_key:
        st.session_state.clicked_points_display = []
        st.session_state.rightmost_point = None
        st.session_state.leftmost_point = None
        st.session_state._last_select_image_key = current_image_key

    # Always-on reliable click capture (no drawable canvas)
    if st.button("Reset selected points"):
        st.session_state.clicked_points_display = []
        st.session_state.rightmost_point = None
        st.session_state.leftmost_point = None
        do_rerun()

    if streamlit_image_coordinates is None:
        st.error(
            "`streamlit-image-coordinates` is not installed. "
            "Add `streamlit-image-coordinates` to requirements.txt and redeploy."
        )
    else:
        # Draw any already-selected points onto the image for feedback
        preview = pil_bg.copy()
        draw = ImageDraw.Draw(preview)
        r = 6
        for (px, py) in st.session_state.clicked_points_display:
            draw.ellipse((px - r, py - r, px + r, py + r), outline=(0, 255, 255), width=3)

        click = streamlit_image_coordinates(preview, key=f"img_click_{current_idx}_{filename_base}")
        if click is not None and "x" in click and "y" in click:
            x_disp = int(click["x"])
            y_disp = int(click["y"])

            # Only collect the first two *distinct* clicks (avoid double-click same spot)
            if len(st.session_state.clicked_points_display) == 0:
                st.session_state.clicked_points_display.append((x_disp, y_disp))
                do_rerun()
            elif len(st.session_state.clicked_points_display) == 1:
                lx, ly = st.session_state.clicked_points_display[0]
                if abs(x_disp - lx) > 2 or abs(y_disp - ly) > 2:
                    st.session_state.clicked_points_display.append((x_disp, y_disp))
                    do_rerun()

        # Convert the 2 selected display points back into original image coords
        if len(st.session_state.clicked_points_display) >= 2:
            (x1d, y1d), (x2d, y2d) = st.session_state.clicked_points_display[:2]

            # Map display->original
            p1 = (int(round(x1d / scale_factor)), int(round(y1d / scale_factor)))
            p2 = (int(round(x2d / scale_factor)), int(round(y2d / scale_factor)))

            # Assign by X so order doesn't matter
            leftmost = p1 if p1[0] < p2[0] else p2
            rightmost = p2 if p1[0] < p2[0] else p1

            st.session_state.leftmost_point = leftmost
            st.session_state.rightmost_point = rightmost

            st.success(f"✅ Leftmost X: {leftmost[0]}, Rightmost X: {rightmost[0]}")

            if st.button("➡️ Next: View Diameter Visualization and Graph"):
                st.session_state.app_step = "diameter"
                do_rerun()
        else:
            st.info("ℹ️ Click two points on the image.")



# --- STEP 3: DIAMETER MEASUREMENT ---

if st.session_state.app_step == "diameter":
    
    
    

    uploaded_files = st.session_state.uploaded_files
    current_idx = st.session_state.current_img_idx
    uploaded_file = uploaded_files[current_idx]
    filename_base = os.path.splitext(uploaded_file.name)[0]
    st.session_state.uploaded_filename = filename_base
    st.write(f"**Image {current_idx + 1} of {len(uploaded_files)}:** `{uploaded_file.name}`")

    st.subheader("Nerve Diameter Measurement")

    # analyze ------------------------

    rightmost_x = st.session_state.rightmost_point
    leftmost_x = st.session_state.leftmost_point

    
    yellow_mask = st.session_state.yellow_mask
    orig_h, orig_w = st.session_state.orig_shape

    radius = int(st.session_state.get("sampling_radius", 30))
    angle_step = 1
    max_steps = 100

    kernel = np.ones((3, 3), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    _, yellow_mask = cv2.threshold(yellow_mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        st.error("❌ No contours found.")
        st.stop()

    main_contour = max(contours, key=cv2.contourArea)
    contour_points = set(tuple(pt[0]) for pt in main_contour)

    rx = rightmost_x[0]
    lx = leftmost_x[0]
    column = yellow_mask[:, rx]
    nonzero_y = np.where(column > 0)[0]
    if len(nonzero_y) < 2:
        st.error("Could not find top and bottom at rightmost_x")
        st.stop()

    top_start = (rx, nonzero_y[0])
    bottom_start = (rx, nonzero_y[-1])
    top_path = [top_start]
    bottom_path = [bottom_start]

    def find_next_contour_point(cx, cy, radius, min_angle, max_angle):
        angles = range(min_angle, max_angle + 1, angle_step) if min_angle <= max_angle else range(min_angle, max_angle - 1, -angle_step)
        for angle in angles:
            rad = np.deg2rad(angle)
            x = int(round(cx + radius * np.cos(rad)))
            y = int(round(cy + radius * np.sin(rad)))
            dist = cv2.pointPolygonTest(main_contour, (x, y), True)
            if abs(dist) < 3:
                return (x, y)
        return None

    for _ in range(max_steps):
        cx, cy = top_path[-1]
        next_pt = find_next_contour_point(cx, cy, radius, 270, 90)
        if not next_pt or next_pt[0] < lx:
            break
        top_path.append(next_pt)

    for _ in range(max_steps):
        cx, cy = bottom_path[-1]
        next_pt = find_next_contour_point(cx, cy, radius, 90, 270)
        if not next_pt or next_pt[0] < lx:
            break
        bottom_path.append(next_pt)


    # === START: Midpoint Intersection Analysis ===
    
    _, binary_mask = cv2.threshold(yellow_mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours (for the outline and internal contours)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create a color version of the mask for visualization
    visualization = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)


    x_start = leftmost_x
    x_end = rightmost_x  


    intersection_counts = []
    midpoints = []  


    top_x_values = [pt[0] for pt in top_path]
    bottom_x_values = [pt[0] for pt in bottom_path]

    red_lines = top_x_values[1:] + bottom_x_values


    for x in red_lines:

        intersections = []
        for contour in contours:
            # Check where the vertical line intersects the contour
            for i in range(len(contour) - 1):
                pt1, pt2 = contour[i][0], contour[i + 1][0]
                if pt1[0] <= x <= pt2[0] or pt2[0] <= x <= pt1[0]:  
                    intersections.append((pt1[1], pt2[1]))


   
        if intersections:
            intersections = sorted(intersections, key=lambda t: min(t[0], t[1]))

            groups = []
            current_group = [intersections[0]]

            for i in range(1, len(intersections)):
                y1, y2 = intersections[i]
                last_y1, last_y2 = current_group[-1]

                # If the y-values are within ±3, group them together
                if abs(y1 - last_y1) <= 3 or abs(y2 - last_y2) <= 3:
                    current_group.append((y1, y2))
                else:
                    groups.append(current_group)
                    current_group = [(y1, y2)]

            groups.append(current_group)


            if len(groups) >= 4:
                group_2 = groups[1]
                group_3 = groups[2]

                # Take the average of the y-values for the 2nd group
                y2_avg = np.mean([y for y, _ in group_2])
                # Take the average of the y-values for the 3rd group
                y3_avg = np.mean([y for y, _ in group_3])

                # Compute the midpoint
                midpoint_y = int((y2_avg + y3_avg) / 2)
                midpoints.append((x, midpoint_y))  # Store the midpoint (x, midpoint_y)

                cv2.circle(visualization, (x, midpoint_y), 5, (0, 0, 255), -1)  # Red point
            elif len(groups) == 3:
                # If there are exactly 3 groups, use the middle y-value
                middle_group = groups[1]
                middle_group_avg_y = np.mean([y for y, _ in middle_group])

                # Add the middle y-value as the midpoint
                midpoints.append((x, int(middle_group_avg_y)))

              
            else:
                if midpoints:
                    # Use the last valid midpoint
                    last_midpoint_y = midpoints[-1][1]
                else:
                    last_midpoint_y = 0  # Fallback in case there's no valid midpoint yet

                # Use the first group intersection's y-value (average of y-values of the first group)
                first_group = groups[0]
                first_group_avg_y = np.mean([y for y, _ in first_group])

                # Use the last group intersection's y-value (average of y-values of the last group)
                last_group = groups[-1]
                last_group_avg_y = np.mean([y for y, _ in last_group])

                # Smarter fallback: avoid weird high midpoints if groups collapse

                # If there is more than one group and first/last are very far apart → trust average
                if len(groups) > 1 and abs(first_group_avg_y - last_group_avg_y) > 10:
                    average_y = int((last_midpoint_y + first_group_avg_y + last_group_avg_y) / 3)
                else:
                    # Groups too close → likely noise, just repeat last good midpoint
                    average_y = last_midpoint_y

                # Add this new midpoint
                midpoints.append((x, average_y))

        


        # Draw the vertical line on visualization for debugging
        color = (255, 0, 0) if not intersections else (0, 255, 0)  # Blue for no intersections, green for any intersections
      

        # Print the x-coordinate of the vertical line if it intersects with the contour
        if intersections:
            print(f"Vertical line at x={x} has intersections.")

    

    # Debugging: print midpoints list to check


    # After gathering midpoints, draw lines between consecutive midpoints to create a continuous line
    if len(midpoints) > 1:
        for i in range(1, len(midpoints)):
            pt1 = midpoints[i - 1]
            pt2 = midpoints[i]
            cv2.line(visualization, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (0, 255, 255), 2)  # Yellow line

    # Extend horizontal lines at the leftmost and rightmost points
    if midpoints:
        # Leftmost point
        leftmost_x_point, leftmost_y = midpoints[-1]

        # Rightmost point
        rightmost_x_point, rightmost_y = midpoints[0]



    st.session_state.yellow_mask_processed = yellow_mask
    st.session_state.top_x_values = top_x_values
    st.session_state.bottom_x_values = bottom_x_values
    st.session_state.midpoints = midpoints





    # analyze ---------------------------







    # Retrieve values from session state
    yellow_mask = st.session_state.get("yellow_mask_processed")
    top_x_values = st.session_state.get("top_x_values")
    bottom_x_values = st.session_state.get("bottom_x_values")
    midpoints = st.session_state.get("midpoints")

    

    if yellow_mask is None or top_x_values is None or bottom_x_values is None or midpoints is None:
        st.error("Required data for diameter measurement is missing. Please run contour analysis first.")
        st.stop()

    # Convert yellow_mask to grayscale if needed
    if len(yellow_mask.shape) == 3:
        yellow_mask = cv2.cvtColor(yellow_mask, cv2.COLOR_BGR2GRAY)

    color_mask = cv2.cvtColor(yellow_mask, cv2.COLOR_GRAY2BGR)
    height, width = yellow_mask.shape
    np.random.seed(42)

    # Storage for top and bottom points with section index
    top_points = []  # Will store (x, y, section_index)
    bottom_points = []  # Will store (x, y, section_index)

    # Track if any intersections are found
    found_intersections = False

    color = (0, 255, 255)  # yellow for points

    # --- Top points ---
    for i, x in enumerate(top_x_values):  # Use enumerate to get index `i`
        column = yellow_mask[:, x]  # Get the column at x-position
        nonzero_y = np.where(column > 0)[0]  # Find y-coordinates where mask is present

        if len(nonzero_y) < 2:
            continue  # Skip if not enough mask pixels

        top_intersection = nonzero_y[0]  # Topmost y

        found_intersections = True

        midpoint_y = midpoints[i][1] if i < len(midpoints) else height // 2

        if top_intersection < midpoint_y:
            top_points.append((x, top_intersection, i))
            cv2.circle(color_mask, (x, top_intersection), 6, color, -1)

    # --- Bottom points ---
    for i, x in enumerate(bottom_x_values):  # Again, use enumerate for index
        column = yellow_mask[:, x]
        nonzero_y = np.where(column > 0)[0]

        if len(nonzero_y) < 2:
            continue

        bottom_intersection = nonzero_y[-1]  # Bottommost y

        found_intersections = True

        midpoint_y = midpoints[i][1] if i < len(midpoints) else height // 2

        if bottom_intersection > midpoint_y:
            bottom_points.append((x, bottom_intersection, i))
            cv2.circle(color_mask, (x, bottom_intersection), 6, color, -1)

    # Function to estimate tangent slope using nearby points
    def estimate_tangent_slope(points, index, search_range=3):
        """Estimate the tangent slope at a given index using nearby points."""
        if len(points) < 2:
            return None  # Not enough points to estimate slope

        x, y, _ = points[index]

        # Find two nearby points for slope estimation
        left_idx = max(0, index - search_range)
        right_idx = min(len(points) - 1, index + search_range)

        x1, y1, _ = points[left_idx]
        x2, y2, _ = points[right_idx]

        if x2 - x1 == 0:
            return None  # Avoid division by zero

        return (y2 - y1) / (x2 - x1)  # Slope = rise / run

    # Function to find intersection with mask boundary along a line with midpoint constraint
    def find_mask_intersection(mask, start_x, start_y, angle, midpoint_y, is_top, max_length=300):
        """Find the intersection point with mask boundary starting from (start_x, start_y)
        and moving in the direction given by angle (in radians).

        Parameters:
        - mask: The binary mask image
        - start_x, start_y: Starting point coordinates
        - angle: Direction angle in radians
        - midpoint_y: Y-coordinate of midpoint line (for stopping condition)
        - is_top: Whether this is for the top nerve (True) or bottom nerve (False)
        - max_length: Maximum ray length to check
        """
        dx = np.cos(angle)
        dy = np.sin(angle)

        # Check points along the ray
        for length in range(1, max_length):
            x = int(start_x + length * dx)
            y = int(start_y + length * dy)

            # Check if out of bounds
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None

            # Check if we've reached the midpoint boundary
            # For top nerve, stop if we go below midpoint (when going down)
            if is_top and dy > 0 and y >= midpoint_y:
                return (x, y)

            # For bottom nerve, stop if we go above midpoint (when going up)
            if not is_top and dy < 0 and y <= midpoint_y:
                return (x, y)

            # Check if we've hit the boundary (pixel value changes from >0 to 0)
            if mask[y, x] == 0:
                return (x, y)

        return None  # No intersection found within max_length

    def vertical_diameter_top_leg(mask, start_x, start_y, midpoint_y, go_down=True, max_length=300):
        """
        For index == 1: Draw a vertical line up or down from (start_x, start_y),
        stopping when reaching midpoint_y or exiting mask.
        """
        direction = 1 if go_down else -1

        for length in range(1, max_length):
            x = start_x
            y = start_y + length * direction  # Move vertically

            # Out of bounds
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None

            # If reached midpoint_y, stop
            if go_down and y >= midpoint_y:
                return (x, y)
            if not go_down and y <= midpoint_y:
                return (x, y)

            # If pixel is outside mask, stop
            if mask[y, x] == 0:
                return (x, y)

        return None

    def vertical_diameter_bottom_leg(mask, start_x, start_y, midpoint_y, go_up=True, max_length=300):
        """
        For bottom leg: Draw a vertical line up or down from (start_x, start_y),
        stopping when reaching midpoint_y or exiting mask.
        """
        direction = -1 if go_up else 1

        for length in range(1, max_length):
            x = start_x
            y = start_y + length * direction  # Move vertically

            # Out of bounds
            if x < 0 or x >= mask.shape[1] or y < 0 or y >= mask.shape[0]:
                return None

            # If reached midpoint_y, stop
            if go_up and y <= midpoint_y:
                return (x, y)
            if not go_up and y >= midpoint_y:
                return (x, y)

            # If pixel is outside mask, stop
            if mask[y, x] == 0:
                return (x, y)

        return None

    # Lists to store diameter measurements
    diameters_top = []
    diameters_bottom = []

    # Process top points
    for i, (x, y, section_idx) in enumerate(top_points):

        slope = estimate_tangent_slope(top_points, i)
        if slope is None:
            continue

        # Get the midpoint y for this section
        midpoint_y = midpoints[section_idx][1] if section_idx < len(midpoints) else height // 2

        # Calculate perpendicular angle (in radians)
        perp_angle = np.arctan(-1/slope) if slope != 0 else np.pi/2

        # Find the intersection with the upper boundary (going up)
        intersection1 = find_mask_intersection(yellow_mask, x, y, perp_angle + np.pi,
                                              midpoint_y, True)

        # Find the intersection with the lower boundary (going down)
        intersection2 = find_mask_intersection(yellow_mask, x, y, perp_angle,
                                              midpoint_y, True)

        if (i == 0):
            intersection1 = vertical_diameter_top_leg(yellow_mask, x, y, midpoint_y, go_down=False)
            intersection2 = vertical_diameter_top_leg(yellow_mask, x, y, midpoint_y, go_down=True)

        # Draw the full diameter line if both intersections found
        if intersection1 and intersection2:
            x1, y1 = intersection1
            x2, y2 = intersection2

            diameter = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if diameter < 5:
                # fallback to vertical:
                intersection1 = vertical_diameter_top_leg(yellow_mask, x, y, midpoint_y, go_down=False)
                intersection2 = vertical_diameter_top_leg(yellow_mask, x, y, midpoint_y, go_down=True)
                
                if intersection1 and intersection2:
                    x1, y1 = intersection1
                    x2, y2 = intersection2
                    diameter = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            diameters_top.append((x, diameter))
            cv2.line(color_mask, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Process bottom points
    for i, (x, y, section_idx) in enumerate(bottom_points):
        
        slope = estimate_tangent_slope(bottom_points, i)
        if slope is None:
            continue

        # Get the midpoint y for this section
        midpoint_y = midpoints[section_idx][1] if section_idx < len(midpoints) else height // 2

        # Calculate perpendicular angle (in radians)
        perp_angle = np.arctan(-1/slope) if slope != 0 else np.pi/2

        # Find the intersection with the lower boundary (going down)
        intersection1 = find_mask_intersection(yellow_mask, x, y, perp_angle,
                                              midpoint_y, False)

        # Find the intersection with the upper boundary (going up)
        intersection2 = find_mask_intersection(yellow_mask, x, y, perp_angle + np.pi,
                                              midpoint_y, False)

        if (i == 0):
            intersection1 = vertical_diameter_bottom_leg(yellow_mask, x, y, midpoint_y, go_up=False)
            intersection2 = vertical_diameter_bottom_leg(yellow_mask, x, y, midpoint_y, go_up=True)

        if intersection1 and intersection2:
            x1, y1 = intersection1
            x2, y2 = intersection2

            diameter = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            if diameter < 5:
                # fallback to vertical:
                intersection1 = vertical_diameter_bottom_leg(yellow_mask, x, y, midpoint_y, go_up=False)
                intersection2 = vertical_diameter_bottom_leg(yellow_mask, x, y, midpoint_y, go_up=True)
                
                if intersection1 and intersection2:
                    x1, y1 = intersection1
                    x2, y2 = intersection2
                    diameter = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            diameters_bottom.append((x, diameter))
            cv2.line(color_mask, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Draw the midpoint dots to visualize the midpoints
    for i, midpoint in enumerate(midpoints):
        cv2.circle(color_mask, (int(midpoint[0]), int(midpoint[1])), 4, (255, 255, 0), -1)

    # Display the result with measurements in Streamlit
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.imshow(cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB))
    ax1.set_xlim(0, width)
    ax1.set_ylim(height, 0)
    ax1.set_xlabel('Position Value (How Far Along the Nerve)')
    ax1.set_ylabel('Y-axis')
    ax1.set_title('Nerve Diameter Measurements Graph')
    st.pyplot(fig1)

      # Plot diameter vs x position in Streamlit
    radius = int(st.session_state.get("sampling_radius", 30))
    MICRONS_PER_PIXEL = float(
        st.session_state.get(
            "microns_per_pixel_confirmed",
            st.session_state.get("microns_per_pixel", 3.07),
        )
    )

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if diameters_top:
        positions_top_px = [i * radius for i in range(len(diameters_top))]
        d_values_top_px = [d for x, d in diameters_top]
        # Convert to microns
        positions_top_um = [p * MICRONS_PER_PIXEL for p in positions_top_px]
        d_values_top_um = [d * MICRONS_PER_PIXEL for d in d_values_top_px]
        ax2.plot(positions_top_um, d_values_top_um, 'go-', label='Top Nerve')

    if diameters_bottom:
        positions_bottom_px = [i * radius for i in range(len(diameters_bottom))]
        d_values_bottom_px = [d for x, d in diameters_bottom]
        # Convert to microns
        positions_bottom_um = [p * MICRONS_PER_PIXEL for p in positions_bottom_px]
        d_values_bottom_um = [d * MICRONS_PER_PIXEL for d in d_values_bottom_px]
        ax2.plot(positions_bottom_um, d_values_bottom_um, 'ro-', label='Bottom Nerve')

    ax2.set_xlabel('Position along Nerve (microns)')
    ax2.set_ylabel('Diameter (microns)')
    ax2.set_title('Nerve Diameter vs Position')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # --- CSV Download Option ---
    import pandas as pd
    from io import StringIO
    
     # Prepare data for CSV: combine top and bottom nerves side by side
    max_len = max(len(diameters_top), len(diameters_bottom))
    MICRONS_PER_PIXEL = float(
        st.session_state.get(
            "microns_per_pixel_confirmed",
            st.session_state.get("microns_per_pixel", 3.07),
        )
    )

    rows = []
    for i in range(max_len):
        # Top nerve data
        if i < len(diameters_top):
            x_top, d_top = diameters_top[i]
            pos_top_px = i * radius
            pos_top_um = pos_top_px * MICRONS_PER_PIXEL
            d_top_um = d_top * MICRONS_PER_PIXEL
        else:
            x_top, d_top, pos_top_px, pos_top_um, d_top_um = [None]*5

        # Bottom nerve data
        if i < len(diameters_bottom):
            x_bot, d_bot = diameters_bottom[i]
            pos_bot_px = i * radius
            pos_bot_um = pos_bot_px * MICRONS_PER_PIXEL
            d_bot_um = d_bot * MICRONS_PER_PIXEL
        else:
            x_bot, d_bot, pos_bot_px, pos_bot_um, d_bot_um = [None]*5

        rows.append({
            "Top_X": x_top,
            "Top_Position_along_nerve_px": pos_top_px,
            "Top_Diameter_px": d_top,
            "Top_Position_along_nerve_um": pos_top_um,
            "Top_Diameter_um": d_top_um,
            "Bottom_X": x_bot,
            "Bottom_Position_along_nerve_px": pos_bot_px,
            "Bottom_Diameter_px": d_bot,
            "Bottom_Position_along_nerve_um": pos_bot_um,
            "Bottom_Diameter_um": d_bot_um
        })

    if rows:
        df = pd.DataFrame(rows)

        # Count how many columns are for top and bottom
        top_cols = [col for col in df.columns if col.startswith("Top_")]
        bottom_cols = [col for col in df.columns if col.startswith("Bottom_")]

        # Build the first header row: repeat "Top Nerve" for all top columns, "Bottom Nerve" for all bottom columns
        multi_header = ["Top Nerve"] * len(top_cols) + ["Bottom Nerve"] * len(bottom_cols)

        csv_buffer = StringIO()
        # Write the multi-header row
        csv_buffer.write(",".join(multi_header) + "\n")
        # Write the actual column headers and data
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Diameter Data as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{getattr(st.session_state, 'uploaded_filename', 'nerve')}_diameters.csv",
            mime="text/csv",
            key="download_diameter_csv"
        )
        csv_filename = f"{getattr(st.session_state, 'uploaded_filename', 'nerve')}_diameters.csv"
        st.session_state.csv_buffers[csv_filename] = StringIO(csv_buffer.getvalue())

        if st.session_state.current_img_idx < len(st.session_state.uploaded_files) - 1:
            if st.button("➡️ Next Image"):
                st.session_state.current_img_idx += 1
                st.session_state.app_step = "model"
                do_rerun()

        # Only show ZIP download on the last image
        import zipfile
        import io
        if st.session_state.current_img_idx == len(st.session_state.uploaded_files) - 1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as zf:
                for fname, csv_buf in st.session_state.csv_buffers.items():
                    zf.writestr(fname, csv_buf.getvalue())
            st.download_button(
                label="⬇️ Download All CSVs as ZIP",
                data=zip_buffer.getvalue(),
                file_name="all_nerve_diameters.zip",
                mime="application/zip"
            )
