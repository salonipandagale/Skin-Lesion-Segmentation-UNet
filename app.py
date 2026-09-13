import os
from pathlib import Path
import io

# Reduce TensorFlow console noise before importing TensorFlow.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import streamlit as st
import tensorflow as tf
import keras
from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Lesion Analysis | AI Image Segmentation",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

tf.get_logger().setLevel("ERROR")


# ============================================================
# AIRA-INSPIRED VISUAL THEME
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Global ---------- */

    .stApp {
        background: #f4f8fa;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ---------- Typography ---------- */

    html,
    body,
    [class*="css"] {
        font-family: "Segoe UI", Arial, sans-serif;
    }

    h1 {
        color: #083f50 !important;
        font-weight: 750 !important;
        letter-spacing: -0.5px;
    }

    h2,
    h3 {
        color: #0a5263 !important;
    }

    p,
    li {
        color: #506772;
    }


    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background: #eaf2f5;
        border-right: 1px solid #d2e0e5;
    }

    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #083f50 !important;
    }


    /* ---------- Metric cards ---------- */

    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8e4e8;
        border-radius: 14px;
        padding: 1rem 1.05rem;
        box-shadow: 0 4px 14px rgba(8, 63, 80, 0.06);
        min-height: 120px;
    }

    [data-testid="stMetricLabel"] {
        color: #607681 !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-size: 0.75rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #07586a !important;
        font-weight: 750 !important;
    }


    /* ---------- Upload area ---------- */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 1px dashed #8bb9c2;
        border-radius: 14px;
        padding: 0.75rem;
    }


    /* ---------- Buttons ---------- */

    .stButton > button,
    .stDownloadButton > button {
        background: #075f70;
        color: white;
        border: 1px solid #075f70;
        border-radius: 8px;
        font-weight: 650;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #064b59;
        border-color: #064b59;
        color: white;
    }


    /* ---------- Images ---------- */

    [data-testid="stImage"] {
        border-radius: 10px;
    }


    /* ---------- Expanders ---------- */

    [data-testid="stExpander"] {
        border: 1px solid #d8e4e8;
        border-radius: 12px;
        background: #ffffff;
    }


    /* ---------- Horizontal rule ---------- */

    hr {
        border-color: #d5e2e6 !important;
    }


    /* ---------- Footer ---------- */

    .footer-text {
        color: #71838c;
        font-size: 0.78rem;
        text-align: center;
        padding-top: 1.25rem;
    }


    /* ---------- Download button ---------- */

    .stDownloadButton > button {
        background-color: #056B78 !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
    }

    .stDownloadButton > button p {
        color: #FFFFFF !important;
    }

    .stDownloadButton > button span {
        color: #FFFFFF !important;
    }

    .stDownloadButton > button:hover {
        background-color: #045A66 !important;
        color: #FFFFFF !important;
    }

    .stDownloadButton > button:hover p,
    .stDownloadButton > button:hover span {
        color: #FFFFFF !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "unet_skin_lesion_best.keras"
)

INPUT_SIZE = (256, 256)

THRESHOLD = 0.50


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

# We persist the uploaded image bytes because Streamlit reruns
# the complete script whenever widget state changes.
#
# This prevents the uploaded image from disappearing between
# reruns and makes the application more robust on Render.

if "uploaded_image_bytes" not in st.session_state:
    st.session_state.uploaded_image_bytes = None

if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None

if "analysis_requested" not in st.session_state:
    st.session_state.analysis_requested = False


# ============================================================
# CUSTOM OBJECTS USED BY THE TRAINED MODEL
# ============================================================

def dice_coefficient(y_true, y_pred, smooth=1e-6):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(
        y_true_f * y_pred_f
    )

    return (
        (2.0 * intersection + smooth)
        / (
            tf.reduce_sum(y_true_f)
            + tf.reduce_sum(y_pred_f)
            + smooth
        )
    )


def dice_loss(y_true, y_pred):

    return 1.0 - dice_coefficient(
        y_true,
        y_pred,
    )


def combined_loss(y_true, y_pred):

    bce = tf.keras.losses.binary_crossentropy(
        y_true,
        y_pred,
    )

    return (
        tf.reduce_mean(bce)
        + dice_loss(y_true, y_pred)
    )


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_segmentation_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model file not found at:\n{MODEL_PATH}\n\n"
            "Make sure the model is present in "
            "models/unet_skin_lesion_best.keras "
            "and has been correctly downloaded from Git LFS."
        )

    # Check that the path is actually a file.
    if not MODEL_PATH.is_file():

        raise FileNotFoundError(
            f"Model path exists but is not a file:\n{MODEL_PATH}"
        )

    # Check for the common Git LFS pointer problem.
    #
    # A Git LFS pointer is usually only a few hundred bytes.
    # The actual model should be hundreds of MB.
    file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)

    if file_size_mb < 1:

        raise RuntimeError(
            "The model file appears to be a Git LFS pointer "
            "instead of the actual .keras model.\n\n"
            f"Model path: {MODEL_PATH}\n"
            f"Detected size: {file_size_mb:.4f} MB\n\n"
            "Verify that the Git LFS object was uploaded "
            "and is available to the deployment."
        )

    model = keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss,
            "combined_loss": combined_loss,
        },
        compile=False,
    )

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image: Image.Image):
    """
    Prepare an uploaded RGB image for U-Net inference.

    Returns:
        original_image:
            Original RGB uint8 image.

        resized_image:
            256 × 256 float32 image in [0, 1].

        model_input:
            NumPy array with shape (1, 256, 256, 3).
    """

    image = image.convert("RGB")

    original_image = np.asarray(
        image
    ).copy()

    resized_pil = image.resize(
        INPUT_SIZE,
        Image.Resampling.LANCZOS,
    )

    resized_image = np.asarray(
        resized_pil,
        dtype=np.float32,
    ) / 255.0

    model_input = np.expand_dims(
        resized_image,
        axis=0,
    )

    return (
        original_image,
        resized_image,
        model_input,
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

def predict_segmentation(
    model,
    model_input,
):
    """
    Generate the lesion probability map
    and binary segmentation mask.
    """

    prediction = model.predict(
        model_input,
        verbose=0,
    )

    # --------------------------------------------------------
    # Validate model output.
    # --------------------------------------------------------

    if prediction is None:

        raise RuntimeError(
            "The model returned no prediction."
        )

    prediction = np.asarray(
        prediction
    )

    if prediction.ndim != 4:

        raise RuntimeError(
            "Unexpected model output shape: "
            f"{prediction.shape}. "
            "Expected shape similar to "
            "(1, 256, 256, 1)."
        )

    if prediction.shape[0] != 1:

        raise RuntimeError(
            "Unexpected batch dimension in model output: "
            f"{prediction.shape}"
        )

    if prediction.shape[-1] != 1:

        raise RuntimeError(
            "Unexpected number of output channels: "
            f"{prediction.shape[-1]}. "
            "The segmentation model should output one "
            "binary lesion channel."
        )

    probability = prediction[
        0, :, :, 0
    ].astype(np.float32)

    # Ensure probability values are in [0, 1].
    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    binary_mask = (
        probability >= THRESHOLD
    ).astype(np.uint8)

    return (
        probability,
        binary_mask,
    )


# ============================================================
# POSTPROCESSING
# ============================================================

def resize_mask_to_original(
    binary_mask,
    original_size,
):
    """
    Resize the predicted 256 × 256 mask back
    to the original uploaded image dimensions.
    """

    mask_image = Image.fromarray(
        (
            binary_mask * 255
        ).astype(np.uint8)
    )

    resized_mask = mask_image.resize(
        original_size,
        Image.Resampling.NEAREST,
    )

    return np.asarray(
        resized_mask
    ) > 127


def create_overlay(
    image_array,
    mask,
):
    """
    Create a teal clinical-style
    segmentation overlay.
    """

    image = (
        image_array.astype(np.float32)
        / 255.0
    )

    overlay = image.copy()

    overlay_color = np.array(
        [0.02, 0.42, 0.48],
        dtype=np.float32,
    )

    alpha = 0.42

    lesion_pixels = mask.astype(bool)

    overlay[lesion_pixels] = (
        (1.0 - alpha)
        * overlay[lesion_pixels]
        + alpha
        * overlay_color
    )

    return np.clip(
        overlay,
        0.0,
        1.0,
    )


def calculate_statistics(
    probability,
    binary_mask,
):
    """
    Calculate lesion area and mean lesion confidence.

    Both probability and binary_mask should correspond
    to the model's 256 × 256 output.
    """

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    binary_mask = np.asarray(
        binary_mask
    )

    probability = np.squeeze(
        probability
    )

    binary_mask = np.squeeze(
        binary_mask
    )

    if probability.size == 0:

        return 0.0, 0.0

    if probability.max() > 1.0:

        probability = (
            probability / 100.0
        )

    probability = np.clip(
        probability,
        0.0,
        1.0,
    )

    if binary_mask.size == 0:

        return 0.0, 0.0

    if binary_mask.max() <= 1:

        binary_mask_bool = (
            binary_mask > 0.5
        )

    else:

        binary_mask_bool = (
            binary_mask > 0
        )

    if (
        binary_mask_bool.shape
        != probability.shape
    ):

        mask_image = Image.fromarray(
            (
                binary_mask_bool.astype(
                    np.uint8
                )
                * 255
            )
        )

        mask_image = mask_image.resize(
            (
                probability.shape[1],
                probability.shape[0],
            ),
            Image.Resampling.NEAREST,
        )

        binary_mask_bool = (
            np.asarray(mask_image)
            > 127
        )

    lesion_area = (
        np.sum(binary_mask_bool)
        / binary_mask_bool.size
        * 100.0
    )

    if np.any(binary_mask_bool):

        mean_confidence = (
            probability[
                binary_mask_bool
            ].mean()
            * 100.0
        )

    else:

        mean_confidence = 0.0

    return (
        float(lesion_area),
        float(mean_confidence),
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## LESION ANALYSIS"
    )

    st.caption(
        "AI-assisted dermoscopic image "
        "segmentation for research and "
        "educational use."
    )

    st.divider()

    st.markdown(
        "### Model Information"
    )

    st.write(
        "**Architecture**"
    )
    st.write("U-Net")

    st.write(
        "**Task**"
    )
    st.write(
        "Binary semantic segmentation"
    )

    st.write(
        "**Input**"
    )
    st.write(
        "256 × 256 RGB"
    )

    st.write(
        "**Output**"
    )
    st.write(
        "Lesion probability mask"
    )

    st.write(
        "**Decision threshold**"
    )
    st.write(
        f"{THRESHOLD:.2f}"
    )

    st.divider()

    st.markdown(
        "### Test Performance"
    )

    st.metric(
        "Dice",
        "78.75%",
    )

    st.metric(
        "IoU",
        "68.72%",
    )

    st.metric(
        "Precision",
        "82.44%",
    )

    st.metric(
        "Recall",
        "75.83%",
    )

    st.divider()

    st.markdown(
        "### Dataset"
    )

    st.write(
        "**ISIC 2016 Task 1**"
    )

    st.caption(
        "900 annotated dermoscopic images"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.caption(
    "ARTIFICIAL INTELLIGENCE  ·  IMAGE ANALYSIS"
)

st.title(
    "Skin Lesion Segmentation"
)

st.write(
    "AI-assisted pixel-level analysis of "
    "dermoscopic images using a custom "
    "U-Net segmentation model."
)

st.divider()


# ============================================================
# MODEL OVERVIEW
# ============================================================

st.subheader(
    "Model Overview"
)

overview_col1, overview_col2, overview_col3, overview_col4 = (
    st.columns(4)
)

with overview_col1:

    st.metric(
        "Architecture",
        "U-Net",
    )

    st.caption(
        "Encoder-decoder network "
        "with skip connections."
    )


with overview_col2:

    st.metric(
        "Input",
        "256 × 256",
    )

    st.caption(
        "RGB dermoscopic image."
    )


with overview_col3:

    st.metric(
        "Test Dice",
        "78.75%",
    )

    st.caption(
        "Held-out test-set "
        "segmentation overlap."
    )


with overview_col4:

    st.metric(
        "Test IoU",
        "68.72%",
    )

    st.caption(
        "Intersection-over-Union "
        "on test images."
    )


st.divider()


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.subheader(
    "Image Analysis"
)

st.write(
    "Upload a dermoscopic image to generate "
    "a pixel-level lesion segmentation mask "
    "and visual overlay."
)


uploaded_file = st.file_uploader(
    "Upload dermoscopic image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    key="lesion_image_uploader",
    help=(
        "Supported formats: JPG, JPEG and PNG. "
        "The model processes the image at "
        "256 × 256 resolution."
    ),
)


# ============================================================
# STORE UPLOADED IMAGE
# ============================================================

# If a new file has just been uploaded,
# immediately copy its bytes into session state.

if uploaded_file is not None:

    try:

        current_bytes = uploaded_file.getvalue()

        if current_bytes:

            st.session_state.uploaded_image_bytes = (
                current_bytes
            )

            st.session_state.uploaded_image_name = (
                uploaded_file.name
            )

            # New upload means the user has a new
            # image that should be analyzed.
            st.session_state.analysis_requested = True

    except Exception as exc:

        st.error(
            "The uploaded file could not be read."
        )

        with st.expander(
            "Technical error details"
        ):

            st.exception(exc)


# ============================================================
# DETERMINE WHETHER AN IMAGE EXISTS
# ============================================================

has_uploaded_image = (
    st.session_state.uploaded_image_bytes
    is not None
    and len(
        st.session_state.uploaded_image_bytes
    ) > 0
)


# ============================================================
# READY STATE
# ============================================================

if not has_uploaded_image:

    st.info(
        "Ready for analysis. Upload a "
        "dermoscopic image above to begin "
        "segmentation."
    )


# ============================================================
# ANALYSIS
# ============================================================

else:

    # --------------------------------------------------------
    # Show uploaded image information.
    # --------------------------------------------------------

    st.success(
        f"Image uploaded successfully: "
        f"{st.session_state.uploaded_image_name}"
    )

    st.caption(
        "The image has been received by the application "
        "and is ready for segmentation."
    )


    # --------------------------------------------------------
    # Analyze button.
    #
    # This makes the inference step explicit and prevents
    # accidental repeated model inference during Streamlit
    # widget reruns.
    # --------------------------------------------------------

    analyze_button = st.button(
        "Analyze Image",
        type="primary",
        use_container_width=False,
    )


    if analyze_button:

        st.session_state.analysis_requested = True


    # --------------------------------------------------------
    # Only run inference after an image exists.
    # --------------------------------------------------------

    if st.session_state.analysis_requested:

        try:

            # =================================================
            # LOAD MODEL
            # =================================================

            with st.spinner(
                "Loading the segmentation model..."
            ):

                model = load_segmentation_model()


            # =================================================
            # READ IMAGE FROM SESSION STATE
            # =================================================

            image_bytes = (
                st.session_state.uploaded_image_bytes
            )

            if image_bytes is None:

                raise RuntimeError(
                    "The uploaded image data is no longer "
                    "available in the current session."
                )


            image = Image.open(
                io.BytesIO(image_bytes)
            )

            image.load()

            (
                original_image,
                resized_image,
                model_input,
            ) = preprocess_image(
                image
            )


            # =================================================
            # RUN INFERENCE
            # =================================================

            with st.spinner(
                "Analyzing the lesion region..."
            ):

                (
                    probability,
                    prediction_mask,
                ) = predict_segmentation(
                    model,
                    model_input,
                )


            # =================================================
            # RESIZE MASK
            # =================================================

            original_mask = (
                resize_mask_to_original(
                    prediction_mask,
                    image.size,
                )
            )


            # =================================================
            # CREATE OVERLAY
            # =================================================

            overlay = create_overlay(
                original_image,
                original_mask,
            )


            # =================================================
            # CALCULATE STATISTICS
            # =================================================

            (
                lesion_percentage,
                mean_confidence,
            ) = calculate_statistics(
                probability,
                prediction_mask,
            )


            # =================================================
            # SUCCESS MESSAGE
            # =================================================

            st.success(
                "Segmentation completed successfully. "
                "The U-Net generated a binary lesion mask."
            )


            # =================================================
            # RESULTS
            # =================================================

            st.subheader(
                "Segmentation Results"
            )

            result_col1, result_col2, result_col3 = (
                st.columns(
                    3,
                    gap="large",
                )
            )


            with result_col1:

                st.markdown(
                    "**Input Image**"
                )

                st.image(
                    original_image,
                    use_container_width=True,
                )


            with result_col2:

                st.markdown(
                    "**Predicted Lesion Mask**"
                )

                st.image(
                    original_mask.astype(
                        np.uint8
                    ) * 255,
                    use_container_width=True,
                )


            with result_col3:

                st.markdown(
                    "**Segmentation Overlay**"
                )

                st.image(
                    overlay,
                    use_container_width=True,
                )


            # =================================================
            # ANALYSIS SUMMARY
            # =================================================

            st.subheader(
                "Analysis Summary"
            )

            (
                metric_col1,
                metric_col2,
                metric_col3,
                metric_col4,
            ) = st.columns(4)


            with metric_col1:

                st.metric(
                    "Estimated lesion area",
                    f"{lesion_percentage:.2f}%",
                )


            with metric_col2:

                st.metric(
                    "Mean lesion confidence",
                    f"{mean_confidence:.1f}%",
                )


            with metric_col3:

                st.metric(
                    "Segmentation threshold",
                    f"{THRESHOLD:.2f}",
                )


            with metric_col4:

                st.metric(
                    "Model output",
                    "Binary mask",
                )


            st.caption(
                "Estimated lesion area is the percentage "
                "of 256 × 256 model-input pixels classified "
                "as lesion. Mean lesion confidence is the "
                "average predicted lesion probability over "
                "pixels classified as lesion."
            )


            # =================================================
            # PROBABILITY MAP
            # =================================================

            with st.expander(
                "View model probability map"
            ):

                st.write(
                    "Each pixel represents the model's "
                    "estimated probability of belonging "
                    "to the lesion region. Higher values "
                    "indicate stronger model confidence."
                )

                st.image(
                    probability,
                    clamp=True,
                    channels="L",
                    caption=(
                        "Lesion probability map"
                    ),
                    use_container_width=True,
                )


            # =================================================
            # EXPORT
            # =================================================

            st.subheader(
                "Export"
            )

            mask_image = Image.fromarray(
                (
                    original_mask.astype(
                        np.uint8
                    )
                    * 255
                )
            )

            buffer = io.BytesIO()

            mask_image.save(
                buffer,
                format="PNG",
            )

            st.download_button(
                label=(
                    "Download segmentation mask"
                ),
                data=buffer.getvalue(),
                file_name=(
                    "skin_lesion_segmentation_mask.png"
                ),
                mime="image/png",
            )


            # =================================================
            # TECHNICAL DETAILS
            # =================================================

            with st.expander(
                "Technical details"
            ):

                st.write(
                    f"Original image size: "
                    f"{image.size[0]} × "
                    f"{image.size[1]}"
                )

                st.write(
                    "Model input size: "
                    "256 × 256 × 3"
                )

                st.write(
                    "Output type: "
                    "Binary lesion mask"
                )

                st.write(
                    f"Decision threshold: "
                    f"{THRESHOLD:.2f}"
                )

                st.write(
                    "Model: Custom U-Net"
                )

                st.write(
                    "Dataset: ISIC 2016 Task 1"
                )


        except Exception as exc:

            st.error(
                "The image could not be processed."
            )

            st.warning(
                "The application received the image, "
                "but an error occurred during model loading "
                "or inference."
            )

            with st.expander(
                "Technical error details",
                expanded=True,
            ):

                st.exception(exc)


# ============================================================
# RESEARCH DISCLAIMER
# ============================================================

st.divider()

st.warning(
    "**Research Use Only.** This application is "
    "an educational and research demonstration "
    "of an AI-based skin lesion segmentation "
    "model. It does not diagnose melanoma, "
    "cancer, or any other medical condition and "
    "must not be used as a substitute for "
    "evaluation by a qualified healthcare "
    "professional. Model predictions can contain "
    "segmentation errors, especially for small, "
    "low-contrast, irregular, or artifact-affected "
    "lesions."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-text">'
    "AI-Assisted Medical Image Analysis · "
    "Skin Lesion Segmentation · "
    "ISIC 2016 Task 1 · U-Net"
    "</div>",
    unsafe_allow_html=True,
)
