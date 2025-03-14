import streamlit as st
import os
import time
import gc
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means
from io import BytesIO
import base64
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Image Compression & Upscaling",
    page_icon="🖼️",
    layout="wide"
)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'intro'  # Default to intro page

# Function to switch to main page
def switch_to_main():
    st.session_state.page = 'main'

# Functions for image processing
def compress_image(image, quality):
    """Compress an image to the specified JPEG quality"""
    img_pil = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    buffer = BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    compressed_img = Image.open(buffer)
    return np.array(compressed_img)

def get_basic_upscaled(image, method, scale_factor):
    """Apply basic upscaling method to an image"""
    new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
    
    if method == "nearest":
        return cv2.resize(image, new_size, interpolation=cv2.INTER_NEAREST)
    elif method == "bilinear":
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    elif method == "bicubic":
        return cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
    elif method == "lanczos":
        return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
    else:
        raise ValueError(f"Unsupported method: {method}")

def get_edi_upscaled(image, scale_factor):
    """Apply Edge-Directed Interpolation to an image"""
    new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
    edge_directed = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    with st.spinner("Applying Edge-Directed Interpolation..."):
        edge_directed = denoise_tv_chambolle(edge_directed, weight=0.1, channel_axis=-1)
    return (edge_directed * 255).astype(np.uint8)

def hybrid_upscaler(compressed_img, scale_factor, esrgan_weight=0.4, bicubic_weight=0.6, detail_threshold=30):
    """
    Hybrid upscaling method combining ESRGAN and Bicubic for optimal PSNR and SSIM
    
    Args:
        compressed_img: Input compressed image (numpy array)
        scale_factor: Desired scaling factor
        esrgan_weight: Weight for ESRGAN contribution (0-1)
        bicubic_weight: Weight for Bicubic contribution (0-1)
        detail_threshold: Threshold for edge detection (higher = fewer edges detected)
        
    Returns:
        Hybrid upscaled image
    """
    # Convert to PIL for ESRGAN
    compressed_pil = Image.fromarray(compressed_img)
    
    # Get bicubic upscaled version
    bicubic_upscaled = get_basic_upscaled(compressed_img, "bicubic", scale_factor)
    
    # Get ESRGAN upscaled version
    upscaler = ESRGANUpscaler()
    esrgan_result = upscaler.upscale(
        compressed_pil,
        patch_size=128,
        max_size=None,  # Don't limit size for better quality
        attempt_whole_image=False  # Always use patch-based for consistency
    )
    esrgan_np = np.array(esrgan_result)
    
    # Resize ESRGAN result to match bicubic dimensions if needed
    if esrgan_np.shape[:2] != bicubic_upscaled.shape[:2]:
        h, w = bicubic_upscaled.shape[:2]
        esrgan_np = cv2.resize(esrgan_np, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create an edge mask to adaptively blend between methods
    gray = cv2.cvtColor(compressed_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, detail_threshold, detail_threshold*2)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=2)
    
    # Resize edge mask to match output dimensions
    edge_mask = cv2.resize(edges, (bicubic_upscaled.shape[1], bicubic_upscaled.shape[0]), 
                          interpolation=cv2.INTER_NEAREST)
    
    # Normalize mask to 0-1 range
    edge_mask = edge_mask.astype(float) / 255.0
    
    # Create a blending mask: higher ESRGAN weight in edge areas, higher Bicubic in smooth areas
    blend_mask = np.expand_dims(edge_mask, axis=2)
    blend_mask = np.repeat(blend_mask, 3, axis=2)
    
    # Calculate dynamic weights based on edge mask
    dynamic_esrgan_weight = esrgan_weight + blend_mask * 0.3  # Increase ESRGAN in edge areas
    dynamic_bicubic_weight = 1.0 - dynamic_esrgan_weight
    
    # Blend images with dynamic weights
    hybrid_result = (dynamic_esrgan_weight * esrgan_np.astype(float) + 
                    dynamic_bicubic_weight * bicubic_upscaled.astype(float))
    
    # Optional: Apply a final detail enhancement
    hybrid_result = cv2.detailEnhance(hybrid_result.astype(np.uint8), sigma_s=0.5, sigma_r=0.1)
    
    # Optional: Apply a mild sharpening filter
    kernel = np.array([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])
    hybrid_result = cv2.filter2D(hybrid_result, -1, kernel)
    
    return hybrid_result.astype(np.uint8)

def get_nlm_upscaled(image, scale_factor):
    """Apply Non-Local Means upscaling to an image"""
    with st.spinner("Applying Non-Local Means denoising..."):
        sigma_est = np.mean(np.std(image, axis=(0, 1)))
        nlm_denoised = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, 
                                      patch_size=5, patch_distance=6, channel_axis=-1)
    new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
    nlm_upscaled = cv2.resize(nlm_denoised, new_size, interpolation=cv2.INTER_LINEAR)
    return (nlm_upscaled * 255).astype(np.uint8)

def get_downloadable_img(img_array, filename):
    """Create a downloadable link for an image"""
    img_pil = Image.fromarray(img_array) if isinstance(img_array, np.ndarray) else img_array
    buffer = BytesIO()
    img_pil.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}.png">Download {filename}</a>'

# def get_image_size_str(img):
#     """Get image dimensions and file size as string"""
#     if isinstance(img, np.ndarray):
#         height, width = img.shape[:2]
#         # Convert to PIL for file size estimation
#         img_pil = Image.fromarray(img)
#     else:
#         # Assume it's a PIL image
#         width, height = img.size
#         img_pil = img
        
#     # Estimate file size
#     buffer = BytesIO()
#     img_pil.save(buffer, format="PNG")
#     size_bytes = buffer.getbuffer().nbytes
    
#     # Convert to appropriate unit
#     if size_bytes < 1024:
#         size_str = f"{size_bytes} bytes"
#     elif size_bytes < 1024 * 1024:
#         size_str = f"{size_bytes / 1024:.2f} KB"
#     else:
#         size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        
#     return f"{width}x{height} pixels, {size_str}"



def get_image_size_str(img, original_file=None):
    """Get image dimensions and file size as string"""
    if isinstance(img, np.ndarray):
        height, width = img.shape[:2]
    else:
        # Assume it's a PIL image
        width, height = img.size
    
    # Use original file size if provided
    if original_file is not None:
        original_file.seek(0, os.SEEK_END)
        size_bytes = original_file.tell()
        original_file.seek(0)
    else:
        # Estimate with a buffer
        buffer = BytesIO()
        if isinstance(img, np.ndarray):
            Image.fromarray(img).save(buffer, format="JPEG", quality=95)
        else:
            img.save(buffer, format="JPEG", quality=95)
        size_bytes = buffer.getbuffer().nbytes
    
    # Convert to appropriate unit
    if size_bytes < 1024:
        size_str = f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.2f} KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        
    return f"{width}x{height} pixels, {size_str}"

def calculate_image_metrics(original, upscaled):
    """Calculate PSNR and SSIM between original and upscaled image"""
    # Resize original to match upscaled dimensions for fair comparison
    if isinstance(original, np.ndarray):
        h_up, w_up = upscaled.shape[:2]
        original_resized = cv2.resize(original, (w_up, h_up), interpolation=cv2.INTER_CUBIC)
    else:
        # Convert PIL to numpy if needed
        original_np = np.array(original)
        upscaled_np = np.array(upscaled)
        h_up, w_up = upscaled_np.shape[:2]
        original_resized = cv2.resize(original_np, (w_up, h_up), interpolation=cv2.INTER_CUBIC)
        
    # Calculate metrics
    try:
        psnr_value = psnr(original_resized, upscaled)
        ssim_value = ssim(original_resized, upscaled, channel_axis=-1 if len(upscaled.shape) > 2 else None)
        return {
            "PSNR": f"{psnr_value:.2f} dB",
            "SSIM": f"{ssim_value:.4f}"
        }
    except Exception as e:
        return {
            "PSNR": "Error",
            "SSIM": "Error",
            "Error": str(e)
        }

# Import the ESRGAN upscaler class
class ESRGANUpscaler:
    """
    Image upscaler using ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)
    with memory-efficient patch-based processing for high-resolution images.
    """

    def __init__(self, model_url="https://tfhub.dev/captain-pool/esrgan-tf2/1"):
        """
        Initialize the ESRGAN upscaler.

        Args:
            model_url: URL or path to the ESRGAN TensorFlow Hub model
        """
        self.model_url = model_url
        self.model = None
        self.scale_factor = 4  # ESRGAN typically upscales 4x

        # Configure TensorFlow for memory efficiency
        self._configure_tensorflow()

    def _configure_tensorflow(self):
        """Configure TensorFlow for memory efficiency"""
        os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"
        
        # Enable memory growth to avoid allocating all GPU memory at once
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"Memory growth enabled on {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"Error setting memory growth: {e}")

    def load_model(self):
        """Load the ESRGAN model from TensorFlow Hub"""
        with st.spinner("Loading ESRGAN model..."):
            start_time = time.time()
            self.model = hub.load(self.model_url)
            print(f"Model loaded in {time.time() - start_time:.2f} seconds")

            # Test the model to determine actual scale factor
            test_tensor = tf.zeros((1, 16, 16, 3))
            output = self.model(test_tensor)
            if output.shape[1] // 16 == 4:
                self.scale_factor = 4
                print(f"Detected scale factor: {self.scale_factor}x")
            else:
                self.scale_factor = output.shape[1] // 16
                print(f"Detected custom scale factor: {self.scale_factor}x")

    def preprocess_image(self, image, max_size=None):
        """
        Load and preprocess image with optional resizing for memory efficiency

        Args:
            image: PIL Image or path to the image file
            max_size: Maximum dimension (width or height) to resize to

        Returns:
            PIL Image of the preprocessed image
        """
        # Load image
        if isinstance(image, str):
            img = Image.open(image)
            print(f"Original image size: {img.size}")
        else:
            # Assume it's already a PIL Image
            img = image

        # Resize if specified and image is larger than max_size
        if max_size is not None and max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            print(f"Resized image to {new_size}")

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        return img

    def _process_image_tensor(self, image_tensor):
        """
        Process a single image tensor through the model

        Args:
            image_tensor: TensorFlow tensor of the image

        Returns:
            Output tensor from the ESRGAN model
        """
        if self.model is None:
            self.load_model()

        # Add batch dimension if needed
        if len(image_tensor.shape) == 3:
            image_tensor = tf.expand_dims(image_tensor, 0)

        # Process through model
        output = self.model(image_tensor)
        output = tf.squeeze(output)

        return output

    def process_whole_image(self, pil_image):
        """
        Process the entire image at once

        Args:
            pil_image: PIL Image to process

        Returns:
            Super-resolution PIL Image
        """
        if self.model is None:
            self.load_model()

        # Convert PIL Image to tensor
        img_array = np.array(pil_image)
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Process image
        with st.spinner("Processing image with ESRGAN..."):
            start_time = time.time()
            output_tensor = self._process_image_tensor(img_tensor)
            print(f"Processed whole image in {time.time() - start_time:.2f} seconds")

        # Convert back to PIL Image
        output_array = tf.cast(tf.clip_by_value(output_tensor, 0, 255), tf.uint8).numpy()
        output_image = Image.fromarray(output_array)

        return output_image

    def process_in_patches(self, pil_image, patch_size=192, overlap=24):
        """
        Process the image in patches to reduce memory usage

        Args:
            pil_image: PIL Image to process
            patch_size: Size of patches to process
            overlap: Overlap between patches to avoid boundary artifacts

        Returns:
            Super-resolution PIL Image
        """
        if self.model is None:
            self.load_model()

        # Get image dimensions
        width, height = pil_image.size

        # Calculate dimensions of the output (upscaled) image
        output_width = width * self.scale_factor
        output_height = height * self.scale_factor

        # Create an empty output image
        output_image = Image.new('RGB', (output_width, output_height))

        # Create a mask for blending overlapping patches
        mask = Image.new('L', (patch_size * self.scale_factor, patch_size * self.scale_factor), 255)
        mask_draw = ImageDraw.Draw(mask)

        # Create gradient edges for the mask to blend overlapping regions
        for i in range(overlap * self.scale_factor):
            # Calculate alpha value for gradient (0 at edge, 255 at overlap distance)
            alpha = int(255 * i / (overlap * self.scale_factor))

            # Draw gradient on all four edges
            mask_draw.rectangle(
                [i, i, patch_size * self.scale_factor - i - 1, patch_size * self.scale_factor - i - 1],
                outline=alpha
            )

        # Convert mask to numpy array for efficient blending
        mask_np = np.array(mask)

        # Initialize a weight map for blending
        weight_map = np.zeros((output_height, output_width), dtype=np.float32)
        output_array = np.zeros((output_height, output_width, 3), dtype=np.float32)

        # Process patches
        total_patches = ((height - 1) // (patch_size - overlap) + 1) * ((width - 1) // (patch_size - overlap) + 1)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        patch_count = 0
        for y in range(0, height, patch_size - overlap):
            for x in range(0, width, patch_size - overlap):
                # Calculate actual patch size (may be smaller at edges)
                h = min(patch_size, height - y)
                w = min(patch_size, width - x)

                # Skip if patch is too small
                if h < 16 or w < 16:
                    patch_count += 1
                    continue

                # Update progress
                patch_count += 1
                progress = int(100 * patch_count / total_patches)
                progress_bar.progress(progress)
                status_text.text(f"Processing patch {patch_count}/{total_patches} ({progress}%)")

                # Extract patch as PIL Image
                patch = pil_image.crop((x, y, x + w, y + h))

                # Make dimensions divisible by 4 (required by ESRGAN)
                patch_width, patch_height = patch.size
                if patch_width % 4 != 0 or patch_height % 4 != 0:
                    new_w = (patch_width // 4) * 4
                    new_h = (patch_height // 4) * 4
                    if new_w == 0:
                        new_w = 4
                    if new_h == 0:
                        new_h = 4
                    patch = patch.resize((new_w, new_h), Image.LANCZOS)

                try:
                    # Convert to tensor
                    patch_array = np.array(patch)
                    patch_tensor = tf.convert_to_tensor(patch_array, dtype=tf.float32)

                    # Clear memory
                    tf.keras.backend.clear_session()
                    gc.collect()

                    # Process patch
                    output_tensor = self._process_image_tensor(patch_tensor)
                    output_patch_array = output_tensor.numpy()

                    # Calculate output coordinates (scaled)
                    out_x = x * self.scale_factor
                    out_y = y * self.scale_factor
                    out_w = output_patch_array.shape[1]
                    out_h = output_patch_array.shape[0]

                    # Create mask for this specific patch size
                    if out_w != patch_size * self.scale_factor or out_h != patch_size * self.scale_factor:
                        # Create custom sized mask
                        patch_mask = np.ones((out_h, out_w), dtype=np.float32)
                        # Apply gradient to edges
                        fade_dist = min(overlap * self.scale_factor, out_w // 4, out_h // 4)
                        for i in range(fade_dist):
                            factor = i / fade_dist
                            # Top and bottom edges
                            patch_mask[i, :] *= factor
                            if i < out_h - fade_dist:
                                patch_mask[out_h - i - 1, :] *= factor
                            # Left and right edges
                            patch_mask[:, i] *= factor
                            if i < out_w - fade_dist:
                                patch_mask[:, out_w - i - 1] *= factor
                    else:
                        # Use pre-computed mask
                        patch_mask = mask_np[:out_h, :out_w] / 255.0

                    # Expand mask to 3 channels
                    patch_mask_3d = np.expand_dims(patch_mask, axis=2)
                    patch_mask_3d = np.repeat(patch_mask_3d, 3, axis=2)

                    # Apply mask to output patch
                    masked_patch = output_patch_array * patch_mask_3d

                    # Add to output array
                    out_y_end = min(out_y + out_h, output_height)
                    out_x_end = min(out_x + out_w, output_width)

                    # Adjust patch size if it would exceed output dimensions
                    if out_y_end - out_y < out_h or out_x_end - out_x < out_w:
                        masked_patch = masked_patch[:(out_y_end - out_y), :(out_x_end - out_x), :]
                        patch_mask = patch_mask[:(out_y_end - out_y), :(out_x_end - out_x)]

                    # Update output and weight map
                    output_array[out_y:out_y_end, out_x:out_x_end] += masked_patch
                    weight_map[out_y:out_y_end, out_x:out_x_end] += patch_mask

                except tf.errors.ResourceExhaustedError:
                    st.warning(f"Out of memory with patch size {patch_size}. Reducing patch size.")
                    if patch_size <= 64:
                        st.error("Cannot reduce patch size further. Skipping this patch.")
                    else:
                        # Try again with smaller patch size
                        new_patch_size = patch_size // 2
                        new_overlap = max(16, overlap // 2)

                        # Create sub-image for this region
                        sub_image = pil_image.crop((x, y, x + w, y + h))

                        # Process with smaller patches
                        sub_result = self.process_in_patches(
                            sub_image,
                            patch_size=new_patch_size,
                            overlap=new_overlap
                        )

                        # Insert result into output
                        sub_width, sub_height = sub_result.size
                        output_image.paste(sub_result, (out_x, out_y))

                        # Update weight map (approximately)
                        sub_weight = np.ones((sub_height, sub_width))
                        weight_map[out_y:out_y+sub_height, out_x:out_x+sub_width] += sub_weight

                except Exception as e:
                    st.error(f"Error processing patch at ({x}, {y}): {e}")

        # Clear the progress bar
        progress_bar.empty()
        status_text.empty()

        # Normalize by weights
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-6
        weight_map = np.expand_dims(weight_map + epsilon, axis=2)
        weight_map = np.repeat(weight_map, 3, axis=2)
        normalized_output = output_array / weight_map

        # Convert back to PIL Image
        output_image = Image.fromarray(np.uint8(np.clip(normalized_output, 0, 255)))

        return output_image

    def upscale(self, image, patch_size=None, max_size=None, attempt_whole_image=True):
        """
        Upscale an image with ESRGAN

        Args:
            image: PIL Image or path to the input image
            patch_size: Size of patches to process (None = auto-determine)
            max_size: Maximum dimension for resizing input
            attempt_whole_image: Whether to try processing the whole image first

        Returns:
            Upscaled PIL Image
        """
        # Preprocess image
        pil_image = self.preprocess_image(image, max_size=max_size)

        # Try processing the whole image first if requested
        if attempt_whole_image:
            try:
                st.info("Attempting to process the whole image...")
                upscaled_image = self.process_whole_image(pil_image)
                st.success("Successfully processed the whole image")
            except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
                st.warning(f"Processing whole image failed: {e}")
                st.info("Falling back to patch-based processing")

                # Auto-determine patch size based on image dimensions and memory
                if patch_size is None:
                    w, h = pil_image.size
                    total_pixels = w * h
                    # Adjust patch size based on image size
                    if total_pixels > 1000000:  # 1 megapixel
                        patch_size = 128
                    elif total_pixels > 500000:  # 0.5 megapixel
                        patch_size = 192
                    elif total_pixels > 250000:  # 0.25 megapixel
                        patch_size = 256
                    else:
                        patch_size = 384

                st.info(f"Using patch size: {patch_size}")
                upscaled_image = self.process_in_patches(pil_image, patch_size=patch_size)
        else:
            # Use patch-based processing directly
            if patch_size is None:
                patch_size = 192  # Default patch size
            st.info(f"Using patch-based processing with patch size: {patch_size}")
            upscaled_image = self.process_in_patches(pil_image, patch_size=patch_size)

        return upscaled_image


# Introduction Page
if st.session_state.page == 'intro':
    st.title("Contrasting Traditional Upscaling and GAN-Based Restoration for Compressed Image Enhancement")
    
    # Project Team Information
    st.header("Project Team")
    team_data = [
        {"Name": "Suvodip Som", "Roll Number": "M23CSA533"},
        {"Name": "Anindya Bandyopadhyay", "Roll Number": "M23CSA508"},
        {"Name": "Swapnil Adak", "Roll Number": "M23CSA534"},
        {"Name": "Tushar Kumar Tiwari", "Roll Number": "M23CSA537"}
    ]
    
    # Create a DataFrame for better display
    team_df = pd.DataFrame(team_data)
    st.table(team_df)
    
    # Project Introduction
    st.header("Project Overview")
    st.markdown("""
    This project explores and compares various image upscaling techniques to enhance compressed images. 
    We examine both traditional interpolation methods and modern deep learning-based approaches, 
    specifically focusing on the differences in quality, artifacts, and overall visual fidelity.
    
    The tool allows you to:
    1. Compress an image to a specified quality level
    2. Apply various upscaling methods
    3. Compare results side-by-side
    4. Analyze image quality metrics
    """)
    
    # Upscaling Methods Explanation
    st.header("Upscaling Methods Explained")
    
    methods_info = [
        {
            "name": "Nearest Neighbor Interpolation",
            "description": """
            The simplest method that assigns each new pixel the value of the nearest original pixel. 
            This preserves sharp edges but creates blocky artifacts.
            
            **Mathematical representation**: For a pixel at position (x, y) in the upscaled image:
            
            f(x, y) = f(round(x/s), round(y/s))
            
            where s is the scaling factor.
            """
        },
        {
            "name": "Bilinear Interpolation",
            "description": """
            Calculates new pixel values by linear interpolation between four surrounding original pixels.
            This creates smoother results than nearest neighbor but can blur sharp edges.
            
            **Mathematical representation**: For a pixel at position (x, y) between four known pixels:
            
            f(x, y) = f(x₁, y₁)(1-a)(1-b) + f(x₂, y₁)a(1-b) + f(x₁, y₂)(1-a)b + f(x₂, y₂)ab
            
            where a = x - x₁ and b = y - y₁ are the fractional distances.
            """
        },
        {
            "name": "Bicubic Interpolation",
            "description": """
            Uses cubic polynomial interpolation on 16 surrounding original pixels (4×4 neighborhood).
            Produces smoother edges than bilinear while preserving more details.
            
            **Mathematical representation**: Uses a cubic kernel function:
            
            K(x) = { 1.5|x|³ - 2.5|x|² + 1, if |x| ≤ 1
                   { -0.5|x|³ + 2.5|x|² - 4|x| + 2, if 1 < |x| < 2
                   { 0, otherwise
            """
        },
        {
            "name": "Lanczos Interpolation",
            "description": """
            Uses a windowed sinc function as the interpolation kernel with a larger sampling area.
            Often produces the best results among traditional methods for photographic images.
            
            **Mathematical representation**: Uses the Lanczos kernel:
            
            L(x) = { sinc(x)·sinc(x/a), if -a < x < a
                   { 0, otherwise
                   
            where sinc(x) = sin(πx)/(πx) and a is typically 2 or 3.
            """
        },
        {
            "name": "Edge-Directed Interpolation (EDI)",
            "description": """
            An advanced technique that attempts to preserve edges by analyzing edge directions.
            This method uses total variation regularization to enhance edges while reducing artifacts.
            
            We implement this using the Total Variation Chambolle algorithm which minimizes the functional:
            
            E(u) = ∫|∇u| + λ/2 ∫(u-f)²
            
            where f is the original image, u is the result, and λ controls the smoothness.
            """
        },
        {
            "name": "Non-Local Means (NLM)",
            "description": """
            An advanced denoising algorithm that preserves image structure by averaging similar patches.
            Unlike local filters, it can find similar patterns across the entire image.
            
            **Mathematical representation**: The NLM algorithm calculates each pixel value as:
            
            NL[f](x) = ∑ w(x,y)f(y)
                       y∈Ω
                       
            where w(x,y) is a weight based on the similarity between patches centered at x and y.
            """
        },
        {
            "name": "ESRGAN (Enhanced Super-Resolution GAN)",
            "description": """
            A deep learning approach using Generative Adversarial Networks (GANs) specifically designed for super-resolution.
            ESRGAN introduces several improvements over earlier SRGAN:
            
            1. Residual-in-Residual Dense Block (RRDB) architecture
            2. Removal of Batch Normalization for better image quality
            3. Improved adversarial loss using Relativistic GAN
            4. Perceptual loss using VGG features
            
            The model is trained to generate photorealistic high-resolution images from low-resolution inputs.
            
            This method typically provides the most detailed and natural-looking results, especially for complex textures.
            """
        }
    ]
    
    # Display each method with expandable details
    for method in methods_info:
        with st.expander(method["name"]):
            st.markdown(method["description"])
    
    # Quality Metrics Explanation
    st.header("Image Quality Metrics")
    st.markdown("""
    We use the following metrics to quantitatively evaluate the upscaling methods:
    
    1. **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible power of a signal and the power of corrupting noise.
       - Higher values indicate better quality
       - Typical values for good quality images range from 30-50 dB
       - Formula: PSNR = 10 · log₁₀(MAX²/MSE) where MAX is the maximum pixel value and MSE is the mean squared error
    
    2. **SSIM (Structural Similarity Index)**: Measures the similarity between two images based on luminance, contrast, and structure.
       - Ranges from -1 to 1, with 1 indicating perfect similarity
       - More aligned with human perception than PSNR
       - Considers local patterns of pixel intensities rather than just intensity differences
    
    3. **File Size**: The compressed and upscaled image sizes are reported to evaluate storage efficiency.
    """)
    
    # Expected Results
    st.header("Expected Results")
    st.markdown("""
    - **Nearest Neighbor**: Fastest but lowest quality with visible blocky artifacts
    - **Bilinear/Bicubic**: Better quality with smoother transitions but may blur fine details
    - **Lanczos**: Generally better preservation of details than other traditional methods
    - **EDI**: Better edge preservation but may introduce ringing artifacts
    - **NLM**: Good detail preservation but may over-smooth textures
    - **ESRGAN**: Best overall visual quality with natural details, but most computationally intensive
    
    The best method depends on your specific needs:
    - For fast, low-complexity upscaling: Use bilinear or bicubic
    - For best visual quality with moderate speed: Use Lanczos
    - For highest possible quality (and sufficient computing resources): Use ESRGAN
    """)
    
    # Continue button
    st.button("Continue to the Application", on_click=switch_to_main)

# Main Application Page
elif st.session_state.page == 'main':
    # Define header
    st.title("Main Contrastive Testing Pipeline")
    st.markdown("""
                
    Upload an image to:
    1. Compress it to a specified quality level
    2. Upscale it using various methods including ESRGAN
    """)

    # Sidebar for parameters
    with st.sidebar:
        st.header("Settings")
        
        # Compression settings
        st.subheader("Compression")
        compression_quality = st.slider("JPEG Compression Quality", 1, 100, 10, 
                                    help="Lower values = smaller file size but lower quality")
        
        # Upscaling settings
        st.subheader("Upscaling")
        scale_factor = st.slider("Scale Factor", 2, 4, 3, 
                                help="How much to enlarge the image")
        
        # Method selection
        st.subheader("Methods")
        use_nearest = st.checkbox("Nearest Neighbor", value=True)
        use_bilinear = st.checkbox("Bilinear", value=True)
        use_bicubic = st.checkbox("Bicubic", value=True)
        use_lanczos = st.checkbox("Lanczos", value=True)
        use_edi = st.checkbox("Edge-Directed (EDI)", value=True)
        use_nlm = st.checkbox("Non-Local Means (NLM)", value=True)
        use_esrgan = st.checkbox("ESRGAN (AI-based)", value=False, 
                                help="High-quality but slower and more memory-intensive")

        
        # ESRGAN settings (only shown if ESRGAN is selected)
        if use_esrgan:
            st.subheader("ESRGAN Settings")
            esrgan_patch_size = st.select_slider(
                "Patch Size", 
                options=[64, 96, 128, 192, 256], 
                value=128,
                help="Smaller values use less memory but may be slower overall"
            )
            esrgan_max_size = st.slider(
                "Max Input Size", 
                256, 1024, 512, 
                help="Larger input images will be resized to this size before processing"
            )
        
        # Buttons
        st.subheader("Actions")
        clear_cache = st.button("Clear Cache")
        if clear_cache:
            # Clear TensorFlow session and garbage collect
            tf.keras.backend.clear_session()
            gc.collect()
            st.success("Cache cleared!")
            
        # Add a button to go back to the introduction page
        if st.button("Back to Project Introduction"):
            st.session_state.page = 'intro'
            st.experimental_rerun()
            
    # Main application flow
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        # Load the uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        
        # Display the original image with size information
        st.subheader("Original Image")
        original_size = get_image_size_str(image)
        st.image(image, caption=f"Original Image - {original_size}", use_column_width=True)
        
        # Compress the image
        with st.spinner(f"Compressing image to {compression_quality}% quality..."):
            compressed_img = compress_image(image_np, compression_quality)
        
        # Display the compressed image with size info
        compressed_size = get_image_size_str(compressed_img)
        st.subheader(f"Compressed Image ({compression_quality}% Quality)")
        st.image(compressed_img, caption=f"Compressed to {compression_quality}% Quality - {compressed_size}", use_column_width=True)
        
        # Provide a download link for the compressed image
        st.markdown(get_downloadable_img(compressed_img, "compressed_image"), unsafe_allow_html=True)
        
        # Upscaling section
        st.subheader(f"Upscaled Images ({scale_factor}x)")
        
        # Create dictionaries to store all upscaled images and metrics
        upscaled_images = {}
        metrics_data = []
        
        # Apply the selected upscaling methods
        col1, col2 = st.columns(2)
        
        # Basic upscaling methods
        if use_nearest:
            with col1:
                nearest = get_basic_upscaled(compressed_img, "nearest", scale_factor)
                upscaled_images["Nearest Neighbor"] = nearest
                nearest_size = get_image_size_str(nearest)
                st.image(nearest, caption=f"Nearest Neighbor - {nearest_size}", use_column_width=True)
                st.markdown(get_downloadable_img(nearest, "nearest_neighbor"), unsafe_allow_html=True)
                
                # Calculate metrics
                metrics = calculate_image_metrics(image_np, nearest)
                metrics_data.append({
                    "Method": "Nearest Neighbor",
                    "Input Size": compressed_size,
                    "Output Size": nearest_size,
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"]
                })
        
        if use_bilinear:
            with col2:
                bilinear = get_basic_upscaled(compressed_img, "bilinear", scale_factor)
                upscaled_images["Bilinear"] = bilinear
                bilinear_size = get_image_size_str(bilinear)
                st.image(bilinear, caption=f"Bilinear - {bilinear_size}", use_column_width=True)
                st.markdown(get_downloadable_img(bilinear, "bilinear"), unsafe_allow_html=True)
                
                # Calculate metrics
                metrics = calculate_image_metrics(image_np, bilinear)
                metrics_data.append({
                    "Method": "Bilinear",
                    "Input Size": compressed_size,
                    "Output Size": bilinear_size,
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"]
                })
        
        if use_bicubic:
            with col1:
                bicubic = get_basic_upscaled(compressed_img, "bicubic", scale_factor)
                upscaled_images["Bicubic"] = bicubic
                bicubic_size = get_image_size_str(bicubic)
                st.image(bicubic, caption=f"Bicubic - {bicubic_size}", use_column_width=True)
                st.markdown(get_downloadable_img(bicubic, "bicubic"), unsafe_allow_html=True)
                
                # Calculate metrics
                metrics = calculate_image_metrics(image_np, bicubic)
                metrics_data.append({
                    "Method": "Bicubic",
                    "Input Size": compressed_size,
                    "Output Size": bicubic_size,
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"]
                })


        
        if use_lanczos:
            with col2:
                lanczos = get_basic_upscaled(compressed_img, "lanczos", scale_factor)
                upscaled_images["Lanczos"] = lanczos
                lanczos_size = get_image_size_str(lanczos)
                st.image(lanczos, caption=f"Lanczos - {lanczos_size}", use_column_width=True)
                st.markdown(get_downloadable_img(lanczos, "lanczos"), unsafe_allow_html=True)
                
                # Calculate metrics
                metrics = calculate_image_metrics(image_np, lanczos)
                metrics_data.append({
                    "Method": "Lanczos",
                    "Input Size": compressed_size,
                    "Output Size": lanczos_size,
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"]
                })
        
        # Advanced upscaling methods
        if use_edi:
            with col1:
                edi = get_edi_upscaled(compressed_img / 255.0, scale_factor)
                upscaled_images["Edge-Directed Interpolation"] = edi
                edi_size = get_image_size_str(edi)
                st.image(edi, caption=f"Edge-Directed Interpolation (EDI) - {edi_size}", use_column_width=True)
                st.markdown(get_downloadable_img(edi, "edge_directed"), unsafe_allow_html=True)
                
                # Calculate metrics
                metrics = calculate_image_metrics(image_np, edi)
                metrics_data.append({
                    "Method": "Edge-Directed Interpolation",
                    "Input Size": compressed_size,
                    "Output Size": edi_size,
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"]
                })
        
        if use_nlm:
            with col2:
                nlm = get_nlm_upscaled(compressed_img / 255.0, scale_factor)
                upscaled_images["Non-Local Means"] = nlm
                nlm_size = get_image_size_str(nlm)
                st.image(nlm, caption=f"Non-Local Means (NLM) - {nlm_size}", use_column_width=True)
                st.markdown(get_downloadable_img(nlm, "non_local_means"), unsafe_allow_html=True)
                
                # Calculate metrics
                metrics = calculate_image_metrics(image_np, nlm)
                metrics_data.append({
                    "Method": "Non-Local Means",
                    "Input Size": compressed_size,
                    "Output Size": nlm_size,
                    "PSNR": metrics["PSNR"],
                    "SSIM": metrics["SSIM"]
                })
        
        # ESRGAN upscaling (if selected)
        if use_esrgan:
            st.subheader("ESRGAN Upscaling (AI-based)")
            
            esrgan_status = st.empty()
            esrgan_status.info("Initializing ESRGAN...")
            
            upscaler = ESRGANUpscaler()
            
            # Convert compressed image back to PIL format for ESRGAN
            compressed_pil = Image.fromarray(compressed_img)
            
            with st.spinner("Applying ESRGAN upscaling (this may take a while)..."):
                esrgan_result = upscaler.upscale(
                    compressed_pil,
                    patch_size=esrgan_patch_size,
                    max_size=esrgan_max_size,
                    attempt_whole_image=esrgan_max_size <= 256  # Only attempt whole image for small images
                )
                
                # Convert to numpy array for display
                esrgan_np = np.array(esrgan_result)
                upscaled_images["ESRGAN"] = esrgan_np
            
            esrgan_status.success("ESRGAN upscaling complete!")
            esrgan_size = get_image_size_str(esrgan_np)
            st.image(esrgan_np, caption=f"ESRGAN (AI-based Upscaling) - {esrgan_size}", use_column_width=True)
            st.markdown(get_downloadable_img(esrgan_np, "esrgan_upscaled"), unsafe_allow_html=True)
            
            # Calculate metrics
            metrics = calculate_image_metrics(image_np, esrgan_np)
            metrics_data.append({
                "Method": "ESRGAN",
                "Input Size": compressed_size,
                "Output Size": esrgan_size,
                "PSNR": metrics["PSNR"],
                "SSIM": metrics["SSIM"]
            })
            
            # Clean up TensorFlow session
            tf.keras.backend.clear_session()
            gc.collect()
        
        # Display metrics table
        if metrics_data:
            st.subheader("Image Quality Metrics")
            metrics_df = pd.DataFrame(metrics_data)
            st.table(metrics_df)
        
        # Show comparison if at least two methods were used
        if len(upscaled_images) >= 2:
            st.subheader("Comparison of Methods")
            
            # Create tabs for different comparisons
            tab_names = ["All Methods"] + list(upscaled_images.keys())
            tabs = st.tabs(tab_names)
            
            # All methods comparison tab
            with tabs[0]:
                methods_row1 = list(upscaled_images.keys())[:3]
                methods_row2 = list(upscaled_images.keys())[3:6]
                methods_row3 = list(upscaled_images.keys())[6:]
                
                # Row 1
                if methods_row1:
                    cols = st.columns(len(methods_row1))
                    for i, method in enumerate(methods_row1):
                        cols[i].image(upscaled_images[method], caption=method, use_column_width=True)
                
                # Row 2
                if methods_row2:
                    cols = st.columns(len(methods_row2))
                    for i, method in enumerate(methods_row2):
                        cols[i].image(upscaled_images[method], caption=method, use_column_width=True)
                
                # Row 3
                if methods_row3:
                    cols = st.columns(len(methods_row3))
                    for i, method in enumerate(methods_row3):
                        cols[i].image(upscaled_images[method], caption=method, use_column_width=True)
            
            # Individual method comparison tabs
            for i, method in enumerate(upscaled_images.keys()):
                with tabs[i+1]:
                    col1, col2 = st.columns(2)
                    col1.image(compressed_img, caption="Compressed Image", use_column_width=True)
                    col2.image(upscaled_images[method], caption=f"{method} Upscaled", use_column_width=True)
                    
                    # Display metrics for this method
                    method_metrics = next((m for m in metrics_data if m["Method"] == method), None)
                    if method_metrics:
                        st.markdown(f"""
                        **Metrics for {method}:**
                        - PSNR: {method_metrics['PSNR']}
                        - SSIM: {method_metrics['SSIM']}
                        - Input Size: {method_metrics['Input Size']}
                        - Output Size: {method_metrics['Output Size']}
                        """)
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Upload an image to begin.")
        st.markdown("""
        ### How this app works:
        1. Upload an image file (JPG, PNG, etc.)
        2. The image will be compressed to your specified quality level
        3. Multiple upscaling methods will be applied to enhance the image
        4. You can compare the results and download any version
        5. View detailed image metrics (PSNR, SSIM) for each method
        
        ### Available upscaling methods:
        - **Basic methods**: Nearest Neighbor, Bilinear, Bicubic, Lanczos
        - **Advanced methods**: Edge-Directed Interpolation, Non-Local Means
        - **AI-based**: ESRGAN (Enhanced Super-Resolution GAN)
        
        Choose which methods to use in the sidebar.
        
        ### Metrics explained:
        - **PSNR (Peak Signal-to-Noise Ratio)**: Measures the ratio between the maximum possible power of an image and the power of corrupting noise. Higher values indicate better quality.
        - **SSIM (Structural Similarity Index)**: Measures the similarity between two images based on luminance, contrast, and structure. Values closer to 1 indicate greater similarity to the original.
        """)