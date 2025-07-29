
"""
根据slopecraft写的，梦到哪里写哪里，有的东西我也搞不太明白，后续还需完善，只写了核心功能
由于没写深度计算，彩色表现的并不理想🤔
别急😋
"""

import json
import numpy as np
from PIL import Image
import tinynbt
import os
from pathlib import Path
import argparse
import sys
from collections import Counter

# --- GPU/CPU Acceleration Imports and Setup ---
GPU_BACKEND = None
GPU_AVAILABLE = False

# --- Try to import and initialize GPU backends in order of preference ---
try:
    import pyopencl as cl

    platforms = cl.get_platforms()
    gpu_device = None
    gpu_context = None
    gpu_queue = None
    gpu_program = None

    for platform in platforms:
        if "Apple" in platform.name:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    gpu_device = devices[0]
                    break
            except:
                pass
        if gpu_device is None:
            try:
                devices = platform.get_devices(device_type=cl.device_type.GPU)
                if devices:
                    gpu_device = devices[0]
                    break
            except:
                pass

    if gpu_device:
        gpu_context = cl.Context([gpu_device])
        gpu_queue = cl.CommandQueue(gpu_context)
        # --- OpenCL Kernel for 256-color matching (simplified placeholder) ---
        # A full implementation would require a large kernel or texture lookup.
        # For this version, we'll indicate GPU is available but logic will fall back for modifiers.
        kernel_code = """
        __kernel void dummy_kernel(__global const float4 *pixels, __global unsigned char *result) {
            int gid = get_global_id(0);
            result[gid] = 0; // Dummy
        }
        """
        gpu_program = cl.Program(gpu_context, kernel_code).build()
        GPU_BACKEND = 'opencl'
        GPU_AVAILABLE = True
        print(f"GPU backend detected (OpenCL) on: {gpu_device.name}. Note: Modifier support falls back to CPU.")
except ImportError:
    pass

# 2. CuPy (for NVIDIA CUDA) - only if OpenCL failed
if not GPU_AVAILABLE:
    try:
        import cupy as cp

        a_cp = cp.array([1, 2, 3])
        cp.cuda.runtime.getDeviceCount()
        GPU_BACKEND = 'cupy'
        GPU_AVAILABLE = True
        print("GPU backend detected (CuPy/CUDA). Note: Modifier support falls back to CPU.")
    except ImportError:
        pass

if not GPU_AVAILABLE:
    print("No GPU backend found. Running on CPU only.")

from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. 定义 MC 地图基色和修饰符 ---
# Minecraft 基础颜色 RGB 表 (baseColor 0-63)
MC_BASIC_COLORS_RGB = [
    [0, 0, 0], [128, 128, 128], [128, 128, 128], [128, 128, 128],
    [127, 178, 56], [247, 233, 163], [199, 199, 199], [255, 0, 0],
    [160, 160, 255], [167, 167, 167], [0, 124, 0], [255, 255, 255],
    [164, 168, 184], [151, 109, 77], [112, 112, 112], [64, 64, 255],
    [143, 119, 72], [255, 252, 245], [216, 127, 51], [178, 76, 216],
    [102, 153, 216], [229, 229, 51], [127, 204, 25], [242, 127, 165],
    [76, 76, 76], [153, 153, 153], [76, 127, 153], [127, 63, 178],
    [51, 76, 178], [102, 76, 51], [153, 51, 51], [25, 25, 25],
    [250, 238, 77], [92, 219, 213], [74, 128, 255], [0, 217, 58],
    [129, 86, 49], [112, 2, 0], [209, 177, 161], [159, 82, 36],
    [149, 87, 108], [112, 108, 138], [186, 133, 36], [103, 117, 53],
    [160, 77, 78], [57, 41, 35], [135, 107, 98], [87, 92, 92],
    [121, 77, 52], [72, 82, 40], [152, 126, 104], [135, 98, 67],
    [184, 163, 145], [104, 72, 54], [87, 58, 41], [54, 41, 28],
    [126, 98, 54], [183, 163, 135], [104, 77, 54], [87, 58, 41],
    [54, 41, 28], [126, 98, 54], [183, 163, 135], [104, 77, 54],
]

# 定义 Modifier 系数
# Modifier 0: LOW (180/255), 1: NORMAL (220/255), 2: HIGH (255/255), 3: LOWEST (135/255)
MODIFIER_FACTORS = np.array([180 / 255.0, 220 / 255.0, 255 / 255.0, 135 / 255.0], dtype=np.float32)


# --- 创建完整的 256 色查找表 (用于精确匹配) ---
def create_full_color_lookup_table():
    """
    创建一个包含所有 64 baseColors * 4 modifiers = 256 种颜色的查找表。
    返回:
        full_colors_rgb (np.ndarray): 形状为 (256, 3) 的 float32 数组，值范围 0-1。
        index_to_bm (list of tuples): 长度为 256 的列表，每个元素是 (base_color, modifier)。
    """
    full_colors_rgb = []
    index_to_bm = []
    for base_color in range(64):
        base_rgb = np.array(MC_BASIC_COLORS_RGB[base_color], dtype=np.float32) / 255.0
        for modifier in range(4):
            factor = MODIFIER_FACTORS[modifier]
            final_rgb = base_rgb * factor
            full_colors_rgb.append(final_rgb)
            index_to_bm.append((base_color, modifier))
    return np.array(full_colors_rgb, dtype=np.float32), index_to_bm


# 在模块加载时创建一次查找表
FULL_COLOR_LOOKUP_RGB, INDEX_TO_BM = create_full_color_lookup_table()
NUM_FULL_COLORS = FULL_COLOR_LOOKUP_RGB.shape[0]  # Should be 256


def load_blocks(json_path, target_version=21):
    """加载方块列表，并过滤出适用于目标版本的方块。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        blocks_data = json.load(f)

    # 不再需要 basic_colors_rgb for matching, we use FULL_COLOR_LOOKUP_RGB
    # 但为了兼容性或打印信息，可以保留或重构
    # basic_colors_rgb = np.array(MC_BASIC_COLORS_RGB, dtype=np.float32) / 255.0

    available_blocks = {}
    for block_info in blocks_data:
        version = block_info.get('version', 0)
        if version <= target_version:
            base_color = block_info['baseColor']
            if not (0 <= base_color <= 63):
                print(f"Warning: Invalid baseColor {base_color} for block {block_info.get('id', 'unknown')}. Skipping.")
                continue
            if base_color not in available_blocks:
                available_blocks[base_color] = []
            available_blocks[base_color].append(block_info)

    return available_blocks, None  # Return None for basic_colors_rgb as it's not used for matching anymore


def preprocess_image(image_path, background_color=(255, 255, 255)):
    """加载图像并处理透明像素。"""
    try:
        img = Image.open(image_path).convert('RGBA')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        raise

    data = np.array(img, dtype=np.float32)

    R, G, B, A = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    mask_pure_transparent = (A == 0)
    data[mask_pure_transparent] = [*background_color, 255]

    mask_semi_transparent = (A > 0) & (A < 255)
    if np.any(mask_semi_transparent):
        alpha = A[mask_semi_transparent] / 255.0
        bg_r, bg_g, bg_b = background_color
        R[mask_semi_transparent] = R[mask_semi_transparent] * alpha + bg_r * (1 - alpha)
        G[mask_semi_transparent] = G[mask_semi_transparent] * alpha + bg_g * (1 - alpha)
        B[mask_semi_transparent] = B[mask_semi_transparent] * alpha + bg_b * (1 - alpha)
        data[mask_semi_transparent, 3] = 255

    processed_img = Image.fromarray(np.uint8(data), 'RGBA')
    return processed_img.convert('RGB')


# --- 3. 颜色转换与抖动 (核心修改部分) ---

def is_image_simple(image_array, unique_color_threshold=100):
    """
    简单判断图像是否为简单图像（纯色或颜色种类很少）。
    Args:
        image_array (np.ndarray): 形状为 (H, W, 3) 的图像数组。
        unique_color_threshold (int): 唯一颜色数量的阈值。
    Returns:
        bool: 如果图像简单则返回 True，否则返回 False。
    """
    print("  -> Analyzing image complexity...")
    h, w, _ = image_array.shape

    if h * w > 500 * 500:
        scale_factor = np.sqrt((500 * 500) / (h * w))
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        img_pil = Image.fromarray(np.uint8(image_array))
        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
        sampled_image = np.array(img_resized)
    else:
        sampled_image = image_array

    pixels = sampled_image.reshape(-1, 3)
    pixel_tuples = [tuple(pixel) for pixel in pixels]
    unique_colors = len(Counter(pixel_tuples))

    print(f"    -> Unique colors in sampled image: {unique_colors}")
    is_simple = unique_colors < unique_color_threshold
    if is_simple:
        print("    -> Image is simple. Skipping dithering for better performance.")
    else:
        print("    -> Image is complex. Precise color matching will be applied.")
    return is_simple


# --- 非抖动的精确颜色映射 (支持 Modifier) ---
def map_colors_simple(image_array, full_colors_rgb, index_to_bm):
    """
    对简单图像进行快速、精确的颜色映射，不使用抖动。
    现在匹配的是 256 种完整的显示颜色。
    """
    print("  -> Mapping colors using fast, non-dithered method with modifiers...")
    height, width, _ = image_array.shape
    img_data = image_array.astype(np.float32) / 255.0
    base_color_map = np.zeros((height, width), dtype=np.uint8)
    modifier_map = np.zeros((height, width), dtype=np.uint8)

    pixels_flat = img_data.reshape(-1, 3)
    diff = pixels_flat[:, np.newaxis, :] - full_colors_rgb[np.newaxis, :, :]
    distances_sq = np.sum(diff ** 2, axis=2)
    best_color_indices = np.argmin(distances_sq, axis=1)

    best_bms = np.array([index_to_bm[i] for i in best_color_indices])
    base_color_map = best_bms[:, 0].reshape(height, width).astype(np.uint8)
    modifier_map = best_bms[:, 1].reshape(height, width).astype(np.uint8)

    print("  -> Fast color mapping with modifiers completed.")
    return base_color_map, modifier_map


# --- CPU Color Finding (支持 Modifier) ---
def find_closest_full_color_cpu_vectorized(image_array, full_colors_rgb, index_to_bm):
    """
    CPU 上矢量化查找图像中所有像素的最接近 (baseColor, modifier) 组合。
    """
    print("  -> Finding closest full colors (with modifiers) using vectorized CPU...")
    h, w, _ = image_array.shape
    img_data_normalized = image_array.astype(np.float32) / 255.0

    pixels_flat = img_data_normalized.reshape(-1, 3)
    diff = pixels_flat[:, np.newaxis, :] - full_colors_rgb[np.newaxis, :, :]
    distances_sq = np.sum(diff ** 2, axis=2)
    best_color_indices = np.argmin(distances_sq, axis=1)

    best_bms = np.array([index_to_bm[i] for i in best_color_indices])
    base_color_map = best_bms[:, 0].reshape(h, w).astype(np.uint8)
    modifier_map = best_bms[:, 1].reshape(h, w).astype(np.uint8)

    return base_color_map, modifier_map


# --- 为复杂图像提供精确匹配 ---
# 对于地图颜色这种索引色系统，找到每个像素最接近的索引通常比抖动效果更好。
# 因此，即使是复杂图像，我们也使用精确匹配而非 FS 抖动。
# 如果未来要实现真正的抖动，逻辑会更复杂。
# goodbye,FS.
# 我草我写了一整天的抖动还是没用上


# --- 4. 导出地图文件 (核心修改部分) ---

def _export_single_map(args):
    """导出单个地图文件的辅助函数，供并行处理调用。"""
    # 解包参数，现在包含 modifier_map
    y_start, x_start, base_color_map, modifier_map, output_dir, base_name, index_offset, map_index, map_height, map_width = args
    try:
        y_end = min(y_start + map_height, base_color_map.shape[0])
        x_end = min(x_start + map_width, base_color_map.shape[1])

        block_b = base_color_map[y_start:y_end, x_start:x_end]
        block_m = modifier_map[y_start:y_end, x_start:x_end]

        if block_b.shape[0] < map_height or block_b.shape[1] < map_width:
            pad_height = map_height - block_b.shape[0]
            pad_width = map_width - block_b.shape[1]
            block_b = np.pad(block_b, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
            block_m = np.pad(block_m, ((0, pad_height), (0, pad_width)), mode='constant',
                             constant_values=1)  # Default to NORMAL

        # --- 关键修改：安全钳位和组合 baseColor 与 modifier ---
        clipped_b = np.clip(block_b, 0, 63).astype(np.uint8)
        clipped_m = np.clip(block_m, 0, 3).astype(np.uint8)

        # 根据公式组合: mapColor = (baseColor << 2) | (modifier & 3)
        map_colors = (clipped_b << 2) | (clipped_m & 3)
        map_colors = map_colors.astype(np.uint8)
        # ----------------------------

        color_data = map_colors.flatten().tolist()

        root_tag = tinynbt.create_map_nbt(
            color_data,
            map_scale=2,
            map_dimension=0,
            x_center=0,
            z_center=0
        )

        filename = f"map_{base_name}_{index_offset + map_index}.dat"
        filepath = os.path.join(output_dir, filename)
        tinynbt.write_nbt_file(root_tag, filepath, gzipped=True)
        return filename
    except Exception as e:
        import traceback
        error_msg = f"Error exporting map {base_name}_{index_offset + map_index}: {e}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg


def export_map_files(base_color_map, modifier_map, output_dir, num, begin_index=0, use_multiprocessing=False):
    """将 baseColor 和 modifier 矩阵分割并导出为 Minecraft 1.21 .dat 文件。"""
    height, width = base_color_map.shape  # Assume modifier_map has same shape
    map_height, map_width = 128, 128

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tasks = []
    map_index = 0
    for y_start in range(0, height, map_height):
        for x_start in range(0, width, map_width):
            # 传递两个 map
            tasks.append(
                (y_start, x_start, base_color_map, modifier_map, output_dir, num, begin_index, map_index, map_height,
                 map_width))
            map_index += 1

    if use_multiprocessing and len(tasks) > 1 and sys.platform != 'win32':
        print(f"  -> Exporting {len(tasks)} map files using multiprocessing...")
        num_workers = min(32, len(tasks), (os.cpu_count() or 1) + 4)
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_export_single_map, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                if result and not result.startswith("Error"):
                    print(f"    Exported {result}")
                elif result:
                    print(f"    {result}")
    else:
        if use_multiprocessing and sys.platform == 'win32':
            print("  -> Multiprocessing disabled on Windows for this script.")
        print(f"  -> Exporting {len(tasks)} map files sequentially...")
        for task in tasks:
            result = _export_single_map(task)
            if result and not result.startswith("Error"):
                print(f"    Exported {result}")
            elif result:
                print(f"    {result}")


# --- Main Function ---
def build(image_path, blocks_json_path, output_dir, num, begin_index=0, use_gpu=True, use_multiprocessing=False,
          quiet=True):
    """Main function to build Minecraft maps from an image."""
    if not quiet: print("--- Step 1: Loading blocks... ---")
    try:
        available_blocks, _ = load_blocks(blocks_json_path)
        # Verification of FULL_COLOR_LOOKUP_RGB is implicit as it's a module constant
    except Exception as e:
        print(f"Failed to load blocks: {e}")
        return

    if not quiet: print("--- Step 2: Loading and preprocessing image... ---")
    try:
        img_rgb = preprocess_image(image_path)
        img_array = np.array(img_rgb)
        if not quiet: print(f"Image loaded: {img_array.shape[1]}x{img_array.shape[0]} pixels")
    except Exception as e:
        print(f"Failed to load or process image: {e}")
        return

    if not quiet: print("--- Step 3: Analyzing and converting colors... ---")
    try:
        base_color_map = None
        modifier_map = None

        # 1. 分析图像复杂度
        if is_image_simple(img_array):
            # 2a. 如果简单，使用快速非抖动映射 (支持 Modifier)
            base_color_map, modifier_map = map_colors_simple(img_array, FULL_COLOR_LOOKUP_RGB, INDEX_TO_BM)
        else:
            # 2b. 如果复杂，使用矢量化的精确匹配 (效果优于传统抖动)
            if not quiet: print("--- Step 3b: Converting colors with precise matching (for rich colors)... ---")
            base_color_map, modifier_map = find_closest_full_color_cpu_vectorized(img_array, FULL_COLOR_LOOKUP_RGB,
                                                                                  INDEX_TO_BM)

        if base_color_map is None or modifier_map is None:
            raise RuntimeError("Color conversion failed to produce a result.")

    except Exception as e:
        print(f"Failed during color analysis/conversion: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 修改：Step 4 现在导出两个 map ---
    if not quiet: print("--- Step 4: Exporting map files... ---")
    try:
        export_map_files(base_color_map, modifier_map, output_dir, num, begin_index,
                         use_multiprocessing=use_multiprocessing)
    except Exception as e:
        print(f"Failed during export: {e}")
        return
    # --- 修改结束 ---

    if not quiet:
        print("--- Conversion complete! ---")
    else:
        print(f"Map '{num}' conversion complete.")


# --- Entry Point with CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert an image to Minecraft 1.21 map files with rich colors and optimizations.")
    parser.add_argument("image_path", help="Path to the input image file.")
    parser.add_argument("blocks_json_path", help="Path to the blocks.json file.")
    parser.add_argument("output_dir", help="Directory to save the output map_*.dat files.")
    parser.add_argument("num", help="Base name for the output files (e.g., '1' for map_1_0.dat, map_1_1.dat).")
    parser.add_argument("--begin_index", type=int, default=0, help="Starting index offset for map files (default: 0).")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Enable GPU acceleration (Metal/CUDA/OpenCL). Note: Modifier logic falls back to CPU.")
    parser.add_argument("--use_multiprocessing", action="store_true",
                        help="Enable CPU multiprocessing for exporting maps.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")

    args = parser.parse_args()

    build(
        args.image_path,
        args.blocks_json_path,
        args.output_dir,
        args.num,
        args.begin_index,
        use_gpu=args.use_gpu,
        use_multiprocessing=args.use_multiprocessing,
        quiet=args.quiet
    )
