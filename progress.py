
import cv2
import numpy as np
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import shutil



sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import converter as mmc




def process_single_tile(args):
    """
    处理单个图像并生成地图文件。
    这个函数被设计为在独立的进程中运行。
    """
    tile_image_path, output_dir, tile_name, use_gpu, use_multiprocessing = args

    try:

        # 启用 quiet=True 以减少子进程的输出混乱
        mmc.build(
            image_path=tile_image_path,
            blocks_json_path="blocks_list.json",  # 假设 blocks_list.json 在当前工作目录
            output_dir=output_dir,
            num=tile_name,  # 使用 tile_name 作为文件名基础
            begin_index=0,  # 每个frame单独处理，索引从0开始
            use_gpu=use_gpu,
            use_multiprocessing=use_multiprocessing,
            quiet=True  # 子进程静默
        )
        # 成功处理后删除临时瓦片文件
        os.remove(tile_image_path)
        return f"Success: Tile {tile_name}"
    except Exception as e:
        # 如果出错，也尝试清理文件
        try:
            os.remove(tile_image_path)
        except:
            pass
        return f"Error processing tile {tile_name}: {e}"


def process_video(input_path, output_dir, use_gpu, use_multiprocessing, max_workers=None):
    """
    处理视频文件，提取帧，分割成瓦片，并并行生成地图。
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp_tiles")
    os.makedirs(temp_dir, exist_ok=True)

    # 打开视频文件
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Err: Unable to open video")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Info: {width}x{height}, {original_fps:.2f}fps, Total Frames: {total_frames}")

    # 目标设置
    target_fps = 20
    canvas_w, canvas_h = 2048, 1152
    tile_size = 128
    tiles_x, tiles_y = 16, 9  # 2048/128=16, 1152/128=9

    # 计算帧采样间隔
    sample_interval = original_fps / target_fps
    frame_count = 0
    saved_frame_index = 0

    # 确定并行工作进程数
    if max_workers is None:
        # 合理估计：CPU核心数 + 4，但不超过总瓦片数
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    print(f"Using up to {max_workers} worker processes.")

    all_tasks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 采样器
        if frame_count % sample_interval < 1.0:
            # 设置分辨率
            resized = cv2.resize(frame, (1920, 1080))

            # 创建背景画布 (带 Alpha 通道)
            canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

            # 放置帧
            x_offset = (canvas_w - 1920) // 2
            y_offset = (canvas_h - 1080) // 2

            # -------- 修正：保持 BGR 顺序，加 alpha --------
            bgra_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
            canvas[y_offset:y_offset + 1080, x_offset:x_offset + 1920] = bgra_frame
            # ---------------------------------------------

            # 为当前帧的frame创建任务列表
            current_frame_tasks = []

            # 分割画布为frame
            for y in range(tiles_y):
                for x in range(tiles_x):
                    tile = canvas[
                           y * tile_size: (y + 1) * tile_size,
                           x * tile_size: (x + 1) * tile_size,
                           ]

                    # 生成唯一的临时文件名和瓦片名称
                    tile_filename = f"temp_tile_{saved_frame_index}_{y}_{x}.png"
                    tile_image_path = os.path.join(temp_dir, tile_filename)

                    # 瓦片名称 (用于地图文件命名)
                    tile_map_name = f"{(saved_frame_index * tiles_x * tiles_y) + (y * tiles_x + x)}"

                    # 保存临时ferame文件
                    cv2.imwrite(tile_image_path, tile)

                    # 创建传递给 process_single_tile 的参数元组
                    task_args = (tile_image_path, output_dir, tile_map_name, use_gpu, use_multiprocessing)
                    current_frame_tasks.append(task_args)

            # 将当前帧的所有frame任务添加到总任务列表
            all_tasks.extend(current_frame_tasks)
            print(
                f"Prepared frame {saved_frame_index + 1}/{int(total_frames / sample_interval)} with {len(current_frame_tasks)} tiles.")
            saved_frame_index += 1

        frame_count += 1

    cap.release()

    # --- 并行执行所有任务 ---
    print(f"Starting parallel processing of {len(all_tasks)} tiles...")
    processed_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_tile, task): task for task in all_tasks}

        # 处理完成的任务
        for future in as_completed(future_to_task):
            task_args = future_to_task[future]
            tile_name = task_args[2]  # 从args中获取tile_name用于日志
            try:
                result = future.result()
                if result.startswith("Success"):
                    processed_count += 1
                    # 每处理一定数量打印一次进度，避免刷屏
                    if processed_count % max(1, len(all_tasks) // 20) == 0 or processed_count == len(all_tasks):
                        print(f"Progress: {processed_count}/{len(all_tasks)} tiles processed.")
                else:
                    print(f"  {result}")  # 打印错误信息
            except Exception as exc:
                print(f'  Tile {tile_name} generated an exception: {exc}')

    # 清理临时目录
    try:
        shutil.rmtree(temp_dir)
        print("Temporary files cleaned up.")
    except Exception as e:
        print(f"Warning: Could not delete temp directory {temp_dir}: {e}")

    print(f"All done! Total tiles processed: {processed_count} / {len(all_tasks)}. Maps saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video into Minecraft maps using GPU/CPU acceleration.")
    parser.add_argument("input", help="Input video file path.")
    parser.add_argument("output", help="Output directory for map_*.dat files.")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration in map generation.")
    parser.add_argument("--use_multiprocessing", action="store_true", help="Enable multiprocessing in map generation.")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Maximum number of worker processes (default: auto).")

    args = parser.parse_args()

    process_video(args.input, args.output, args.use_gpu, args.use_multiprocessing, args.max_workers)
