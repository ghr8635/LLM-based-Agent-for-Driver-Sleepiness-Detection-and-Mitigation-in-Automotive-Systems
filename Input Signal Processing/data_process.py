import os, cv2, json
import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copyfile

SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp'}

# -------- STEP 1: CLEAN --------
def clean_ir_images(input_dir, output_dir, check_black_white=True):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_files = []
    total, dropped = 0, 0

    for file in input_dir.glob("*"):
        if file.suffix.lower() not in SUPPORTED_FORMATS:
            continue
        total += 1

        if file.stat().st_size == 0:
            dropped += 1
            continue

        img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if img is None or np.isnan(img).any() or np.isinf(img).any():
            dropped += 1
            continue
        if check_black_white and (np.all(img == 0) or np.all(img == 255)):
            dropped += 1
            continue

        dest_path = output_dir / file.name
        copyfile(file, dest_path)
        valid_files.append(dest_path.name)

    print(f"[Clean] Total={total}, Dropped={dropped}, Kept={len(valid_files)}")
    return valid_files

# -------- STEP 2: LOAD JSONs --------
def load_json_data(ir_json_path, driving_json_path):
    with open(ir_json_path, 'r') as f:
        ir_data = pd.DataFrame(json.load(f))

    with open(driving_json_path, 'r') as f:
        drive_data = pd.DataFrame(json.load(f))

    #drive_data.rename(columns={'lane_position': 'lane_offset'}, inplace=True)

    ir_data['timestamp'] = pd.to_datetime(ir_data['timestamp'], unit='s')
    drive_data['timestamp'] = pd.to_datetime(drive_data['timestamp'], unit='s')

    return ir_data, drive_data

# -------- STEP 3: SYNC --------
def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

def sync_data(valid_ir_files, ir_df, drive_df, output_path, window_seconds=5):
    timestamp_map = dict(zip(ir_df['filename'], ir_df['timestamp']))
    synced_rows = []

    for fname in valid_ir_files:
        ir_ts = timestamp_map.get(fname)
        if ir_ts is None:
            continue

        closest_row = drive_df.iloc[(drive_df['timestamp'] - ir_ts).abs().argmin()]

        synced_rows.append({
            "timestamp": ir_ts,
            "ir_filename": fname,
            "steering_angle": closest_row['steering_angle'],
            "lane_offset": closest_row['lane_offset']
        })

    df_out = pd.DataFrame(synced_rows)
    df_out['steering_angle'] = normalize_series(df_out['steering_angle'])
    df_out['lane_offset'] = normalize_series(df_out['lane_offset'])

    df_out = df_out.sort_values('timestamp').reset_index(drop=True)
    df_out['window_id'] = -1
    window_ranges = []

    for i in range(len(df_out)):
        start_time = df_out.loc[i, 'timestamp']
        end_time = start_time + pd.Timedelta(seconds=window_seconds)
        mask = (df_out['timestamp'] >= start_time) & (df_out['timestamp'] < end_time) & (df_out['window_id'] == -1)

        if mask.any():
            start_row = df_out[mask].index.min()
            end_row = df_out[mask].index.max()
            df_out.loc[mask, 'window_id'] = i
            window_ranges.append({
                'window_id': i,
                'start_row': int(start_row),
                'end_row': int(end_row),
                'start_time': start_time,
                'end_time': end_time
            })

    df_out.to_csv(output_path, index=False)
    pd.DataFrame(window_ranges).to_csv(Path(output_path).with_name("window_ranges.csv"), index=False)
    print(f"[Sync] Synced rows with window IDs written to {output_path}")

# -------- STEP 4: GAUSSIAN BLUR --------
def apply_gaussian_blur(input_dir, output_dir, kernel_size=(5, 5), sigma=0):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    blurred_files = []
    for file in input_dir.glob("*"):
        if file.suffix.lower() not in SUPPORTED_FORMATS:
            continue

        img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        output_path = output_dir / file.name
        cv2.imwrite(str(output_path), blurred)
        blurred_files.append(output_path.name)

    print(f"[Blur] {len(blurred_files)} images blurred → {output_dir.name}")
    return blurred_files

# -------- STEP 5: RESIZE --------
def resize_images(input_dir, output_dir, target_size=(224, 224)):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resized_files = []
    for file in input_dir.glob("*"):
        if file.suffix.lower() not in SUPPORTED_FORMATS:
            continue

        img = cv2.imread(str(file), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        output_path = output_dir / file.name
        cv2.imwrite(str(output_path), resized)
        resized_files.append(output_path.name)

    print(f"[Resize] {len(resized_files)} images resized to {target_size} → {output_dir.name}")
    return resized_files

# -------- MASTER RUNNER --------
def run_pipeline(
    step_clean=True,
    step_sync=True,
    step_blur=True,
    step_resize=True,
    raw_ir_dir=r"C:\Path\to\raw_ir_images",
    cleaned_ir_dir=r"C:\Path\to\cleaned_ir_images",
    blurred_ir_dir=r"C:\Path\to\blurred_ir_images",
    resized_ir_dir=r"C:\Path\to\resized_ir_images",
    ir_timestamps_json="ir_timestamps.json",
    driving_data_json="carla_data_log.json",
    sync_output_csv=r"C:\Path\to\synced_output.csv",
    resize_target=(224, 224),
    window_seconds=0.005
):
    valid_files = []

    if step_clean:
        valid_files = clean_ir_images(raw_ir_dir, cleaned_ir_dir)
    else:
        valid_files = [f.name for f in Path(cleaned_ir_dir).glob("*") if f.suffix.lower() in SUPPORTED_FORMATS]

    if step_sync:
        ir_df, drive_df = load_json_data(ir_timestamps_json, driving_data_json)
        sync_data(valid_files, ir_df, drive_df, Path(sync_output_csv), window_seconds)

    if step_blur:
        apply_gaussian_blur(cleaned_ir_dir, blurred_ir_dir)

    if step_resize:
        resize_images(blurred_ir_dir, resized_ir_dir, resize_target)

# -------- ENTRY POINT --------
if __name__ == "__main__":
    run_pipeline()
