import pandas as pd
from pathlib import Path

# -------- STEP 1: LOAD CSVs --------
def load_csv_data(camera_csv_path, driving_csv_path):
    cam_df = pd.read_csv(camera_csv_path)
    drive_df = pd.read_csv(driving_csv_path)

    # Convert timestamp to datetime
    cam_df['timestamp'] = pd.to_datetime(cam_df['timestamp'], unit='s')
    drive_df['timestamp'] = pd.to_datetime(drive_df['timestamp'], unit='s')

    return cam_df, drive_df

# -------- STEP 2: NORMALIZATION --------
def normalize_series(series):
    return (series - series.min()) / (series.max() - series.min())

# -------- STEP 3: SYNC (camera high-freq, sync to steering/lane) --------
def sync_data(cam_df, drive_df, output_path, window_seconds=5):
    synced_rows = []

    for _, drive_row in drive_df.iterrows():
        drive_ts = drive_row['timestamp']

        # Find closest camera frame
        closest_cam = cam_df.iloc[(cam_df['timestamp'] - drive_ts).abs().argmin()]
        synced_rows.append({
            "timestamp": drive_ts,
            "ir_filename": closest_cam['image_name'],
            "steering_angle": drive_row['steering_angle'],
            "lane_offset": drive_row['lane_offset']
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
    print(f"[Sync] {len(df_out)} steering/lane rows synced with closest camera frames.")

# -------- RUNNER --------
def run_csv_sync(
    camera_csv_path,
    driving_csv_path,
    output_csv_path,
    window_seconds=0.005
):
    cam_df, drive_df = load_csv_data(camera_csv_path, driving_csv_path)
    sync_data(cam_df, drive_df, output_csv_path, window_seconds)

# -------- ENTRY POINT --------
if __name__ == "__main__":
    run_csv_sync(
        camera_csv_path="camera_data.csv",              # CSV with columns: timestamp, image_name
        driving_csv_path="driving_data.csv",            # CSV with columns: timestamp, steering_angle, lane_offset
        output_csv_path="synced_output.csv",            # Output combined synced CSV
        window_seconds=0.005
    )
