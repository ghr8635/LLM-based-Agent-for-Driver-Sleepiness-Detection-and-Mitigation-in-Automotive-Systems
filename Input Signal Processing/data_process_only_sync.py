import pandas as pd

# -------- STEP 1: LOAD CSVs AND CONVERT TIMESTAMPS --------
def load_csv_data(camera_csv_path, driving_csv_path):
    cam_df = pd.read_csv(camera_csv_path)
    drive_df = pd.read_csv(driving_csv_path)

    # Convert float UNIX timestamps to datetime
    cam_df['timestamp'] = pd.to_datetime(cam_df['timestamp'], unit='s')
    drive_df['timestamp'] = pd.to_datetime(drive_df['timestamp'], unit='s')

    return cam_df, drive_df

# -------- STEP 2: SYNC (Reference = Steering/Lane data) --------
def sync_data(cam_df, drive_df, output_path):
    synced_rows = []

    for _, drive_row in drive_df.iterrows():
        drive_ts = drive_row['timestamp']

        # Find closest camera image by timestamp
        closest_cam = cam_df.iloc[(cam_df['timestamp'] - drive_ts).abs().argmin()]

        synced_rows.append({
            "timestamp": drive_ts,
            "timestamp_float": drive_ts.timestamp(),  # float with full precision
            "ir_filename": closest_cam['image_filename'],
            "steering_angle": drive_row['steering'],
            "lane_offset": drive_row['offset']
        })

    df_out = pd.DataFrame(synced_rows)

    df_out.to_csv(
        output_path,
        index=False,
        date_format='%Y-%m-%d %H:%M:%S.%f',
        float_format='%.9f'
    )

    print(f"[Sync] Saved {len(df_out)} synced rows to {output_path}")

# -------- RUNNER --------
def run_csv_sync(
    camera_csv_path,
    driving_csv_path,
    output_csv_path
):
    cam_df, drive_df = load_csv_data(camera_csv_path, driving_csv_path)
    sync_data(cam_df, drive_df, output_csv_path)

# -------- ENTRY POINT --------
if __name__ == "__main__":
    run_csv_sync(
        camera_csv_path=r"C:\Users\hussa\OneDrive\Desktop\Projects\LLM Project\image_metadata.csv",  # CSV with columns: timestamp, image_filename
        driving_csv_path=r"C:\Users\hussa\OneDrive\Desktop\Projects\LLM Project\data_capture.csv",  # CSV with columns: timestamp, steering, offset
        output_csv_path="synced_output.csv"  # Output combined synced CSV
    )
