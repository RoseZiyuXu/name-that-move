"""

PLOT DATA AFTER RECORDING !!!!!!! PLOT PLOT PLOT PLOT CHECK CHECK CHECK CHECK PLEASE

"""


from receive_data import start_osc_server

if __name__ == "__main__":
    IP = "0.0.0.0"           # OSC listener IP
    PORT = 10000               # OSC listener port
    SAMPLE_RATE = 48         # Hz
    CHUNK_DURATION = 2       # seconds
    PAD_TO_LENGTH = None     # e.g., 167 or None to disable

    # Channel paths for 4 IMUs (acc + gyro, 6 channels each)

    CHANNEL_PATHS = [
        # "/m/2/acc/x", "/m/2/acc/y", "/m/2/acc/z", "/m/2/gyro/x", "/m/2/gyro/y", "/m/2/gyro/z",
        # "/m/3/acc/x", "/m/3/acc/y", "/m/3/acc/z", "/m/3/gyro/x", "/m/3/gyro/y", "/m/3/gyro/z",
        # "/m/4/acc/x", "/m/4/acc/y", "/m/4/acc/z", "/m/4/gyro/x", "/m/4/gyro/y", "/m/4/gyro/z",
        # "/m/5/acc/x", "/m/5/acc/y", "/m/5/acc/z", "/m/5/gyro/x", "/m/5/gyro/y", "/m/5/gyro/z",
        # "/m/6/acc/x", "/m/6/acc/y", "/m/6/acc/z", "/m/6/gyro/x", "/m/6/gyro/y", "/m/6/gyro/z",
        "/m/7/acc/x", "/m/7/acc/y", "/m/7/acc/z", "/m/7/gyro/x", "/m/7/gyro/y", "/m/7/gyro/z"
    ]


    # ML inference server
    DO_INFERENCE = True
    MODEL_URL = "http://10.158.221.208:8891/process"

    # TouchDesigner OSC output
    TD_IP = "127.0.0.1"
    TD_PORT = 6000

    # Start OSC server and data pipeline
    start_osc_server(
        ip=IP,
        port=PORT,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
        channel_paths=CHANNEL_PATHS,
        pad_to_length=PAD_TO_LENGTH,
        do_inference=DO_INFERENCE,
        model_url=MODEL_URL,
        td_ip=TD_IP,
        td_port=TD_PORT
    )
