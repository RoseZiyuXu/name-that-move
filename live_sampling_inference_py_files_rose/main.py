"""
Multi-IMU OSC listener and ML inference broadcaster
Each IMU's result is sent to a unique OSC port.
"""

from receive_data import start_multi_imu_server

if __name__ == "__main__":
    IP = "0.0.0.0"
    PORT = 10000            # single listener for all IMUs
    SAMPLE_RATE = 48        # Hz
    CHUNK_DURATION = 2      # seconds
    PAD_TO_LENGTH = None

    # IMU IDs you’re using and their output OSC ports
    IMU_PORT_MAP = {
        2: 1000,
        3: 2000,
        4: 3000,
        5: 4000,
        6: 5000,
        7: 6000,
        # Add more if needed
    }

    DO_INFERENCE = True
    MODEL_URL = "http://10.158.221.208:8891/process"

    start_multi_imu_server(
        ip=IP,
        port=PORT,
        sample_rate=SAMPLE_RATE,
        chunk_duration=CHUNK_DURATION,
        pad_to_length=PAD_TO_LENGTH,
        do_inference=DO_INFERENCE,
        model_url=MODEL_URL,
        imu_port_map=IMU_PORT_MAP,
        base_path="/python"
    )
