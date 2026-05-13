from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from threading import Thread
from collections import deque
from label_to_TD_mult import stream_to_td
import numpy as np
import pickle
import time
import os
from datetime import datetime
import requests


def start_multi_imu_server(ip, port, sample_rate, chunk_duration,
                           pad_to_length=None, do_inference=False,
                           model_url=None, imu_port_map=None, base_path="/python"):
    """
    Listens to multiple IMUs (/m/2/, /m/3/, etc.), buffers data for each IMU,
    runs inference per IMU, and sends each result to its corresponding port.
    """

    imu_buffers = {}
    current_values = {}
    samples_per_chunk = int(sample_rate * chunk_duration)

    # initialize buffers for each IMU
    for imu_id in imu_port_map.keys():
        paths = [
            f"/m/{imu_id}/acc/x", f"/m/{imu_id}/acc/y", f"/m/{imu_id}/acc/z",
            f"/m/{imu_id}/gyro/x", f"/m/{imu_id}/gyro/y", f"/m/{imu_id}/gyro/z"
        ]
        imu_buffers[imu_id] = deque(maxlen=samples_per_chunk)
        for p in paths:
            current_values[p] = 0.0

    print(f"📡 Listening for IMUs: {', '.join(str(i) for i in imu_port_map.keys())}")
    print(f"🔧 Sampling at {sample_rate} Hz | Chunk duration: {chunk_duration}s")

    def update_handler(address, *args):
        if address in current_values and len(args) == 1:
            current_values[address] = args[0]

    def sampling_loop():
        while True:
            for imu_id in imu_port_map.keys():
                paths = [
                    f"/m/{imu_id}/acc/x", f"/m/{imu_id}/acc/y", f"/m/{imu_id}/acc/z",
                    f"/m/{imu_id}/gyro/x", f"/m/{imu_id}/gyro/y", f"/m/{imu_id}/gyro/z"
                ]
                snapshot = [current_values[p] for p in paths]
                imu_buffers[imu_id].append(snapshot)

                if len(imu_buffers[imu_id]) == samples_per_chunk:
                    array = np.array(imu_buffers[imu_id]).T  # (6, N)
                    array = array[np.newaxis, :, :]          # (1, 6, N)

                    if pad_to_length and array.shape[2] < pad_to_length:
                        pad_width = pad_to_length - array.shape[2]
                        array = np.pad(array, ((0,0),(0,0),(0,pad_width)), mode='constant')

                    # Save chunk for debugging
                    os.makedirs("output", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    fname = os.path.join("output", f"imu{imu_id}_{timestamp}.pkl")
                    with open(fname, "wb") as f:
                        pickle.dump(array, f)
                    print(f"💾 IMU {imu_id} saved {fname}")

                    label, confidence = None, None
                    if do_inference and model_url:
                        try:
                            with open(fname, "rb") as f:
                                response = requests.post(model_url, files={'file': f})
                            if response.ok:
                                result = response.json()
                                label = result.get('result', [['0'], [0.0]])[0][0]
                                confidence = result.get('result', [['0'], [0.0]])[1][0]
                                print(f"✅ IMU {imu_id}: label={label}, conf={confidence:.3f}")
                            else:
                                print(f"❌ IMU {imu_id} inference error {response.status_code}")
                        except Exception as e:
                            print(f"❌ IMU {imu_id} inference exception: {e}")

                    # Send result to this IMU’s port
                    if label is not None:
                        target_port = imu_port_map[imu_id]
                        try:
                            stream_to_td(label, confidence,
                                         ip="127.0.0.1", port=target_port, base_path=base_path)
                            print(f"📤 IMU {imu_id} → {target_port}")
                        except Exception as e:
                            print(f"⚠️  Send error IMU {imu_id}: {e}")

                    imu_buffers[imu_id].clear()

            time.sleep(1.0 / sample_rate)

    # Dispatcher maps all IMU paths
    dispatcher = Dispatcher()
    for path in current_values.keys():
        dispatcher.map(path, update_handler)

    Thread(target=sampling_loop, daemon=True).start()

    server = BlockingOSCUDPServer((ip, port), dispatcher)
    server.allow_reuse_port = True
    print(f"🚀 Listening for IMU data on {ip}:{port} ...")
    server.serve_forever()
