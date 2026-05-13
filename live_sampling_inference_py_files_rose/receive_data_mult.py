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
import requests  # for ML inference


def start_osc_server(ip, port, sample_rate, chunk_duration, channel_paths,
                    pad_to_length=None, do_inference=False, model_url=None,
                    send_targets=None, base_path="/python"):
    """
    OSC receiver that samples IMU data, runs optional inference,
    and broadcasts results to multiple OSC destinations.

    send_targets: list of (ip, port) tuples for broadcast outputs.
    """

    # --- Data setup ---
    current_values = {path: 0.0 for path in channel_paths}
    samples_per_chunk = int(sample_rate * chunk_duration)
    buffer = deque(maxlen=samples_per_chunk)

    # --- Output setup ---
    if send_targets:
        print(f"📤 Output targets: {', '.join([f'{a}:{p}' for a, p in send_targets])}")
    else:
        print("⚠️ No send_targets specified — results won't be sent out.")

    # --- Incoming OSC updates ---
    def update_handler(address, *args):
        if address in current_values and len(args) == 1:
            current_values[address] = args[0]

    # --- Sampling + inference loop ---
    def sampling_loop():
        while True:
            snapshot = [current_values[path] for path in channel_paths]
            buffer.append(snapshot)

            if len(buffer) == samples_per_chunk:
                array = np.array(buffer).T  # shape: (6, N)
                array = array[np.newaxis, :, :]  # shape: (1, 6, N)

                # Optional zero-padding
                if pad_to_length and array.shape[2] < pad_to_length:
                    pad_width = pad_to_length - array.shape[2]
                    array = np.pad(array, ((0, 0), (0, 0), (0, pad_width)), mode='constant')
                    print(f"Padded array to (1, 6, {pad_to_length})")

                # Save chunk
                os.makedirs("output", exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = os.path.join("output", f"chunk_{timestamp_str}.pkl")
                with open(filename, "wb") as f:
                    pickle.dump(array, f)
                print(f"💾 Saved {filename}")

                # Run inference (if enabled)
                label, confidence = None, None
                if do_inference and model_url:
                    try:
                        with open(filename, "rb") as f:
                            response = requests.post(model_url, files={'file': f})
                        if response.ok:
                            result = response.json()
                            label = result.get('result', [['0'], [0.0]])[0][0]
                            confidence = result.get('result', [['0'], [0.0]])[1][0]
                            print(f"✅ Inference: label={label}, confidence={confidence:.3f}")
                        else:
                            print(f"❌ Inference error {response.status_code}: {response.text}")
                    except Exception as e:
                        print(f"❌ Inference exception: {e}")

                # Broadcast results
                if label is not None:
                    for ip_, port_ in (send_targets or []):
                        try:
                            stream_to_td(label, confidence, ip=ip_, port=port_, base_path=base_path)
                            print(f"📡 Sent to {ip_}:{port_}")
                        except Exception as e:
                            print(f"⚠️  Error sending to {ip_}:{port_}: {e}")

                buffer.clear()

            time.sleep(1.0 / sample_rate)

    # --- Dispatcher setup ---
    dispatcher = Dispatcher()
    for path in channel_paths:
        dispatcher.map(path, update_handler)

    print(f"🔧 Sampling at {sample_rate} Hz | Chunk duration: {chunk_duration}s")
    print("📡 Listening on:")
    for path in channel_paths:
        print(f"  - {path}")

    Thread(target=sampling_loop, daemon=True).start()

    # --- Start server ---
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    server.allow_reuse_port = True
    print(f"🚀 Listening for IMU data on {ip}:{port} ...")
    server.serve_forever()
