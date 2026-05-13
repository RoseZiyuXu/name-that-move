from pythonosc.udp_client import SimpleUDPClient

def stream_to_td(label, confidence, ip="127.0.0.1", port=8000, base_path="/python"):
    """
    Sends the ML inference result to TouchDesigner via OSC.
    """
    client = SimpleUDPClient(ip, port)
    client.send_message(f"{base_path}/label", int(label))
    client.send_message(f"{base_path}/confidence", float(confidence))
