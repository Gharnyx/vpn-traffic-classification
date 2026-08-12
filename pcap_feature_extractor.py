import dpkt
import socket
import pandas as pd
import numpy as np
from collections import defaultdict
import sys

def parse_pcap(filepath):
    flows = defaultdict(lambda: {
        "src_ip": "", "dst_ip": "", "src_port": 0, "dst_port": 0,
        "protocol": 0, "packet_count": 0, "byte_count": 0,
        "pkt_sizes": [], "timestamps": [],
    })

    with open(filepath, "rb") as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
                if ip.p not in (dpkt.ip.IP_PROTO_TCP, dpkt.ip.IP_PROTO_UDP):
                    continue

                transport = ip.data
                src_ip = socket.inet_ntoa(ip.src)
                dst_ip = socket.inet_ntoa(ip.dst)
                src_port = transport.sport
                dst_port = transport.dport

                key = (src_ip, dst_ip, src_port, dst_port, ip.p)
                flow = flows[key]

                flow["src_ip"] = src_ip
                flow["dst_ip"] = dst_ip
                flow["src_port"] = src_port
                flow["dst_port"] = dst_port
                flow["protocol"] = ip.p
                flow["packet_count"] += 1
                flow["byte_count"] += len(buf)
                flow["pkt_sizes"].append(len(buf))
                flow["timestamps"].append(ts)

            except:
                continue

    return flows

def extract_features(flows):
    records = []
    for f in flows.values():
        duration = 0
        if len(f["timestamps"]) > 1:
            duration = f["timestamps"][-1] - f["timestamps"][0]

        records.append({
            "src_ip": f["src_ip"],
            "dst_ip": f["dst_ip"],
            "src_port": f["src_port"],
            "dst_port": f["dst_port"],
            "protocol": f["protocol"],
            "duration": duration,
            "packet_count": f["packet_count"],
            "byte_count": f["byte_count"],
            "mean_pkt_size": np.mean(f["pkt_sizes"]),
            "std_pkt_size": np.std(f["pkt_sizes"]),
        })

    return pd.DataFrame(records)

if __name__ == "__main__":
    pcap_file = sys.argv[1]
    flows = parse_pcap(pcap_file)
    df = extract_features(flows)
    df.to_csv("traffic_features.csv", index=False)
    print("Feature extraction complete.")