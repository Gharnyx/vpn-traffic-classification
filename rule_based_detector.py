import pandas as pd
import sys

VPN_PORTS = {1194, 51820, 500, 4500, 1701, 1723, 8388, 1080}

def classify(df):
    df["vpn_label"] = "Non-VPN"

    for i, row in df.iterrows():
        if row["src_port"] in VPN_PORTS or row["dst_port"] in VPN_PORTS:
            df.at[i, "vpn_label"] = "VPN"

        elif row["mean_pkt_size"] > 1200 and row["std_pkt_size"] < 50:
            df.at[i, "vpn_label"] = "VPN"

    return df

if __name__ == "__main__":
    df = pd.read_csv("traffic_features.csv")
    df = classify(df)
    df.to_csv("traffic_rule_labeled.csv", index=False)
    print("Rule-based classification complete.")