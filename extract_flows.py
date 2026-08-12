# Safe for Windows multiprocessing
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()

    from nfstream import NFStreamer

    print("Starting flow extraction from D:\\traffic.pcap ...")
    print("This may take 15-90 minutes. Monitor CPU usage.")

    # Use built-in CSV export - no need for loop or dict conversion
    total_flows = NFStreamer(
        source="D:\\traffic.pcap",
        statistical_analysis=True,   # Enables IAT, active/idle, rates, etc.
        n_dissections=0,             # Faster mode
        performance_report=True      # Shows summary at end
    ).to_csv(
        path="D:\\traffic_nfstream.csv",   # Output file
        flows_per_file=0,                  # 0 = all in one file
        columns_to_anonymize=[]            # No anonymization
    )

    print(f"Finished! Extracted {total_flows} flows.")
    print("CSV saved at D:\\traffic_nfstream.csv")