# Campaign of 23 May 2026

telemetry.csv is extracted without deduplication from the MQTT log 01_mosquitto_sub_spvs_power.log: **22,678 data records** plus a header (22,679 CSV lines). Null values are empty cells. Timestamps are Unix seconds; power is in W, current in A and bus_voltage in V.

logs/06_spvs_emulator_RT.log.gz records ANOMALY STARTED/ENDED intervals, providing detection ground truth. logs/04_identificazione_anomalie.log.gz contains detector events. These logs are compressed in gzip (`.gz`) format; see [logs/README.md](logs/README.md) for decompression instructions. episodes/ contains the 64 original contexts and raw responses. runs/ contains contexts and responses for six configurations and five runs: 1,920 inference records. Episode indices follow log order; use episode_key for metric joins, not the index as a timestamp.

05_pv_sim_RT.log records received sensor data and poses.
