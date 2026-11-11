# Campaign logs

Log files larger than 25 MB (25,000,000 bytes) are compressed in gzip (`.gz`) format:

- `04_identificazione_anomalie.log.gz`: detector events.
- `06_spvs_emulator_RT.log.gz`: ANOMALY STARTED/ENDED intervals, providing detection ground truth.

Decompress these files before using tools that expect plain `.log` files:

```sh
gzip -dk 04_identificazione_anomalie.log.gz 06_spvs_emulator_RT.log.gz
```

The command keeps the compressed files and restores the original `.log` files.
All other logs are stored uncompressed. The decompressed contents of both gzip
files have been verified against the originals using SHA-256.
