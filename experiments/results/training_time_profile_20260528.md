# Training Time Profile 20260528

Each run drops the configured warmup rows and summarizes the measured rows from `log_r0.csv`.

| env | backend | runs | step wall ms | CV % | data fetch ms | epoch min | 50 epoch h | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| maze | raw | 3 | 848.516 | 1.297 | 68.998 | 10.734 | 8.945 | stable |
| maze | swm_lance | 3 | 231.150 | 0.165 | 0.183 | 2.924 | 2.437 | stable |
| mw | raw | 3 | 841.051 | 1.926 | 164.908 | 66.233 | 55.194 | stable |
| mw | swm_lance | 3 | 236.766 | 0.091 | 0.230 | 18.645 | 15.538 | stable |
| pusht | raw | 5 | 1384.452 | 7.221 | 325.485 | 238.149 | 198.457 | needs_review |
| pusht | swm_lance | 3 | 230.221 | 0.202 | 0.195 | 39.602 | 33.001 | stable |
| wall | raw | 3 | 307.057 | 1.120 | 22.815 | 1.428 | 1.190 | stable |
| wall | swm_lance | 3 | 235.870 | 0.444 | 3.264 | 1.097 | 0.914 | stable |
