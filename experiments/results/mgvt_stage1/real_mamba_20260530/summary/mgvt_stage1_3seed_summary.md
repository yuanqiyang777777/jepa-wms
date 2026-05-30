| Task | Model | n | Skill mean | Skill std | Change skill mean | Change skill std | Params | Step ms | Mamba backend | Seeds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| METAWORLD_HF:mw-reach-wall | AdaLN | 3 | 0.2797 | 0.0044 | 0.5800 | 0.0051 | 279416 | 46.38 |  | 1 2 3 |
| METAWORLD_HF:mw-reach-wall | mgvt_convmixer | 3 | 0.2682 | 0.0035 | 0.5701 | 0.0077 | 283325 | 38.98 |  | 1 2 3 |
| METAWORLD_HF:mw-reach-wall | mgvt_mamba | 3 | 0.2305 | 0.0034 | 0.5730 | 0.0043 | 294392 | 45.70 | mamba_ssm | 1 2 3 |
| METAWORLD_HF:mw-reach-wall | mgvt_mlp | 3 | 0.2485 | 0.0022 | 0.5348 | 0.0079 | 283934 | 37.20 |  | 1 2 3 |
| PushT | AdaLN | 3 | 0.3742 | 0.0043 | 0.5885 | 0.0054 | 278376 | 46.60 |  | 234 235 236 |
| PushT | mgvt_convmixer | 3 | 0.3324 | 0.0056 | 0.5546 | 0.0045 | 282295 | 38.78 |  | 234 235 236 |
| PushT | mgvt_mamba | 3 | 0.3160 | 0.0031 | 0.5523 | 0.0033 | 293512 | 43.06 | mamba_ssm | 234 235 236 |
| PushT | mgvt_mlp | 3 | 0.2940 | 0.0024 | 0.4865 | 0.0043 | 282874 | 35.15 |  | 234 235 236 |
