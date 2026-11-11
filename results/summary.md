| config | trials | pass 0 acc | final acc (mean) | best acc (mean) | final accs | passes used |
|---|---|---|---|---|---|---|
| random-search | 3 | 0.331 | 0.586 | 0.661 | 0.635, 0.669, 0.455 | 30, 30, 30 |
| haiku-low | 3 | 0.331 | 0.983 | 0.983 | 0.983, 0.983, 0.983 | 30, 30, 30 |
| haiku-high | 3 | 0.331 | 0.981 | 0.983 | 0.983, 0.978, 0.983 | 26, 25, 24 |
| sonnet-low | 3 | 0.331 | 0.987 | 0.987 | 0.989, 0.994, 0.978 | 30, 30, 30 |
| sonnet-high | 3 | 0.331 | 0.985 | 0.987 | 0.983, 0.983, 0.989 | 30, 30, 30 |
| opus-low | 3 | 0.331 | 0.991 | 0.991 | 1.000, 0.989, 0.983 | 30, 30, 30 |
| opus-high | 3 | 0.331 | 0.985 | 0.991 | 0.989, 0.972, 0.994 | 8, 15, 30 |

| run | final acc | hidden1 weights changed | hidden2 weights changed | output weights changed |
|---|---|---|---|---|
| haiku-high/trial0 | 0.983 | 39/130 | 99/100 | 30/30 |
| haiku-high/trial1 | 0.978 | 39/130 | 30/100 | 30/30 |
| haiku-high/trial2 | 0.983 | 130/130 | 100/100 | 30/30 |
| haiku-low/trial0 | 0.983 | 130/130 | 100/100 | 30/30 |
| haiku-low/trial1 | 0.983 | 129/130 | 100/100 | 30/30 |
| haiku-low/trial2 | 0.983 | 39/130 | 100/100 | 30/30 |
| opus-high/trial0 | 0.989 | 130/130 | 100/100 | 30/30 |
| opus-high/trial1 | 0.972 | 129/130 | 100/100 | 30/30 |
| opus-high/trial2 | 0.994 | 130/130 | 100/100 | 30/30 |
| opus-low/trial0 | 1.000 | 130/130 | 100/100 | 30/30 |
| opus-low/trial1 | 0.989 | 130/130 | 100/100 | 30/30 |
| opus-low/trial2 | 0.983 | 130/130 | 100/100 | 30/30 |
| sonnet-high/trial0 | 0.983 | 130/130 | 100/100 | 30/30 |
| sonnet-high/trial1 | 0.983 | 129/130 | 100/100 | 30/30 |
| sonnet-high/trial2 | 0.989 | 130/130 | 100/100 | 30/30 |
| sonnet-low/trial0 | 0.989 | 130/130 | 100/100 | 30/30 |
| sonnet-low/trial1 | 0.994 | 129/130 | 100/100 | 30/30 |
| sonnet-low/trial2 | 0.978 | 130/130 | 100/100 | 30/30 |
