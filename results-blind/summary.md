| config | trials | pass 0 acc | final acc (mean) | best acc (mean) | final accs | passes used |
|---|---|---|---|---|---|---|
| random-search | 3 | 0.322 | 0.584 | 0.661 | 0.506, 0.646, 0.601 | 30, 30, 30 |
| haiku-low | 3 | 0.322 | 0.459 | 0.459 | 0.433, 0.534, 0.410 | 30, 30, 30 |
| haiku-medium | 3 | 0.322 | 0.431 | 0.434 | 0.360, 0.545, 0.388 | 30, 30, 30 |
| haiku-high | 3 | 0.322 | 0.545 | 0.545 | 0.596, 0.500, 0.539 | 30, 30, 30 |
| sonnet-low | 3 | 0.322 | 0.513 | 0.522 | 0.556, 0.545, 0.438 | 30, 30, 30 |
| sonnet-medium | 3 | 0.322 | 0.579 | 0.584 | 0.399, 0.635, 0.702 | 30, 30, 30 |
| sonnet-high | 3 | 0.322 | 0.684 | 0.685 | 0.770, 0.601, 0.680 | 30, 30, 30 |
| opus-low | 3 | 0.322 | 0.743 | 0.743 | 0.904, 0.685, 0.640 | 30, 30, 30 |
| opus-medium | 3 | 0.322 | 0.841 | 0.843 | 0.865, 0.916, 0.742 | 30, 30, 30 |
| opus-high | 3 | 0.322 | 0.860 | 0.860 | 0.978, 0.933, 0.669 | 30, 30, 30 |

| run | final acc | hidden1 weights changed | hidden2 weights changed | output weights changed |
|---|---|---|---|---|
| haiku-high/trial0 | 0.596 | 0/130 | 0/100 | 30/30 |
| haiku-high/trial1 | 0.500 | 0/130 | 0/100 | 30/30 |
| haiku-high/trial2 | 0.539 | 0/130 | 0/100 | 30/30 |
| haiku-low/trial0 | 0.433 | 0/130 | 0/100 | 10/30 |
| haiku-low/trial1 | 0.534 | 0/130 | 0/100 | 6/30 |
| haiku-low/trial2 | 0.410 | 0/130 | 0/100 | 30/30 |
| haiku-medium/trial0 | 0.360 | 0/130 | 0/100 | 30/30 |
| haiku-medium/trial1 | 0.545 | 0/130 | 0/100 | 30/30 |
| haiku-medium/trial2 | 0.388 | 0/130 | 0/100 | 30/30 |
| opus-high/trial0 | 0.978 | 130/130 | 100/100 | 30/30 |
| opus-high/trial1 | 0.933 | 129/130 | 100/100 | 30/30 |
| opus-high/trial2 | 0.669 | 0/130 | 0/100 | 30/30 |
| opus-low/trial0 | 0.904 | 130/130 | 100/100 | 30/30 |
| opus-low/trial1 | 0.685 | 0/130 | 0/100 | 23/30 |
| opus-low/trial2 | 0.640 | 0/130 | 0/100 | 30/30 |
| opus-medium/trial0 | 0.865 | 130/130 | 100/100 | 30/30 |
| opus-medium/trial1 | 0.916 | 130/130 | 100/100 | 30/30 |
| opus-medium/trial2 | 0.742 | 0/130 | 0/100 | 30/30 |
| sonnet-high/trial0 | 0.770 | 130/130 | 100/100 | 30/30 |
| sonnet-high/trial1 | 0.601 | 0/130 | 50/100 | 20/30 |
| sonnet-high/trial2 | 0.680 | 0/130 | 0/100 | 30/30 |
| sonnet-low/trial0 | 0.556 | 0/130 | 0/100 | 30/30 |
| sonnet-low/trial1 | 0.545 | 0/130 | 0/100 | 10/30 |
| sonnet-low/trial2 | 0.438 | 0/130 | 0/100 | 10/30 |
| sonnet-medium/trial0 | 0.399 | 0/130 | 2/100 | 30/30 |
| sonnet-medium/trial1 | 0.635 | 0/130 | 0/100 | 10/30 |
| sonnet-medium/trial2 | 0.702 | 0/130 | 0/100 | 30/30 |
