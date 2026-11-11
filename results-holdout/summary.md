| config | trials | pass 0 acc | final acc (mean) | best acc (mean) | final accs | passes used |
|---|---|---|---|---|---|---|
| random-search | 3 | 0.384 | 0.579 | 0.648 | 0.642, 0.585, 0.509 | 30, 30, 30 |
| haiku-low | 3 | 0.384 | 0.447 | 0.447 | 0.377, 0.566, 0.396 | 30, 30, 30 |
| haiku-medium | 3 | 0.384 | 0.497 | 0.509 | 0.340, 0.528, 0.623 | 30, 30, 30 |
| haiku-high | 3 | 0.384 | 0.528 | 0.547 | 0.415, 0.585, 0.585 | 30, 30, 30 |
| sonnet-low | 3 | 0.384 | 0.491 | 0.509 | 0.434, 0.642, 0.396 | 30, 30, 30 |
| sonnet-medium | 3 | 0.384 | 0.553 | 0.560 | 0.604, 0.566, 0.491 | 30, 30, 30 |
| sonnet-high | 3 | 0.384 | 0.654 | 0.692 | 0.736, 0.528, 0.698 | 30, 30, 30 |
| opus-low | 3 | 0.384 | 0.698 | 0.723 | 0.774, 0.830, 0.491 | 30, 30, 30 |
| opus-medium | 3 | 0.384 | 0.830 | 0.855 | 0.774, 0.887, 0.830 | 30, 30, 30 |
| opus-high | 3 | 0.384 | 0.855 | 0.868 | 0.887, 0.830, 0.849 | 30, 30, 30 |

| run | final acc | hidden1 weights changed | hidden2 weights changed | output weights changed |
|---|---|---|---|---|
| haiku-high/trial0 | 0.415 | 0/130 | 40/100 | 30/30 |
| haiku-high/trial1 | 0.585 | 0/130 | 0/100 | 7/30 |
| haiku-high/trial2 | 0.585 | 0/130 | 0/100 | 30/30 |
| haiku-low/trial0 | 0.377 | 0/130 | 0/100 | 7/30 |
| haiku-low/trial1 | 0.566 | 0/130 | 0/100 | 6/30 |
| haiku-low/trial2 | 0.396 | 0/130 | 2/100 | 5/30 |
| haiku-medium/trial0 | 0.340 | 2/130 | 3/100 | 5/30 |
| haiku-medium/trial1 | 0.528 | 0/130 | 0/100 | 10/30 |
| haiku-medium/trial2 | 0.623 | 0/130 | 0/100 | 30/30 |
| opus-high/trial0 | 0.887 | 130/130 | 100/100 | 30/30 |
| opus-high/trial1 | 0.830 | 129/130 | 100/100 | 30/30 |
| opus-high/trial2 | 0.849 | 130/130 | 100/100 | 30/30 |
| opus-low/trial0 | 0.774 | 130/130 | 100/100 | 30/30 |
| opus-low/trial1 | 0.830 | 129/130 | 100/100 | 30/30 |
| opus-low/trial2 | 0.491 | 0/130 | 0/100 | 30/30 |
| opus-medium/trial0 | 0.774 | 130/130 | 100/100 | 30/30 |
| opus-medium/trial1 | 0.887 | 129/130 | 100/100 | 30/30 |
| opus-medium/trial2 | 0.830 | 130/130 | 100/100 | 30/30 |
| sonnet-high/trial0 | 0.736 | 130/130 | 100/100 | 30/30 |
| sonnet-high/trial1 | 0.528 | 39/130 | 0/100 | 30/30 |
| sonnet-high/trial2 | 0.698 | 0/130 | 0/100 | 30/30 |
| sonnet-low/trial0 | 0.434 | 0/130 | 0/100 | 21/30 |
| sonnet-low/trial1 | 0.642 | 0/130 | 1/100 | 16/30 |
| sonnet-low/trial2 | 0.396 | 39/130 | 10/100 | 10/30 |
| sonnet-medium/trial0 | 0.604 | 65/130 | 30/100 | 30/30 |
| sonnet-medium/trial1 | 0.566 | 0/130 | 0/100 | 14/30 |
| sonnet-medium/trial2 | 0.491 | 2/130 | 0/100 | 12/30 |
