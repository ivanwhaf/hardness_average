# hardness_average

A Curriculum Learning Method: Hardness Average

## Usage

### 1.Pretrain

### 2.Score

### 3.Average

### 4.Retrain

## Experiment

| Name                                                                              | Test acc                                    |
|-----------------------------------------------------------------------------------|---------------------------------------------|
| cifar10 clean 9cnn 50000 data                                                     | 90%                                         |
| cifar10 clean 9cnn 40000 data                                                     | 89.09%                                      |
| cifar10 clean 9cnn 30000 data                                                     | 87.19%                                      |
| cifar10 clean 9cnn 20000 data lr0.005                                             | 72.68%                                      |
| cifar10 clean 9cnn 20000 data lr0.01                                              | 81.48%                                      |
| cifar10 clean 9cnn 10000 data lr0.005                                             | 52.40%                                      |
| cifar10 clean 9cnn 10000 data lr0.01                                              | 67.98%                                      |
| cifar10 noise s0.05 9cnn 50000 data lr0.01                                        | 89.00%                                      |
| cifar10 noise s0.1 9cnn 50000 data lr0.01                                         | 87.03%                                      |
| cifar10 noise s0.2 9cnn 50000 data lr0.01                                         | 86.12%                                      |
| cifar10 noise s0.4 9cnn 50000 data lr0.01                                         | 80.76%                                      |
| cifar10 noise s0.6 9cnn 50000 data lr0.01                                         | 66.93%                                      |
| cifar10 clean retrain no-curriculum lrnodecay 9cnn 50000 data lr0.01              | 88.33%                                      |
| cifar10 clean retrain simple-hard lrnodecay 9cnn 50000 data lr0.01                | 87.80%                                      |
| cifar10 clean retrain hard-simple lrnodecay 9cnn 50000 data lr0.01                | 90.05%                                      |
| cifar10 clean retrain hardavg lrnodecay 9cnn 50000 data lr0.01                    | 89.38%                                      |
| cifar10 clean retrain hardavg-cuthard lrnodecay 9cnn 50000 data lr0.01            | 88.81%(1%)                                  |
| cifar10 clean retrain hardavg-cuthardsimple lrnodecay 9cnn 50000 data lr0.01      | 89.11%(1%,1%)                               |
| cifar10 noise s0.2 retrain no-curriculum 9cnn 50000 data lr0.01                   | 70.35%                                      |
| cifar10 noise s0.2 retrain simple-hard 9cnn 50000 data lr0.01                     | 77.50%                                      |
| cifar10 noise s0.2 retrain hard-simple 9cnn 50000 data lr0.01                     | 10.00%                                      |
| cifar10 noise s0.2 retrain hardavg 9cnn 50000 data lr0.01                         | 70%                                         |
| cifar10 noise s0.2 retrain hardavg-cuthard 9cnn 50000 data lr0.01                 | %                                           |
| cifar10 noise s0.2 retrain hardavg-cuthardsimple 9cnn 50000 data lr0.01           | %                                           |
| cifar10 noise s0.2 retrain no-curriculum lrnodecay 9cnn 50000 data lr0.01         | 83.51%                                      |
| cifar10 noise s0.2 retrain simple-hard lrnodecay 9cnn 50000 data lr0.01           | 85.72%                                      |
| cifar10 noise s0.2 retrain hard-simple lrnodecay 9cnn 50000 data lr0.01           | 80.49%                                      |
| cifar10 noise s0.2 retrain hardavg lrnodecay 9cnn 50000 data lr0.01               | 84.37%                                      |
| cifar10 noise s0.2 retrain hardavg-cuthard lrnodecay 9cnn 50000 data lr0.01       | 86.67%/87.45%                               |
| cifar10 noise s0.2 retrain hardavg-cuthardsimple lrnodecay 9cnn 50000 data lr0.01 | 84.41%/84.71%/84.39%(1%,10%)/87.53%(1%,20%) |
| cifar10 noise s0.4 retrain no-curriculum lrnodecay 9cnn 50000 data lr0.01         | 77.77%                                      |
| cifar10 noise s0.4 retrain simple-hard lrnodecay 9cnn 50000 data lr0.01           | 77.91%                                      |
| cifar10 noise s0.4 retrain hard-simple lrnodecay 9cnn 50000 data lr0.01           | 63.53%                                      |
| cifar10 noise s0.4 retrain hardavg lrnodecay 9cnn 50000 data lr0.01               | 78.14%                                      |
| cifar10 noise s0.4 retrain hardavg-cuthard lrnodecay 9cnn 50000 data lr0.01       | 84.42%                                      |
| cifar10 noise s0.4 retrain hardavg-cuthardsimple lrnodecay 9cnn 50000 data lr0.01 | 70.81%/84.21%/80.86%(1%,20%)                |
| cifar10 noise s0.6 retrain no-curriculum lrnodecay 9cnn 50000 data lr0.01         | 62.45%                                      |
| cifar10 noise s0.6 retrain simple-hard lrnodecay 9cnn 50000 data lr0.01           | 62.35%                                      |
| cifar10 noise s0.6 retrain hard-simple lrnodecay 9cnn 50000 data lr0.01           | 64.21%                                      |
| cifar10 noise s0.6 retrain hardavg lrnodecay 9cnn 50000 data lr0.01               | 63.40%                                      |
| cifar10 noise s0.6 retrain hardavg-cuthard lrnodecay 9cnn 50000 data lr0.01       | 53.58%/58.81%/63.16%(10%)                   |
| cifar10 noise s0.6 retrain hardavg-cuthardsimple lrnodecay 9cnn 50000 data lr0.01 | 61.82%(5%,10%),65.49%(1%,10%)               |
