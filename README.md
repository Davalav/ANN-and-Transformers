# ANN & Transformers
## Performance List
### Small Dataset
| Model       | Accuracy |
|------------|---------|
| ANN        | 88%     |
| LSTM       | 86%     |
| DistilBERT | 94%     |
| GPT2       | 88%     |
| RoBERTa    | 93.33%  |

### 25k Dataset
| Model       | Accuracy |
|------------|---------|
| ANN        | 89.64%  | 
| LSTM       | 89.72%  |
| DistilBERT | 92.29%  |
| GPT2       | 91.12%  |
| RoBERTa    | 93.65%  |

## Model Training Experiments

| Model | Small Dataset | 25k Dataset |
|------|--------------|------------|
| ANN | [Small](https://www.comet.com/kanskejoanna/lab1/943f219c2d7a473885bc25fb79ad1f9b?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step) | [25K](https://www.comet.com/kanskejoanna/lab1/7edbce63e96047c09a7d562a510795bc?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step) |
| LSTM | [Small](https://www.comet.com/kanskejoanna/lab1/53a64f2f9ec442e0bc24f4f3e40ce1be?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step) | [25K](https://www.comet.com/kanskejoanna/lab1/a26e41f8985648ec803d375434c68473?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&viewId=new&xAxis=step) |
| DistilBERT | [Small](https://www.comet.com/kanskejoanna/lab1/fb83e000c48c47af9fd5730af5086344?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=1&viewId=new&xAxis=step) | [25K](https://www.comet.com/kanskejoanna/lab1/875d9a275b434f22a2a9d79c8480a441?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=1&viewId=new&xAxis=step) |
| GPT2 | [Small](https://www.comet.com/kanskejoanna/lab1/0c20a33e67bc42eb863ba711ec97fce7?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=1&viewId=new&xAxis=step) | [25K](https://www.comet.com/kanskejoanna/lab1/a2e091ba62664a1d9e39d36357ee594f?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=1&viewId=new&xAxis=step) |
| RoBERTa | [Small](https://www.comet.com/kanskejoanna/lab1/88a105157e5d4007acf10fb94981bffd?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=1&viewId=new&xAxis=step) | [25K](https://www.comet.com/kanskejoanna/lab1/15c86444df464e5fa51970893bd20ec2?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=1&viewId=new&xAxis=step) |

### Model Architecture

- ANN takes one vector input
- LSTM takes in Sequence, step by step (Hidden state).
- Transformer processes all tokens simultaenously and uses self-attention to understand the relationship between all words in the sequence.