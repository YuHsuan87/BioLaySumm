# BioLaySumm
Finetune model in BioLaySumm dataset
### Init Analysis
#### Relevance
|        | Rouge-1 | Rouge-2 | Rouge-L | BERTScore |
|:------ |:------- | ------- |:------- |:--------- |
| BART | 0.4785  | 0.1524  | 0.4451  | 0.8486    |
| LED | 0.4805  | 0.1562  | 0.4488  | 0.8544    |
| T5   | 0.4358  | 0.1214  | 0.4095  | 0.8398    |
#### Readability
|        | FKGL    | DCRS    |
|:------ | ------- |:------- |
| BART | 12.3616 | 9.9344  |
| LED | 12.1540 | 9.8266  |
| T5   | 10.1728 | 9.1107  |
#### Factuality
|        | BARTScore |
|:------ |:--------- |
| BART | -2.7568   |
| LED | -2.2716   |
| T5   | -3.7528   |

