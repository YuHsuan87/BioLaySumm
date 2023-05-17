# BioLaySumm
Finetune model in BioLaySumm dataset
### Init Analysis
PLOS and eLife val dataset mean
#### Relevance
|        | Rouge-1 | Rouge-2 | Rouge-L | BERTScore |
|:------ |:------- | ------- |:------- |:--------- |
| BART | 0.4786  | 0.1525  | 0.4452  | 0.8486    |
| LED | **0.4858**  | **0.1552**  | **0.4502**  | **0.8571**    |
| T5   | 0.4358  | 0.1214  | 0.4095  | 0.8398    |
#### Readability
|        | FKGL    | DCRS    |
|:------ | ------- |:------- |
| BART | 12.3617 | 9.9345  |
| LED | 11.8577 | 9.8441  |
| T5   | **10.1728** | **9.1107**  |
#### Factuality
|        | BARTScore |
|:------ |:--------- |
| BART | -2.7569   |
| LED | **-2.0367**   |
| T5   | -3.7528   |

