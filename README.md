# BioLaySumm
Finetune model in BioLaySumm dataset

### Evaluation_scripts
```
python evaluata_st1.py [pred_folder] ans
```
- pred_folder: there are plos.txt and elfie.txt in the folder.

#### Relevance
|        | Rouge-1 | Rouge-2 | Rouge-L | BERTScore |
|:------ |:------- | ------- |:------- |:--------- |
| BART-2 | 0.4785  | 0.1524  | 0.4451  | 0.8486    |
| Long-2 | 0.4805  | 0.1562  | 0.4488  | 0.8544    |
| T5-1   | 0.4358  | 0.1214  | 0.4095  | 0.8398    |
| LongT5 | 0.3862  | 0.1157  | 0.3457  | 0.8377    |
#### Readability
|        | FKGL    | DCRS    |
|:------ | ------- |:------- |
| BART-2 | 12.3616 | 9.9344  |
| Long-2 | 12.1540 | 9.8266  |
| T5-1   | 10.1728 | 9.1107  |
| LongT5 | 15.9498 | 11.2925 |
#### Factuality
|        | BARTScore |
|:------ |:--------- |
| BART-2 | -2.7568   |
| Long-2 | -2.2716   |
| T5-1   | -3.7528   |
| LongT5 | -2.9590   |
