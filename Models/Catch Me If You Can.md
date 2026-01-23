- Features: 'session_id', 'site1', 'time1', 'site2', 'time2', 'site3', 'time3',
       'site4', 'time4', 'site5', 'time5', 'site6', 'time6', 'site7', 'time7',
       'site8', 'time8', 'site9', 'time9', 'site10', 'time10'
- Label:  'target'

``` python
  ## for each row
# day of the week
# total session length -sess
# biggest diff, stayed the longest - long_gap
# at which hours, days - start, end - at which time of the day
# smallest stay - short_gap



def make(train_s):
    
    trTrain = train_s.copy()
    
    
                
    for idx, row in trTrain.iterrows():
        start_hr = -1
        end_hr = -1
        start = -1
        st = -1
        longS = 0
        shortS = 10**4
        end = -1
        day = -1
    
        for col in trTrain.columns:
            if col.startswith("time") and pd.notnull(row[col]):
                sec = (
                    row[col].hour * 3600 +
                    row[col].minute * 60 +
                    row[col].second
                )
    
                if st == -1:
                    start = sec
                    day = row[col].dayofweek
                    st = sec
                    start_hr = row[col].hour
                else:
                    longS = max(longS, sec - st)
                    shortS = min(shortS, sec - st)
                    st = sec
                end = sec
                end_hr = row[col].hour
        if shortS == 10**4:
            shortS = -1
        # ✅ write back to the DataFrame
        trTrain.at[idx, "long_gap"] = longS
        trTrain.at[idx, "short_gap"] = shortS
        trTrain.at[idx, "end_time"] = end
        trTrain.at[idx, "day"] = day
        trTrain.at[idx, "sess"] = end-start
        trTrain.at[idx, "start"] = start_hr
        trTrain.at[idx, "end"] = end_hr

    return trTrain
```

- made a model on only faetures : cols = "target","day","sess","start","end","long_gap","short_gap"
- with balanced learning for the LR model model was bad it was not working well comp : acc : 73
``` python
"""

Accuracy: 0.7176329468212715

Confusion Matrix:
 [[8884 3506]
 [  25   90]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.72      0.83     12390
           1       0.03      0.78      0.05       115

    accuracy                           0.72     12505
   macro avg       0.51      0.75      0.44     12505
weighted avg       0.99      0.72      0.83     12505

ROC AUC: 0.837684668561603

"""

```

- added extra features cols = ["target","day","sess","start","end","long_gap","short_gap","fsess","lsess","danger","notDanger"]
``` python
"""
Accuracy: 0.8385445821671331

Confusion Matrix:
 [[10381  2009]
 [   10   105]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      0.84      0.91     12390
           1       0.05      0.91      0.09       115

    accuracy                           0.84     12505
   macro avg       0.52      0.88      0.50     12505
weighted avg       0.99      0.84      0.90     12505

ROC AUC: 0.9518265080534793
"""

```
model on compi : 86 % acc
        
