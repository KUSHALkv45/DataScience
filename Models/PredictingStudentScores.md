``` python
"""
Initial features (given)
"""
target = "exam_score"
features = [ 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
       'facility_rating', 'exam_difficulty']


"""
Did no FE or EDA just Linear Regression with these features  
MAE: 7.093343642903409
MSE: 78.97132319255964
RMSE: 8.886581074438
R²: 0.7779503590963764

and for the compi - score was 8.87 (1-8.537)
"""
added study_quality + study_method - interaction term and removed (study_quality , study_method)
MAE: 7.093143819955258
MSE: 78.97022899796043
RMSE: 8.88651950979462
R²: 0.7779534357260696


with study_quality and study_method added to the above

MAE: 7.093143982563964
MSE: 78.97022989908724
RMSE: 8.886519560496518
R²: 0.7779534331923031



Now with XGBoostRegresser

XGBoost MAE: 6.997633938011109
XGBoost MSE: 76.95169514743661
XGBoost RMSE: 8.772211531161147
XGBoost R²: 0.7836290999867241

```
