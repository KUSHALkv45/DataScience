from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df_train, val = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

target = "exam_score"
features = [ 'gender', 'course', 'study_hours', 'class_attendance',
       'internet_access', 'sleep_hours', 'sleep_quality', 'study_method',
       'facility_rating', 'exam_difficulty']

X_train = df_train[features]
y_train = df_train[target]

X_val = val[features]
y_val = val[target]

numeric_features = ["study_hours", "class_attendance", "sleep_hours"]
categorical_features = [ 'gender', 'course',  
       'internet_access',  'sleep_quality', 'study_method',
       'facility_rating', 'exam_difficulty']

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

lr_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

lr_model.fit(X_train, y_train)

y_pred = lr_model.predict(X_val)



print("MAE:", mean_absolute_error(y_val, y_pred))
print("MSE:", mean_squared_error(y_val, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("R²:", r2_score(y_val, y_pred))


# for coefficients
coefficients = lr_model.named_steps["regressor"].coef_
intercept = lr_model.named_steps["regressor"].intercept_

print("Coefficients:", coefficients)
print("Intercept:", intercept)


testData = pd.read_csv("/kaggle/input/playground-series-s6e1/test.csv")
# Predict on test data
test_preds = lr_model.predict(testData)
