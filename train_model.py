import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
df = pd.read_csv("laptop_data.csv")
print("Data loaded. Shape:", df.shape)

# Data preprocessing
df.drop(columns=['Unnamed: 0'], inplace=True)

df['Ram'] = df['Ram'].str.replace('GB', '')
df['Weight'] = df['Weight'].str.replace('kg', '')
df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')

# Feature engineering
df['Touchscreen'] = df['ScreenResolution'].apply(lambda x: 1 if 'Touchscreen' in x else 0)
df['Ips'] = df['ScreenResolution'].apply(lambda x: 1 if 'IPS' in x else 0)

new = df['ScreenResolution'].str.split('x', n=1, expand=True)
df['X_res'] = new[0]
df['Y_res'] = new[1]
df['X_res'] = df['X_res'].str.replace(',', '').str.findall(r'(\d+\.?\d+)').apply(lambda x: x[0])
df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')
df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5 / df['Inches']).astype('float')

df.drop(columns=['ScreenResolution', 'Inches', 'X_res', 'Y_res'], inplace=True)

# CPU feature
df['Cpu Name'] = df['Cpu'].apply(lambda x: " ".join(x.split()[0:3]))
def FetchProcessor(text):
    if text == 'Intel Core i7' or text == 'Intel Core i5' or text == 'Intel Core i3':
        return text
    else:
        if text.split()[0] == 'Intel':
            return 'Other Intel Processor'
        else:
            return 'AMD Processor'
df['Cpu brand'] = df['Cpu Name'].apply(FetchProcessor)
df.drop(columns=['Cpu', 'Cpu Name'], inplace=True)

# Memory feature
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df['Memory'] = df['Memory'].str.replace('GB', '')
df['Memory'] = df['Memory'].str.replace('TB', '000')
new = df['Memory'].str.split("+", n=1, expand=True)
df['first'] = new[0]
df['second'] = new[1]
df['first'] = df['first'].str.strip()
df['Layer1HDD'] = df['first'].apply(lambda x: 1 if 'HDD' in x else 0)
df['Layer1SSD'] = df['first'].apply(lambda x: 1 if 'SSD' in x else 0)
df['Layer1Hybrid'] = df['first'].apply(lambda x: 1 if 'Hybrid' in x else 0)
df['Layer1Flash_Storage'] = df['first'].apply(lambda x: 1 if 'Flash Storage' in x else 0)
df['first'] = df['first'].str.replace(r'\D', '', regex=True)
df['second'] = df['second'].str.replace(r'\D', '', regex=True)
df['second'].fillna("0", inplace=True)
df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)
df['Layer2HDD'] = df['second'].apply(lambda x: 1 if 'HDD' in str(x) else 0)
df['Layer2SSD'] = df['second'].apply(lambda x: 1 if 'SSD' in str(x) else 0)
df['Layer2Hybrid'] = df['second'].apply(lambda x: 1 if 'Hybrid' in str(x) else 0)
df['Layer2Flash_Storage'] = df['second'].apply(lambda x: 1 if 'Flash Storage' in str(x) else 0)
df["HDD"] = df["first"] * df["Layer1HDD"] + df["second"] * df["Layer2HDD"]
df["SSD"] = df["first"] * df["Layer1SSD"] + df["second"] * df["Layer2SSD"]
df["Hybrid"] = df["first"] * df["Layer1Hybrid"] + df["second"] * df["Layer2Hybrid"]
df["Flash_Storage"] = df["first"] * df["Layer1Flash_Storage"] + df["second"] * df["Layer2Flash_Storage"]
df.drop(columns=['first', 'second', 
                 'Layer1HDD', 'Layer1SSD', 'Layer1Hybrid', 'Layer1Flash_Storage', 
                 'Layer2HDD', 'Layer2SSD', 'Layer2Hybrid', 'Layer2Flash_Storage'], inplace=True)
df.drop(columns=['Memory', 'Hybrid', 'Flash_Storage'], inplace=True)

# GPU feature
df['Gpu brand'] = df['Gpu'].apply(lambda x: x.split()[0])
df = df[df['Gpu brand'] != 'ARM']
df.drop(columns=['Gpu'], inplace=True)

# OS feature
def Category_Os(input):
    if input == 'Windows 10' or input == 'Windows 7' or input == 'Windows 10 S':
        return 'Windows'
    elif input == 'macOS' or input == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'
df['OS'] = df['OpSys'].apply(Category_Os)
df.drop(columns=['OpSys'], inplace=True)

print("Preprocessing done. Final shape:", df.shape)
print("Columns:", df.columns.tolist())

# Prepare features and target
X = df.drop(columns=['Price'])
Y = np.log(df['Price'])

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=2)

# Create pipeline with Random Forest
step1 = ColumnTransformer(transformers=[ 
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first'), [0, 1, 7, 10, 11]) 
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.85,
                              max_features=0.45,
                              max_depth=26)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

# Train the model
print("Training model...")
pipe.fit(X_train, Y_train)

# Evaluate
Y_pred = pipe.predict(X_test)
print('R2 score:', r2_score(Y_test, Y_pred))
print('MAE:', mean_absolute_error(Y_test, Y_pred))

# Save the model and dataframe
print("Saving model...")
pickle.dump(df, open('df.pkl', 'wb'))
pickle.dump(pipe, open('pipe.pkl', 'wb'))
print("Model saved successfully!")
print("Run 'streamlit run app.py' to start the app.")
