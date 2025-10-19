import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


CSV_FILE = 'sensor_data.csv'
LOOKBACK = 24
EPOCHS = 100
BATCH_SIZE = 16


print("Loading data...")
df = pd.read_csv(CSV_FILE)
print(f"Loaded {len(df)} rows")

timestamp_col = None
if 'timestamp_x' in df.columns:
    timestamp_col = 'timestamp_x'
elif 'timestamp' in df.columns:
    timestamp_col = 'timestamp'
else:
    timestamp_col = df.columns[0]

print(f"\nUsing timestamp column: {timestamp_col}")

df['timestamp'] = pd.to_datetime(df[timestamp_col], format='%d-%m-%Y %H:%M')
df = df.sort_values('timestamp').reset_index(drop=True)

# Extract time 
df['hour'] = df['timestamp'].dt.hour
df['day'] = df['timestamp'].dt.dayofweek

# DETECT FEA
has_outdoor = 'outdoor_temp' in df.columns

if has_outdoor:
    print("\n all ok, Outdoor data detected - training with 7 features")
    features = ['temperature', 'humidity', 'outdoor_temp', 'outdoor_humidity', 
                'outdoor_pressure', 'hour', 'day']
else:
    print("\n error, No outdoor data - training with 4 indoor-only features")
    features = ['temperature', 'humidity', 'hour', 'day']

target = 'temperature'
X_data = df[features].values

# Normalize values
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_data)

# Save scaler 
print("\n COPY THESE VALUES TO ESP32 CODE")
for i, feat in enumerate(features):
    print(f"const float {feat.upper()}_MIN = {scaler.data_min_[i]:.2f};")
    print(f"const float {feat.upper()}_MAX = {scaler.data_max_[i]:.2f};")
print("wait\n")

# Create sequences 
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        # Flat the sequence 
        X.append(data[i-lookback:i, :].flatten())
        y.append(data[i, 0]) 
    return np.array(X), np.array(y)

X, y = create_sequences(X_normalized, LOOKBACK)
print(f"Created {len(X)} sequences")
print(f"Input shape: {X.shape}")  

# Split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

#BUILD SIMPLE MODEL 
print("\nBuilding simple feedforward model.")

input_size = LOOKBACK * len(features)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_size,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

#  TRAIN 
print("\nTraining")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    verbose=1
)

# EVALUATE 
print("\nEvaluating")
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {test_mae:.6f}")

predictions = model.predict(X_test)

# 
temp_min = scaler.data_min_[0]
temp_max = scaler.data_max_[0]
y_test_real = y_test * (temp_max - temp_min) + temp_min
pred_real = predictions.flatten() * (temp_max - temp_min) + temp_min

rmse = np.sqrt(np.mean((y_test_real - pred_real)**2))
print(f"\nActual Temperature RMSE: {rmse:.2f}°C")

print("\nSample Predictions:")
for i in range(min(10, len(y_test_real))):
    print(f"  Actual: {y_test_real[i]:.2f}°C, Predicted: {pred_real[i]:.2f}°C")

# SAVE MODEL 
print("\nSaving model")
model.save('temp_model_simple.h5')

#  TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('temp_model.tflite', 'wb') as f:
    f.write(tflite_model)

#  C header
def convert_to_c_array(model_data):
    hex_array = []
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_str = ', '.join([f'0x{b:02x}' for b in chunk])
        hex_array.append(f'  {hex_str}')
    return ',\n'.join(hex_array)

with open('temp_model.h', 'w') as f:
    f.write("const unsigned char temp_model[] = {\n")
    f.write(convert_to_c_array(tflite_model))
    f.write("\n};\n")
    f.write(f"const unsigned int temp_model_len = {len(tflite_model)};\n")

print(f"\n Model saved! Size: {len(tflite_model)/1024:.2f} KB")
print("Files: temp_model.h, temp_model.tflite, temp_model_simple.h5")
print("\n please include temp_model.h in your Arduino project")
print(f"Input shape for ESP32: [{LOOKBACK}, {len(features)}]")