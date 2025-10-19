

#include <Wire.h>

#include <LiquidCrystal_I2C.h>

#include <DHT.h>

// TensorFlow Lite 

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "tensorflow/lite/micro/micro_interpreter.h"

#include "tensorflow/lite/schema/schema_generated.h"

#include "temp_model.h"

#define DHTPIN 19

#define DHTTYPE DHT11

DHT dht(DHTPIN, DHTTYPE);

LiquidCrystal_I2C lcd(0x27, 16, 2);

const float TEMPERATURE_MIN = 23.80;

const float TEMPERATURE_MAX = 27.60;

const float HUMIDITY_MIN = 48.00;

const float HUMIDITY_MAX = 73.00;

const float HOUR_MIN = 0.00;

const float HOUR_MAX = 23.00;

const float DAY_MIN = 0.00;

const float DAY_MAX = 6.00;

const int LOOKBACK = 24;

const int N_FEATURES = 4;

const int TENSOR_ARENA_SIZE = 70 * 1024; // Reduced to 70KB to fit in RAM

// TFLite 

alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model = nullptr;

tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input_tensor = nullptr;

TfLiteTensor* output_tensor = nullptr;

float temp_buffer[LOOKBACK];

float humidity_buffer[LOOKBACK];

int hour_buffer[LOOKBACK];

int day_buffer[LOOKBACK];

int buffer_idx = 0;

bool buffer_full = false;

byte degreeSymbol[8] = {

0b00110, 0b01001, 0b01001, 0b00110,

0b00000, 0b00000, 0b00000, 0b00000

};

float normalize(float value, float min_val, float max_val) {

if (max_val == min_val) return 0.0;

return (value - min_val) / (max_val - min_val);

}

float denormalize(float value, float min_val, float max_val) {

return value * (max_val - min_val) + min_val;

}

void setup() {

Serial.begin(115200);

delay(2000);



lcd.init();

lcd.backlight();

lcd.createChar(0, degreeSymbol);

dht.begin();



lcd.clear();

lcd.print("Feed forward Predictor");

delay(2000);



lcd.clear();

lcd.print("Loading...");



Serial.println("\nFeed forward Temperature Predictor ===");



// Load model

model = tflite::GetModel(temp_model);

if (model->version() != TFLITE_SCHEMA_VERSION) {

    lcd.clear();

    lcd.print("Model Error!");

    Serial.println("Model version mismatch!");

    while(1) delay(1000);

}



// Add operations feedforward neural network

static tflite::MicroMutableOpResolver<10> micro_op_resolver;



micro_op_resolver.AddFullyConnected();

micro_op_resolver.AddReshape();

micro_op_resolver.AddQuantize();

micro_op_resolver.AddDequantize();

micro_op_resolver.AddLogistic();

micro_op_resolver.AddRelu();



// interpreter

static tflite::MicroInterpreter static_interpreter(

    model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE);

interpreter = &static_interpreter;



// Allocate tensors

Serial.println("Allocating tensors");

TfLiteStatus allocate_status = interpreter->AllocateTensors();

if (allocate_status != kTfLiteOk) {

    lcd.clear();

    lcd.print("Tensor Error!");

    Serial.println("Tensor allocation failed!");

    Serial.println("Model needs more ops or memory or check header file");

    while(1) delay(1000);

}



// Get  pointers

input_tensor = interpreter->input(0);

output_tensor = interpreter->output(0);



Serial.println("Model OK!, great");

Serial.printf("Model: %d bytes\n", temp_model_len);

Serial.printf("Arena: %d bytes\n", TENSOR_ARENA_SIZE);

Serial.printf("Input shape: [%d, %d, %d]\n", 

              input_tensor->dims->data[0],

              input_tensor->dims->data[1], 

              input_tensor->dims->data[2]);



// Initialize buffers

for (int i = 0; i < LOOKBACK; i++) {

    temp_buffer[i] = 25.0;

    humidity_buffer[i] = 60.0;

    hour_buffer[i] = 12;

    day_buffer[i] = 0;

}



lcd.clear();

lcd.print("Ready!");

lcd.setCursor(0, 1);

lcd.print("Accuracy:0.12C");

delay(2000);



Serial.println("System Ready!\n");

}

void storeSensorReading(float temp, float humidity, int hour, int day) {

temp_buffer[buffer_idx] = temp;

humidity_buffer[buffer_idx] = humidity;

hour_buffer[buffer_idx] = hour;

day_buffer[buffer_idx] = day;



buffer_idx++;

if (buffer_idx >= LOOKBACK) {

    buffer_idx = 0;

    buffer_full = true;

}

}

float predictNextHour() {

if (!buffer_full) return -999.0;



int start_idx = buffer_idx;



// Fill input tensor flattened sequence for feedforward network

for (int i = 0; i < LOOKBACK; i++) {

    int idx = (start_idx + i) % LOOKBACK;

    int base = i * N_FEATURES;

    

    input_tensor->data.f[base + 0] = normalize(temp_buffer[idx], TEMPERATURE_MIN, TEMPERATURE_MAX);

    input_tensor->data.f[base + 1] = normalize(humidity_buffer[idx], HUMIDITY_MIN, HUMIDITY_MAX);

    input_tensor->data.f[base + 2] = normalize(hour_buffer[idx], HOUR_MIN, HOUR_MAX);

    input_tensor->data.f[base + 3] = normalize(day_buffer[idx], DAY_MIN, DAY_MAX);

}



// Run inference

unsigned long start = millis();

TfLiteStatus invoke_status = interpreter->Invoke();

unsigned long inference_time = millis() - start;



if (invoke_status != kTfLiteOk) {

    Serial.println("Inference failed!");

    return -999.0;

}



// Get prediction

float predicted_norm = output_tensor->data.f[0];

float predicted = denormalize(predicted_norm, TEMPERATURE_MIN, TEMPERATURE_MAX);



Serial.printf("Inference: %lu ms\n", inference_time);



return predicted;

}

void updateLCD(float current_temp, float current_humidity, float predicted_temp) {

lcd.clear();



lcd.setCursor(0, 0);

lcd.print("Now:");

lcd.print(current_temp, 1);

lcd.write(byte(0));

lcd.print("C ");

lcd.print((int)current_humidity);

lcd.print("%");



lcd.setCursor(0, 1);

if (predicted_temp == -999.0) {

    lcd.print("Data:");

    lcd.print(buffer_idx);

    lcd.print("/24");

} else {

    lcd.print("Next:");

    lcd.print(predicted_temp, 1);

    lcd.write(byte(0));

    lcd.print("C");

    

    float diff = predicted_temp - current_temp;

    lcd.setCursor(14, 1);

    lcd.print(diff > 0.2 ? "^" : diff < -0.2 ? "v" : "=");

}

}

int getCurrentHour() {

return (millis() / (1000 * 60 * 60)) % 24;

}

int getCurrentDay() {

return (millis() / (1000 * 60 * 60 * 24)) % 7;

}

void loop() {

float current_temp = dht.readTemperature();

float current_humidity = dht.readHumidity();



if (isnan(current_temp) || isnan(current_humidity)) {

    lcd.clear();

    lcd.print("Sensor Error!");

    Serial.println("DHT read failed!");

    delay(2000);

    return;

}



int current_hour = getCurrentHour();

int current_day = getCurrentDay();



storeSensorReading(current_temp, current_humidity, current_hour, current_day);

float predicted = predictNextHour();



updateLCD(current_temp, current_humidity, predicted);



Serial.println("=== Reading ===");

Serial.printf("Temp: %.2f C\n", current_temp);

Serial.printf("Humidity: %.1f %%\n", current_humidity);



if (predicted != -999.0) {

    Serial.printf("Predicted: %.2f C\n", predicted);

    Serial.printf("Change: %+.2f C\n", predicted - current_temp);

} else {

    Serial.printf("Collecting: %d/24\n", buffer_idx);

}

Serial.println("plese wait \n");



delay(10000);

}