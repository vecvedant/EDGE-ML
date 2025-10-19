#include <Arduino.h>
#include <WiFi.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <DHT.h>
#include <FS.h>
#include <SPIFFS.h>
#include <time.h>


#define DHTPIN 19
#define DHTTYPE DHT11
#define LOG_FILE "/data.csv"
const unsigned long LOG_INTERVAL = 60000;    // 1 min delay
const unsigned long SERIAL_INTERVAL = 5000;  // 5 sec seril print value

const char* ssid = "wifi name";
const char* password = "passcode";


DHT dht(DHTPIN, DHTTYPE);
unsigned long lastLog = 0;
unsigned long lastSerial = 0;

unsigned long startMillis = 0;
time_t startEpoch = 0;


WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP, "pool.ntp.org", 19800); // IST = UTC +5:30 Mumbai India


void setup() {
  Serial.begin(115200);
  delay(1000);
  dht.begin();

  Serial.println("\n ESP32 DHT22 Data Logger ");


  WiFi.begin(ssid, password);
  Serial.print("Connecting to Wi-Fi");
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if(WiFi.status() != WL_CONNECTED){
    Serial.println("\nWi-Fi connection failed! Restarting");
    ESP.restart();
  }

  Serial.println("\nWi-Fi Connected");


  timeClient.begin();
  timeClient.setUpdateInterval(60000);

  Serial.print("Syncing time");
  int syncAttempts = 0;
  while(!timeClient.update() && syncAttempts < 10) {
    timeClient.forceUpdate();
    delay(500);
    Serial.print(".");
    syncAttempts++;
  }
  Serial.println();

  startEpoch = timeClient.getEpochTime(); // save epoch for sync with real time computers
  startMillis = millis();                  // save  millis for clock
  Serial.println("Time synced: " + timeClient.getFormattedTime());
  Serial.print("Epoch: "); Serial.println(startEpoch);


  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  Serial.println("Wi-Fi turned off. Logging continues offline.");


  if(!SPIFFS.begin(true)){
    Serial.println("SPIFFS Mount Failed! Restarting");
    delay(1000);
    ESP.restart();
  }
  Serial.println("SPIFFS mounted successfully");

 
  if(!SPIFFS.exists(LOG_FILE)){
    File file = SPIFFS.open(LOG_FILE, FILE_WRITE);
    if(file){
      file.println("timestamp,temperature,humidity");
      file.close();
      Serial.println("Created new log file with header");
    } else {
      Serial.println("Failed to create log file!");
    }
  } else {
    Serial.println("Log file already exists");
  }

  Serial.println("\n=== Setup Complete ===");
  Serial.println("Starting logging...\n");
}


void loop() {
  unsigned long now = millis();

  
  if (now - lastSerial >= SERIAL_INTERVAL) {
    lastSerial = now;
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    if(!isnan(temp) && !isnan(hum)){
      Serial.printf(">> Temp: %.2f°C | Humidity: %.2f%%\n", temp, hum);
    } else {
      Serial.println("ERROR: Failed to read DHT!");
    }
  }

 
  if (now - lastLog >= LOG_INTERVAL) {
    lastLog = now;
    float temp = dht.readTemperature();
    float hum = dht.readHumidity();
    if(isnan(temp) || isnan(hum)) return;


    time_t currentEpoch = startEpoch + (millis() - startMillis) / 1000;
    struct tm *ptm = gmtime(&currentEpoch);

    char timestamp[25];
    sprintf(timestamp, "%04d-%02d-%02d %02d:%02d:%02d", 
            ptm->tm_year + 1900, 
            ptm->tm_mon + 1, 
            ptm->tm_mday,
            ptm->tm_hour, 
            ptm->tm_min, 
            ptm->tm_sec);

    File file = SPIFFS.open(LOG_FILE, FILE_APPEND);
    if(file){
      file.printf("%s,%.2f,%.2f\n", timestamp, temp, hum);
      file.close();
      Serial.printf("✓ Logged: %s | Temp: %.2f°C | Humidity: %.2f%%\n", timestamp, temp, hum);
    } else {
      Serial.println("ERROR: Failed to open log file for writing!");
    }

    yield(); 
  }
}
