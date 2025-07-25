/*
 * Fall Detection System - ESP32C3 + Grove Vision AI V2
 * Two-Step Verification: IMU + Vision AI
 * 
 * Hardware Setup:
 * - XIAO ESP32C3 as main controller
 * - Grove Vision AI V2 with fall detection model
 * - MPU6050 IMU sensor
 * - I2C communication between devices
 */

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <Wire.h>

// WiFi credentials - UPDATE THESE!
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// I2C addresses
#define GROVE_AI_ADDRESS 0x62  // Grove Vision AI V2
#define MPU6050_ADDRESS 0x68   // MPU6050 IMU

// Pin definitions
#define LED_PIN 8              // Built-in LED on ESP32C3
#define ALARM_PIN 10           // Emergency alarm output

// Fall detection parameters
const float ACCEL_THRESHOLD = 2.5;  // g-force threshold
const float GYRO_THRESHOLD = 250;   // degrees/second
const unsigned long FALL_WINDOW = 3000; // 3 seconds confirmation window
const unsigned long DEBOUNCE_TIME = 5000; // 5 seconds between detections

// Web server
WebServer server(80);

// System state
bool fallDetected = false;
unsigned long lastFallTime = 0;
unsigned long lastDebounceTime = 0;
bool visionConfirmed = false;
int totalFallsDetected = 0;

struct SensorData {
  float accelX, accelY, accelZ;
  float gyroX, gyroY, gyroZ;
  float magnitude;
  unsigned long timestamp;
};

void setup() {
  Serial.begin(115200);
  Wire.begin();
  
  // Initialize pins
  pinMode(LED_PIN, OUTPUT);
  pinMode(ALARM_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  digitalWrite(ALARM_PIN, LOW);
  
  // Initialize MPU6050
  Serial.println("Initializing MPU6050...");
  Wire.beginTransmission(MPU6050_ADDRESS);
  Wire.write(0x6B);  // PWR_MGMT_1 register
  Wire.write(0);     // set to zero (wakes up the MPU-6050)
  Wire.endTransmission(true);
  
  Wire.beginTransmission(MPU6050_ADDRESS);
  if (Wire.endTransmission() == 0) {
    Serial.println("MPU6050 found and initialized");
  } else {
    Serial.println("MPU6050 not found on I2C bus");
  }
  
  // Connect to WiFi
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  
  int wifi_attempts = 0;
  while (WiFi.status() != WL_CONNECTED && wifi_attempts < 20) {
    delay(1000);
    Serial.print(".");
    wifi_attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connection failed. Continuing without WiFi.");
  }
  
  // Setup web server routes
  server.on("/", handleRoot);
  server.on("/status", handleStatus);
  server.on("/data", handleData);
  server.on("/reset", handleReset);
  server.begin();
  
  // Startup indication
  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(200);
    digitalWrite(LED_PIN, LOW);
    delay(200);
  }
  
  Serial.println("Fall Detection System Ready!");
  Serial.println("Two-step verification: IMU + Vision AI");
}

void loop() {
  server.handleClient();
  
  // Check debounce time
  if (millis() - lastDebounceTime < DEBOUNCE_TIME) {
    delay(100);
    return;
  }
  
  // Read IMU data
  SensorData data = readIMUData();
  
  // Step 1: IMU-based fall detection
  bool imuFallDetected = detectFallIMU(data);
  
  if (imuFallDetected && !fallDetected) {
    Serial.println("\n*** IMU FALL DETECTED! ***");
    Serial.println("Requesting vision confirmation...");
    
    // Indicate processing
    digitalWrite(LED_PIN, HIGH);
    
    // Step 2: Request vision AI confirmation
    visionConfirmed = requestVisionConfirmation();
    
    if (visionConfirmed) {
      fallDetected = true;
      lastFallTime = millis();
      triggerFallAlert(data);
    } else {
      Serial.println("Vision AI: No fall confirmed - False positive filtered");
      digitalWrite(LED_PIN, LOW);
    }
  }
  
  // Reset fall detection after window
  if (fallDetected && (millis() - lastFallTime > FALL_WINDOW)) {
    fallDetected = false;
    visionConfirmed = false;
    digitalWrite(LED_PIN, LOW);
    digitalWrite(ALARM_PIN, LOW);
    lastDebounceTime = millis();
    Serial.println("Fall alert ended. System ready.");
  }
  
  delay(100); // 10Hz sampling rate
}

SensorData readIMUData() {
  SensorData data;
  data.timestamp = millis();
  
  // Try to read from MPU6050
  Wire.beginTransmission(MPU6050_ADDRESS);
  if (Wire.endTransmission() == 0) {
    // MPU6050 is available - read accelerometer data
    Wire.beginTransmission(MPU6050_ADDRESS);
    Wire.write(0x3B);  // starting with register 0x3B (ACCEL_XOUT_H)
    Wire.endTransmission(false);
    Wire.requestFrom(MPU6050_ADDRESS, 14, true); // request a total of 14 registers
    
    if (Wire.available() >= 14) {
      // Read accelerometer data
      int16_t accelXRaw = Wire.read() << 8 | Wire.read();
      int16_t accelYRaw = Wire.read() << 8 | Wire.read();
      int16_t accelZRaw = Wire.read() << 8 | Wire.read();
      
      // Skip temperature data
      Wire.read(); Wire.read();
      
      // Read gyroscope data
      int16_t gyroXRaw = Wire.read() << 8 | Wire.read();
      int16_t gyroYRaw = Wire.read() << 8 | Wire.read();
      int16_t gyroZRaw = Wire.read() << 8 | Wire.read();
      
      // Convert to g and degrees/second
      data.accelX = accelXRaw / 16384.0;  // For ±2g range
      data.accelY = accelYRaw / 16384.0;
      data.accelZ = accelZRaw / 16384.0;
      
      data.gyroX = gyroXRaw / 131.0;      // For ±250°/s range
      data.gyroY = gyroYRaw / 131.0;
      data.gyroZ = gyroZRaw / 131.0;
    } else {
      // Default values if read fails
      data.accelX = 0.0; data.accelY = 0.0; data.accelZ = 1.0;
      data.gyroX = 0.0; data.gyroY = 0.0; data.gyroZ = 0.0;
    }
  } else {
    // MPU6050 not available, use default values
    data.accelX = 0.0;
    data.accelY = 0.0;
    data.accelZ = 1.0; // Normal gravity
    data.gyroX = 0.0;
    data.gyroY = 0.0;
    data.gyroZ = 0.0;
  }
  
  // Calculate acceleration magnitude
  data.magnitude = sqrt(data.accelX*data.accelX + 
                       data.accelY*data.accelY + 
                       data.accelZ*data.accelZ);
  
  return data;
}

bool detectFallIMU(SensorData data) {
  // Enhanced threshold-based fall detection
  bool highAccel = data.magnitude > ACCEL_THRESHOLD;
  bool lowAccel = data.magnitude < 0.5; // Free fall detection
  bool highGyro = (abs(data.gyroX) > GYRO_THRESHOLD || 
                   abs(data.gyroY) > GYRO_THRESHOLD || 
                   abs(data.gyroZ) > GYRO_THRESHOLD);
  
  // Print sensor data for debugging (every 5 seconds)
  if (millis() % 5000 < 100) {
    Serial.printf("IMU: Accel=%.2f g, Gyro=%.1f°/s\n", 
                  data.magnitude, max({abs(data.gyroX), abs(data.gyroY), abs(data.gyroZ)}));
  }
  
  return highAccel || lowAccel || highGyro;
}

bool requestVisionConfirmation() {
  // Check if Grove Vision AI V2 is available
  Wire.beginTransmission(GROVE_AI_ADDRESS);
  if (Wire.endTransmission() != 0) {
    Serial.println("Grove Vision AI V2 not found on I2C bus");
    return false; // Conservative: no confirmation without vision AI
  }
  
  // Send inference request command
  Wire.beginTransmission(GROVE_AI_ADDRESS);
  Wire.write(0x01); // Request inference command
  Wire.endTransmission();
  
  delay(200); // Wait for processing
  
  // Read result
  Wire.requestFrom(GROVE_AI_ADDRESS, 3);
  if (Wire.available() >= 3) {
    uint8_t result = Wire.read();     // 0=non-fall, 1=fall
    uint8_t confidence = Wire.read(); // Confidence score (0-255)
    uint8_t status = Wire.read();     // Status byte
    
    Serial.printf("Vision AI - Result: %s, Confidence: %d/255, Status: 0x%02X\n", 
                  result ? "FALL" : "NO-FALL", confidence, status);
    
    // Return true if fall detected with high confidence
    return (result == 1 && confidence > 150); // confidence > ~60%
  }
  
  Serial.println("No response from Vision AI");
  return false;
}

void triggerFallAlert(SensorData data) {
  totalFallsDetected++;
  
  Serial.println("\n");
  Serial.println("🚨 FALL CONFIRMED! 🚨");
  Serial.println("Two-step verification completed.");
  Serial.printf("Total falls detected: %d\n", totalFallsDetected);
  
  // Activate alarm
  digitalWrite(ALARM_PIN, HIGH);
  
  // Create alert JSON
  StaticJsonDocument<300> alert;
  alert["type"] = "fall_confirmed";
  alert["timestamp"] = data.timestamp;
  alert["fall_count"] = totalFallsDetected;
  alert["verification"]["imu"] = true;
  alert["verification"]["vision"] = visionConfirmed;
  alert["sensor_data"]["accel_magnitude"] = data.magnitude;
  alert["sensor_data"]["accel_x"] = data.accelX;
  alert["sensor_data"]["accel_y"] = data.accelY;
  alert["sensor_data"]["accel_z"] = data.accelZ;
  alert["sensor_data"]["gyro_x"] = data.gyroX;
  alert["sensor_data"]["gyro_y"] = data.gyroY;
  alert["sensor_data"]["gyro_z"] = data.gyroZ;
  
  String alertString;
  serializeJson(alert, alertString);
  
  // Send alert via Serial
  Serial.println("Alert JSON:");
  Serial.println(alertString);
  
  // Here you can add code to send alerts via:
  // - HTTP POST to server
  // - MQTT
  // - Email (via SMTP)
  // - SMS (via API)
  // - IoT platform (AWS IoT, Azure IoT, etc.)
  // - Telegram bot
  
  // LED pattern for fall alert
  for (int i = 0; i < 10; i++) {
    digitalWrite(LED_PIN, HIGH);
    delay(100);
    digitalWrite(LED_PIN, LOW);
    delay(100);
  }
}

void handleRoot() {
  String html = "<!DOCTYPE html><html><head>";
  html += "<title>Fall Detection System</title>";
  html += "<meta name='viewport' content='width=device-width, initial-scale=1'>";
  html += "<style>body{font-family:Arial;margin:20px;background:#f0f0f0;}";
  html += ".container{background:white;padding:20px;border-radius:10px;box-shadow:0 2px 10px rgba(0,0,0,0.1);}";
  html += ".status{padding:10px;margin:10px 0;border-radius:5px;text-align:center;font-weight:bold;}";
  html += ".normal{background:#d4edda;color:#155724;}";
  html += ".alert{background:#f8d7da;color:#721c24;}";
  html += "a{display:inline-block;margin:5px;padding:10px 15px;background:#007bff;color:white;text-decoration:none;border-radius:5px;}";
  html += "</style></head><body>";
  html += "<div class='container'>";
  html += "<h1>🛡️ Fall Detection System</h1>";
  html += "<h2>Two-Step Verification: IMU + Vision AI</h2>";
  
  if (fallDetected) {
    html += "<div class='status alert'>🚨 FALL DETECTED 🚨</div>";
  } else {
    html += "<div class='status normal'>✅ System Normal</div>";
  }
  
  html += "<p><strong>Total Falls Detected:</strong> " + String(totalFallsDetected) + "</p>";
  html += "<p><strong>WiFi:</strong> " + (WiFi.status() == WL_CONNECTED ? "Connected" : "Disconnected") + "</p>";
  html += "<p><strong>Uptime:</strong> " + String(millis()/1000) + " seconds</p>";
  
  html += "<div>";
  html += "<a href='/status'>📊 Status JSON</a>";
  html += "<a href='/data'>📈 Live Data</a>";
  html += "<a href='/reset'>🔄 Reset Counter</a>";
  html += "</div>";
  
  html += "</div></body></html>";
  
  server.send(200, "text/html", html);
}

void handleStatus() {
  StaticJsonDocument<200> status;
  status["fall_detected"] = fallDetected;
  status["vision_confirmed"] = visionConfirmed;
  status["total_falls"] = totalFallsDetected;
  status["last_fall_time"] = lastFallTime;
  status["uptime_seconds"] = millis() / 1000;
  status["wifi_connected"] = (WiFi.status() == WL_CONNECTED);
  status["free_heap"] = ESP.getFreeHeap();
  
  String response;
  serializeJson(status, response);
  
  server.send(200, "application/json", response);
}

void handleData() {
  SensorData data = readIMUData();
  
  StaticJsonDocument<250> jsonData;
  jsonData["timestamp"] = data.timestamp;
  jsonData["accelerometer"]["x"] = data.accelX;
  jsonData["accelerometer"]["y"] = data.accelY;
  jsonData["accelerometer"]["z"] = data.accelZ;
  jsonData["accelerometer"]["magnitude"] = data.magnitude;
  jsonData["gyroscope"]["x"] = data.gyroX;
  jsonData["gyroscope"]["y"] = data.gyroY;
  jsonData["gyroscope"]["z"] = data.gyroZ;
  jsonData["thresholds"]["accel"] = ACCEL_THRESHOLD;
  jsonData["thresholds"]["gyro"] = GYRO_THRESHOLD;
  
  String response;
  serializeJson(jsonData, response);
  
  server.send(200, "application/json", response);
}

void handleReset() {
  totalFallsDetected = 0;
  fallDetected = false;
  visionConfirmed = false;
  lastFallTime = 0;
  lastDebounceTime = 0;
  
  digitalWrite(LED_PIN, LOW);
  digitalWrite(ALARM_PIN, LOW);
  
  Serial.println("System reset - Fall counter cleared");
  
  server.send(200, "text/plain", "System reset successfully");
}