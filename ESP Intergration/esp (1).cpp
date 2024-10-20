#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "....";//wifi ssid
const char* password = "....";//wifi pass
const char* serverUrl = "http://148.100.108.170:38888/lab/signal";//according to end address of AI

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting...");
  }
  Serial.println("Connected!");

  // Set up GPIO pins for each signal
  pinMode(2, OUTPUT); // Junction 1, Lane 1, Left
  pinMode(4, OUTPUT); // Junction 1, Lane 1, Straight
  pinMode(5, OUTPUT); // Junction 1, Lane 1, Right
  pinMode(14, OUTPUT); // Junction 1, Lane 2, Left
  // Add more pins for other lanes and junctions
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    
    int httpResponseCode = http.GET();
    
    if (httpResponseCode > 0) {
      String payload = http.getString();
      Serial.println(payload);

      // Assuming the payload is a JSON string with signal status
      DynamicJsonDocument doc(1024);
      deserializeJson(doc, payload);

      // Process signals for each junction and lane
      if (doc["junction1"]["lane1"]["left"] == "GREEN") {
        digitalWrite(2, HIGH);  // Left signal green
      } else {
        digitalWrite(2, LOW);
      }

      if (doc["junction1"]["lane1"]["straight"] == "GREEN") {
        digitalWrite(4, HIGH);  // Straight signal green
      } else {
        digitalWrite(4, LOW);
      }

      if (doc["junction1"]["lane1"]["right"] == "GREEN") {
        digitalWrite(5, HIGH);  // Right signal green
      } else {
        digitalWrite(5, LOW);
      }

      if (doc["junction1"]["lane2"]["left"] == "GREEN") {
        digitalWrite(14, HIGH);  // Junction 1 Lane 2 Left signal green
      } else {
        digitalWrite(14, LOW);
      }

      // Add more conditions for other lanes and junctions
    } else {
      Serial.println("Error in HTTP request");
    }
    http.end();
  }
  delay(10000); // Delay for 10 seconds before the next request
}
