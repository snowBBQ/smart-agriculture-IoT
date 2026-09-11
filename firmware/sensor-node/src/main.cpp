#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "esp_wpa2.h"
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <Wire.h>
#include <BH1750.h>
#include "secrets.h"


// CONFIGURATION

// Wi-Fi Credentials
const char* mqtt_server = MQTT_HOST;

// Pin Definitions
#define PIN_DHT 4
#define PIN_SOIL 34
#define PIN_RELAY 5

// Objects initialization
DHT dht(PIN_DHT, DHT22);
BH1750 lightMeter;
WiFiClientSecure espClient;
PubSubClient client(espClient);

// Timing variables
unsigned long lastMsg = 0;
const long interval = 10000; 

// CONNECT TO WIFI
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.println("Connecting to ConcordiaUniversity");

  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);

  #ifdef EAP_IDENTITY
    esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
    esp_wifi_sta_wpa2_ent_set_username((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
    esp_wifi_sta_wpa2_ent_set_password((uint8_t *)EAP_PASSWORD, strlen(EAP_PASSWORD));
    esp_wifi_sta_wpa2_ent_enable();
    WiFi.begin(WIFI_SSID2);
  #else
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  #endif

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

// MQTT CALLBACK
// This function runs when the server sends a message to the ESP32
void callback(char* topic, byte* message, unsigned int length) {
  Serial.print("Message arrived on topic: ");
  Serial.print(topic);
  Serial.print(". Message: ");
  
  String messageTemp;
  
  for (int i = 0; i < length; i++) {
    Serial.print((char)message[i]);
    messageTemp += (char)message[i];
  }
  Serial.println();

  // Simple check to turn pump ON/OFF via MQTT
  // Topic: "smartplant/control/pump" -> Payload: "ON" or "OFF"
  if (String(topic) == "smartplant/control/pump") {
    if(messageTemp == "ON"){
      Serial.println("Turning Pump ON");
      digitalWrite(PIN_RELAY, HIGH); // Assuming HIGH enables the relay
    }
    else if(messageTemp == "OFF"){
      Serial.println("Turning Pump OFF");
      digitalWrite(PIN_RELAY, LOW);
    }
  }
}

// CONNECT TO MQTT
void reconnect() {
  // Loop until we're reconnected
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection");
    
    // Create a random client ID
    String clientId = "ESP32Client-";
    clientId += String(random(0xffff), HEX);
    
    // Attempt to connect
    if (client.connect(clientId.c_str(), MQTT_USER, MQTT_PASSWORD)) {
      Serial.println("connected");
      // Subscribe to control topics here
      client.subscribe("smartplant/control/pump");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);

  analogSetAttenuation(ADC_11db);

  // Init Sensors
  dht.begin();
  
  Wire.begin();
  lightMeter.begin();

  // Init Relay
  pinMode(PIN_RELAY, OUTPUT);
  digitalWrite(PIN_RELAY, LOW); // Start with pump OFF

  // Init Network
  setup_wifi();
  espClient.setInsecure();

  client.setServer(mqtt_server, 8883);
  client.setBufferSize(512);
  client.setCallback(callback); // Register the listener function
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
  Serial.println("WiFi lost, reconnecting");
  setup_wifi();
  }

  // Keep MQTT alive
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Send sensor data periodically
  unsigned long now = millis();
  if (now - lastMsg > interval) {
    lastMsg = now;

    // Read Data
    float h = dht.readHumidity();
    float t = dht.readTemperature();
    int soil = analogRead(PIN_SOIL);
    float lux = lightMeter.readLightLevel();

    // Prepare JSON Payloads and Publish
    // We use a small buffer for JSON generation
    char buffer[256];

    // Publish Soil
    JsonDocument docSoil;
    docSoil["value"] = soil;
    docSoil["unit"] = "raw";
    serializeJson(docSoil, buffer);
    client.publish("smartplant/sensors/soil_moisture", buffer);

    // Publish Light
    if (lux >= 0) {
      JsonDocument docLight;
      docLight["value"] = lux;
      docLight["unit"] = "lx";
      serializeJson(docLight, buffer);
      client.publish("smartplant/sensors/light_lux", buffer);
    }

    // Publish DHT only if readings are valid
    if (!isnan(h) && !isnan(t)) {
      JsonDocument docTemp;
      docTemp["value"] = t;
      docTemp["unit"] = "C";
      serializeJson(docTemp, buffer);
      client.publish("smartplant/sensors/air_temp", buffer);

      JsonDocument docHum;
      docHum["value"] = h;
      docHum["unit"] = "%";
      serializeJson(docHum, buffer);
      client.publish("smartplant/sensors/air_hum", buffer);
    } else {
      Serial.println("[WARN] DHT22 read failed! Soil and Light sent successfully.");
    }

    Serial.println("Data sent to MQTT Server");
  }
}