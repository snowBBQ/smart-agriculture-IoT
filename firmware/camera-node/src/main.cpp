#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include "esp_camera.h"
#include "esp_wpa2.h" 
#include "HTTPClient.h"
#include "secrets.h"

// =================== PIN DEFINITIONS (FREENOVE WROVER) ===================
#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    21
#define SIOD_GPIO_NUM    26
#define SIOC_GPIO_NUM    27

#define Y9_GPIO_NUM      35
#define Y8_GPIO_NUM      34
#define Y7_GPIO_NUM      39
#define Y6_GPIO_NUM      36
#define Y5_GPIO_NUM      19
#define Y4_GPIO_NUM      18
#define Y3_GPIO_NUM       5
#define Y2_GPIO_NUM       4
#define VSYNC_GPIO_NUM   25
#define HREF_GPIO_NUM    23
#define PCLK_GPIO_NUM    22

// =================== SETTINGS ===================
// How often to take a photo in milliseconds
// 300000 = 5 minutes
const long TIMER_INTERVAL = 3600000; 
unsigned long previousMillis = 0;

WiFiClientSecure espClient;

void setup_camera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Quality settings for PSRAM
  if(psramFound()){
    config.frame_size = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 10;          // 0-63, the lower the number, the better the quality
    config.fb_count = 2;
  } else {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
  }

  // Start of the camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  Serial.println("Camera Ready!");

  sensor_t * s = esp_camera_sensor_get();

  s->set_vflip(s, 1);
  s->set_hmirror(s, 1);

  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_wb_mode(s, 0); 
}

void sendPhoto() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi not connected");
    return;
  }

  Serial.println("Stabilizing sensor");

  for (int i = 0; i < 5; i++) {
    camera_fb_t * fb_temp = esp_camera_fb_get();
    esp_camera_fb_return(fb_temp);
    delay(200); 
  }

  Serial.println("Taking real picture");
  camera_fb_t * fb = esp_camera_fb_get();

  // Get the picture from the camera
  if (!fb) {
    Serial.println("Camera capture failed");
    return;
  }

  Serial.println("Connecting to server");
  HTTPClient http;
  
  // Start connection
  http.begin(espClient, SERVER_URL); 

  // Prepare Multipart Boundary
  String boundary = "SmartVineyardBoundary";
  http.addHeader("Content-Type", "multipart/form-data; boundary=" + boundary);

  // Construct the Body header and tail
  String head = "--" + boundary + "\r\nContent-Disposition: form-data; name=\"file\"; filename=\"esp32-cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
  String tail = "\r\n--" + boundary + "--\r\n";

  // Calculate total length: head + image bytes + tail
  uint32_t totalLen = head.length() + fb->len + tail.length();
  
  // Need to send this as a single buffer for simple POST
  uint8_t* whole_payload = nullptr;
  if (psramFound()) {
    whole_payload = (uint8_t*)ps_malloc(totalLen);
  } else {
    whole_payload = (uint8_t*)malloc(totalLen);
  }

  if (!whole_payload) {
    Serial.println("[ERROR] Failed to allocate memory for HTTP payload");
    esp_camera_fb_return(fb);
    http.end();
    return;
  }

  // Assemble the package: [HEAD][IMAGE][TAIL]
  memcpy(whole_payload, head.c_str(), head.length());
  memcpy(whole_payload + head.length(), fb->buf, fb->len);
  memcpy(whole_payload + head.length() + fb->len, tail.c_str(), tail.length());

  // Send the request
  Serial.println("Sending image");
  int httpResponseCode = http.POST(whole_payload, totalLen);

  // Handle response
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("HTTP Response code: %d\n", httpResponseCode);
    Serial.println("Response: " + response);
  } else {
    Serial.printf("Error occurred during POST: %s\n", http.errorToString(httpResponseCode).c_str());
  }

  // Cleanup memory 
  free(whole_payload);
  esp_camera_fb_return(fb);
  http.end();
}

// CONNECT TO WIFI
void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.println("Connecting to ConcordiaUniversity");

  WiFi.disconnect(true);
  WiFi.mode(WIFI_STA);

  esp_wifi_sta_wpa2_ent_set_identity((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
  esp_wifi_sta_wpa2_ent_set_username((uint8_t *)EAP_IDENTITY, strlen(EAP_IDENTITY));
  esp_wifi_sta_wpa2_ent_set_password((uint8_t *)EAP_PASSWORD, strlen(EAP_PASSWORD));
  esp_wifi_sta_wpa2_ent_enable();

  WiFi.begin(WIFI_SSID2);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setup() {
  Serial.begin(115200);
  
  // Wi-Fi Connection
  setup_wifi();

  espClient.setInsecure();

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");

  // Camera Init
  setup_camera();
  
  // We take the first photo immediately when we turn it on
  sendPhoto();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
  Serial.println("WiFi lost, reconnecting");
  setup_wifi();
}

  // Timer for next photos
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= TIMER_INTERVAL) {
    previousMillis = currentMillis;
    sendPhoto();
  }
}