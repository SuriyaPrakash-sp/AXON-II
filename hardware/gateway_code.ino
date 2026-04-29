// ============================================================
//  gateway_code.ino  —  LoRa Gateway (ESP32 + WiFi + HTTP)
//  Receives LoRa packets and forwards them to the Flask backend
// ============================================================

#include <SPI.h>
#include <LoRa.h>
#include <WiFi.h>
#include <HTTPClient.h>

// ── WiFi credentials (edit before flashing) ────────────────
const char* WIFI_SSID     = "YOUR_WIFI";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";

// ── Backend endpoint (edit before flashing) ────────────────
const char* SERVER_URL    = "http://<YOUR_SERVER_IP>:5000/data";

// ── LoRa pin mapping ────────────────────────────────────────
#define LORA_NSS   5
#define LORA_RST   14
#define LORA_DIO0  26
#define LORA_FREQ  433E6

// ── Tuning constants ────────────────────────────────────────
#define WIFI_CONNECT_TIMEOUT_MS  15000   // give up after 15 s, retry next loop
#define WIFI_RETRY_INTERVAL_MS   10000   // check WiFi health every 10 s
#define HTTP_TIMEOUT_MS          8000    // HTTP request timeout
#define LORA_INIT_RETRIES        5

// ── State ───────────────────────────────────────────────────
unsigned long lastWifiCheck = 0;
uint32_t      packetCount   = 0;


// ════════════════════════════════════════════════════════════
//  WiFi helpers
// ════════════════════════════════════════════════════════════

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;

  Serial.printf("[WiFi] Connecting to \"%s\" ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    if (millis() - start > WIFI_CONNECT_TIMEOUT_MS) {
      Serial.println("[WiFi] Timeout — will retry later.");
      return;
    }
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("[WiFi] Connected  ✓");
  Serial.println("[WiFi] IP address : " + WiFi.localIP().toString());
  Serial.printf( "[WiFi] RSSI       : %d dBm\n", WiFi.RSSI());
}

// Periodically re-check and reconnect if dropped.
void maintainWiFi() {
  unsigned long now = millis();
  if (now - lastWifiCheck < WIFI_RETRY_INTERVAL_MS) return;
  lastWifiCheck = now;

  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] Connection lost — reconnecting ...");
    WiFi.disconnect();
    connectWiFi();
  }
}


// ════════════════════════════════════════════════════════════
//  HTTP helper
// ════════════════════════════════════════════════════════════

// POST a JSON string to the Flask backend.
// Returns HTTP status code, or -1 on connection failure.
int postToBackend(const String& json) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[HTTP] Cannot send — WiFi not connected.");
    return -1;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(HTTP_TIMEOUT_MS);

  Serial.println("[HTTP] POST → " + String(SERVER_URL));
  Serial.println("[HTTP] Body : " + json);

  int httpCode = http.POST(json);

  if (httpCode > 0) {
    String response = http.getString();
    Serial.printf("[HTTP] Response %d : %s\n", httpCode, response.c_str());
  } else {
    Serial.printf("[HTTP] Error: %s\n", http.errorToString(httpCode).c_str());
  }

  http.end();
  return httpCode;
}


// ════════════════════════════════════════════════════════════
//  LoRa helper
// ════════════════════════════════════════════════════════════

void initLoRa() {
  LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);

  for (int attempt = 1; attempt <= LORA_INIT_RETRIES; attempt++) {
    Serial.printf("[LoRa] Init attempt %d / %d ...\n", attempt, LORA_INIT_RETRIES);
    if (LoRa.begin(LORA_FREQ)) {
      // Must match node_code settings exactly
      LoRa.setSpreadingFactor(9);
      LoRa.setSignalBandwidth(125E3);
      LoRa.setCodingRate4(5);
      Serial.println("[LoRa] Initialised in receive mode  ✓");
      return;
    }
    Serial.println("[LoRa] Failed. Retrying in 2 s ...");
    delay(2000);
  }

  Serial.println("[LoRa] FATAL: Could not initialise LoRa. Halting.");
  while (true) { delay(1000); }
}


// ════════════════════════════════════════════════════════════
//  setup()
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }
  delay(500);

  Serial.println("========================================");
  Serial.println("  FloodWatch Gateway");
  Serial.println("========================================");

  // WiFi
  connectWiFi();

  // LoRa
  SPI.begin();
  initLoRa();

  Serial.println("[SYS] Gateway ready — listening for LoRa packets.");
}


// ════════════════════════════════════════════════════════════
//  loop()
// ════════════════════════════════════════════════════════════
void loop() {
  // Keep WiFi alive
  maintainWiFi();

  // Non-blocking LoRa receive
  int packetSize = LoRa.parsePacket();

  if (packetSize > 0) {
    packetCount++;

    // Read full packet
    String incoming = "";
    while (LoRa.available()) {
      incoming += (char)LoRa.read();
    }

    int   rssi = LoRa.packetRssi();
    float snr  = LoRa.packetSnr();

    Serial.println("========================================");
    Serial.printf( "[RX]  Packet #%u  (%d bytes)\n", packetCount, packetSize);
    Serial.printf( "[RX]  RSSI : %d dBm   SNR : %.1f dB\n", rssi, snr);
    Serial.println("[RX]  Data : " + incoming);

    // Validate: must look like a JSON object
    incoming.trim();
    if (incoming.startsWith("{") && incoming.endsWith("}")) {
      int httpCode = postToBackend(incoming);
      if (httpCode == 200 || httpCode == 201) {
        Serial.println("[FWD] Forwarded successfully  ✓");
      } else {
        Serial.printf("[FWD] Forward failed (HTTP %d)  ✗\n", httpCode);
      }
    } else {
      Serial.println("[RX]  Malformed packet — discarded.");
    }

    Serial.println("========================================");
  }

  delay(10);   // yield to RTOS
}
