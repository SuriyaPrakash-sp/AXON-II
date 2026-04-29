// ============================================================
//  node_code.ino  —  Flood Sensor Node (ESP32 + Ultrasonic + LoRa)
//  Node ID  : N3
//  Measures water level via HC-SR04, transmits JSON over LoRa
// ============================================================

#include <SPI.h>
#include <LoRa.h>

// ── LoRa pin mapping ────────────────────────────────────────
#define LORA_NSS   5
#define LORA_RST   14
#define LORA_DIO0  26
#define LORA_FREQ  433E6

// ── Ultrasonic sensor pins ──────────────────────────────────
#define TRIG_PIN   25
#define ECHO_PIN   33

// ── Tuning constants ────────────────────────────────────────
#define TANK_HEIGHT_CM   100      // maximum measurable depth
#define SEND_INTERVAL_MS 5000     // transmit every 5 s
#define SOUND_SPEED_CM   0.0343f  // cm per microsecond at ~20 °C
#define MAX_ECHO_US      30000UL  // timeout: ~5 m range ceiling
#define LORA_INIT_RETRIES 5

const char* NODE_ID = "N3";

// ── State ───────────────────────────────────────────────────
float previousLevel    = 0.0f;
unsigned long lastSend = 0;


// ════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════

// Measure distance in cm using HC-SR04.
// Returns -1.0 on timeout / out-of-range.
float measureDistanceCm() {
  // Ensure trigger is low before pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(4);

  // 10 µs HIGH pulse
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Read echo pulse width (µs)
  unsigned long duration = pulseIn(ECHO_PIN, HIGH, MAX_ECHO_US);

  if (duration == 0) {
    Serial.println("[SENSOR] Echo timeout — sensor out of range or disconnected.");
    return -1.0f;
  }

  float distance = (duration * SOUND_SPEED_CM) / 2.0f;
  return distance;
}

// Convert distance to water level, clamped 0–100 cm.
float distanceToWaterLevel(float distance) {
  if (distance < 0) return previousLevel;           // keep last known on error
  float level = (float)TANK_HEIGHT_CM - distance;
  if (level < 0.0f)   level = 0.0f;
  if (level > 100.0f) level = 100.0f;
  return level;
}

// Build JSON payload manually (no ArduinoJson dependency).
String buildJSON(float waterLevel, float rateOfRise) {
  String json = "{";
  json += "\"type\":\"sensor\",";
  json += "\"node_id\":\"" + String(NODE_ID) + "\",";
  json += "\"water_level\":" + String(waterLevel, 2) + ",";
  json += "\"rate_of_rise\":" + String(rateOfRise, 2);
  json += "}";
  return json;
}

// Send a string via LoRa.
bool loRaSend(const String& payload) {
  LoRa.beginPacket();
  LoRa.print(payload);
  bool ok = LoRa.endPacket();
  return ok;
}

// Attempt LoRa initialisation with retries.
void initLoRa() {
  LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);

  for (int attempt = 1; attempt <= LORA_INIT_RETRIES; attempt++) {
    Serial.printf("[LoRa] Init attempt %d / %d ...\n", attempt, LORA_INIT_RETRIES);
    if (LoRa.begin(LORA_FREQ)) {
      // Optional RF tuning for better range
      LoRa.setSpreadingFactor(9);
      LoRa.setSignalBandwidth(125E3);
      LoRa.setCodingRate4(5);
      LoRa.setTxPower(17);
      Serial.println("[LoRa] Initialised successfully.");
      return;
    }
    Serial.println("[LoRa] Failed. Retrying in 2 s ...");
    delay(2000);
  }

  Serial.println("[LoRa] FATAL: Could not initialise LoRa after all retries. Halting.");
  while (true) { delay(1000); }   // halt — hardware problem
}


// ════════════════════════════════════════════════════════════
//  setup()
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }
  delay(500);

  Serial.println("========================================");
  Serial.println("  FloodWatch Sensor Node — " + String(NODE_ID));
  Serial.println("========================================");

  // Ultrasonic pins
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  digitalWrite(TRIG_PIN, LOW);
  Serial.printf("[SENSOR] Ultrasonic TRIG=GPIO%d  ECHO=GPIO%d\n", TRIG_PIN, ECHO_PIN);

  // LoRa
  SPI.begin();
  initLoRa();

  Serial.println("[SYS] Setup complete. Transmitting every " + String(SEND_INTERVAL_MS / 1000) + " s.");
}


// ════════════════════════════════════════════════════════════
//  loop()
// ════════════════════════════════════════════════════════════
void loop() {
  unsigned long now = millis();

  if (now - lastSend >= SEND_INTERVAL_MS) {
    lastSend = now;

    // 1. Measure
    float distance   = measureDistanceCm();
    float waterLevel = distanceToWaterLevel(distance);
    float rateOfRise = waterLevel - previousLevel;
    previousLevel    = waterLevel;

    Serial.println("----------------------------------------");
    Serial.printf("[SENSOR] Raw distance  : %.2f cm\n", distance);
    Serial.printf("[SENSOR] Water level   : %.2f cm\n", waterLevel);
    Serial.printf("[SENSOR] Rate of rise  : %.2f cm/cycle\n", rateOfRise);

    // 2. Build JSON
    String payload = buildJSON(waterLevel, rateOfRise);
    Serial.println("[TX]    Payload  : " + payload);

    // 3. Transmit
    bool sent = loRaSend(payload);
    if (sent) {
      Serial.println("[TX]    Status   : OK ✓");
    } else {
      Serial.println("[TX]    Status   : FAILED ✗  (packet dropped)");
    }
    Serial.println("----------------------------------------");
  }

  // Small yield — avoids WDT resets
  delay(10);
}
