// ============================================================
//  sos_code.ino  —  Emergency SOS Button Device (ESP32 + LoRa)
//  Push button on GPIO 4 → sends SOS alert via LoRa
// ============================================================

#include <SPI.h>
#include <LoRa.h>

// ── LoRa pin mapping ────────────────────────────────────────
#define LORA_NSS   5
#define LORA_RST   14
#define LORA_DIO0  26
#define LORA_FREQ  433E6

// ── Button pin ──────────────────────────────────────────────
#define BUTTON_PIN  4

// ── Debounce & repeat-guard tuning ─────────────────────────
#define DEBOUNCE_MS       50     // minimum stable press duration
#define SOS_COOLDOWN_MS   3000  // ignore further presses for 3 s after an SOS send

// ── LoRa RF settings (must match gateway) ──────────────────
#define LORA_INIT_RETRIES 5

// ── Identity ────────────────────────────────────────────────
const char* NODE_ID      = "N3";
const char* SOS_MESSAGE  = "Flood emergency detected";

// ── Button state machine ────────────────────────────────────
enum ButtonState { BTN_IDLE, BTN_DEBOUNCING, BTN_HELD };
ButtonState btnState          = BTN_IDLE;
unsigned long btnEventTime    = 0;    // timestamp of last edge
unsigned long lastSosSentAt   = 0;    // timestamp of last SOS transmission
bool          prevRawLevel    = HIGH; // last raw digitalRead result


// ════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════

// Build SOS JSON payload.
String buildSosJSON() {
  String json = "{";
  json += "\"type\":\"sos\",";
  json += "\"node_id\":\"" + String(NODE_ID) + "\",";
  json += "\"message\":\"" + String(SOS_MESSAGE) + "\"";
  json += "}";
  return json;
}

// Send payload via LoRa. Returns true on success.
bool loRaSend(const String& payload) {
  LoRa.beginPacket();
  LoRa.print(payload);
  return LoRa.endPacket();
}

// Attempt LoRa initialisation with retries.
void initLoRa() {
  LoRa.setPins(LORA_NSS, LORA_RST, LORA_DIO0);

  for (int attempt = 1; attempt <= LORA_INIT_RETRIES; attempt++) {
    Serial.printf("[LoRa] Init attempt %d / %d ...\n", attempt, LORA_INIT_RETRIES);
    if (LoRa.begin(LORA_FREQ)) {
      LoRa.setSpreadingFactor(9);
      LoRa.setSignalBandwidth(125E3);
      LoRa.setCodingRate4(5);
      LoRa.setTxPower(20);          // max power for emergency transmissions
      Serial.println("[LoRa] Initialised successfully  ✓");
      return;
    }
    Serial.println("[LoRa] Failed. Retrying in 2 s ...");
    delay(2000);
  }

  Serial.println("[LoRa] FATAL: Could not initialise LoRa. Halting.");
  while (true) { delay(1000); }
}

// Send SOS and print confirmation.
void sendSOS() {
  String payload = buildSosJSON();

  Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
  Serial.println("  SOS ALERT TRIGGERED");
  Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
  Serial.println("[SOS] Payload : " + payload);

  bool ok = loRaSend(payload);

  if (ok) {
    Serial.println("[SOS] Transmitted successfully  ✓");
    Serial.println("[SOS] Gateway should forward to backend.");
  } else {
    Serial.println("[SOS] Transmission FAILED  ✗  — check LoRa module.");
  }

  Serial.println("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
  lastSosSentAt = millis();
}


// ════════════════════════════════════════════════════════════
//  setup()
// ════════════════════════════════════════════════════════════
void setup() {
  Serial.begin(9600);
  while (!Serial) { delay(10); }
  delay(500);

  Serial.println("========================================");
  Serial.println("  FloodWatch SOS Device — " + String(NODE_ID));
  Serial.println("========================================");

  // Button — internal pull-up: LOW = pressed, HIGH = released
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.printf("[BTN]  Button on GPIO%d (active LOW, internal pull-up)\n", BUTTON_PIN);

  // LoRa
  SPI.begin();
  initLoRa();

  Serial.println("[SYS] Ready. Press button to send SOS.");
}


// ════════════════════════════════════════════════════════════
//  loop()
// ════════════════════════════════════════════════════════════
void loop() {
  unsigned long now     = millis();
  bool          rawLevel = digitalRead(BUTTON_PIN);  // LOW when pressed

  // ── Cooldown guard (global, regardless of state) ─────────
  bool inCooldown = (now - lastSosSentAt) < SOS_COOLDOWN_MS;

  // ── State machine ─────────────────────────────────────────
  switch (btnState) {

    case BTN_IDLE:
      if (rawLevel == LOW && prevRawLevel == HIGH) {
        // Falling edge detected — start debounce timer
        btnEventTime = now;
        btnState     = BTN_DEBOUNCING;
        Serial.println("[BTN] Press detected — debouncing ...");
      }
      break;

    case BTN_DEBOUNCING:
      if (rawLevel == HIGH) {
        // Button released before debounce window — noise, ignore
        Serial.println("[BTN] Bounce filtered — ignoring.");
        btnState = BTN_IDLE;
      } else if (now - btnEventTime >= DEBOUNCE_MS) {
        // Signal stable for full debounce window → confirmed press
        btnState = BTN_HELD;
        Serial.println("[BTN] Press confirmed.");

        if (inCooldown) {
          unsigned long remaining = SOS_COOLDOWN_MS - (now - lastSosSentAt);
          Serial.printf("[SOS] Cooldown active — %.1f s remaining. Ignoring.\n",
                        remaining / 1000.0f);
        } else {
          sendSOS();
        }
      }
      break;

    case BTN_HELD:
      if (rawLevel == HIGH) {
        // Button released — return to idle, ready for next press
        btnState = BTN_IDLE;
        Serial.println("[BTN] Button released — idle.");
      }
      break;
  }

  prevRawLevel = rawLevel;

  delay(5);   // 5 ms polling — fine-grained enough for a 50 ms debounce
}
