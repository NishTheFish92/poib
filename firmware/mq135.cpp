// Standalone MQ135 test sketch — kept for reference.
// Active code moved to main.cpp.

// #include <EEPROM.h>
//
// const int MQ135_PIN = 7;
// const float RL = 10.0;
// const float Vc = 3.3;
// float R0;
//
// void setup() {
//   Serial.begin(115200);
//   EEPROM.begin(512);
//   EEPROM.get(0, R0);
//   if(isnan(R0) || R0 <= 0) {
//     Serial.println("ERROR: No valid R0 found! Run calibration first.");
//     while(1);
//   }
//   Serial.print("Loaded R0: "); Serial.println(R0);
// }
//
// void loop() {
//   float Vout = analogRead(MQ135_PIN) * (Vc / 4095.0);
//   float Rs = ((Vc / Vout) - 1) * RL;
//   float ratio = Rs / R0;
//   float ppm = 116.6020682 * pow(ratio, -2.769034857);
//   Serial.print("CO2: "); Serial.print(ppm); Serial.println(" ppm");
//   delay(1000);
// }
