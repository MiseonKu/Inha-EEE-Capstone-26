#include <Wire.h>

const uint8_t MPU_ADDR_1 = 0x68;
const uint8_t MPU_ADDR_2 = 0x69;
const uint8_t WHO_AM_I_REG = 0x75;

bool consumeTimeoutFlag() {
  const bool timedOut = Wire.getWireTimeoutFlag();
  if (timedOut) {
    Wire.clearWireTimeoutFlag();
  }
  return timedOut;
}

uint8_t probeAddress(uint8_t address, bool &timedOut) {
  Wire.beginTransmission(address);
  const uint8_t error = Wire.endTransmission(true);
  timedOut = consumeTimeoutFlag();
  return error;
}

bool readWhoAmI(uint8_t address, uint8_t &value, bool &timedOut) {
  Wire.beginTransmission(address);
  Wire.write(WHO_AM_I_REG);
  if (Wire.endTransmission(false) != 0) {
    timedOut = consumeTimeoutFlag();
    return false;
  }
  timedOut = consumeTimeoutFlag();
  if (timedOut) {
    return false;
  }

  const uint8_t bytesRead = Wire.requestFrom(address, static_cast<uint8_t>(1), static_cast<uint8_t>(true));
  timedOut = consumeTimeoutFlag();
  if (bytesRead != 1) {
    return false;
  }

  value = Wire.read();
  return true;
}

void printProbeResult(uint8_t address) {
  Serial.print(F("Checking 0x"));
  if (address < 16) {
    Serial.print('0');
  }
  Serial.print(address, HEX);
  Serial.print(F(" ... "));

  bool timedOut = false;
  const uint8_t error = probeAddress(address, timedOut);
  if (timedOut) {
    Serial.println(F("timeout"));
    return;
  }

  if (error != 0) {
    Serial.print(F("no response (err="));
    Serial.print(error);
    Serial.println(')');
    return;
  }

  uint8_t whoAmI = 0;
  if (readWhoAmI(address, whoAmI, timedOut)) {
    Serial.print(F("responded, WHO_AM_I = 0x"));
    if (whoAmI < 16) {
      Serial.print('0');
    }
    Serial.println(whoAmI, HEX);
  } else if (timedOut) {
    Serial.println(F("responded, but WHO_AM_I read timed out"));
  } else {
    Serial.println(F("responded, but WHO_AM_I read failed"));
  }
}

void scanI2cBus() {
  Serial.println(F("Full I2C scan:"));

  uint8_t foundCount = 0;
  for (uint8_t address = 1; address < 127; address++) {
    Wire.beginTransmission(address);
    const uint8_t error = Wire.endTransmission(true);
    const bool timedOut = consumeTimeoutFlag();

    if (timedOut) {
      Serial.print(F("  Timeout at 0x"));
      if (address < 16) {
        Serial.print('0');
      }
      Serial.println(address, HEX);
      continue;
    }

    if (error == 0) {
      Serial.print(F("  Found device at 0x"));
      if (address < 16) {
        Serial.print('0');
      }
      Serial.println(address, HEX);
      foundCount++;
    }
  }

  if (foundCount == 0) {
    Serial.println(F("  No I2C devices found."));
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Wire.begin();
  Wire.setWireTimeout(3000, true);

  Serial.println();
  Serial.println(F("=== MPU6050 Address Check ==="));
  Serial.println(F("Wire timeout: 3000us"));
  printProbeResult(MPU_ADDR_1);
  printProbeResult(MPU_ADDR_2);
  scanI2cBus();
  Serial.println(F("============================="));
  Serial.println(F("Reset the board to scan again."));
}

void loop() {
}
