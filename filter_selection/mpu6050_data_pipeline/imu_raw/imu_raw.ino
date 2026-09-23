#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;
const unsigned long SAMPLE_INTERVAL_MS = 333;

int16_t axRaw = 0;
int16_t ayRaw = 0;
int16_t azRaw = 0;
int16_t tempRaw = 0;
int16_t gxRaw = 0;
int16_t gyRaw = 0;
int16_t gzRaw = 0;

bool readImuSample() {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x3B);
  if (Wire.endTransmission(false) != 0) {
    return false;
  }

  const uint8_t expectedBytes = 14;
  const uint8_t bytesRead = Wire.requestFrom(
    static_cast<uint8_t>(MPU_ADDR),
    static_cast<uint8_t>(expectedBytes),
    static_cast<uint8_t>(true)
  );
  if (bytesRead != expectedBytes) {
    return false;
  }

  axRaw = (Wire.read() << 8) | Wire.read();
  ayRaw = (Wire.read() << 8) | Wire.read();
  azRaw = (Wire.read() << 8) | Wire.read();
  tempRaw = (Wire.read() << 8) | Wire.read();
  gxRaw = (Wire.read() << 8) | Wire.read();
  gyRaw = (Wire.read() << 8) | Wire.read();
  gzRaw = (Wire.read() << 8) | Wire.read();

  return true;
}

void printCsvHeader() {
  Serial.println(F("millis,ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw,temp_raw,temp_c"));
}

void printCsvRow(unsigned long timestampMs) {
  Serial.print(timestampMs);
  Serial.print(',');
  Serial.print(axRaw);
  Serial.print(',');
  Serial.print(ayRaw);
  Serial.print(',');
  Serial.print(azRaw);
  Serial.print(',');
  Serial.print(gxRaw);
  Serial.print(',');
  Serial.print(gyRaw);
  Serial.print(',');
  Serial.print(gzRaw);
  Serial.print(',');
  Serial.print(tempRaw);
  Serial.print(',');
  Serial.println((tempRaw / 340.0f) + 36.53f, 2);
}

void setup() {
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  Serial.begin(115200);
  delay(200);

  Serial.println(F("# MPU6050 raw CSV stream started"));
  printCsvHeader();
}

void loop() {
  Serial.println("1");

  if (!readImuSample()) {
    Serial.println(F("# Failed to read MPU6050 sample"));
    delay(SAMPLE_INTERVAL_MS);
    return;
  }

  printCsvRow(millis());
  delay(SAMPLE_INTERVAL_MS);
}
