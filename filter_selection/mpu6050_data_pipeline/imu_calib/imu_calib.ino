#include <EEPROM.h>
#include <Wire.h>
#include <ctype.h>
#include <math.h>
#include <string.h>

const uint8_t MPU_ADDR = 0x68;
const unsigned long SAMPLE_INTERVAL_MS = 333;
const unsigned long CALIBRATION_DURATION_MS = 10000;
const unsigned long CALIBRATION_DELAY_MS = 5;
const uint32_t CALIBRATION_MAGIC = 0x494D5531UL;
const size_t COMMAND_BUFFER_SIZE = 32;
const float ACCEL_1G_RAW = 16384.0f;

struct ImuSample {
  int16_t ax;
  int16_t ay;
  int16_t az;
  int16_t temp;
  int16_t gx;
  int16_t gy;
  int16_t gz;
};

struct CalibrationData {
  uint32_t magic;
  float axOffset;
  float ayOffset;
  float azOffset;
  float gxOffset;
  float gyOffset;
  float gzOffset;
};

CalibrationData calibrationData = {0, 0, 0, 0, 0, 0, 0};
bool calibrationLoaded = false;

char commandBuffer[COMMAND_BUFFER_SIZE];
size_t commandLength = 0;
unsigned long lastSampleTimeMs = 0;

int16_t axRaw = 0;
int16_t ayRaw = 0;
int16_t azRaw = 0;
int16_t tempRaw = 0;
int16_t gxRaw = 0;
int16_t gyRaw = 0;
int16_t gzRaw = 0;

void resetCalibration(bool keepMagic) {
  calibrationData.magic = keepMagic ? CALIBRATION_MAGIC : 0;
  calibrationData.axOffset = 0.0f;
  calibrationData.ayOffset = 0.0f;
  calibrationData.azOffset = 0.0f;
  calibrationData.gxOffset = 0.0f;
  calibrationData.gyOffset = 0.0f;
  calibrationData.gzOffset = 0.0f;
}

bool loadCalibrationFromEeprom() {
  EEPROM.get(0, calibrationData);
  if (calibrationData.magic != CALIBRATION_MAGIC) {
    resetCalibration(false);
    calibrationLoaded = false;
    return false;
  }

  calibrationLoaded = true;
  return true;
}

void saveCalibrationToEeprom() {
  calibrationData.magic = CALIBRATION_MAGIC;
  EEPROM.put(0, calibrationData);
  calibrationLoaded = true;
}

void clearCalibrationFromEeprom() {
  resetCalibration(false);
  EEPROM.put(0, calibrationData);
  calibrationLoaded = false;
}

bool readImuSampleRaw(ImuSample& sample) {
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

  sample.ax = (Wire.read() << 8) | Wire.read();
  sample.ay = (Wire.read() << 8) | Wire.read();
  sample.az = (Wire.read() << 8) | Wire.read();
  sample.temp = (Wire.read() << 8) | Wire.read();
  sample.gx = (Wire.read() << 8) | Wire.read();
  sample.gy = (Wire.read() << 8) | Wire.read();
  sample.gz = (Wire.read() << 8) | Wire.read();

  return true;
}

int16_t applyOffsetToValue(int16_t rawValue, float offset) {
  float corrected = static_cast<float>(rawValue) - offset;

  if (corrected > 32767.0f) {
    corrected = 32767.0f;
  } else if (corrected < -32768.0f) {
    corrected = -32768.0f;
  }

  return static_cast<int16_t>(corrected >= 0.0f ? corrected + 0.5f : corrected - 0.5f);
}

void applyCalibration(const ImuSample& rawSample, ImuSample& calibratedSample) {
  calibratedSample = rawSample;

  if (!calibrationLoaded) {
    return;
  }

  calibratedSample.ax = applyOffsetToValue(rawSample.ax, calibrationData.axOffset);
  calibratedSample.ay = applyOffsetToValue(rawSample.ay, calibrationData.ayOffset);
  calibratedSample.az = applyOffsetToValue(rawSample.az, calibrationData.azOffset);
  calibratedSample.gx = applyOffsetToValue(rawSample.gx, calibrationData.gxOffset);
  calibratedSample.gy = applyOffsetToValue(rawSample.gy, calibrationData.gyOffset);
  calibratedSample.gz = applyOffsetToValue(rawSample.gz, calibrationData.gzOffset);
}

void copySampleToGlobals(const ImuSample& sample) {
  axRaw = sample.ax;
  ayRaw = sample.ay;
  azRaw = sample.az;
  tempRaw = sample.temp;
  gxRaw = sample.gx;
  gyRaw = sample.gy;
  gzRaw = sample.gz;
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

void printHelp() {
  Serial.println(F("# Commands: CALIBRATE (or CAL), SHOW, CLEAR, HEADER, HELP"));
}

void printCalibration() {
  if (!calibrationLoaded) {
    Serial.println(F("# Calibration: not set"));
    return;
  }

  Serial.print(F("# accel_offset_raw="));
  Serial.print(calibrationData.axOffset, 3);
  Serial.print(',');
  Serial.print(calibrationData.ayOffset, 3);
  Serial.print(',');
  Serial.println(calibrationData.azOffset, 3);

  Serial.print(F("# gyro_offset_raw="));
  Serial.print(calibrationData.gxOffset, 3);
  Serial.print(',');
  Serial.print(calibrationData.gyOffset, 3);
  Serial.print(',');
  Serial.println(calibrationData.gzOffset, 3);
}

void calibrateImu() {
  Serial.println(F("# CALIBRATE command received. Keep the IMU completely still for 10 seconds."));

  long axSum = 0;
  long aySum = 0;
  long azSum = 0;
  long gxSum = 0;
  long gySum = 0;
  long gzSum = 0;
  ImuSample rawSample;
  uint16_t collected = 0;
  unsigned long calibrationStartMs = millis();
  unsigned long lastProgressReportMs = calibrationStartMs;

  while (millis() - calibrationStartMs < CALIBRATION_DURATION_MS) {
    if (!readImuSampleRaw(rawSample)) {
      Serial.println(F("# Read failed during calibration, retrying..."));
      delay(20);
      continue;
    }

    axSum += rawSample.ax;
    aySum += rawSample.ay;
    azSum += rawSample.az;
    gxSum += rawSample.gx;
    gySum += rawSample.gy;
    gzSum += rawSample.gz;
    collected++;

    const unsigned long now = millis();
    if (now - lastProgressReportMs >= 5000) {
      Serial.print(F("# Calibration progress: "));
      Serial.print((now - calibrationStartMs) / 1000);
      Serial.print(F("s / "));
      Serial.print(CALIBRATION_DURATION_MS / 1000);
      Serial.println(F("s"));
      lastProgressReportMs = now;
    }

    delay(CALIBRATION_DELAY_MS);
  }

  if (collected == 0) {
    Serial.println(F("# Calibration failed: no valid samples collected."));
    return;
  }

  const float axAvg = axSum / static_cast<float>(collected);
  const float ayAvg = aySum / static_cast<float>(collected);
  const float azAvg = azSum / static_cast<float>(collected);
  const float gxAvg = gxSum / static_cast<float>(collected);
  const float gyAvg = gySum / static_cast<float>(collected);
  const float gzAvg = gzSum / static_cast<float>(collected);

  float expectedAx = 0.0f;
  float expectedAy = 0.0f;
  float expectedAz = 0.0f;
  const char* gravityAxisName = "az_raw";

  const float absAx = fabs(axAvg);
  const float absAy = fabs(ayAvg);
  const float absAz = fabs(azAvg);

  if (absAx >= absAy && absAx >= absAz) {
    expectedAx = axAvg >= 0.0f ? ACCEL_1G_RAW : -ACCEL_1G_RAW;
    gravityAxisName = "ax_raw";
  } else if (absAy >= absAx && absAy >= absAz) {
    expectedAy = ayAvg >= 0.0f ? ACCEL_1G_RAW : -ACCEL_1G_RAW;
    gravityAxisName = "ay_raw";
  } else {
    expectedAz = azAvg >= 0.0f ? ACCEL_1G_RAW : -ACCEL_1G_RAW;
    gravityAxisName = "az_raw";
  }

  resetCalibration(true);
  calibrationData.axOffset = axAvg - expectedAx;
  calibrationData.ayOffset = ayAvg - expectedAy;
  calibrationData.azOffset = azAvg - expectedAz;
  calibrationData.gxOffset = gxAvg;
  calibrationData.gyOffset = gyAvg;
  calibrationData.gzOffset = gzAvg;
  saveCalibrationToEeprom();

  Serial.print(F("# Gravity axis detected: "));
  Serial.println(gravityAxisName);
  printCalibration();
  Serial.println(F("# Calibration complete and saved to EEPROM."));
}

void processCommand(const char* command) {
  if (strcmp(command, "CAL") == 0 || strcmp(command, "CALIBRATE") == 0) {
    calibrateImu();
    printCsvHeader();
    return;
  }

  if (strcmp(command, "SHOW") == 0 || strcmp(command, "PRINTCAL") == 0 || strcmp(command, "STATUS") == 0) {
    printCalibration();
    return;
  }

  if (strcmp(command, "CLEAR") == 0) {
    clearCalibrationFromEeprom();
    Serial.println(F("# Calibration cleared from EEPROM."));
    return;
  }

  if (strcmp(command, "HEADER") == 0) {
    printCsvHeader();
    return;
  }

  if (strcmp(command, "HELP") == 0) {
    printHelp();
    return;
  }

  Serial.print(F("# Unknown command: "));
  Serial.println(command);
}

void handleSerialInput() {
  while (Serial.available() > 0) {
    const char incoming = Serial.read();

    if (incoming == '\n' || incoming == '\r') {
      if (commandLength > 0) {
        commandBuffer[commandLength] = '\0';
        processCommand(commandBuffer);
        commandLength = 0;
      }
      continue;
    }

    if (commandLength < (COMMAND_BUFFER_SIZE - 1)) {
      commandBuffer[commandLength++] = static_cast<char>(toupper(static_cast<unsigned char>(incoming)));
    }
  }
}

void setup() {
  Wire.begin();
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(0x6B);
  Wire.write(0);
  Wire.endTransmission(true);

  Serial.begin(115200);
  delay(200);

  Serial.println(F("# MPU6050 calibrated CSV stream started"));
  if (loadCalibrationFromEeprom()) {
    Serial.println(F("# Saved calibration loaded from EEPROM."));
    printCalibration();
  } else {
    Serial.println(F("# No saved calibration found. Send CALIBRATE while the IMU is still."));
  }
  printHelp();
  printCsvHeader();
}

void loop() {
  handleSerialInput();

  const unsigned long now = millis();
  if (now - lastSampleTimeMs < SAMPLE_INTERVAL_MS) {
    return;
  }
  lastSampleTimeMs = now;

  ImuSample rawSample;
  if (!readImuSampleRaw(rawSample)) {
    Serial.println(F("# Failed to read MPU6050 sample"));
    return;
  }

  ImuSample outputSample;
  applyCalibration(rawSample, outputSample);
  outputSample.temp = rawSample.temp;

  copySampleToGlobals(outputSample);
  printCsvRow(now);
}
