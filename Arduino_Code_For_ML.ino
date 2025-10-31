#include <Wire.h>
#include <Adafruit_MotorShield.h>

// Initialize motor shield
Adafruit_MotorShield AFMS = Adafruit_MotorShield(); 
Adafruit_DCMotor *motor1 = AFMS.getMotor(1); 
Adafruit_DCMotor *motor3 = AFMS.getMotor(3); 

bool isRunning = false;
unsigned long lastRun = 0;
const unsigned long DEBOUNCE = 6000; // 6s minimum between runs

void setup() {
  Serial.begin(9600);
  AFMS.begin();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "0" && !isRunning && millis() - lastRun > DEBOUNCE) {
        isRunning = true;
        lastRun = millis();

        // --- Open ---
        motor1->setSpeed(80);
        motor3->setSpeed(80);
        motor1->run(BACKWARD);
        motor3->run(BACKWARD);
        delay(275);
        motor1->run(RELEASE);
        motor3->run(RELEASE);
        delay(5000); // stay open

        // --- Close ---
        motor1->setSpeed(40);
        motor3->setSpeed(40);
        motor1->run(FORWARD);
        motor3->run(FORWARD);
        delay(275);
        motor1->run(RELEASE);
        motor3->run(RELEASE);

        Serial.println("Done");
        isRunning = false;
    }
  }
}
