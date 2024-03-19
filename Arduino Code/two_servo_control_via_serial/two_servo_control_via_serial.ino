#include <ArduinoJson.h>
#include <Servo.h>

Servo panServo;
Servo tiltServo;

void setup()
{
  Serial.begin(9600);
  
  panServo.attach(9); // attach servo on pin 9
  tiltServo.attach(10); // attach servo on pin 10
}

void loop()
{
  if (Serial.available())
  {
    DynamicJsonDocument doc(1024); // 1KB

    // Parse the JSON document
    DeserializationError error = deserializeJson(doc, Serial);
    
    if (error) {
      Serial.print(F("deserializeJson() failed: "));
      Serial.println(error.c_str());
      return;
    }
    
    int pan = doc["pan"];  // Get pan value
    int tilt = doc["tilt"]; // Get tilt value
    
    // Limit values (Assuming servos are 160 degree type)
    pan = constrain(pan , 0, 160);
    tilt = constrain(tilt, 0, 160);

    // Move servos
    panServo.write(pan);
    tiltServo.write(tilt);
  }
}