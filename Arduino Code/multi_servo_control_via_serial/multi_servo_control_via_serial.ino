/*************************************************** 
  This is uses a Adafruit 16-channel PWM & Servo driver board PCA9685
  drivers use I2C to communicate, 2 pins are required to  
  interface.
*/
#include <Wire.h>  // Include the Wire library to communicate with I2C devices
#include <Adafruit_PWMServoDriver.h> // Include the Adafruit PWMServoDriver library to control servos

// Create an Adafruit_PWMServoDriver object called pwm
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

#define SERVOMIN  125  // minimum pulse length count (out of 4096)
#define SERVOMAX  600  // maximum pulse length count (out of 4096)
#define SERVOMID  312  // middle pulse length count (out of 4096)
#define SERVO_FREQ 50 // Analog servos run at ~50 Hz updates

// Servo mapping: array index corresponds to the letter code - 65
int servos[8] = {0, 1, 2, 3, 4, 5, 6, 7};  

// Setup method that runs once when the program starts
void setup() {
  Serial.begin(9600); // Start serial communication at 9600 bits per second
  pwm.begin(); // Initialize the pwm object
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ); // Set the frequency of the pwm signal to 60 Hz

// Set all the servo motors to the middle position initially
  for (int i = 0; i < 8; i++) {
    pwm.setPWM(servos[i], 0, SERVOMID);
  }
}
/*
SERVO LETTER MAPPING
A - wrist
B - thumb
C - pinky
D - ring
E - middle
F - index
G - Tilt
H - pan
*/
void loop() {
  //listen to incoming serial messages
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); // expects input like A75,B120,C45 with no spaces
    // Convert all the letters in the input string to uppercase
    input.toUpperCase();
     // Tokenize the input string with a space delimiter
    char * strs = strtok(input.c_str(), ",");
    while (strs) {
      // Map the current token to a position value between the SERVOMIN and SERVOMAX
      int pos = map(atoi(strs + 1), 0, 150, SERVOMIN, SERVOMAX);
      // Update the position of the corresponding servo motor
      pwm.setPWM(servos[strs[0] - 'A'], 0, pos);
      //Serial.println(String() + "Moved servo " + strs[0] + " to position " + atoi(strs + 1));
       // Get the next token
      strs = strtok(NULL, ",");
    }
  }
}