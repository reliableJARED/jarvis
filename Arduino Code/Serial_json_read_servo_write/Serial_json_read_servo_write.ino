/*
#include <ArduinoJson.h>
#include <Servo.h>

// Define the maximum length of the incoming JSON message
const int MAX_JSON_SIZE = 256;

Servo vodka_servo;  // create servo object to control a servo
Servo orange_juice_servo;  // create servo object to control a servo


void setup() {
  Serial.begin(9600); // Initialize serial communication
  vodka_servo.attach(9);  // attaches the servo on pin 9 to the servo object.  ONly 9 and 10 have servo
  orange_juice_servo.attach(10); //attach servo to pin 10
  //reset to position 0
  vodka_servo.write(0);
  orange_juice_servo.write(0);
}

void loop() {
  if (Serial.available()) {
    // Read the incoming JSON message until a newline character is received
    char jsonMessage[MAX_JSON_SIZE];
    int jsonLength = Serial.readBytesUntil('\n', jsonMessage, MAX_JSON_SIZE);
    jsonMessage[jsonLength] = '\0'; // Null-terminate the received message

    // Parse the JSON message
    StaticJsonDocument<MAX_JSON_SIZE> jsonDoc;
    DeserializationError error = deserializeJson(jsonDoc, jsonMessage);

    if (error) {
      Serial.print("Deserialization failed: ");
      Serial.println(error.c_str());
      return;
    }

    // Access the "ingredients" object
    JsonObject ingredients = jsonDoc["ingredients"];

    // Loop through each ingredient and assign the values to variables
    for (JsonPair ingredient : ingredients) {
      const char *name = ingredient.key().c_str();
      int value = ingredient.value().as<int>();

      // Print to the serial connection
      Serial.println("Ingredients received");
      Serial.flush();

      // Assign the ingredient value to a variable
      if (strcmp(name, "vodka") == 0) {
        int vodkaValue = value;
        //servo max is 170
          if(vodkaValue >= 169){
            vodkaValue = 169;
          }
        // TODO: Do something with the vodka variable
          vodka_servo.write(vodkaValue);
          //Serial.flush();
          Serial.println("vodka done"); //println auto adds \n\r
          delay(3000);//finish pour

           //reset to position 0
          vodka_servo.write(0);

      } else if (strcmp(name, "orange juice") == 0) {
        int ojValue = value;
        //servo max is 170
          if(ojValue >= 169){
            ojValue = 169;
          }
        // TODO: Do something with the juiceValue variable
        orange_juice_servo.write(ojValue);
        //Serial.flush();
        Serial.println("orange juice done"); //println auto adds \n\r
        delay(3000); //finish pour

        orange_juice_servo.write(0);
      }
      // Add more conditions for other ingredients as needed
    }
  }
}
*/

#include <ArduinoJson.h>
#include <Servo.h>

#define MAX_JSON_SIZE 64

Servo vodka_servo;  // create servo object to control a servo
Servo orange_juice_servo;  // create servo object to control a servo

void setup() {
  Serial.begin(9600);
  
  vodka_servo.attach(9);
  orange_juice_servo.attach(10);
  
  vodka_servo.write(0);
  orange_juice_servo.write(0);
}

void loop() {
  if (Serial.available()) {
    char jsonMessage[MAX_JSON_SIZE];

    // Read the incoming JSON message until a newline character is received
    Serial.readBytesUntil('\n', jsonMessage, MAX_JSON_SIZE);

    // Parse the JSON message
    StaticJsonDocument<MAX_JSON_SIZE> jsonDoc;
    DeserializationError error = deserializeJson(jsonDoc, jsonMessage);

    if (error) {
      Serial.print("Deserialization failed: ");
      Serial.println(error.c_str());
      return;
    }

    if (jsonDoc.containsKey("vodka")) {
      int vodkaValue = jsonDoc["vodka"];
      Serial.println(vodkaValue);
      //servo max is 170
      if(vodkaValue >= 169){
            vodkaValue = 169;
          }
      // TODO: Do something with the vodka variable
      vodka_servo.write(vodkaValue);
    }
    Serial.flush();
  }
}