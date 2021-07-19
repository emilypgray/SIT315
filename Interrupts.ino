const uint8_t LED = 10;
const uint8_t  magSensor = 2;

void setup() {
  pinMode(LED, OUTPUT);
  pinMode(magSensor, INPUT);
   
  attachInterrupt(digitalPinToInterrupt(magSensor), sensorChange, CHANGE);
  
  Serial.begin(9600);
}

void loop() {
}

void sensorChange(){
    digitalWrite(LED, digitalRead(magSensor));
    Serial.print("Sensor ActivatedState, LED: ");
    Serial.println(digitalRead(magSensor));
}
