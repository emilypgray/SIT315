
// declare two output pins as volatile
volatile int LEDState = 1;
volatile int LED2State = 1;


void setup() {

  // clear all interrupts
  cli();

  // enable port b for interrupts
  PCICR |= 0b00000001;

  //enable pin PB0 (digital pin 8)
  PCMSK0 |= 0b00000001;

  // clear the registers
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;

  // set the led to flash every 0.2 seconds. 
  // 5Hz = (16,000,000/((3124+1)*1024))
  // set output compare register 
  OCR1A = 3124;

  // enable prescalar of 1024
  TCCR1B |= (1 << CS12) | (1 << CS10);

  // output compare match A interrupt enable
  TIMSK1 |= (1 << OCIE1A);

  // CTC
  TCCR1B |= (1 << WGM12);

  // enable interrupts
  sei();

  // configure input and output pins
  pinMode(8, INPUT_PULLUP);
  pinMode(3, INPUT);
  pinMode(2, INPUT);
  
  pinMode(13, OUTPUT);
  pinMode(11, OUTPUT);

  // attach interrupts to external interrupt digitals pins 2 and 3
  attachInterrupt(digitalPinToInterrupt(2), buttonChange, CHANGE);
  attachInterrupt(digitalPinToInterrupt(3), sensorChange, CHANGE);
  
  Serial.begin(9600);
}

void loop() {
}

// declare ISRs

void buttonChange(){
    // flip LED state
    LEDState = !LEDState;

    // write to pin
    digitalWrite(13, LEDState);
    
    Serial.print("Touch Sensor Activated, LED State: ");
    Serial.println(LEDState);
}

void sensorChange(){
    // read pin input
    LEDState = digitalRead(3);

    // write to pin
    digitalWrite(13, LEDState);
    
    Serial.print("Mag Sensor Activated, LED State: ");
    Serial.println(LEDState);
}

ISR(PCINT0_vect){
  // flip LED state
  LEDState = !LEDState;

  // write to pin
  digitalWrite(13, LEDState);

  Serial.print("Button Sensor Activated, LED State: ");
  Serial.println(LEDState);
}

ISR(TIMER1_COMPA_vect){
  // flip LED state
  LED2State = !LED2State;

  // write to pin
  digitalWrite(11, LED2State);

  Serial.print("LED 2 State: ");
  Serial.println(LED2State);
}
