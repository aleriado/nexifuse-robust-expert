# HL7 v2 Common Segments Reference

## MSH - Message Header
The MSH segment defines the intent, source, destination, and some specifics of the syntax of a message.

Key fields:
- MSH-1: Field Separator (|)
- MSH-2: Encoding Characters (^~\&)
- MSH-3: Sending Application
- MSH-4: Sending Facility
- MSH-5: Receiving Application
- MSH-6: Receiving Facility
- MSH-7: Date/Time of Message
- MSH-9: Message Type (e.g., ADT^A01, ORU^R01)
- MSH-10: Message Control ID
- MSH-11: Processing ID (P=Production, T=Training, D=Debugging)
- MSH-12: Version ID (e.g., 2.5.1)

## PID - Patient Identification
- PID-3: Patient Identifier List (MRN, SSN, etc.)
- PID-5: Patient Name (Family^Given^Middle)
- PID-7: Date of Birth
- PID-8: Administrative Sex (M, F, U)
- PID-11: Patient Address
- PID-13: Phone Number - Home
- PID-18: Patient Account Number
- PID-19: SSN Number

## PV1 - Patient Visit
- PV1-2: Patient Class (I=Inpatient, O=Outpatient, E=Emergency)
- PV1-3: Assigned Patient Location
- PV1-7: Attending Doctor
- PV1-10: Hospital Service
- PV1-19: Visit Number
- PV1-44: Admit Date/Time
- PV1-45: Discharge Date/Time

## OBX - Observation/Result
- OBX-2: Value Type (NM=Numeric, ST=String, CE=Coded Entry)
- OBX-3: Observation Identifier (LOINC code)
- OBX-5: Observation Value
- OBX-6: Units
- OBX-7: Reference Range
- OBX-8: Abnormal Flags (H=High, L=Low, A=Abnormal)
- OBX-11: Observation Result Status (F=Final, P=Preliminary)

## ORC - Common Order
- ORC-1: Order Control (NW=New, CA=Cancel, SC=Status Changed)
- ORC-2: Placer Order Number
- ORC-3: Filler Order Number
- ORC-5: Order Status
- ORC-12: Ordering Provider

## IN1 - Insurance
- IN1-1: Set ID
- IN1-2: Insurance Plan ID
- IN1-3: Insurance Company ID
- IN1-4: Insurance Company Name
- IN1-16: Name of Insured
- IN1-36: Policy Number

## Common Trigger Events
- ADT^A01: Admit/Visit Notification
- ADT^A02: Transfer a Patient
- ADT^A03: Discharge/End Visit
- ADT^A04: Register a Patient
- ADT^A08: Update Patient Information
- ORU^R01: Unsolicited Observation Result
- ORM^O01: General Order Message
- SIU^S12: Notification of New Appointment Booking
- VXU^V04: Unsolicited Vaccination Record Update
