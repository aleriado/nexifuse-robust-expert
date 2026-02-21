# FHIR R4 Patient Resource

## Overview
The Patient resource covers data about patients and animals involved in healthcare activities.

## Key Elements (US Core Must Support)

### identifier
- system: OID or URI for the identifier namespace
- value: The actual identifier value (e.g., MRN)
- type: CodeableConcept (MR=Medical Record, SS=Social Security)

### name
- use: usual | official | temp | nickname | anonymous | old | maiden
- family: Family name (surname)
- given: Given names (first, middle)
- prefix: Mr., Mrs., Dr.
- suffix: Jr., III

### telecom
- system: phone | fax | email | pager | url | sms
- value: The actual contact point
- use: home | work | temp | old | mobile

### gender
- male | female | other | unknown

### birthDate
- YYYY-MM-DD format

### address
- use: home | work | temp | old | billing
- line: Street address lines
- city, state, postalCode, country

### communication
- language: CodeableConcept (BCP-47)
- preferred: boolean

## Example
```json
{
  "resourceType": "Patient",
  "id": "example",
  "identifier": [{
    "type": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v2-0203", "code": "MR"}]},
    "system": "http://hospital.example.org",
    "value": "12345"
  }],
  "name": [{"use": "official", "family": "Smith", "given": ["John", "Michael"]}],
  "gender": "male",
  "birthDate": "1970-01-01",
  "address": [{"use": "home", "line": ["123 Main St"], "city": "Anytown", "state": "CA", "postalCode": "90210"}]
}
```

## HL7 v2 to FHIR Mapping
| HL7 v2 Field | FHIR Element |
|---|---|
| PID-3 | Patient.identifier |
| PID-5 | Patient.name |
| PID-7 | Patient.birthDate |
| PID-8 | Patient.gender |
| PID-11 | Patient.address |
| PID-13 | Patient.telecom |
| PID-18 | Patient.identifier (account) |
