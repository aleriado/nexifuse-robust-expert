# Epic FHIR R4 API Reference

## Base URL
- Sandbox: https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4
- Production: Varies by organization

## Authentication
Epic uses SMART on FHIR (OAuth 2.0) for authentication.

### Backend Services (System-to-System)
1. Register app at App Orchard (open.epic.com)
2. Generate RSA key pair, upload public key
3. Create signed JWT assertion
4. Exchange JWT for access token at token endpoint
5. Use Bearer token for API calls

### Token Request
```
POST /oauth2/token
Content-Type: application/x-www-form-urlencoded

grant_type=client_credentials
&client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer
&client_assertion={signed_jwt}
```

## Common Endpoints

### Patient Search
```
GET /Patient?identifier=http://hospital.org|{MRN}
GET /Patient?family={lastName}&given={firstName}&birthdate={YYYY-MM-DD}
```

### AllergyIntolerance
```
GET /AllergyIntolerance?patient={patientId}
```

### Condition (Problems)
```
GET /Condition?patient={patientId}&category=problem-list-item
```

### MedicationRequest
```
GET /MedicationRequest?patient={patientId}&status=active
```

### Observation (Labs/Vitals)
```
GET /Observation?patient={patientId}&category=laboratory
GET /Observation?patient={patientId}&category=vital-signs
```

### Bulk Data Export
```
# Kick-off
GET /Patient/$export
Accept: application/fhir+json
Prefer: respond-async

# Poll status (from Content-Location header)
GET /{status-url}

# Download NDJSON files from output links
GET /{file-url}
```

## Rate Limits
- 100 requests per second per client
- Bulk exports: 1 concurrent export per client
