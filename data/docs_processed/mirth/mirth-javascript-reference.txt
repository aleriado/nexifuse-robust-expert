# Mirth Connect JavaScript Transformer Reference

## Accessing Message Data

### E4X XML Access (Mirth 3.x+)
```javascript
// Access HL7 segments
var patientName = msg['PID']['PID.5']['PID.5.1'].toString();  // Family name
var mrn = msg['PID']['PID.3']['PID.3.1'].toString();          // MRN
var dob = msg['PID']['PID.7']['PID.7.1'].toString();          // Date of birth

// Access repeating segments
for each (var obx in msg['OBX']) {
    var resultValue = obx['OBX.5']['OBX.5.1'].toString();
    var units = obx['OBX.6']['OBX.6.1'].toString();
}

// Access repeating fields
for each (var id in msg['PID']['PID.3']) {
    var idValue = id['PID.3.1'].toString();
    var idType = id['PID.3.5'].toString();
}
```

### Channel Maps
```javascript
// Set values for downstream use
channelMap.put('patientMRN', mrn);
channelMap.put('patientName', patientName);

// Global map (shared across channels)
globalMap.put('facilityCode', 'HOSP01');

// Retrieve values
var mrn = channelMap.get('patientMRN');
```

### Database Operations
```javascript
var dbConn = DatabaseConnectionFactory.createDatabaseConnection(
    'org.postgresql.Driver',
    'jdbc:postgresql://localhost:5432/clinical',
    'mirth_user',
    $('db_password')
);

try {
    var result = dbConn.executeCachedQuery(
        "SELECT patient_id FROM patients WHERE mrn = ?", [mrn]
    );
    if (result.next()) {
        var patientId = result.getString('patient_id');
    }
} finally {
    dbConn.close();
}
```

### HTTP Requests
```javascript
// Using Mirth's built-in HTTP
var url = 'https://fhir.example.com/Patient?identifier=' + mrn;
var headers = new java.util.HashMap();
headers.put('Authorization', 'Bearer ' + $('access_token'));
headers.put('Accept', 'application/fhir+json');

var response = router.routeMessageByChannelId('http-sender-channel-id', msg.toString());
```

### Date Formatting
```javascript
var formatter = new java.text.SimpleDateFormat('yyyyMMddHHmmss');
var date = formatter.parse(msg['PID']['PID.7']['PID.7.1'].toString());

var outputFormatter = new java.text.SimpleDateFormat('yyyy-MM-dd');
var fhirDate = outputFormatter.format(date);
```

### Error Handling
```javascript
try {
    // Processing logic
    var result = processMessage(msg);
    if (!result) {
        throw new Error('Processing failed');
    }
} catch (e) {
    logger.error('Transform error: ' + e.message);
    // Route to error channel
    router.routeMessageByChannelId('error-handler-id', msg.toString());
    return; // Stop processing
}
```

### Logging
```javascript
logger.info('Processing message: ' + msg['MSH']['MSH.10'].toString());
logger.error('Failed to process: ' + e.message);
logger.debug('Full message: ' + msg.toString());
```
