# NHTSA crash-test signals

Public data from the NHTSA Vehicle Crash Test Database
(https://www.nhtsa.gov/research-data/research-testing-databases).

| File | Test | Vehicle | Curve | Channel |
|---|---|---|---|---|
| `v02320tsv.078` | 2320 | 1996 Dodge Neon, NCAP 56.5 km/h rigid barrier, 1354 kg | 78 | Seat – left rear, X (global), g, 12.5 kHz, as measured |

Source: `https://nrd-static.nhtsa.dot.gov/tsv/vehdb/v00000/v02300/v02320/v02320tsv.078`
(test metadata via `https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/get-instrumentation-info/2320`).

Format: `time[s] <TAB> value`, time 0 at first contact.
