# NHTSA crash-test signals

Public data from the NHTSA Vehicle Crash Test Database
(https://www.nhtsa.gov/research-data/research-testing-databases).

| File | Test | Vehicle | Curve | Channel |
|---|---|---|---|---|
| `v02320tsv.078` | 2320 | 1996 Dodge Neon, NCAP 56.5 km/h rigid barrier, 1354 kg | 78 | Seat – left rear, X (global), g, 12.5 kHz, as measured |
| `v03124tsv.031/.032` | 3124 | 1999 Ford Expedition, flat rigid barrier 48.5 km/h, 2460 kg | 31, 32 | Frame crossmember – rear, X, g, 12.5 kHz |
| `v04429tsv.089/.092` | 4429 | 1996 Dodge Neon (vehicle 2, 1378 kg) struck head-on by a 1999 Lincoln Navigator (vehicle 1, 2873 kg), 48.5 km/h each, 0 % offset | 89, 92 | Sill – left / right rear, X, g |
| `v04429tsv.179/.182` | 4429 | 1999 Lincoln Navigator (vehicle 1) | 179, 182 | Sill – left / right rear, X, g |

Test 4429 (TRC, 2002-09-28, "1999 NAVIGATOR INTO 1996 NEON; 0% OFFSET; 0 DEGREES"): NHTSA VAX crush Navigator 550 mm, Neon 864 mm.

Source: `https://nrd-static.nhtsa.dot.gov/tsv/vehdb/v00000/v0NN00/v0NNNN/v0NNNNtsv.CCC`
(test metadata via `https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/get-instrumentation-info/2320`).

Format: `time[s] <TAB> value`, time 0 at first contact.
