\## Current source status



LYCOS-IDS2017 remains the intended primary dataset.



On 17 August 2026, the official LYCOS-IDS2017 download and

associated Le Mans University Git service were unavailable,

returning an Apache service-unavailable error.



To avoid blocking implementation work, the official CIC-IDS2017

CSV dataset has been obtained as the upstream source dataset.



No final substitution of CIC-IDS2017 for LYCOS-IDS2017 has yet

been made. The project preprocessing pipeline is being developed

so that the intended LYCOS dataset can be substituted when the

official source becomes available.



CIC-IDS2017 fallback source file:

MachineLearningCSV.zip



SHA-256:

C3F26274B36C837CCF28FFD2DBF4582941C30B3EE70A635C6E5B2F87C4727928



Download date: 7 August 2026

Source: University of New Brunswick Canadian Institute for Cybersecurity



\## Experimental Data Split



The final cleaned dataset contains 2,520,798 records and was

split using a fixed random seed of 42.



The binary target was used for stratification to preserve the

overall benign/attack class distribution.



Final partitions:



\- Training: 1,764,558 records (70%)

\- Validation: 378,120 records (15%)

\- Test: 378,120 records (15%)



Class distribution:



\### Training



\- BENIGN: 1,466,539 (83.1108%)

\- ATTACK: 298,019 (16.8892%)



\### Validation



\- BENIGN: 314,259 (83.1109%)

\- ATTACK: 63,861 (16.8891%)



\### Test



\- BENIGN: 314,259 (83.1109%)

\- ATTACK: 63,861 (16.8891%)



Automated validation confirmed that all original attack categories

remain represented in each partition.



\## Partition Integrity



Random seed: 42



train.csv  

SHA256: 87673554EA49AE6B8239C743FCBB13EC4340D0F57CA4F0EEEC8FF908FDEC0C94



validation.csv  

SHA256: BD5B40F9787ECF8B912BA307E2739EFE779F0A67F71A89F575B6142A397B8CEC



test.csv  

SHA256: EB9118DA9D87345B0F13DA32B0581BE303463B99A7B6D7350C957623B612AD98



The test partition is treated as held-out data and will not be used

during model training, hyperparameter selection or baseline development.

