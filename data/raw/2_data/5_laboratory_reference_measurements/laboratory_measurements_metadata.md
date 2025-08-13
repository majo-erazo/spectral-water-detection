## lab_measurements.csv
**Brief description**
Table containing the measurements of conventional pollution indicators done in the laboratory after grabbing samples from the flume
**File structure**
|index|timestamp_iso|indicator_1|...|indicator_i|sampling_method
|--|--|--|--|--|--|
|*integer*|*string*|*float*|...|*float*|*string*|
|Index of the measurement|Timestamp of the sample collection, in isoformat|Pollution indicator 1|...|Pollution indicator i|
**Naming convention for the pollution indicators**
- **lab_[pollution indicator]_[unit]**

with:

 - [pollution indicator]: 
	 - doc: dissolved organic carbon concentration
	 - po4: phosphate concentration
	 - so4: sulfate concentration
	 - nh4: ammonium concentration
	 - nsol: dissolved nitrogen concentration
	 - toc: total organic carbon concentration
	 - ntot: total nitrogen concentration
	 - tss: total suspended solids concentration
	 - turbidity: nephelometric turbidity
 - [unit]:
	 - mg_l: mg/l
	 - NTU: nephelometric turbidity unit


## laboratroy_measurements_organic_chemicals.csv
**Brief description**
Table containing the measurements of organic chemicals done in the laboratory by LC-HRMS/MS analysis after grabbing samples from the flume. When a chemical was not detected (below the limit of detection) in a sample, the value "<LOQ" was set.
**File structure**
|index|timestamp_iso|chemical_1|...|chemical_i|sampling_method
|--|--|--|--|--|--|
|*integer*|*string*|*float* or *string*|...|*float* or *string*|*string*|
|Index of the measurement|Timestamp of the sample collection, in isoformat|Organic chemical 1|...|Organic chemical i|
**Naming convention for organic chemicals**
- **lab_[organic chemical]_[unit]**

with:

 - [organic chemical]: 
	 - acesulfame: Acesulfame
	 - caffeine: Caffeine
	 - cyclamate: Cyclamate
	 - candesartan: Candesartan
	 - citalopram: Citalopram
	 - diclofenac: Diclofenac
	 - hydrochlorothiazide: Hydrochlorothiazide
	 - triclosan: Triclosan
	 - 13-diphenylguanidine: 1,3-Diphenylguanidine
	 - 6ppd-quinone: 6PPD-Quinone
	 - hmmm: Hexa(methoxymethy)melamine (HMMMM)
	 - 24-d: (2,4-Dichlorophenoxy)acetic acid (2,4-D)
	 - carbendazim: Carbendazim
	 - diuron: Diuron
	 - mcpa: (4-Chloro-2-methylphenoxy)acetic acid (MCPA)
	 - mecoprop: Mecoprop-p
	 - oit: 2-n-Octyl-4-isothiazolin-3-on (OIT)
	 - 4-&5-methylbenzotriazole: 4-&5-Methylbenzotriazole
	 - benzotriazole: 1H-Benzotriazole
	 - deet: N-N-diethyl-3-methylbenzamide (DEET)

 - [unit]:
	 - ng_l: ng/l


## laboratory_measurements_loq_organic_chemicals.csv
**Brief description**
Table containing the limits of detection (LOQ) for the organic chemicals measured in the laboratory by LC-HRMS/MS analysis after grabbing samples in the flume. The LOQs are corrected with relative recoveries and matrix factors. For each rain event, different LOQs were calculated. Thus, the timestamp of the first sample taken was set for each rain event.
**File structure**
|index|timestamp_iso|loq_chemical_1|...|loq_chemical_i|sampling_method
|--|--|--|--|--|--|
|*integer*|*string*|*float*|...|*float*|*string*|
|Index of the measurement|Timestamp of first sample collection, in isoformat|LOQ of organic chemical 1|...|LOQ of organic chemical i|
**Naming convention for organic chemicals**
- **lab_loq_[organic chemical]_[unit]**

with:

 - [organic chemical]: 
	 - acesulfame: Acesulfame
	 - caffeine: Caffeine
	 - cyclamate: Cyclamate
	 - candesartan: Candesartan
	 - citalopram: Citalopram
	 - diclofenac: Diclofenac
	 - hydrochlorothiazide: Hydrochlorothiazide
	 - triclosan: Triclosan
	 - 13-diphenylguanidine: 1,3-Diphenylguanidine
	 - 6ppd-quinone: 6PPD-Quinone
	 - hmmm: Hexa(methoxymethy)melamine (HMMMM)
	 - 24-d: (2,4-Dichlorophenoxy)acetic acid (2,4-D)
	 - carbendazim: Carbendazim
	 - diuron: Diuron
	 - mcpa: (4-Chloro-2-methylphenoxy)acetic acid (MCPA)
	 - mecoprop: Mecoprop-p
	 - oit: 2-n-Octyl-4-isothiazolin-3-on (OIT)
	 - 4-&5-methylbenzotriazole: 4-&5-Methylbenzotriazole
	 - benzotriazole: 1H-Benzotriazole
	 - deet: N-N-diethyl-3-methylbenzamide (DEET)

 - [unit]:
	 - ng_l: ng/l

