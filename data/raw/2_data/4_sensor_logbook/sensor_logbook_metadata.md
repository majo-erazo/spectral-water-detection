

## Overview
For each of the eight sensors used during the experiment, we compiled in a .csv file all relevant information to determine whether a data point is valid or not. 

## File naming convention
Each file in this folder, except flume_logbook.csv, is named following the convention: 
**[sensor location]_[sensor name]_logbook.csv**
With: 

 - [sensor location]:
	 - flume: the flume in which wastewater was circulated
	 - hall: the experimental hall in which the flume is installed
 - [sensor name]: 
	 - e53: electrical conductivity
	 - isa: spectrometer
	 - isamax: ion selective electrode
	 - mvx: hyperspectral camera measuring hyperspectral datacubes
	 - scan: spectrometer
	 - turbimax: turbidimeter
	 - c2s: flowmeter 
	 - i3: levelmeter 

The file flume_logbook.csv is structured in the same way as the other files and contains information about when the flume was stopped for maintenance or other reasons. 
## File structure
Each logbook follows the same structure, organized by events: 
|event|start|end|data_valid|comment|
|--|--|--|--|--|
|*integer*|*string*|*string*|*0/1*|*string*|
|number of the event|starting timestamp, in isoformat|ending timestamp, in isoformat|1 if the data of this event can be considered as valid|a brief description of the event|
Each event refers to a period with a start and an end timestamp, during which something unusual happened to the sensor.
## Sensor without logbook
The only sensor without a logbook is the pluvioS sensor installed in the catchment. We did not manage the sensor ourselves, and therefore received the data already validated.