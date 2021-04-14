## System requirements
- Ubuntu 18.04 or higher
- OPPT (https://github.com/RDLLab/oppt) v0.5 or higher

## Building
Once OPPT has been successfully built and installed, navigate to the `` <ABTLite_root_directory>``directory and execute
		
	mkdir build && cd build
	cmake ..
	make

Once the build process has been successfully completed, you'll find the abtLite executable inside the ``<ABTLite_root_directory>/bin`` directory.	

## Usage
First, make sure that your OPPT environment is properly set up. Please refer to the "Configuring the OPPT runtime environment" section at https://github.com/RDLLab/oppt

To run ABTLite on your POMDP problem, navigate to the ``<ABTLite_root_directory>/bin`` directory and execute
		
	./abtLite --cfg <ProblemConfigurationFile>.cfg

OPPT ships with a set of example POMDP problems with corresponding problem configuration files.
