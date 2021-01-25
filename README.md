# Intro
The CICFlowMeter is an open source tool that generates Biflows from pcap files, and extracts features from these flows.

# Installation and executing CICFlowMeter:

For Linux

> $ sudo apt-get install libpcap-dev

## executing
Go to the extracted directory,enter the 'bin' folder

### linux
Open a terminal and run this command
```
//For GUI:
sudo ./CICFlowMeter

//For Command line:
./cfm "inputFolder" "outputFolder"

# For executing GOAAB:

For Linux

## For training mode run
> $ python3 GOAAB.py [--csv <csv file name>] [--train]

## For getting Flow data from a '.pcap' source run
> $ python3 GOAAB.py  [--pcap <pcap file name>]
// Note: you cannot train in pcap mode because labeling is too inconsistent accross datasets


