# Intro
The CICFlowMeter is an open source tool that generates Biflows from pcap files, and extracts features from these flows.

# Installation and executing CICFlowMeter:

For Linux (prereq)

> $ sudo apt-get install libpcap-dev

## Executing
Go to the extracted directory,enter the 'bin' folder

### linux
Open a terminal and run this command

For GUI:
> $ sudo ./CICFlowMeter

For Command line:
> $ ./cfm "inputFolder" "outputFolder"

# For executing GOAAB in Linux:

usage: GOAAB.py [-h] [--pcap \<pcap file name\>] [--csv \<csv file name\>] [--train]
  
## For training mode run:
> $ python3 GOAAB.py [--csv \<csv file name\>] [--train]

## For getting Flow data from a '.pcap' source run:
> $ python3 GOAAB.py  [--pcap \<pcap file name\>] [--train]

## For running a '.pcap' file from source to classiciation run:
> $ python3 GOAAB.py  [--pcap \<pcap file name\>]

Note: you cannot train in pcap mode because labeling is too inconsistent accross datasets

# To get some training data:

1. Visit "https://www.unb.ca/cic/datasets/botnet.html"
2. Scroll to bottom and select "Download this dataset"
3. Fill out the request form
4. Download the following:
> ISCX_Botnet-Testing.pcap	 
> ISCX_Botnet-Training.pcap
> listofmaliciousips.docx
5. Run with GOAAB the following command 
> $ python3 GOAAB.py ISCX_Botnet-Testing.pcap --train
> $ python3 GOAAB.py ISCX_Botnet-Training.pcap --train

### This will give you two processed datasets for labeling (may take some time)
### *SEE MY DATAANALYSIS NOTEBOOK FOR REFERENCE TO THE FOLLOWING*

6. Now merge the two datasets in jupyter notebooks
7. In jupyter notebooks, hash the malicious ips from listofmaliciousips.docx. I copied the list at the bottom of this file to a spreadsheet and saved as a CSV file.
8. For each ip address in the merged dataset, check it against the hashes of malicious ips, if it hits, that ip is malicious, otherwise its not, add to the 'Label' column its appropriate Label (malicious == 1, benign =-0). 
7. Export the labeled merged dataset as a csv file for use in training/testing.


