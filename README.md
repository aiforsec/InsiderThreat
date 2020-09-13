## InsiderThreat
This research 

To learn more about this - [CMU Insider Threat](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=91513)

## Dataset

The dataset is available at Impact Cyber Trust and can be requested [here](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)

## How to execute code
Order of execution:
1. git clone https://github.com/aiforsec/InsiderThreat.git
2. Download dataset in that folder (recommend r5.2).
3. Download answers file in that folder.
4. Create files - http_preprocessed.csv, logon_preprocessed.csv in "_output"
5. Run code in this order
	- python pre_process_data.py "../Code/" "../Code/ftp.sei.cmu.edu/pub/cert-data/r5.2"
