## InsiderThreat
This research 

To learn more about this - [CMU Insider Threat](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=91513)

## Dataset

The dataset is available at Impact Cyber Trust and can be requested [here](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)

## How to execute code
Order of execution:
1. git clone https://github.com/aiforsec/InsiderThreat.git
2. Download Insider Dataset and copy it in the code folder (we recommend r5.2).
3. Copy answers folder into the code folder.
4. Run code in this order
	- python pre_process_data.py [dataset answer directory] [dataset directory]
    - get "action id" with 3 action_id_*.ipynb
    - merge_processed.ipynb
    - model.ipynb
    
discover_dataset.ipynb to get detailed raw idea of the current dataset
