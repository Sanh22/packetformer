# Data

## CICIDS2017

Download the **MachineLearningCSV.zip** from the University of New Brunswick:

https://www.unb.ca/cic/datasets/ids-2017.html

You need to fill out a short form to get the download link. Download the ZIP, not the MD5 files — those are just checksums for verifying integrity after download (optional to use).

After downloading, extract into `data/raw/`:

```
data/
└── raw/
    ├── Monday-WorkingHours.pcap_ISCX.csv
    ├── Tuesday-WorkingHours.pcap_ISCX.csv
    ├── Wednesday-workingHours.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
    ├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
    ├── Friday-WorkingHours-Morning.pcap_ISCX.csv
    ├── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
    └── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

The `data/raw/` folder is gitignored — do not commit the CSVs.

## Optional: Verify with MD5

If you want to confirm the download isn't corrupted:

```bash
md5sum -c <downloaded_md5_file>
```

## Phase 2+ Datasets

- **CICIDS2018**: https://www.unb.ca/cic/datasets/ids-2018.html
- **UNSW-NB15**: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- **CAIDA** (unlabeled, for pre-training): https://www.caida.org/catalog/datasets/
