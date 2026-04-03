# Data Directory

This folder stores datasets. All data subdirectories are excluded from Git by `.gitignore`.

## Dataset: RadioMapSeer

- **Homepage**: https://radiomapseer.github.io/
- **Download**: Follow the instructions on the dataset homepage.

## Expected Layout

After downloading and preprocessing, organize the data as follows:

```text
data/
├── samples/            # radio map images (ground truth)
├── buildings_complete/ # building footprint masks
├── antennas/           # antenna position maps
└── val_images/         # validation split images
```

Do not commit raw datasets or generated outputs to Git.
