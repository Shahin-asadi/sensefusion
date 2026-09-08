# Bundled data dictionary

| Example / field | Meaning |
|---|---|
| `sample_id` | Unique physical vial/ageing-step observation, or formulation/domain observation |
| `group` | Oil bottle ID (24 groups) or formulation ID (nine groups) |
| Oil `target` | Thermal ageing time in days at 60 °C; 0, 2, 4, 7, 9, 18, 27, 36, 45, 53 |
| `FL__ex*_em*` | Fixed fluorescence window mean; excitation/emission labels in nm, deposited instrument intensity units |
| `UV__K232`, `K264`, `K268`, `K272` | Deposited specific extinction coefficients |
| Temperature `target` | Acetaminophen % w/w |
| `LOW__*`, `HIGH__*` | Source-domain NIR absorbance below / at-or-above 1300 nm; one device split into two bands |

The oil file has 240 UV-matched observations. Fixed fluorescence windows exclude stated first/second scatter regions, preserve negative intensities and do not correct undiluted-oil inner-filter effects. Four repeated fluorescence acquisitions were averaged at their physical keys before splitting. Unmatched fluorescence observations were excluded. Full preparation details and duplicate source-file names are in `olive_preparation.json`.

Source attribution and licenses: `DATA_SOURCES.json`. Transformations: `scripts/reproduce_examples.py`.
