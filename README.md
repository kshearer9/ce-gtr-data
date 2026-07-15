## Running the Data Pipeline

Run the following scripts in order from the project root directory. Each step generates the data required for the following stage.

### 1. Collect UKRI Gateway to Research (GtR) Projects

Collects the latest UKRI GtR project metadata.

```bash
python3 -m scripts.collection.collect_gtr_projects --outcomes --sectors
```

### 2. Collect UKRI Gateway to Research (GtR) Outcomes

Collects the latest UKRI GtR outcome metadata.

```bash
python3 -m scripts.collection.collect_gtr_outcomes
```

### 3. Clean UKRI Gateway to Research (GtR) Projects

Cleans and standardises UKRI GtR project data.

```bash
python3 -m scripts.cleaning.clean_gtr_projects
```

### 4. Clean UKRI Gateway to Research (GtR) Outcomes

Cleans and standardises UKRI GtR outcome data for downstream analysis and NLP.

```bash
python3 -m scripts.cleaning.clean_gtr_outcomes
```

### 5. Collect OpenAlex Data

Matches UKRI projects to OpenAlex records and retrieves associated project and research output metadata.

```bash
python3 -m scripts.collection.collect_openalex
```

### 6. Clean OpenAlex Projects

Cleans and standardises OpenAlex project metadata.

```bash
python3 -m scripts.cleaning.clean_openalex_projects
```

### 7. Clean OpenAlex Outcomes

Cleans and standardises OpenAlex research output metadata.

```bash
python3 -m scripts.cleaning.clean_openalex_outcomes
```

### 8. Merge project and outcome datasets

Merges UKRI and OpenAlex projects and outcomes into one dataset each.

```bash
python3 -m scripts.cleaning.merge_datasets.py
```
After completing all steps, cleaned and merged datasets will be available in the `data/cleaned/merged` directory and individual outcome types for OpenAlex including extra metadata are available in 'data/cleaned/outcomes'.
