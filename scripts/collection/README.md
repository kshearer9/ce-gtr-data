## UKRI GtR Collection

**collect_gtr_projects.py** - collects all projects from UKRI GtR database and their metadata.
* Outputs:
  * gtr_projects_latest.csv - contains all projects that passed screening and their metadata.
  * gtr_all_with_decision_{timestamp}.csv - contains all projects fetched and their filter decisions (for audit).
  * gtr_validation_sample_{timestamp}.csv - contains a validation sample for hand-coding.
  * gtr_outcome_hrefs.csv - contains all of the outcome hrefs to be used in gtr_outcomes.py.

**collect_gtr_outcomes.py** - collects all outcomes associated with projects collected by collect_gtr.py and their metadata.
* Outputs:
  * gtr_outcomes_latest.csv - contains all of the outcomes and metadata.

## OpenAlex Outcome Collection

**collect_openalex.py** - uses list of CE projects from GtR and matches to funded outputs in OpenAlex database.
* Outputs:
  * openalex_outputs.csv - contains all outcomes attached to GtR CE funded projects and their metadata.
  * Optional: openalex_missing_outputs.csv - contains all projects that weren't found in OpenAlex database and the reason (for testing code).