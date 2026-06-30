## UKRI GtR Collection

**collect_gtr.py** - collects all projects from UKRI GtR database and their metadata.
* Outputs:
  * gtr_ce_projects_latest.csv - contains all projects that passed screening and their metadata.
  * gtr_ce_all_with_decision_{timestamp}.csv - contains all projects fetched and their filter decisions (for audit).
  * gtr_validation_sample_{timestamp}.csv - contains a validation sample for hand-coding.
  * gtr_outcome_hrefs.csv - contains all of the outcome hrefs to be used in gtr_outcomes.py.

**gtr_outcomes.py** - collects all outcomes associated with projects collected by collect_gtr.py and their metadata.
* Outputs:
  * gtr_outcomes_latest.csv - contains all of the outcomes and metadata.
 
To run the entire pipeline, use the following commands:
1. python3 collect_gtr.py --outcomes
2. python3 gtr_outcomes.py
