## Circular Economy Research Ecosystem Project - Durham University

**collect_gtr.py** - collects list of all projects and metadata from UKRI GtR database.
* Outputs:
  * gtr_ce_projects_enriched.csv

**openalex_api.py** -  collects all works and metadata associated with the list of UKRI projects in gtr_ce_projects_enriched.csv
* Outputs:
  * openalex_outputs.csv
  * Optional: openalex_missing_outputs.csv - saves all projects that have no matches for inspection
* Cache files:
  * award_cache.json
  * work_cache.json
