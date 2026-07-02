## OpenAlex Outcome Collection

**collect_openalex.py** - uses list of CE projects from GtR and matches to funded outputs in OpenAlex database.
* Outputs:
  * openalex_outputs.csv - contains all outcomes attached to GtR CE funded projects and their metadata.
  * Optional: openalex_missing_outputs.csv - contains all projects that weren't found in OpenAlex database and the reason (for testing code).
 
To run the entire pipeline, you must first run:
1. python3 collect_gtr.py --outcomes
2. python3 gtr_outcomes.py

Then:
3. python3 collect_openalex.py

Optionally, if you want to evaluate projects that were skipped:
3. python3 collect_openalex.py --save-skipped
