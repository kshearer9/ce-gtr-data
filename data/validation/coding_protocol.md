# Discipline coding protocol

Written as a task specification. It is the instruction set a second coder would
be handed, and it doubles as the codebook for the methodology appendix.

---

## Task

You are given 100 UK research projects. For each one, decide which single
research field the project belongs to, choosing from a fixed list of twelve.
Record your answer and how confident you were.

You are establishing a human reference standard. A model has already assigned a
field to every one of these projects, and your judgements will be compared
against it. You will not see the model's answers while you work, and that is
deliberate: the comparison is only meaningful if your judgement is independent.

## Input

One row per project, with:

- `sample_id` — the row identifier, do not change it
- `project_id` — the GtR identifier
- `funder` — the funding body, for example Innovate UK, EPSRC
- `title` — the project title
- `abstract` — the project abstract, truncated at 2,000 characters

## Output

Fill in two columns per row, leaving everything else untouched:

- `your_field` — one of the twelve fields, or UNCLEAR
- `your_confidence` — certain, fairly sure, or guess
- `notes` — optional, free text

---

## The decision rule

**Code the field the research work sits in, not the sector it is applied to.**

This is the single rule that resolves most hard cases. A project developing a
chemical process to recover fibres from waste textiles is Chemical Engineering,
not Business, even though the customer is the clothing industry. A project
building software to track material flows in a construction firm is Computer
Science, not Engineering.

**Where a project spans several fields, choose the one holding the largest share
of the actual research effort.** Most circular economy projects are
interdisciplinary by nature, so this is normal rather than a sign you have
misread it. Ask what the researchers spend their time doing, not what the
project is ultimately for.

---

## The twelve fields

Definitions follow the crosswalk used to build this study's ground truth. Where
a placement looks counterintuitive, the crosswalk's convention wins, because
consistency with the reference standard is what is being measured. The bracketed
terms are the funder subject headings that map to each field.

**Agricultural and Biological Sciences** — agri-environmental science, food
science and nutrition, animal science, plant and crop science. Anything about
growing, farming, food production or the biology of organisms in an applied
setting.

**Biochemistry, Genetics and Molecular Biology** — biomolecules and
biochemistry, omics, cell biology, genetics. Research at the molecular and
cellular level. Distinguish from Agricultural and Biological Sciences by scale:
molecules and cells here, organisms and systems there.

**Business, Management and Accounting** — management and business studies.
Business models, supply chain management, organisational practice, consumer
behaviour as a management problem.

**Chemical Engineering** — process engineering, catalysis and surfaces,
bioengineering. Reactors, separation, scale-up, catalytic processes,
bioprocessing. Note that bioengineering sits here, not in Engineering.

**Chemistry** — chemical synthesis, chemical measurement, reaction dynamics and
mechanisms. Bench-scale molecular chemistry. The boundary with Chemical
Engineering is scale and intent: making and measuring molecules is Chemistry,
running and scaling a process is Chemical Engineering.

**Computer Science** — information and communication technology. Software,
algorithms, data infrastructure, machine learning, digital platforms.

**Economics, Econometrics and Finance** — economics. Market analysis, pricing,
economic modelling, investment. Distinguish from Business by whether the object
is an economy or market (here) or a firm or its operations (Business).

**Energy** — energy. Generation, storage, conversion, distribution, efficiency,
and energy systems as a whole. Use this when energy is the object of the
research. If energy is merely the application of a mechanical or manufacturing
advance, code Engineering.

**Engineering** — manufacturing, civil engineering and the built environment,
materials processing, design, mechanical engineering, systems engineering,
instrumentation and sensors, electrical engineering. The broadest field. Note
that materials **processing** sits here, while materials **science** does not.

**Environmental Science** — pollution, waste and resources, environmental
engineering, climate and climate change, terrestrial and freshwater
environments, environmental planning, ecology and biodiversity, marine
environments. **Waste management, recycling systems, pollution control and
resource recovery belong here, not in Engineering**, even when the work is
engineering in character. This is the most consequential convention in the
scheme for a circular economy corpus.

**Materials Science** — materials sciences. The composition, structure and
properties of materials. If the project is about how a material behaves, code
here. If it is about forming, shaping or manufacturing with it, code
Engineering.

**Social Sciences** — development studies, sociology, human geography, social
anthropology, science and technology studies, law and legal studies, education.
Policy, regulation, behaviour, institutions, public engagement, skills.

---

## Boundary cases, resolved

These five pairs account for most of the difficulty in this corpus. Apply them
consistently rather than case by case.

**Environmental Science or Engineering.** If the object is waste, pollution,
emissions, resource recovery or a recycling system, code Environmental Science
even where the method is engineering. If the object is a machine, structure,
process line or product, code Engineering.

**Chemical Engineering or Chemistry.** Reactors, separations, catalysis in use,
process scale-up and bioprocessing are Chemical Engineering. Synthesising
compounds, characterising them and studying reaction mechanisms is Chemistry.

**Materials Science or Engineering.** Understanding a material is Materials
Science. Processing, forming or manufacturing with it is Engineering.

**Energy or Engineering.** If the research question is about energy itself, code
Energy. If energy is the setting for a mechanical, electrical or manufacturing
question, code Engineering.

**Business or Social Sciences or Economics.** A firm and its operations is
Business. A market or an economy is Economics. Policy, regulation, institutions
and behaviour at societal level is Social Sciences.

---

## When to use UNCLEAR

Use it in three situations.

The abstract is too thin, too generic or too truncated to support a judgement.

Two or more fields have a genuinely equal claim and you cannot separate them.

**The project clearly belongs to a field that is not on the list.** The dropdown
holds only the twelve fields the classifier can produce. Six further fields exist
in the full scheme but were too rare to model: Arts and Humanities, Earth and
Planetary Sciences, Immunology and Microbiology, Mathematics, Physics and
Astronomy, and Psychology. If a project is plainly one of those, mark UNCLEAR
and write the field in `notes`. Do not force it into the nearest of the twelve,
because that would record a coder error where the real issue is the scheme's
coverage.

UNCLEAR rows are excluded from the agreement figures and reported separately.
Using it honestly costs nothing. Using it to avoid hard calls costs a lot,
because it shrinks an already small sample.

## Confidence

**certain** — the abstract states the field or makes it obvious.

**fairly sure** — you had to weigh two options but one clearly won.

**guess** — you were close to a coin flip, or the abstract barely supported a
decision.

Record this honestly. If agreement turns out to be much lower on the rows you
marked certain than on the ones you guessed, that means something quite
different from the reverse, and it can only be seen if the column is accurate.

---

## Constraints

Do not open `discipline_verification_KEY.csv` until every row is coded. It holds
the model's answers and reading it destroys the independence the exercise
depends on.

Do not look up the project in GtR or search for it online. Code from the title
and abstract alone, which is exactly what the classifier had. Giving yourself
information the model did not have would make the comparison unfair in your
favour.

Do not revise earlier rows once you have moved on. Drift is real, and going back
to make the set look tidier is how a coding pass loses its independence. If you
change your mind about a rule partway through, note it and carry on, and say so
afterwards.

Code every row. A partial sample is much weaker than a complete one at this
size.

## What happens next

`score_verification.py` joins the coding back to the key and reports agreement
per tier with confidence intervals, top-2 agreement, Cohen's kappa, and where
the disagreements cluster. The tier 2 figure is the one that matters: it tests
whether the 80 per cent accuracy target declared before the threshold was set
actually holds on the projects the classifier was applied to.
