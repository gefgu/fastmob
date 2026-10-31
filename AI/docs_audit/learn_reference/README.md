# Learn and Reference improvement brief

This is a prompt pack for a new improvement campaign, not a record of completed
documentation improvements. Preserve previous campaigns under `AI/docs_audit/runs/`.

Use `implementer_prompt.md` to start the campaign, `critic_prompt.md` for each
independent review, and `LOOP.md` to control rounds and acceptance. These files
supersede the old appearance-only scope **for this campaign only**.

## Agreed audience and outcome

Readers are primarily mobility researchers who understand their research goals
but have limited programming skills. Give them concise, professionally precise
concepts, executable steps, and help interpreting results. Technical readers
must also be able to scan for a task, function, parameter, or example quickly.

Learn is a guided book grouped by mobility themes. Reference is a complete,
compact API catalogue with access to full function details. Every executable
learning example must be a notebook that can run independently in Google Colab,
with a Binder launch option and internet-accessible data. Binder portability
review is sufficient; a hosted Binder execution is optional. At least 80% of
public functions must have meaningful executed example coverage.

No maximum runtime or download size has been agreed. Measure and disclose both;
prefer small representative data and avoid inventing limits or runtime claims.

## Why a new campaign

The previous critic explicitly excluded content depth and mainly reviewed
top-level pages. Round 10 gave Learn 8.0 while Reference received 7.3. Those
scores are historical design evidence, not acceptance for this campaign. The
current Learn index is a flat guide/card collection, with separate notebooks;
the reference overview includes a toy walkthrough and a module diagram ahead
of the catalogue. Re-evaluate their roles and the detail pages together.

Scikit-learn provides the structural benchmark:

- [User guide](https://scikit-learn.org/stable/user_guide.html): theme-based
  chapters and predictable subsections.
- [API reference](https://scikit-learn.org/stable/api/index.html): module-oriented
  discovery and individual API details.

Adapt these principles to this smaller mobility library. Do not copy its scale,
introduce empty chapters, or reproduce its wording.

## Starting the campaign

Paste `implementer_prompt.md` into the implementation session and provide this
directory as context. The first deliverable is a baseline inventory and proposed
navigation, followed by a representative pilot, then systematic expansion.
This brief authorizes no commits, deployment, or publishing. Follow repository
instructions for any separately requested Git operations.
