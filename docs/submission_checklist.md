# Submission checklist (IFJC4002, Case 4)

Status on 2026-10-28. "Done" means it exists in the repository; items in the last section need the group.

## Done
- [x] Report PDF: `docs/report.pdf` (cover, 9 pages of main text, references, appendices). Source: `docs/report_final.md`; rebuild with `python docs/build_report.py`.
- [x] Code: `src/`, `app.py`, `requirements.txt`, saved model and preprocessing in `models/`, `README.md` with reproduction and deployment steps; `notebooks/analysis.ipynb` for EDA and error analysis.
- [x] Data: `data/processed/building_energy.csv` plus `data/README.md` (source, columns, limits). The Excel file is course-provided synthetic data.
- [x] Links: GitHub https://github.com/kitthakarn-sawasdipab/AI-Project and live app https://ai-project-awvytzqc3vsqxywkv6yesy.streamlit.app/
- [x] Evidence: model comparison (report section 3), dashboard screenshots (`docs/screenshots/`), test log (`docs/test_log.md`), monitoring record (report section 5.5 and test log C.4).
- [x] Prompt record: `docs/prompt_record.md` (9 entries, real prompts, with what was checked).
- [x] 35 automated tests pass (`pytest`).

## Needs the group before submitting
- [ ] **Read and understand the whole project.** The demonstration marks code understanding, and every member must be able to explain the preprocessing, the split, why the log target helps, the model comparison, the validation and the monitoring.
- [ ] **Check the AI-use rules.** The brief names ChatGPT as an example of allowed coding support. This project was built with Claude Code, and the report text was drafted with it. Confirm with the lecturer that this is acceptable, say so in the report, and review the text so it reflects what the group understands.
- [ ] **Contribution record.** The brief asks for all five members with student IDs, specific tasks and supporting evidence. The repository only lists the names (the IDs and tasks were removed on request). Add them to the report's Appendix A if the lecturer requires them.
- [ ] **Repository access.** Give the lecturer access if the repository is private.
- [ ] **Re-run the checks by hand on the live app** (a typical case, a zero floor area, a temperature of 99, and a 50,000 m² building) and add tester names and dates to `docs/test_log.md`. The recorded checks were run by script.
- [ ] **Demonstration (10 minutes).** Prepare: a live prediction, the model comparison, one invalid or boundary input, the monitoring evidence, and answers about the code. Open the app a few minutes before, because free hosting sleeps when idle.
- [ ] **Confirm the deadline and the demonstration schedule** with the lecturer.

## Known limits to mention if asked
- The data are synthetic, so accuracy figures show how well the models recover the simulation, not real-world performance.
- All temperatures in the data are 23–37 °C; outside this range the model extrapolates by formula (for example 111 kWh/day at −40 °C).
- The test set was not used to choose the final candidates, but the earlier BDG2 version of the project did see test results while iterating (Git history).
