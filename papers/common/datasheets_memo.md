# Common Reading Memo: Datasheets + Data Cards (Gebru et al. 2021; Pushkarna et al. FAccT 2022)

**Papers:**
- Datasheets for Datasets (Gebru et al., 2021)
- Data Cards: Purposeful and Transparent Dataset Documentation (Pushkarna et al., FAccT 2022)

**Role:** Documentation standard for the published dataset (train + dev splits on HuggingFace)

---

## Key Contributions

**Gebru et al. — Datasheets** defines seven mandatory sections for any published dataset: Motivation, Composition, Collection Process, Preprocessing/Cleaning/Labeling, Uses, Distribution, and Maintenance. The central argument is that undocumented datasets propagate harms invisibly — consumers cannot audit what they cannot see.

**Pushkarna et al. — Data Cards** extends the Datasheet standard with a layered documentation model: telescopic (one-paragraph summary for quick triage), periscopic (structured metadata for technical users), and microscopic (full field-level detail for auditors). The motivation is that a single flat document serves no audience well — different readers need different depths.

---

## Application to Tenacious-Bench

### `dataset/datasheet.md`

The datasheet at `dataset/datasheet.md` implements all seven Gebru sections. It also adopts the telescopic/periscopic/microscopic layering from Data Cards:

- **Telescopic:** one-paragraph summary at the top — what the dataset is, who it's for, and what it should not be used for
- **Periscopic:** structured tables for authoring mode distribution, failure mode frequencies, and per-segment breakdowns
- **Microscopic:** full field-level schema documented in `dataset/schema.json` with type, constraints, and example values for every field

### Specific Gebru requirements satisfied

| Section | Where documented |
|---|---|
| Motivation | `datasheet.md` §1 — Tenacious brand-voice evaluation, B2B sales domain |
| Composition | `datasheet.md` §2 — 200 tasks, 4 authoring modes, 5 failure modes, 3 segments |
| Collection Process | `datasheet.md` §3 — generation scripts, judge filter, IRA protocol |
| Preprocessing | `datasheet.md` §4 — contamination checks, profile-level grouping |
| Uses | `datasheet.md` §5 — intended: evaluation + SFT; not intended: general email generation |
| Distribution | `datasheet.md` §6 — train+dev on HuggingFace; held_out sealed |
| Maintenance | `datasheet.md` §7 — version 0.1, contact, update policy |

### One Pushkarna requirement pending

Data Cards recommends a "known limitations" section with quantified scope. Currently missing from the datasheet: a statement that Tenacious-Bench is limited to English-language B2B SaaS outreach and that performance on non-English or non-SaaS domains is untested. This should be added before publishing.

---

*~380 words | Key standards applied: 7 Gebru sections, telescopic/periscopic/microscopic layering. Gap: quantified limitations section pending before HuggingFace publish.*
