# Task 0: the five Phase 2 questions against association and sequential mining

The question wording is the final reformulated set from Phase 2, Task 4.

| # | Question | Association / sequential? | Status in Phase 3 |
|---|---|---|---|
| 1 | **Magnitude from structure** | Neither on its own. It is regression/classification and was answered in Phase 2 (confirmed strongly through clustering). | Corroborated as a side effect: Task 1, Supplementary Rule B (size_category x reaction_category inside Financial/Earnings). |
| 2 | **Bad-news asymmetry** | Not applicable. | Closed. Retired in Phase 2 ("resolved... recorded as a finding"), so it is out of scope here. |
| 3 | **The content gap** | Not association- or sequential-shaped; it is the project's central synthesis question. | Supporting evidence: the Task 1 rules have lift close to 1 (1.006 to 1.111). The structural attributes co-occur with reactions at about the rate independence predicts. |
| 4 | **Filing timeliness, narrowed** | **Association.** Phase 2 asks for a residual, independent association once company and category effects are controlled. | Tested by Task 1, Supplementary Rule A: Late => reaction_category inside a fixed event_category, with lift against that category's base rate. It also gives lift against a size x volatility matched rate. |
| 5 | **Reaction persistence** | **Sequential, reframed.** Phase 2 defines persistence as CAR[-1,+1] vs CAR[-1,+5] for the *same* filing. That is within-event decay, a paired-correlation question and not a sequence of events. | Task 2 deliberately reframes it as **cross-event** persistence: does one filing's reaction predict the company's *next* filing's reaction? It does not answer the original within-event framing unchanged. |

## Caveats to carry into the write-up

- **Q4.** reaction_category is measured from day 0, which is the first trading day on or after **report_date** (the event date), not filing_date.
    - For Late filings the median gap is 27 days before the filing, so Rule A tests whether late filers had different *event-day* reactions. It does not test whether filing late moved the price.
    - Company identity itself is not controlled; only event_category and the size x volatility profile are.
- **Q5.** Some consecutive pairs share CAR days, which inflates persistence:
    - 18,888 pairs have the same day 0. They share the same CAR, so they repeat by construction.
    - Gaps of 1 to 2 trading days overlap partly.
    - Once these are removed, most of what remains matches a within-company shuffle. That points to company mix rather than order. See the Task 2 notes.
