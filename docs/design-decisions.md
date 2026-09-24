# Design decisions and ethics

TraceAI is a research prototype on synthetic data. This note explains the choices that matter most, the
alternatives I rejected, and the risks I could not remove. It is written to be argued with.

## What could go wrong, and for whom

| Harm | Who is hurt | How the design responds |
|---|---|---|
| Someone who fled an abuser is located through a "missing person" report | A person at risk of violence | Cases open only from a police reference plus a family-violence screening attestation. The public cannot search, see matches or see exact locations. |
| A wrong lead sends police or a searcher to the wrong person | A bystander who resembles the photo | Leads are ranked prompts for human review, never identifications. Face matches are similarity scores, never names. |
| Fake tips waste effort or frame someone | A named or described person, and the investigators | Rate limits, consent and false-report notice, flagging of repeat sources. A tip can only join a review queue. |
| Staff browse cases they have no need to see | People in the cases | Per-case access (even admins are blocked) and an audit log that records every read. |
| Biometric data outlives its purpose | Everyone whose photo was stored | Closing a case deletes its photo, embedding and leads. Public tips expire after 90 days. |

## Decisions

**Two separate processes, not two sets of routes.** The public app does not contain the officer routes at
all, so a bug or a misconfigured route cannot expose them. It costs a second deployment. I judged that
worth it because the failure it prevents is the worst one.

**A tip returns only a receipt.** The natural design shows the tipster "your report matched case X". That
turns the form into a search tool for anyone with a photo. Returning only a random reference removes that,
at the price of a worse experience for well-meaning reporters. I chose the safer side.

**Hand-set scoring, not a trained model.** There is no real labelled data and I would not collect any. A
learned model on synthetic data would only learn my generator. The heuristic is explainable end to end,
which matters when a person has to decide whether to act on a lead. Its weakness is that the weights are my
judgement, not evidence.

**Identity evidence scales the score.** Early versions averaged the signals, and a well-located, plausible
report about someone else outranked real matches whenever a photo was missing. Scaling by the strongest
identity evidence (face or description) fixed that. Two hard rules also force a score to zero: an
impossible journey and a sighting before the last-known time.

**Face matching is opt-in per case, and lazy.** It runs only if the officer attests the photo was lawfully
obtained, and a tip's photo is turned into an embedding only if some open case has matching enabled. The
alternative, embedding everything always, collects biometrics nobody needs.

**A hash-chained audit log.** It makes edits detectable. It does not stop someone with database admin
rights from rewriting the whole chain, so production needs write-once storage as well.

**Synthetic data only, and no real faces on the public demo.** The face dataset (LFW) is photos of real
public figures. They are used to test matching, but never shown as "missing persons" on a public page: the
demo and README screenshots use placeholder avatars.

**A static demo site, not a hosted app.** A live face-matching service with uploads would need real
hosting, abuse controls and a privacy position. A static page of pre-computed results shows the work with
nothing to abuse.

## What I deliberately did not build

- A public search over cases, or anything that names a person from a photo.
- Scraping of social media or other photo sources.
- Automatic alerts or actions triggered by a tip.
- Location tracking of any individual over time beyond the movement corridor for a single open case.

## Known gaps and open questions

- **The attestations are self-declared.** A screening checkbox does not stop a determined bad actor
  inside an agency. Real use needs the agency's own case workflow and accountability.
- **Fairness is untested.** Face recognition error rates differ across demographic groups. I have not
  measured this and would not deploy without it.
- **The benchmark is easy.** Synthetic reports and the extraction rules were written together. See the
  README for the results and the harder setting that removes text as a shortcut.
- **Retention periods are my guess**, not a legal position. So is the family-violence workflow.
- **Tipster verification is weak.** Stronger resistance to fake reports needs phone or email verification,
  which brings its own privacy cost.
- **Legal review is missing.** Biometric data is sensitive information under the Privacy Act 1988. A
  privacy impact assessment and legal advice come before any real data.
