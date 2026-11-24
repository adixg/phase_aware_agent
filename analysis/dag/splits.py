"""The 16-week split, fixed before any model was fitted on it.

The release holds 16 weeks, May 17 to Sep 5, 2026. Through Sep 1 that is 10,983 sessions and 1.09 million
requests (analysis/dag/build.py --week all), against the 2,220 sessions and 242,000 requests of Aug 23 to Sep 1
that the first models used. Volume grows over time: weeks 1 to 4 hold 440 to 930 turns each, weeks 13 to 15
over 10,000, so the split is in time order and puts the large recent weeks where they test best.

  fit         May 17 - Aug 8     weeks 1-12, 6,106 sessions
  tuning      Aug 9 - Aug 15     week 13, 1,354 sessions
  validation  Aug 16 - Aug 22    week 14, 1,303 sessions: read ONCE per model, after it and every setting are
                                 frozen, then the model is refitted on fit + tuning
  exposed     Aug 23 - Sep 1     read by the earlier models' validations (the first online, session and tool
                                 work tuned and validated on these days); no decision of a 16-week model uses them
  held out    Sep 2 - Sep 5      never built here (the offline predictor's frozen split)

Every window precedes the next, so a model is never fitted on data later than what it is scored on. A user's
account history is every earlier session or turn in the release, which the one continuous folder
(data/freeinference/dag_all) gives without any carry-over step.
"""

import datetime


def days(first: str, last: str) -> tuple[str, ...]:
    a, b = datetime.date.fromisoformat(first), datetime.date.fromisoformat(last)
    return tuple(str(a + datetime.timedelta(n)) for n in range((b - a).days + 1))


FIT_DAYS = days("2026-05-17", "2026-08-08")
TUNE_DAYS = days("2026-08-09", "2026-08-15")
VALIDATION_DAYS = days("2026-08-16", "2026-08-22")
EXPOSED_DAYS = days("2026-08-23", "2026-09-01")
HELDOUT_DAYS = days("2026-09-02", "2026-09-05")

assert FIT_DAYS[-1] < TUNE_DAYS[0] and TUNE_DAYS[-1] < VALIDATION_DAYS[0] < VALIDATION_DAYS[-1] < EXPOSED_DAYS[0]
assert EXPOSED_DAYS[-1] < HELDOUT_DAYS[0]
