"""Work-day calendar: quiet days, outbox flush time and reminder times.

All computations go through zoneinfo; naive datetimes are rejected because a
naive value has no instant, so any comparison with it would be a guess.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
_DAY_INDEX = {name: i for i, name in enumerate(_DAYS)}

FLUSH_TIME = time(7, 0)
EVENING_BEFORE = time(19, 0)
MORNING_OF = time(9, 0)
LEAD_BEFORE_EVENT = timedelta(hours=2)


def _parse_days(spec: str) -> frozenset[int]:
    """Parse 'mon-fri', 'mon,tue,thu' or a mix such as 'mon-wed,fri' into weekday numbers."""
    days: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            raise ValueError(f"invalid work_days spec {spec!r}: empty entry")
        start_name, sep, end_name = token.partition("-")
        start_name, end_name = start_name.strip(), end_name.strip()
        try:
            start = _DAY_INDEX[start_name]
            end = _DAY_INDEX[end_name] if sep else start
        except KeyError:
            raise ValueError(f"invalid work_days spec {spec!r}: bad day in {token!r}") from None
        # A range may wrap over the week end (sat-mon).
        i = start
        while True:
            days.add(i)
            if i == end:
                break
            i = (i + 1) % 7
    return frozenset(days)


def _require_aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


class WorkCalendar:
    def __init__(self, work_days: str, tz: str):
        self.days = _parse_days(work_days)
        try:
            self.tz = ZoneInfo(tz)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"invalid timezone {tz!r}") from exc

    def _at(self, day: date, at: time) -> datetime:
        return datetime.combine(day, at, tzinfo=self.tz)

    def is_work_time(self, now: datetime) -> bool:
        """True when the local calendar date of `now` is a work day."""
        _require_aware(now, "now")
        return now.astimezone(self.tz).weekday() in self.days

    def next_flush_at(self, now: datetime) -> datetime:
        """When the outbox may next be flushed: now on a work day after 07:00, else 07:00 local."""
        _require_aware(now, "now")
        local = now.astimezone(self.tz)
        if local.weekday() in self.days:
            if local.timetz().replace(tzinfo=None) >= FLUSH_TIME:
                return now
            return self._at(local.date(), FLUSH_TIME)
        day = local.date()
        for _ in range(7):
            day += timedelta(days=1)
            if day.weekday() in self.days:
                return self._at(day, FLUSH_TIME)
        raise ValueError("no work days configured")  # unreachable: spec is non-empty

    def reminder_times(self, event_at: datetime, has_time: bool, now: datetime) -> list[datetime]:
        """19:00 the evening before, 09:00 the day of, plus event_at - 2h when a time is known.

        Sorted ascending, de-duplicated, and only instants strictly after `now`.
        """
        _require_aware(event_at, "event_at")
        _require_aware(now, "now")
        local = event_at.astimezone(self.tz)
        day = local.date()
        times = {
            self._at(day - timedelta(days=1), EVENING_BEFORE).astimezone(timezone.utc),
            self._at(day, MORNING_OF).astimezone(timezone.utc),
        }
        if has_time:
            times.add(event_at.astimezone(timezone.utc) - LEAD_BEFORE_EVENT)
        return sorted(t.astimezone(self.tz) for t in times if t > now)
