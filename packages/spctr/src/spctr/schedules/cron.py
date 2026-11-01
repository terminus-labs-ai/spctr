"""5-field cron expression parser (spc-185, FR25).

Fields: minute hour day-of-month month day-of-week.
Supported forms per field: ``*``, comma-separated lists, ranges (a-b),
steps (*/n, a-b/n). Standard cron semantics: day-of-week 0=Sunday.

No external dependency — pure stdlib.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


class CronParseError(ValueError):
    """Raised for an invalid cron expression."""


# (field_name, min, max)
_FIELDS = (
    ("minute", 0, 59),
    ("hour", 0, 23),
    ("day_of_month", 1, 31),
    ("month", 1, 12),
    ("day_of_week", 0, 6),
)


def _parse_field(spec: str, lo: int, hi: int, name: str) -> frozenset[int]:
    """Parse one cron field into the set of matching integers."""
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            raise CronParseError(f"empty sub-expression in {name!r} field")
        step = 1
        if "/" in part:
            range_part, step_str = part.rsplit("/", 1)
            try:
                step = int(step_str)
            except ValueError:
                raise CronParseError(
                    f"invalid step {step_str!r} in {name!r} field"
                ) from None
            if step < 1:
                raise CronParseError(
                    f"step must be >= 1 in {name!r} field"
                )
            part = range_part
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            parts = part.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                raise CronParseError(
                    f"invalid range {part!r} in {name!r} field"
                ) from None
        else:
            try:
                start = end = int(part)
            except ValueError:
                raise CronParseError(
                    f"invalid value {part!r} in {name!r} field"
                ) from None
        if start < lo or end > hi:
            raise CronParseError(
                f"value out of range [{lo},{hi}] in {name!r} field: {part!r}"
            )
        if start > end:
            raise CronParseError(
                f"range start > end in {name!r} field: {part!r}"
            )
        values.update(range(start, end + 1, step))
    return frozenset(values)


class CronSpec:
    """A parsed 5-field cron expression."""

    def __init__(self, expression: str) -> None:
        self.expression = expression
        parts = expression.split()
        if len(parts) != 5:
            raise CronParseError(
                f"expected 5 fields, got {len(parts)}: {expression!r}"
            )
        self.minutes = _parse_field(parts[0], *_FIELDS[0][1:], _FIELDS[0][0])
        self.hours = _parse_field(parts[1], *_FIELDS[1][1:], _FIELDS[1][0])
        self.days_of_month = _parse_field(parts[2], *_FIELDS[2][1:], _FIELDS[2][0])
        self.months = _parse_field(parts[3], *_FIELDS[3][1:], _FIELDS[3][0])
        self.days_of_week = _parse_field(parts[4], *_FIELDS[4][1:], _FIELDS[4][0])

    def next_after(self, dt: datetime, tz: ZoneInfo) -> datetime:
        """Return the next occurrence strictly after *dt* in timezone *tz*.

        The result is a timezone-aware datetime in *tz*.
        """
        # Work in local time
        local = dt.astimezone(tz)
        # Start from the next minute
        candidate = local.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Search up to ~4 years ahead to handle rare combinations
        for _ in range(4 * 366 * 24 * 60):
            if candidate.month not in self.months:
                # Advance to first day of next month
                candidate = _next_month(candidate)
                continue
            # day_of_week: Python weekday() is Mon=0, cron is Sun=0
            cron_dow = (candidate.weekday() + 1) % 7
            if candidate.day not in self.days_of_month or cron_dow not in self.days_of_week:
                candidate = candidate.replace(hour=0, minute=0) + timedelta(days=1)
                continue
            if candidate.hour not in self.hours:
                candidate = candidate.replace(minute=0) + timedelta(hours=1)
                continue
            if candidate.minute not in self.minutes:
                candidate = candidate + timedelta(minutes=1)
                continue
            return candidate
        raise CronParseError(
            f"no next occurrence found within 4 years for {self.expression!r}"
        )


def _next_month(dt: datetime) -> datetime:
    """Advance to the first day of the next month at 00:00."""
    if dt.month == 12:
        return dt.replace(year=dt.year + 1, month=1, day=1, hour=0, minute=0)
    return dt.replace(month=dt.month + 1, day=1, hour=0, minute=0)


def parse_cron(expression: str) -> CronSpec:
    """Parse a 5-field cron expression. Raises CronParseError on invalid input."""
    return CronSpec(expression)
