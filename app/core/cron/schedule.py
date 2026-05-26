from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_MONTH_NAMES = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_WEEKDAY_NAMES = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}


@dataclass(frozen=True)
class CronField:
    values: frozenset[int]
    wildcard: bool = False

    def matches(self, value: int) -> bool:
        return value in self.values


@dataclass(frozen=True)
class CronExpression:
    """Small 5-field cron parser used by runtime cron jobs.

    Supported syntax per field: ``*``, ``*/n``, ``a``, ``a-b``, ``a,b`` and
    ``a-b/n``. Month and weekday names are accepted in English short form.
    Weekday follows cron convention where Sunday can be 0 or 7.
    """

    raw: str
    minute: CronField
    hour: CronField
    day_of_month: CronField
    month: CronField
    day_of_week: CronField

    @classmethod
    def parse(cls, expression: str) -> CronExpression:
        parts = expression.split()
        if len(parts) != 5:
            raise ValueError("Cron expression must contain exactly 5 fields: minute hour day month weekday.")
        return cls(
            raw=expression,
            minute=_parse_field(parts[0], minimum=0, maximum=59),
            hour=_parse_field(parts[1], minimum=0, maximum=23),
            day_of_month=_parse_field(parts[2], minimum=1, maximum=31),
            month=_parse_field(parts[3], minimum=1, maximum=12, names=_MONTH_NAMES),
            day_of_week=_parse_field(parts[4], minimum=0, maximum=7, names=_WEEKDAY_NAMES, map_seven_to_zero=True),
        )

    def next_after(self, after: datetime, timezone_name: str) -> datetime:
        tz = _zoneinfo(timezone_name)
        current = after.astimezone(tz).replace(second=0, microsecond=0) + timedelta(minutes=1)
        max_minutes = 366 * 24 * 60
        for _ in range(max_minutes):
            if self.matches(current):
                return current.astimezone(UTC)
            current += timedelta(minutes=1)
        raise ValueError(f"Unable to find next run within 366 days for cron expression: {self.raw}")

    def matches(self, value: datetime) -> bool:
        if not self.minute.matches(value.minute):
            return False
        if not self.hour.matches(value.hour):
            return False
        if not self.month.matches(value.month):
            return False
        return self._matches_day(value)

    def _matches_day(self, value: datetime) -> bool:
        dom_matches = self.day_of_month.matches(value.day)
        cron_weekday = 0 if value.isoweekday() == 7 else value.isoweekday()
        dow_matches = self.day_of_week.matches(cron_weekday)
        if self.day_of_month.wildcard and self.day_of_week.wildcard:
            return True
        if self.day_of_month.wildcard:
            return dow_matches
        if self.day_of_week.wildcard:
            return dom_matches
        return dom_matches or dow_matches


def _parse_field(
    raw: str,
    *,
    minimum: int,
    maximum: int,
    names: dict[str, int] | None = None,
    map_seven_to_zero: bool = False,
) -> CronField:
    if not raw.strip():
        raise ValueError("Cron field must not be empty.")
    values: set[int] = set()
    wildcard = raw == "*"
    for part in raw.split(","):
        values.update(
            _parse_part(
                part.strip().lower(),
                minimum=minimum,
                maximum=maximum,
                names=names or {},
                map_seven_to_zero=map_seven_to_zero,
            )
        )
    if not values:
        raise ValueError(f"Cron field has no values: {raw}")
    return CronField(values=frozenset(values), wildcard=wildcard)


def _parse_part(
    raw: str,
    *,
    minimum: int,
    maximum: int,
    names: dict[str, int],
    map_seven_to_zero: bool,
) -> set[int]:
    if not raw:
        raise ValueError("Cron field segment must not be empty.")
    base, step = _split_step(raw)
    if base == "*":
        start, end = minimum, maximum
    elif "-" in base:
        start_text, end_text = base.split("-", 1)
        start = _parse_value(start_text, names=names, map_seven_to_zero=map_seven_to_zero)
        end = _parse_value(end_text, names=names, map_seven_to_zero=map_seven_to_zero)
    else:
        value = _parse_value(base, names=names, map_seven_to_zero=map_seven_to_zero)
        start, end = value, value
    if start > end:
        raise ValueError(f"Cron range start must be <= end: {raw}")
    if start < minimum or end > maximum:
        raise ValueError(f"Cron value out of range {minimum}-{maximum}: {raw}")
    values = set(range(start, end + 1, step))
    if map_seven_to_zero and 7 in values:
        values.remove(7)
        values.add(0)
    return values


def _split_step(raw: str) -> tuple[str, int]:
    if "/" not in raw:
        return raw, 1
    base, step_text = raw.split("/", 1)
    if not base:
        raise ValueError(f"Cron step is missing a base: {raw}")
    try:
        step = int(step_text)
    except ValueError as exc:
        raise ValueError(f"Cron step must be an integer: {raw}") from exc
    if step <= 0:
        raise ValueError(f"Cron step must be positive: {raw}")
    return base, step


def _parse_value(raw: str, *, names: dict[str, int], map_seven_to_zero: bool) -> int:
    if raw in names:
        return names[raw]
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Cron value must be an integer or known name: {raw}") from exc
    if map_seven_to_zero and value == 7:
        return 7
    return value


def _zoneinfo(timezone_name: str) -> ZoneInfo:
    try:
        return ZoneInfo(timezone_name or "UTC")
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {timezone_name}") from exc
