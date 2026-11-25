from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from clawdia.schedule import WorkCalendar

LX = ZoneInfo("Europe/Lisbon")
cal = WorkCalendar("mon-fri", "Europe/Lisbon")


def test_friday_night_is_work_and_saturday_is_not():
    assert cal.is_work_time(datetime(2026, 10, 30, 23, 59, tzinfo=LX)) is True  # Friday
    assert cal.is_work_time(datetime(2026, 10, 31, 0, 1, tzinfo=LX)) is False  # Saturday


def test_next_flush_from_saturday_is_monday_0700_local():
    nxt = cal.next_flush_at(datetime(2026, 10, 31, 15, 0, tzinfo=LX))
    assert nxt.astimezone(LX).isoformat() == "2026-11-02T07:00:00+00:00"  # after DST end, Lisbon = UTC


def test_next_flush_on_workday_after_0700_is_now():
    now = datetime(2026, 10, 29, 10, 0, tzinfo=LX)
    assert cal.next_flush_at(now) == now


def test_next_flush_on_workday_before_0700_is_today_0700():
    nxt = cal.next_flush_at(datetime(2026, 10, 29, 5, 30, tzinfo=LX))
    assert nxt.astimezone(LX).isoformat() == "2026-10-29T07:00:00+00:00"


def test_next_flush_friday_night_is_now_and_saturday_goes_to_monday():
    fri = datetime(2026, 10, 30, 23, 0, tzinfo=LX)
    assert cal.next_flush_at(fri) == fri


def test_reminder_times_default_and_with_time():
    now = datetime(2026, 11, 1, 10, 0, tzinfo=LX)
    ev = datetime(2026, 11, 7, 21, 0, tzinfo=LX)
    ts = [t.astimezone(LX).strftime("%d %H:%M") for t in cal.reminder_times(ev, True, now)]
    assert ts == ["06 19:00", "07 09:00", "07 19:00"]
    assert len(cal.reminder_times(ev, False, now)) == 2


def test_reminder_times_drop_past_strictly_and_sorted():
    ev = datetime(2026, 11, 7, 10, 0, tzinfo=LX)  # event_at - 2h = 08:00, before 09:00
    now = datetime(2026, 11, 6, 19, 0, tzinfo=LX)  # equals the evening reminder: dropped
    ts = cal.reminder_times(ev, True, now)
    assert [t.astimezone(LX).strftime("%d %H:%M") for t in ts] == ["07 08:00", "07 09:00"]
    assert ts == sorted(ts)


def test_reminder_times_use_event_local_date_not_utc_date():
    tokyo = WorkCalendar("mon-fri", "Asia/Tokyo")
    now = datetime(2026, 11, 1, 0, 0, tzinfo=timezone.utc)
    ev = datetime(2026, 11, 7, 20, 0, tzinfo=timezone.utc)  # 05:00 on 8 Nov in Tokyo
    ts = [t.astimezone(ZoneInfo("Asia/Tokyo")).strftime("%d %H:%M") for t in tokyo.reminder_times(ev, False, now)]
    assert ts == ["07 19:00", "08 09:00"]


def test_comma_list_and_case_insensitive():
    c = WorkCalendar("Mon,TUE, thu", "Europe/Lisbon")
    assert c.is_work_time(datetime(2026, 11, 3, 12, 0, tzinfo=LX)) is True  # Tue
    assert c.is_work_time(datetime(2026, 11, 4, 12, 0, tzinfo=LX)) is False  # Wed
    assert c.is_work_time(datetime(2026, 11, 5, 12, 0, tzinfo=LX)) is True  # Thu


def test_ranges_inside_comma_list_and_wraparound():
    c = WorkCalendar("mon-wed,fri", "Europe/Lisbon")
    assert [c.is_work_time(datetime(2026, 11, d, 12, 0, tzinfo=LX)) for d in range(2, 9)] == [
        True, True, True, False, True, False, False,
    ]
    w = WorkCalendar("sat-mon", "Europe/Lisbon")  # wraps over the week end
    assert w.is_work_time(datetime(2026, 11, 7, 12, 0, tzinfo=LX)) is True
    assert w.is_work_time(datetime(2026, 11, 3, 12, 0, tzinfo=LX)) is False


@pytest.mark.parametrize("spec", ["", "funday", "mon-", "mon-xyz", ",,", "mon,,tue"])
def test_invalid_spec_raises(spec):
    with pytest.raises(ValueError):
        WorkCalendar(spec, "Europe/Lisbon")


def test_invalid_timezone_raises():
    with pytest.raises(ValueError):
        WorkCalendar("mon-fri", "Not/AZone")


def test_naive_datetimes_raise():
    naive = datetime(2026, 11, 2, 12, 0)
    with pytest.raises(ValueError):
        cal.is_work_time(naive)
    with pytest.raises(ValueError):
        cal.next_flush_at(naive)
    with pytest.raises(ValueError):
        cal.reminder_times(naive, True, datetime(2026, 11, 1, tzinfo=LX))
    with pytest.raises(ValueError):
        cal.reminder_times(datetime(2026, 11, 7, 21, 0, tzinfo=LX), True, naive)


def test_utc_input_is_converted_to_local_weekday():
    # Friday 23:30 UTC is still Friday in Lisbon (winter); Sat 00:30 UTC is Saturday.
    assert cal.is_work_time(datetime(2026, 10, 30, 23, 30, tzinfo=timezone.utc)) is True
    assert cal.is_work_time(datetime(2026, 10, 31, 0, 30, tzinfo=timezone.utc)) is False
    # Summer (UTC+1): Sunday 23:30 UTC is already Monday 00:30 in Lisbon.
    assert cal.is_work_time(datetime(2026, 7, 5, 23, 30, tzinfo=timezone.utc)) is True


def test_reminder_times_dedupe_when_two_hours_before_equals_morning():
    ev = datetime(2026, 11, 7, 11, 0, tzinfo=LX)  # 11:00 - 2h = 09:00
    ts = cal.reminder_times(ev, True, datetime(2026, 11, 1, 10, 0, tzinfo=LX))
    assert [t.astimezone(LX).strftime("%d %H:%M") for t in ts] == ["06 19:00", "07 09:00"]


def test_utc_input_saturday_half_in_summer():
    # Summer (UTC+1): Friday 23:30 UTC is already Saturday 00:30 in Lisbon.
    assert cal.is_work_time(datetime(2026, 7, 3, 23, 30, tzinfo=timezone.utc)) is False
    assert cal.is_work_time(datetime(2026, 7, 3, 22, 30, tzinfo=timezone.utc)) is True


def test_next_flush_across_dst_weekends():
    # DST ends Sun 2026-10-25 (Lisbon goes +01:00 -> +00:00); starts Sun 2026-03-29.
    end = cal.next_flush_at(datetime(2026, 10, 24, 15, 0, tzinfo=LX))
    assert end.isoformat() == "2026-10-26T07:00:00+00:00"
    start = cal.next_flush_at(datetime(2026, 3, 28, 15, 0, tzinfo=LX))
    assert start.isoformat() == "2026-03-30T07:00:00+01:00"
