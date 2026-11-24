"""Tests for the SessionMapper base class."""

from datetime import datetime, timedelta, timezone
from typing import Any

from strands_evals.mappers.session_mapper import SessionMapper
from strands_evals.types.trace import Session


class _ConcreteMapper(SessionMapper):
    """Minimal concrete subclass for testing base-class methods."""

    def map_to_session(self, data: Any, session_id: str) -> Session:
        raise NotImplementedError


class TestParseTimestamp:
    """Tests for parse_timestamp handling various formats and ensuring UTC output."""

    def setup_method(self):
        self.mapper = _ConcreteMapper()

    def test_none_returns_aware_utc(self):
        result = self.mapper.parse_timestamp(None)
        assert result.tzinfo == timezone.utc

    def test_iso_string_with_z(self):
        result = self.mapper.parse_timestamp("2026-07-22T16:34:19.917561Z")
        assert result.tzinfo == timezone.utc
        assert result.year == 2026 and result.month == 7

    def test_nanosecond_epoch_int(self):
        """Nanosecond epoch integer is converted to aware UTC datetime."""
        result = self.mapper.parse_timestamp(1700000000000000000)
        assert result.tzinfo == timezone.utc
        assert result.year == 2023

    def test_string_nanosecond_epoch(self):
        """String-encoded nanosecond epoch (OTLP JSON uint64) is correctly parsed."""
        result = self.mapper.parse_timestamp("1700000000000000000")
        assert result.tzinfo == timezone.utc
        assert result.year == 2023 and result.month == 11

    def test_millisecond_epoch_int_and_string(self):
        """Integer and string millisecond epochs are converted to the encoded instant."""
        expected = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        assert self.mapper.parse_timestamp(1700000000000) == expected
        assert self.mapper.parse_timestamp("1700000000000") == expected

    def test_microsecond_epoch_int_and_string(self):
        """Integer and string microsecond epochs are converted to the encoded instant."""
        expected = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
        assert self.mapper.parse_timestamp(1700000000000000) == expected
        assert self.mapper.parse_timestamp("1700000000000000") == expected

    def test_digit_like_unicode_string_falls_back(self):
        """A non-ASCII digit-like string falls back instead of escaping the parser."""
        result = self.mapper.parse_timestamp("²")
        assert result.tzinfo == timezone.utc

    def test_out_of_range_numeric_string_falls_back(self):
        """An out-of-range numeric string falls back instead of escaping the parser."""
        result = self.mapper.parse_timestamp("9" * 25)
        assert result.tzinfo == timezone.utc

    def test_aware_datetime_passthrough(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = self.mapper.parse_timestamp(dt)
        assert result.tzinfo == timezone.utc
        assert result == dt

    def test_naive_datetime_gets_utc(self):
        """A naive datetime is assumed UTC and returned aware."""
        naive = datetime(2024, 6, 15, 12, 0, 0)
        result = self.mapper.parse_timestamp(naive)
        assert result.tzinfo == timezone.utc
        assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_non_utc_datetime_normalized(self):
        """An aware datetime in a non-UTC tz is converted to UTC."""
        plus5 = timezone(timedelta(hours=5))
        dt = datetime(2024, 6, 15, 17, 0, 0, tzinfo=plus5)
        result = self.mapper.parse_timestamp(dt)
        assert result.tzinfo == timezone.utc
        assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_iso_string_without_offset_gets_utc(self):
        """An ISO string with no offset is assumed UTC."""
        result = self.mapper.parse_timestamp("2024-01-01T00:00:00")
        assert result.tzinfo == timezone.utc
        assert result == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_iso_string_with_offset_normalized(self):
        """An ISO string with a non-UTC offset is converted to UTC."""
        result = self.mapper.parse_timestamp("2024-06-15T17:00:00+05:00")
        assert result.tzinfo == timezone.utc
        assert result == datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

    def test_invalid_string_returns_aware_utc(self):
        """An unparseable string falls back to current UTC time."""
        result = self.mapper.parse_timestamp("not-a-timestamp")
        assert result.tzinfo == timezone.utc

    def test_boundary_iso_string_overflow_falls_back(self):
        """An ISO string near datetime.max with non-UTC offset falls back to now(utc)."""
        result = self.mapper.parse_timestamp("9999-12-31T23:59:59-05:00")
        assert result.tzinfo == timezone.utc

    def test_boundary_datetime_overflow_falls_back(self):
        """A datetime near datetime.min with non-UTC offset falls back to now(utc)."""
        plus5 = timezone(timedelta(hours=5))
        dt = datetime(1, 1, 1, 0, 0, 0, tzinfo=plus5)
        result = self.mapper.parse_timestamp(dt)
        assert result.tzinfo == timezone.utc

    def test_float_seconds_epoch(self):
        """A float epoch in seconds (< 1e12) is converted to aware UTC datetime."""
        result = self.mapper.parse_timestamp(1700000000.5)
        assert result.tzinfo == timezone.utc
        assert result.year == 2023

    def test_unsupported_type_falls_back(self):
        """An unsupported type (e.g. list) falls back to current UTC time."""
        result = self.mapper.parse_timestamp([1, 2, 3])
        assert result.tzinfo == timezone.utc
