use std::time::{SystemTime, UNIX_EPOCH};

// u64 milliseconds reach at most a nine-digit year, plus the fractional suffix.
pub(crate) const MAX_TIMESTAMP_BYTES: usize = 29;

/// Returns the current UTC time in the engine's timestamp format. The formatter
/// omits a zero millisecond fraction and uses three digits for a nonzero fraction.
pub fn now_rfc3339() -> String {
    format_unix_millis(unix_millis_now())
}

pub(crate) fn unix_millis_now() -> u64 {
    // Persisted clock values are unsigned milliseconds. Map a clock reading before
    // the Unix epoch to zero and cap a wider duration before converting it to the
    // stored width.
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u64::MAX as u128) as u64)
        .unwrap_or(0)
}

pub(crate) fn parse_rfc3339_millis(value: &str) -> Option<u64> {
    // The seconds parser also serves signed calendar comparisons. A timestamp
    // before the epoch has no unsigned millisecond encoding, even if its date and
    // time are valid there.
    let seconds = u64::try_from(parse_rfc3339_seconds(value)?).ok()?;
    // The storage codec keeps at most millisecond precision. Re-read the fraction
    // because the seconds parser deliberately accepts and discards any number of
    // fractional digits.
    let fraction = match value.get(19..)? {
        "Z" => 0,
        suffix if suffix.starts_with('.') && suffix.ends_with('Z') => {
            let digits = &suffix[1..suffix.len() - 1];
            if digits.is_empty() || digits.len() > 3 || !digits.bytes().all(|b| b.is_ascii_digit())
            {
                return None;
            }
            // Scale by the number of supplied digits. For example .1Z adds one
            // hundred milliseconds and .001Z adds one; accepting a longer fraction
            // would require a rounding policy.
            digits.parse::<u64>().ok()? * 10u64.pow(3 - digits.len() as u32)
        }
        _ => return None,
    };
    seconds.checked_mul(1000)?.checked_add(fraction)
}

pub(crate) fn format_unix_millis(millis: u64) -> String {
    // The common four-digit year with milliseconds fits the initial allocation.
    // Formatting a larger u64 timestamp can grow this owned string; callers
    // reserving their own buffer use MAX_TIMESTAMP_BYTES.
    let mut output = String::with_capacity(24);
    write_unix_millis(&mut output, millis).expect("writing to String is infallible");
    output
}

pub(crate) fn write_unix_millis(
    output: &mut (impl std::fmt::Write + ?Sized),
    millis: u64,
) -> std::fmt::Result {
    // Split the value while it is still milliseconds. Even the largest u64
    // millisecond clock fits in i64 after division, and its remainder always fits
    // in three decimal digits.
    write_timestamp(output, (millis / 1000) as i64, (millis % 1000) as u16)
}

/// Converts the engine's UTC timestamp spelling to whole Unix seconds. An accepted
/// fractional suffix is discarded. Returns `None` for an invalid date, time or
/// suffix, including a numeric timezone offset.
pub fn parse_rfc3339_seconds(value: &str) -> Option<i64> {
    if value.len() < 20 {
        return None;
    }
    // The stored layout places each calendar component at a fixed byte offset.
    // Checked str slices return None when an arbitrary UTF-8 input would split a
    // character at one of those boundaries.
    let year = value.get(0..4)?.parse::<i32>().ok()?;
    let month = value.get(5..7)?.parse::<u32>().ok()?;
    let day = value.get(8..10)?.parse::<u32>().ok()?;
    let hour = value.get(11..13)?.parse::<u32>().ok()?;
    let minute = value.get(14..16)?.parse::<u32>().ok()?;
    let second = value.get(17..19)?.parse::<u32>().ok()?;
    if value.as_bytes().get(4) != Some(&b'-')
        || value.as_bytes().get(7) != Some(&b'-')
        || value.as_bytes().get(10) != Some(&b'T')
        || value.as_bytes().get(13) != Some(&b':')
        || value.as_bytes().get(16) != Some(&b':')
    {
        return None;
    }
    // Validate the Gregorian date before converting it to a day count. The
    // arithmetic conversion assumes the selected month and day already exist and
    // could otherwise normalize an invalid date.
    let days_in_month = match month {
        2 if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        _ => return None,
    };
    // The day starts at midnight and stops after second 59 of its final minute.
    // Reject hour 24 and leap-second notation before applying the fixed
    // 86,400-second day calculation.
    if !(1..=days_in_month).contains(&day) || hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    // Only UTC Z input is consumed here. For whole-second comparisons, the
    // fractional digits must be syntactically valid, but neither their precision
    // nor their value changes the returned second.
    match value.get(19..)? {
        "Z" => {}
        suffix if suffix.starts_with('.') && suffix.ends_with('Z') => {
            let fraction = &suffix[1..suffix.len() - 1];
            if fraction.is_empty() || !fraction.bytes().all(|byte| byte.is_ascii_digit()) {
                return None;
            }
        }
        _ => return None,
    }
    // Keep the epoch-day contribution signed. Calendar dates before 1970 produce
    // negative seconds; the unsigned storage decoder applies its own range check.
    Some(
        days_from_civil(year, month, day) * 86_400
            + hour as i64 * 3600
            + minute as i64 * 60
            + second as i64,
    )
}

// Render the shared UTC timestamp form into a caller-owned writer. The caller can
// use an admitted buffer without first allocating an intermediate String.
fn write_timestamp(
    output: &mut (impl std::fmt::Write + ?Sized),
    seconds: i64,
    millis: u16,
) -> std::fmt::Result {
    // Split into a whole day and a nonnegative time of day. Euclidean division
    // preserves that decomposition even when a timestamp precedes the epoch.
    let days = seconds.div_euclid(86_400);
    let rem = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    // Turn the within-day remainder into fixed-width time fields. This does not
    // use a local timezone or daylight-saving rule.
    let hour = rem / 3600;
    let minute = (rem % 3600) / 60;
    let second = rem % 60;
    write!(
        output,
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}"
    )?;
    // Omit the fractional part for exact seconds. When it is present the format
    // always writes three millisecond digits.
    if millis != 0 {
        write!(output, ".{millis:03}")?;
    }
    output.write_char('Z')
}

// Both conversions count Gregorian eras from March, putting leap day at the
// end of a year. The epoch shift maps that day count to Unix days.
fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    // Treat January and February as the last months of the preceding arithmetic
    // year. Each shifted year starts in March and ends at the location of any leap
    // day.
    let year = year - (month <= 2) as i32;
    // Use floor division for negative shifted years as well as positive ones. The
    // adjusted numerator keeps the year-of-era in the expected nonnegative range.
    let era = (if year >= 0 { year } else { year - 399 }) / 400;
    let yoe = year - era * 400;
    let mp = month as i32 + if month > 2 { -3 } else { 9 };
    // Compute the day offset within the shifted year from a March-relative month,
    // then add completed years and leap days within the era.
    let doy = (153 * mp + 2) / 5 + day as i32 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    // Subtract the common epoch offset to produce Unix-relative days. The inverse
    // conversion must add the same constant.
    (era * 146097 + doe - 719468) as i64
}

// Invert the March-based Gregorian day count. Restore the ordinary month and year
// order only after resolving the day within its era.
fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719468;
    let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
    let doe = z - era * 146097;
    // Derive the year in the era with corrections at the last days of four-year,
    // century and full-era periods. This keeps leap day inside the year it belongs
    // to.
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    // Remove completed years to obtain a zero-based day within the March-start
    // year. Invert the month offsets before restoring January and February to the
    // next year.
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    // Translate the shifted month back to the ordinary calendar. January and
    // February advance the calendar year relative to the March-based count.
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + (m <= 2) as i64;
    (year as i32, m as u32, d as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserved_timestamp_buffer_covers_full_millisecond_range_without_growth() {
        use crate::json::{JsonOperationContext, JsonProcessingPolicyV1, ReservedString};
        for millis in [0, 999, 1000, 253_402_300_799_999, u64::MAX]
            .into_iter()
            .chain((0..100u64).map(|seed| seed.wrapping_mul(6364136223846793005)))
        {
            let expected = format_unix_millis(millis);
            assert!(expected.len() <= MAX_TIMESTAMP_BYTES);
            let context = JsonOperationContext::standalone();
            let limit = JsonProcessingPolicyV1::WORK_BUDGET_BYTES;
            let mut held = context.workspace_reservation();
            held.grow_to(
                limit - MAX_TIMESTAMP_BYTES,
                limit - MAX_TIMESTAMP_BYTES,
                "operation_scratch",
            )
            .unwrap();
            let mut value =
                ReservedString::with_capacity(MAX_TIMESTAMP_BYTES, &context, "batch_prepare")
                    .unwrap();
            let pointer = value.as_ref().as_ptr();
            value.append(|out| write_unix_millis(out, millis)).unwrap();
            assert_eq!(value.as_ref(), expected);
            assert_eq!(value.as_ref().as_ptr(), pointer);
            let (value, work) = value.into_parts();
            assert_eq!(value.capacity(), MAX_TIMESTAMP_BYTES);
            assert_eq!(work.reserved_bytes(), MAX_TIMESTAMP_BYTES);
            drop(value);
            drop(work);
            held.grow_to(limit, limit, "operation_scratch").unwrap();
        }
    }

    #[test]
    fn streaming_timestamps_preserve_format_and_exact_capacity() {
        let mut values = vec![0, 1, 999, 1000, 253_402_300_799_000, 253_402_300_799_999];
        for seed in 0..100_u64 {
            values.push(seed.wrapping_mul(6364136223846793005) % 253_402_300_800_000);
        }
        for millis in values {
            let seconds = millis / 1000;
            let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
            let base = format!(
                "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}",
                seconds % 86_400 / 3600,
                seconds % 3600 / 60,
                seconds % 60
            );
            let expected = if millis % 1000 == 0 {
                format!("{base}Z")
            } else {
                format!("{base}.{:03}Z", millis % 1000)
            };
            let mut output = String::with_capacity(expected.len());
            let pointer = output.as_ptr();
            write_unix_millis(&mut output, millis).unwrap();
            assert_eq!(output, expected);
            assert_eq!(output.as_ptr(), pointer);
            assert_eq!(output.capacity(), expected.len());
            assert_eq!(format_unix_millis(millis), expected);
            assert_eq!(parse_rfc3339_millis(&output), Some(millis));
        }
    }

    #[test]
    fn rfc3339_epoch_round_trip() {
        assert_eq!(parse_rfc3339_seconds("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(format_unix_millis(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn milliseconds_round_trip_without_loss() {
        for millis in [0, 1, 999, 1000, 1_791_234_567_890] {
            assert_eq!(
                parse_rfc3339_millis(&format_unix_millis(millis)),
                Some(millis)
            );
        }
        assert_eq!(parse_rfc3339_millis("1970-01-01T00:00:00.1Z"), Some(100));
        for invalid in [
            "1970-01-01T00:00:00Zjunk",
            "1970-01-01T00:00:00.1234Z",
            "1969-01-01T00:00:00Z",
        ] {
            assert_eq!(parse_rfc3339_millis(invalid), None);
        }
    }

    #[test]
    fn utc_timestamps_validate_calendar_and_entire_suffix() {
        for value in [
            "2024-02-29T23:59:59Z",
            "2000-02-29T00:00:00Z",
            "2026-04-30T00:00:00.123Z",
        ] {
            assert!(parse_rfc3339_millis(value).is_some(), "{value}");
        }
        for value in [
            "2023-02-29T00:00:00Z",
            "1900-02-29T00:00:00Z",
            "2026-04-31T00:00:00Z",
            "2026-02-30T00:00:00Z",
            "2026-01-01T24:00:00Z",
            "2026-01-01T00:60:00Z",
            "2026-01-01T00:00:60Z",
            "2026-01-01T00:00:00Zjunk",
            "2026-01-01T00:00:00.Z",
            "2026-01-01T00:00:00.abcZ",
            "2026-01-01 00:00:00Z",
            "2026-01-01T00:00:00+bad",
        ] {
            assert_eq!(parse_rfc3339_seconds(value), None, "{value}");
            assert_eq!(parse_rfc3339_millis(value), None, "{value}");
        }
        for day in 1..=366 {
            let millis = (1_704_067_200 + (day - 1) * 86_400) * 1000;
            assert_eq!(
                parse_rfc3339_millis(&format_unix_millis(millis)),
                Some(millis)
            );
        }
    }
}
