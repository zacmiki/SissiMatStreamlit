import unittest
from datetime import datetime, timezone

from sissimat.instruments.lakeshore331 import (
    LakeShore331,
    TemperatureReading,
)
from sissimat.pages.temperature_reader import _temperature_table_row


class FakeSerial:
    def __init__(self, responses, **kwargs):
        self.responses = list(responses)
        self.kwargs = kwargs
        self.writes = []
        self.is_open = True

    def reset_input_buffer(self):
        pass

    def write(self, payload):
        self.writes.append(payload)

    def flush(self):
        pass

    def read_until(self, _terminator):
        return self.responses.pop(0)

    def close(self):
        self.is_open = False


class LakeShore331Tests(unittest.TestCase):
    def test_identification_and_temperature_queries(self):
        fake = FakeSerial(
            [b"LSCI,MODEL331S,334187,061404\r\n", b"+123.45\r\n", b"+234.56\r\n"]
        )
        controller = LakeShore331(
            "COM4", query_delay=0, serial_factory=lambda **kwargs: fake
        )

        self.assertEqual(controller.identify(), "LSCI,MODEL331S,334187,061404")
        reading = controller.read_temperatures()

        self.assertEqual(reading.channel_a_kelvin, 123.45)
        self.assertEqual(reading.channel_b_kelvin, 234.56)
        self.assertEqual(fake.writes, [b"*IDN?\r\n", b"KRDG? A\r\n", b"KRDG? B\r\n"])

    def test_temperature_table_row_uses_selected_channel(self):
        reading = TemperatureReading(
            datetime(2026, 10, 26, 12, 30, tzinfo=timezone.utc), 100.0, 200.0
        )
        row_a = _temperature_table_row(reading, 1, "A")
        row_b = _temperature_table_row(reading, 2, "B")

        self.assertEqual(
            row_a,
            {
                "Point n": 1,
                "Temperature": 100.0,
                "Time": "2026-10-26T12:30:00+00:00",
            },
        )
        self.assertEqual(row_b["Point n"], 2)
        self.assertEqual(row_b["Temperature"], 200.0)


if __name__ == "__main__":
    unittest.main()
